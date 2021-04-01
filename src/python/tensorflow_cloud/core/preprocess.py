# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module that makes the `entry_point` distribution ready."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import logging
import os
import sys
import tempfile

from . import machine_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from nbconvert import PythonExporter  # pylint: disable=g-import-not-at-top
except ImportError:
    PythonExporter = None

try:
    # Available in a colab environment.
    from google.colab import _message  # pylint: disable=g-import-not-at-top
except ImportError:
    _message = None


def get_preprocessed_entry_point(
    entry_point,
    chief_config,
    worker_config,
    worker_count,
    distribution_strategy,
    called_from_notebook=False,
    return_file_descriptor=False
):
    """Creates python script for distribution based on the given `entry_point`.

    This utility creates a new python script called `preprocessed_entry_point`
    based on the given `entry_point` and `distribution_strategy` inputs. This
    script will become the new Docker entry point python program.

    1. If `entry_point` is a python file name and `distribution_strategy` is
    auto, then `preprocessed_entry_point` will have the user given
    `entry_point` code wrapped in a Tensorflow distribution strategy.

    2. If `entry_point` is None and `run` is invoked inside of a python script,
    then `preprocessed_entry_point` will be this python script (sys.args[0]).

    3. If `entry_point` is an `ipynb` file, then `preprocessed_entry_point`
    will be the code from the notebook. This utility uses `nbconvert`
    to get the code from notebook.

    4. If `entry_point` is None and `run` is invoked inside of an `ipynb`
    notebook, then `preprocessed_entry_point` will be the code from the
    notebook. This urility uses `google.colab` client API to fetch the code.

    For cases 2, 3 & 4, if `distribution_strategy` is auto, then this script
    will be wrapped in a Tensorflow distribution strategy.

    The distribution strategy instance created is based on the machine
    configurations provided using the `chief_config`, `worker_count` params.
    - If the number of workers > 0,
        - If accelerator type is TPU, we will create an instance of
        `tf.distribute.experimental.TPUStrategy`.
        - Otherwise, we will create a default instance of
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
    - If number of GPUs > 0, we will create a default instance of
        `tf.distribute.MirroredStrategy`
    - Otherwise, we will use `tf.distribute.OneDeviceStrategy`

    Args:
        entry_point: Optional string. File path to the python file or iPython
            notebook that contains the TensorFlow code.
            Note) This path must be in the current working directory tree.
            Example) 'train.py', 'training/mnist.py', 'mnist.ipynb'
            If `entry_point` is not provided, then
            - If you are in an iPython notebook environment, then the
                current notebook is taken as the `entry_point`.
            - Otherwise, the current python script is taken as the
                `entry_point`.
        chief_config: `MachineConfig` that represents the configuration
            for the chief worker in a distribution cluster.
        worker_config: `MachineConfig` that represents the configuration
            for the workers in a distribution cluster.
        worker_count: Integer that represents the number of general workers
            in a distribution cluster. This count does not include the chief
            worker.
        distribution_strategy: 'auto' or None. Defaults to 'auto'.
            'auto' means we will take care of creating a Tensorflow
            distribution strategy instance based on the machine configurations
            provided using the `chief_config`, `worker_config` and
            `worker_count` params.
        called_from_notebook: Boolean. True if the API is run in a
            notebook environment.
        return_file_descriptor: Boolean. True if the file descriptor for the
            temporary file is also returned.

    Returns:
        The `preprocessed_entry_point` file path.

    Raises:
        RuntimeError: If invoked from Notebook but unable to access it.
            Typically, this is due to missing the `nbconvert` package.
    """

    # Set `TF_KERAS_RUNNING_REMOTELY` env variable. This is required in order
    # to prevent running `tfc.run` if we are already in a cloud environment.
    # This is applicable only when `entry_point` is None.
    script_lines = [
        "import os\n",
        "import tensorflow as tf\n",
        'os.environ["TF_KERAS_RUNNING_REMOTELY"]="1"\n',
    ]

    # Setting default Tuner_ID if one is provided in args
    script_lines.extend([
        "import sys\n",
        "for flag in sys.argv[1:]:\n",
        '  if flag.startswith("TUNER_ID"):\n',
        '    os.environ["KERASTUNER_TUNER_ID"]=flag\n',
    ])

    # Auto wrap in distribution strategy.
    if distribution_strategy == "auto":
        if worker_count > 0:
            if machine_config.is_tpu_config(worker_config):
                strategy = get_tpu_cluster_resolver_fn()
                strategy.extend(
                    [
                        "resolver = wait_for_tpu_cluster_resolver_ready()\n",
                        "tf.config.experimental_connect_to_cluster(resolver)\n",
                        "tf.tpu.experimental.initialize_tpu_system(resolver)\n",
                        "strategy = tf.distribute.experimental.TPUStrategy("
                        "resolver)\n",
                    ]
                )
            else:
                strategy = [
                    "strategy = tf.distribute.experimental."
                    "MultiWorkerMirroredStrategy()\n"
                ]
        elif chief_config.accelerator_count > 1:
            strategy = ["strategy = tf.distribute.MirroredStrategy()\n"]
        else:
            strategy = [
                "strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')\n"]
        script_lines.extend(strategy)
        script_lines.append(
            "tf.distribute.experimental_set_strategy(strategy)\n")

    # If `entry_point` is not provided, detect if we are in a notebook
    # or a python script. Fetch the `entry_point`.
    if entry_point is None and not called_from_notebook:
        # Current python script is assumed to be the entry_point.
        entry_point = sys.argv[0]

    # Add user's code.
    if entry_point is not None and entry_point.endswith("py"):
        # We are using exec here to execute the user code object.
        # This will support use case where the user's program has a
        # main method.
        _, entry_point_file_name = os.path.split(entry_point)
        script_lines.append(
            'exec(open("{}").read())\n'.format(entry_point_file_name))
    else:
        if called_from_notebook:
            # Kaggle integration
            if os.getenv("KAGGLE_CONTAINER_NAME"):
                logger.info("Preprocessing Kaggle notebook...")
                py_content = _get_kaggle_notebook_content()
            else:
                # Colab integration
                py_content = _get_colab_notebook_content()
        else:
            if PythonExporter is None:
                raise RuntimeError(
                    "Unable to access iPython notebook. "
                    "Please make sure you have installed `nbconvert` package."
                )

            # Get the python code from the iPython notebook.
            (py_content, _) = PythonExporter().from_filename(entry_point)
            py_content = py_content.splitlines(keepends=True)

        # Remove any iPython special commands and add the python code
        # to script_lines.
        for line in py_content:
            if not (
                line.startswith("!") or
                line.startswith("%") or
                line.startswith("#")
            ):
                script_lines.append(line)

    # Create a tmp wrapped entry point script file.
    file_descriptor, output_file = tempfile.mkstemp(suffix=".py")
    with open(output_file, "w") as f:
        f.writelines(script_lines)

    # Returning file descriptor could be necessary for some os.close calls
    if return_file_descriptor:
      return (output_file, file_descriptor)
    else:
      return output_file


def _get_colab_notebook_content():
    """Returns the colab notebook python code contents."""
    response = _message.blocking_request("get_ipynb",
                                         request="",
                                         timeout_sec=200)
    if response is None:
        raise RuntimeError("Unable to get the notebook contents.")
    cells = response["ipynb"]["cells"]
    py_content = []
    for cell in cells:
        if cell["cell_type"] == "code":
            # Add newline char to the last line of a code cell.
            cell["source"][-1] += "\n"

            # Combine all code cells.
            py_content.extend(cell["source"])
    return py_content


def _get_kaggle_notebook_content():
    """Returns the kaggle notebook python code contents."""
    if PythonExporter is None:
        raise RuntimeError(
            # This should never occur.
            # `nbconvert` is always installed on Kaggle.
            "Please make sure you have installed `nbconvert` package."
        )
    from kaggle_session import UserSessionClient  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    kaggle_session_client = UserSessionClient()
    try:
        response = kaggle_session_client.get_exportable_ipynb()
        ipynb_stream = io.StringIO(response["source"])
        py_content, _ = PythonExporter().from_file(ipynb_stream)
        return py_content.splitlines(keepends=True)
    except:
        raise RuntimeError("Unable to get the notebook contents.")


def get_tpu_cluster_resolver_fn():
    """Returns the fn required for runnning custom container on cloud TPUs.

    This function is added to the user code in the custom container before
    running it on the cloud. With this function, we wait for the TPU to be
    provisioned before calling TpuClusterResolver.

    https://cloud.devsite.corp.google.com/ai-platform/training/docs/
    using-tpus#custom-containers
    """
    return [
        "import json\n",
        "import logging\n",
        "import time\n",
        "logger = logging.getLogger(__name__)\n",
        "logging.basicConfig(level=logging.INFO)\n",
        "def wait_for_tpu_cluster_resolver_ready():\n",
        "  tpu_config_env = os.environ.get('TPU_CONFIG')\n",
        "  if not tpu_config_env:\n",
        "    logging.info('Missing TPU_CONFIG, use CPU/GPU for training.')\n",
        "    return None\n",
        "  tpu_node = json.loads(tpu_config_env)\n",
        "  logging.info('Waiting for TPU to be ready: %s.', tpu_node)\n",
        "  num_retries = 40\n",
        "  for i in range(num_retries):\n",
        "    try:\n",
        "      tpu_cluster_resolver = (\n",
        "          tf.distribute.cluster_resolver.TPUClusterResolver(\n",
        "              tpu=[tpu_node['tpu_node_name']],\n",
        "              zone=tpu_node['zone'],\n",
        "              project=tpu_node['project'],\n",
        "              job_name='worker'))\n",
        "      tpu_cluster_resolver_dict = "
        "tpu_cluster_resolver.cluster_spec().as_dict()\n",
        "      if 'worker' in tpu_cluster_resolver_dict:\n",
        ("        logging.info('Found TPU worker: %s', "
         "tpu_cluster_resolver_dict)\n"),
        "        return tpu_cluster_resolver\n",
        "    except Exception as e:\n",
        "      if i < num_retries - 1:\n",
        ("        logging.info('Still waiting for provisioning of TPU VM "
         "instance.')\n"),
        "      else:\n",
        "        # Preserves the traceback.\n",
        "        raise RuntimeError('Failed to schedule TPU: {}'.format(e))\n",
        "    time.sleep(10)\n",
        "  raise RuntimeError('Failed to schedule TPU.')\n",
    ]
