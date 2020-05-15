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

import os
import sys
import tempfile

from .machine_config import AcceleratorType

try:
    from nbconvert import PythonExporter
except ImportError:
    PythonExporter = None

try:
    # Available in a colab environment.
    from google.colab import _message
except ImportError:
    _message = None


def get_preprocessed_entry_point(entry_point,
                                 chief_config,
                                 worker_count,
                                 distribution_strategy,
                                 called_from_notebook=False):
    """Creates python script for distribution based on the given `entry_point`.

    This utility creates a new python script called `preprocessed_entry_point`
    based on the given `entry_point` and `distribution_strategy` inputs. This
    script will become the new docker entry point python program.

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
    - If the number of workers > 0, we will create a default instance of
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
    - If number of GPUs > 0, we will create a default instance of
        `tf.distribute.MirroredStrategy`
    - Otherwise, we will use `tf.distribute.OneDeviceStrategy`

    Args:
        entry_point: Optional string. File path to the python file or iPython
            notebook that contains the TensorFlow code.
            Note: This path must be in the current working directory tree.
            Example: 'train.py', 'training/mnist.py', 'mnist.ipynb'
            If `entry_point` is not provided, then
            - If you are in an iPython notebook environment, then the
                current notebook is taken as the `entry_point`.
            - Otherwise, the current python script is taken as the
                `entry_point`.
        chief_config: `MachineConfig` that represents the configuration
            for the chief worker in a distribution cluster.
        worker_config: `MachineConfig` that represents the configuration
            for the workers in a distribution cluster.
        distribution_strategy: 'auto' or None. Defaults to 'auto'.
            'auto' means we will take care of creating a Tensorflow
            distribution strategy instance based on the machine configurations
            provided using the `chief_config`, `worker_config` and
            `worker_count` params.
        called_from_notebook: Boolean. True if the API is run in a
            notebook environment.

    Returns:
        The `preprocessed_entry_point` file path.
    """

    # Set `TF_KERAS_RUNNING_REMOTELY` env variable. This is required in order
    # to prevent running `tfc.run` if we are already in a cloud environment.
    # This is applicable only when `entry_point` is None.
    script_lines = [
        'import os\n',
        'import tensorflow as tf\n',
        'os.environ["TF_KERAS_RUNNING_REMOTELY"]="1"\n'
    ]

    # Auto wrap in distribution strategy.
    if ((chief_config.accelerator_type != AcceleratorType.NO_ACCELERATOR) and
            distribution_strategy == 'auto'):
        if worker_count > 0:
            strategy = (
                'strategy = tf.distribute.experimental.'
                'MultiWorkerMirroredStrategy()\n')
        elif chief_config.accelerator_count > 1:
            strategy = 'strategy = tf.distribute.MirroredStrategy()\n'
        else:
            strategy = (
                'strategy = tf.distribute.OneDeviceStrategy('
                'device="/gpu:0")\n')
        script_lines.extend([
            strategy,
            'tf.distribute.experimental_set_strategy(strategy)\n',
        ])

    # If `entry_point` is not provided, detect if we are in a notebook
    # or a python script. Fetch the `entry_point`.
    if entry_point is None and not called_from_notebook:
        # Current python script is assumed to be the entry_point.
        entry_point = sys.argv[0]

    # Add user's code.
    if entry_point is not None and entry_point.endswith('py'):
        # We are using exec here to execute the user code object.
        # This will support use case where the user's program has a
        # main method.
        _, entry_point_file_name = os.path.split(entry_point)
        script_lines.append(
            'exec(open("{}").read())\n'.format(entry_point_file_name))
    else:
        if called_from_notebook:
            py_content = _get_colab_notebook_content()
        else:
            if PythonExporter is None:
                raise RuntimeError(
                    'Unable to access iPython notebook. '
                    'Please make sure you have installed `nbconvert` package.')

            # Get the python code from the iPython notebook.
            (py_content, _) = PythonExporter().from_filename(entry_point)
            py_content = py_content.splitlines(keepends=True)

        # Remove any iPython special commands and add the python code
        # to script_lines.
        for line in py_content:
            if not (line.startswith('!') or line.startswith('%') or
                    line.startswith('#')):
                script_lines.append(line)

    # Create a tmp wrapped entry point script file.
    _, output_file = tempfile.mkstemp(suffix='.py')
    with open(output_file, 'w') as f:
        f.writelines(script_lines)
    return output_file


def _get_colab_notebook_content():
    """Returns the colab notebook python code contents."""
    response = _message.blocking_request(
        'get_ipynb', request='', timeout_sec=200)
    if response is None:
        raise RuntimeError('Unable to get the notebook contents.')
    cells = response['ipynb']['cells']
    py_content = []
    for cell in cells:
        if cell['cell_type'] == 'code':
            # Add newline char to the last line of a code cell.
            cell['source'][-1] += '\n'

            # Combine all code cells.
            py_content.extend(cell['source'])
    return py_content
