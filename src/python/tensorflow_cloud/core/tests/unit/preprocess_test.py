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
"""Tests for the cloud preprocessing module."""

import os

from absl.testing import absltest
import mock

from tensorflow_cloud.core import machine_config
from tensorflow_cloud.core import preprocess


class TestPreprocess(absltest.TestCase):

    def setup_py(self):
        self.entry_point_name = "sample_compile_fit.py"
        self.entry_point = "sample_compile_fit.py"

    def setup_ipython(self):
        self.entry_point_name = "mnist_example_using_fit.ipynb"
        self.entry_point = "mnist_example_using_fit.ipynb"

    def get_preprocessed_entry_point(
        self,
        chief_config=machine_config.COMMON_MACHINE_CONFIGS["CPU"],
        worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
        worker_count=0,
        distribution_strategy="auto",
        called_from_notebook=False,
    ):
        self.wrapped_entry_point = preprocess.get_preprocessed_entry_point(
            self.entry_point,
            chief_config,
            worker_config,
            worker_count,
            distribution_strategy,
            called_from_notebook,
        )

        with open(self.wrapped_entry_point, "r") as f:
            script_lines = f.readlines()
        return script_lines

    def assert_and_cleanup(self, expected_lines, script_lines):
        self.assertListEqual(expected_lines, script_lines)
        os.remove(self.wrapped_entry_point)

    def test_auto_one_device_strategy(self):
        self.setup_py()
        script_lines = self.get_preprocessed_entry_point()
        expected_lines = [
            "import os\n",
            "import tensorflow as tf\n",
            'os.environ["TF_KERAS_RUNNING_REMOTELY"]="1"\n',
            "import sys\n",
            "for flag in sys.argv[1:]:\n",
            '  if flag.startswith("TUNER_ID"):\n',
            '    os.environ["KERASTUNER_TUNER_ID"]=flag\n',
            "strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')\n",
            "tf.distribute.experimental_set_strategy(strategy)\n",
            'exec(open("{}").read())\n'.format(self.entry_point_name),
        ]
        self.assert_and_cleanup(expected_lines, script_lines)

    def test_auto_mirrored_strategy(self):
        self.setup_py()
        chief_config = machine_config.COMMON_MACHINE_CONFIGS["K80_4X"]
        script_lines = self.get_preprocessed_entry_point(
            chief_config=chief_config)
        expected_lines = [
            "import os\n",
            "import tensorflow as tf\n",
            'os.environ["TF_KERAS_RUNNING_REMOTELY"]="1"\n',
            "import sys\n",
            "for flag in sys.argv[1:]:\n",
            '  if flag.startswith("TUNER_ID"):\n',
            '    os.environ["KERASTUNER_TUNER_ID"]=flag\n',
            "strategy = tf.distribute.MirroredStrategy()\n",
            "tf.distribute.experimental_set_strategy(strategy)\n",
            'exec(open("{}").read())\n'.format(self.entry_point_name),
        ]
        self.assert_and_cleanup(expected_lines, script_lines)

    def test_auto_multi_worker_strategy(self):
        self.setup_py()
        script_lines = self.get_preprocessed_entry_point(worker_count=2)
        expected_lines = [
            "import os\n",
            "import tensorflow as tf\n",
            'os.environ["TF_KERAS_RUNNING_REMOTELY"]="1"\n',
            "import sys\n",
            "for flag in sys.argv[1:]:\n",
            '  if flag.startswith("TUNER_ID"):\n',
            '    os.environ["KERASTUNER_TUNER_ID"]=flag\n',
            ("strategy = tf.distribute.experimental."
             "MultiWorkerMirroredStrategy()\n"),
            "tf.distribute.experimental_set_strategy(strategy)\n",
            'exec(open("{}").read())\n'.format(self.entry_point_name),
        ]
        self.assert_and_cleanup(expected_lines, script_lines)

    def test_auto_tpu_strategy(self):
        self.setup_py()
        worker_config = machine_config.COMMON_MACHINE_CONFIGS["TPU"]
        script_lines = self.get_preprocessed_entry_point(
            worker_config=worker_config, worker_count=1
        )
        expected_lines = [
            "import os\n",
            "import tensorflow as tf\n",
            'os.environ["TF_KERAS_RUNNING_REMOTELY"]="1"\n',
            "import sys\n",
            "for flag in sys.argv[1:]:\n",
            '  if flag.startswith("TUNER_ID"):\n',
            '    os.environ["KERASTUNER_TUNER_ID"]=flag\n',
            "import json\n",
            "import logging\n",
            "import time\n",
            "logger = logging.getLogger(__name__)\n",
            "logging.basicConfig(level=logging.INFO)\n",
            "def wait_for_tpu_cluster_resolver_ready():\n",
            "  tpu_config_env = os.environ.get('TPU_CONFIG')\n",
            "  if not tpu_config_env:\n",
            ("    logging.info('Missing TPU_CONFIG, "
             "use CPU/GPU for training.')\n"),
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
            "        logging.info('Still waiting for provisioning of TPU VM"
            " instance.')\n",
            "      else:\n",
            "        # Preserves the traceback.\n",
            ("        raise RuntimeError('Failed to schedule TPU: "
             "{}'.format(e))\n"),
            "    time.sleep(10)\n",
            "  raise RuntimeError('Failed to schedule TPU.')\n",
            "resolver = wait_for_tpu_cluster_resolver_ready()\n",
            "tf.config.experimental_connect_to_cluster(resolver)\n",
            "tf.tpu.experimental.initialize_tpu_system(resolver)\n",
            "strategy = tf.distribute.experimental.TPUStrategy(resolver)\n",
            "tf.distribute.experimental_set_strategy(strategy)\n",
            'exec(open("{}").read())\n'.format(self.entry_point_name),
        ]
        self.assert_and_cleanup(expected_lines, script_lines)

    @mock.patch("tensorflow_cloud.core.preprocess.PythonExporter")  # pylint: disable=line-too-long
    def test_ipython_notebook(self, mock_python_exporter):
        file_contents = (
            "num_train_examples = info.splits['train'].num_examples\n"
            "eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)\n"
        )
        mock_python_exporter.return_value.from_filename.return_value = (
            file_contents,
            None,
        )
        self.setup_ipython()
        script_lines = self.get_preprocessed_entry_point()
        expected_lines = [
            "import os\n",
            "import tensorflow as tf\n",
            'os.environ["TF_KERAS_RUNNING_REMOTELY"]="1"\n',
            "import sys\n",
            "for flag in sys.argv[1:]:\n",
            '  if flag.startswith("TUNER_ID"):\n',
            '    os.environ["KERASTUNER_TUNER_ID"]=flag\n',
            "strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')\n",
            "tf.distribute.experimental_set_strategy(strategy)\n",
        ]
        for el in expected_lines:
            self.assertIn(el, script_lines)
        self.assertIn(
            "num_train_examples = info.splits['train'].num_examples\n",
            script_lines
        )
        self.assertIn(
            "eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)\n",
            script_lines
        )


if __name__ == "__main__":
    absltest.main()
