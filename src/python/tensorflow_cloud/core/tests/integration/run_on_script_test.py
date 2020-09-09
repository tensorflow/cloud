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
"""Integration tests for calling tfc.run on a script with keras."""

import os
from typing import Text
import mock

import tensorflow as tf
import tensorflow_cloud as tfc

# The staging bucket to use for cloudbuild as well as save the model and data.
_TEST_BUCKET = os.environ["TEST_BUCKET"]


class RunOnScriptTest(tf.test.TestCase):

    def setUp(self):
        super(RunOnScriptTest, self).setUp()
        # To keep track of content that needs to be deleted in teardown clean up
        self.test_folders = []
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../testdata/"
        )

        self._mock_sys_exit = mock.patch("sys.exit", autospec=True).start()

    def tearDown(self):
        super(RunOnScriptTest, self).tearDown()
        # Clean up any temporary file or folder created during testing.
        for folder in self.test_folders:
            self.delete_dir(folder)

    def delete_dir(self, path: Text) -> None:
        """Deletes a directory if exists."""
        if tf.io.gfile.isdir(path):
            tf.io.gfile.rmtree(path)

    def test_auto_mirrored_strategy(self):
        tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            chief_config=tfc.MachineConfig(
                cpu_cores=8,
                memory=30,
                accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
                accelerator_count=2,
            ),
        )
        self._mock_sys_exit.assert_called_once_with(0)

    def test_auto_tpu_strategy(self):
        tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
            worker_count=1,
            worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"],
            docker_base_image="tensorflow/tensorflow:2.1.0",
        )
        self._mock_sys_exit.assert_called_once_with(0)

    def test_auto_one_device_strategy(self):
        tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
        )
        self._mock_sys_exit.assert_called_once_with(0)

    def test_auto_one_device_strategy_bucket_build(self):
        tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_image_bucket_name=_TEST_BUCKET,
        )
        self._mock_sys_exit.assert_called_once_with(0)

    def test_auto_multi_worker_strategy(self):
        tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            worker_count=1,
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
        )
        self._mock_sys_exit.assert_called_once_with(0)

    def test_none_dist_strat_multi_worker_strategy(self):
        tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_ctl.py"),
            distribution_strategy=None,
            worker_count=2,
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
        )
        self._mock_sys_exit.assert_called_once_with(0)

    def test_none_dist_strat_mwms_custom_img(self):
        tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            distribution_strategy="auto",
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_base_image=(
                "gcr.io/deeplearning-platform-release"
                "/tf2-gpu.2-2:latest"),
        )
        self._mock_sys_exit.assert_called_once_with(0)

    def test_auto_one_device_job_labels(self):
        tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            job_labels={"job": "on_script_tests", "team": "keras"},
        )
        self._mock_sys_exit.assert_called_once_with(0)


if __name__ == "__main__":
    tf.test.main()
