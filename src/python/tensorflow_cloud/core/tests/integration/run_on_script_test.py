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
import sys
from typing import Text
from unittest import mock
import tensorflow as tf
import tensorflow_cloud as tfc

# Following are the env variables available in test infrastructure:
#
# The staging bucket to use for cloudbuild as well as save the model and data.
# TEST_BUCKET = os.environ['TEST_BUCKET']
#
# The project id to use to run tests.
# PROJECT_ID = os.environ['PROJECT_ID']
#
# The GCP region in which the end-to-end test is run.
# REGION = os.environ['REGION']
#
# Unique ID for this build, can be used as a label for an AI Platform training job.
# BUILD_ID = os.environ['BUILD_ID']


class RunOnScriptTest(tf.test.TestCase):
    def setUp(self):
        super(RunOnScriptTest, self).setUp()
        # To keep track of content that needs to be deleted in teardown clean up
        self.test_folders = []
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../testdata/"
        )

    def tearDown(self):
        super(RunOnScriptTest, self).tearDown()
        # Clean up any temporary file or folder created during testing.
        for folder in self.test_folders:
            self.delete_dir(folder)

    def delete_dir(self, path: Text) -> None:
        """Deletes a directory if exists."""
        if tf.io.gfile.isdir(path):
            tf.io.gfile.rmtree(path)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_auto_mirrored_strategy(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
            chief_config=tfc.COMMON_MACHINE_CONFIGS['T4_2X'],
        )
        mock_exit.assert_called_once_with(0)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_auto_one_device_strategy(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
        )
        mock_exit.assert_called_once_with(0)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_auto_one_device_strategy_bucket_build(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
            docker_image_bucket_name="TEST_BUCKET",
        )
        mock_exit.assert_called_once_with(0)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_auto_multi_worker_strategy(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_fit.py"),
            worker_count=1,
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
        )
        mock_exit.assert_called_once_with(0)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_none_dist_strat_multi_worker_strategy(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_ctl.py"),
            distribution_strategy=None,
            worker_count=2,
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
        )
        mock_exit.assert_called_once_with(0)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_none_dist_strat_multi_worker_strategy_bucket_build(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_ctl.py"),
            worker_count=2,
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
            docker_image_bucket_name="TEST_BUCKET",
        )
        mock_exit.assert_called_once_with(0)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_auto_tpu(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_fit.py"),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
            worker_count=1,
            worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"],
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
        )
        mock_exit.assert_called_once_with(0)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_auto_one_device_stream_logs(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
            stream_logs=True,
        )
        mock_exit.assert_called_once_with(0)

    @mock.patch.object(sys, "exit", autospec=True)
    def test_auto_one_device_job_labels(self, mock_exit):
        tfc.run(
            entry_point=os.path.join(self.test_data_path, "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path, "requirements.txt"),
            job_labels={"job": "on_script_tests", "team": "keras"},
        )
        mock_exit.assert_called_once_with(0)

if __name__ == "__main__":
    tf.test.main()
