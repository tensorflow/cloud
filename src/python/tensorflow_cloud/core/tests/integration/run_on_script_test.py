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
import mock

import tensorflow as tf
import tensorflow_cloud as tfc
from tensorflow_cloud.utils import google_api_client

# The staging bucket to use for cloudbuild as well as save the model and data.
_TEST_BUCKET = os.environ["TEST_BUCKET"]
_PROJECT_ID = os.environ["PROJECT_ID"]


class RunOnScriptTest(tf.test.TestCase):

    def setUp(self):
        super(RunOnScriptTest, self).setUp()
        # To keep track of content that needs to be deleted in teardown clean up
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../testdata/"
        )

    def tearDown(self):
        mock.patch.stopall()
        super(RunOnScriptTest, self).tearDown()

    def auto_mirrored_strategy(self):
        return tfc.run(
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

    def auto_tpu_strategy(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements_tpu_strategy.txt"),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
            worker_count=1,
            worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"],
            docker_config=tfc.DockerConfig(
                base_image="tensorflow/tensorflow:2.1.0"),
        )

    def auto_one_device_strategy(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
        )

    def auto_one_device_strategy_cloud_build(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_config=tfc.DockerConfig(image_build_bucket=_TEST_BUCKET),
        )

    def auto_multi_worker_strategy(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            worker_count=1,
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
        )

    def none_dist_strat_multi_worker_strategy(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_ctl.py"),
            distribution_strategy=None,
            worker_count=2,
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
        )

    def auto_dist_strat_mwms_custom_img(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            distribution_strategy="auto",
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_config=tfc.DockerConfig(
                base_image="gcr.io/deeplearning-platform-release"
                           "/tf2-gpu.2-2:latest"),
        )

    def auto_one_device_job_labels(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.py"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            job_labels={"job": "on_script_tests", "team": "keras"},
        )

    def test_run_on_script(self):
        track_status = {
            "auto_mirrored_strategy": self.auto_mirrored_strategy(),
            "auto_tpu_strategy": self.auto_tpu_strategy(),
            "auto_one_device_strategy": self.auto_one_device_strategy(),
            "auto_one_device_strategy_cloud_build":
                self.auto_one_device_strategy_cloud_build(),
            "auto_multi_worker_strategy": self.auto_multi_worker_strategy(),
            "none_dist_strat_multi_worker_strategy":
                self.none_dist_strat_multi_worker_strategy(),
            "auto_dist_strat_mwms_custom_img":
                self.auto_dist_strat_mwms_custom_img(),
            "auto_one_device_job_labels": self.auto_one_device_job_labels(),
        }

        for test_name, job_id in track_status.items():
            self.assertTrue(
                google_api_client.wait_for_api_training_job_completion(
                    job_id, _PROJECT_ID),
                "Job {} generated from the test: {} has failed".format(
                    job_id, test_name))


if __name__ == "__main__":
    tf.test.main()
