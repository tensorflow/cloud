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
"""Integration tests for calling tfc.run on a notebook with keras."""

import os
import mock

import tensorflow as tf
import tensorflow_cloud as tfc
from tensorflow_cloud.utils import google_api_client

# The staging bucket to use for cloudbuild as well as save the model and data.
_TEST_BUCKET = os.environ["TEST_BUCKET"]
_PROJECT_ID = os.environ["PROJECT_ID"]


# Note: Using K80 GPUs for all but one test case and the TPU test case as it
# provisions faster for integration testing purposes.
class RunOnNotebookTest(tf.test.TestCase):

    def setUp(self):
        super(RunOnNotebookTest, self).setUp()
        # To keep track of content that needs to be deleted in teardown clean up
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../testdata/"
        )

    def tearDown(self):
        mock.patch.stopall()
        super(RunOnNotebookTest, self).tearDown()

    def auto_mirrored_strategy(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            chief_config=tfc.MachineConfig(
                cpu_cores=8,
                memory=30,
                accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_K80,
                accelerator_count=2,
            ),
            job_labels={
                "job": "auto_mirrored_strategy",
                "team": "on_notebook_tests",
            },
        )

    def auto_tpu_strategy(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements_tpu_strategy.txt"),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
            worker_count=1,
            worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"],
            docker_config=tfc.DockerConfig(
                parent_image="tensorflow/tensorflow:2.1.0"),
            job_labels={
                "job": "auto_tpu_strategy",
                "team": "on_notebook_tests",
            },
        )

    def auto_one_device_strategy(self):
        # Using the default T4 GPU for this test.
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            job_labels={
                "job": "auto_one_device_strategy",
                "team": "on_notebook_tests",
            },
        )

    def auto_multi_worker_strategy(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            worker_count=1,
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            worker_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            job_labels={
                "job": "auto_multi_worker_strategy",
                "team": "on_notebook_tests",
            },
        )

    def docker_config_cloud_build(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_config=tfc.DockerConfig(image_build_bucket=_TEST_BUCKET),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            job_labels={
                "job": "docker_config_cloud_build",
                "team": "on_notebook_tests",
            },
        )

    def docker_config_parent_img(self):
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_config=tfc.DockerConfig(
                parent_image="gcr.io/deeplearning-platform-release"
                             "/tf2-gpu.2-2:latest"),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            job_labels={
                "job": "docker_config_parent_img",
                "team": "on_notebook_tests",
            },
        )

    def docker_config_image(self):
        ret_val = tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            job_labels={
                "job": "docker_config_image",
                "team": "on_notebook_tests",
            },
        )
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_config=tfc.DockerConfig(image=ret_val["docker_image"]),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            job_labels={
                "job": "docker_config_image",
                "team": "on_notebook_tests",
            },
        )

    def docker_config_cache_from(self):
        ret_val = tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_config=tfc.DockerConfig(image_build_bucket=_TEST_BUCKET),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            job_labels={
                "job": "docker_config_cache_from",
                "team": "on_notebook_tests",
            },
        )
        return tfc.run(
            entry_point=os.path.join(self.test_data_path,
                                     "mnist_example_using_fit.ipynb"),
            requirements_txt=os.path.join(self.test_data_path,
                                          "requirements.txt"),
            docker_config=tfc.DockerConfig(
                image_build_bucket=_TEST_BUCKET,
                image=ret_val["docker_image"],
                cache_from=ret_val["docker_image"]),
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            job_labels={
                "job": "docker_config_cache_from",
                "team": "on_notebook_tests",
            },
        )

    def test_run_on_notebook(self):
        track_status = {
            "auto_mirrored_strategy": self.auto_mirrored_strategy(),
            # TODO(b/178123173) Enable tests after b/193022465 is resolved.
            # "auto_tpu_strategy": self.auto_tpu_strategy(),
            "auto_one_device_strategy": self.auto_one_device_strategy(),
            "auto_multi_worker_strategy": self.auto_multi_worker_strategy(),
            "docker_config_cloud_build": self.docker_config_cloud_build(),
            "docker_config_parent_img": self.docker_config_parent_img(),
            "docker_config_image": self.docker_config_image(),
            "docker_config_cache_from": self.docker_config_cache_from(),
        }

        for test_name, ret_val in track_status.items():
            self.assertTrue(
                google_api_client.wait_for_aip_training_job_completion(
                    ret_val["job_id"], _PROJECT_ID),
                "Job {} generated from the test: {} has failed".format(
                    ret_val["job_id"], test_name))


if __name__ == "__main__":
    tf.test.main()
