# Lint as: python3
# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Integration tests for calling run_experiment_cloud."""

import os
import uuid

import tensorflow as tf
import tensorflow_cloud as tfc
from tensorflow_cloud.core.experimental import models
from tensorflow_cloud.utils import google_api_client
from official.core import task_factory
from official.utils.testing import mock_task

# The staging bucket to use for cloudbuild as well as save the model and data.
_TEST_BUCKET = os.environ["TEST_BUCKET"]
_PROJECT_ID = os.environ["PROJECT_ID"]
_PARENT_IMAGE = "gcr.io/deeplearning-platform-release/tf2-gpu.2-5"
_BASE_PATH = f"gs://{_TEST_BUCKET}/{uuid.uuid4()}"


class RunExperimentCloudTest(tf.test.TestCase):

    def setUp(self):
        super(RunExperimentCloudTest, self).setUp()
        test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../testdata/"
        )
        self.requirements_txt = os.path.join(test_data_path,
                                             "requirements.txt")

        test_config = {
            "trainer": {
                "checkpoint_interval": 10,
                "steps_per_loop": 10,
                "summary_interval": 10,
                "train_steps": 10,
                "validation_steps": 5,
                "validation_interval": 10,
                "continuous_eval_timeout": 1,
                "validation_summary_subdir": "validation",
                "optimizer_config": {
                    "optimizer": {
                        "type": "sgd",
                    },
                    "learning_rate": {
                        "type": "constant"
                    }
                }
            },
        }

        params = mock_task.mock_experiment()
        params.override(test_config, is_strict=False)
        self.run_experiment_kwargs = dict(
            params=params,
            task=task_factory.get_task(params.task),
            mode="train_and_eval",
        )
        self.docker_config = tfc.DockerConfig(
            parent_image=_PARENT_IMAGE,
            image_build_bucket=_TEST_BUCKET
        )

    def tpu_strategy(self):
        run_kwargs = dict(
            chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
            worker_count=1,
            worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"],
            requirements_txt=self.requirements_txt,
            job_labels={
                "job": "tpu_strategy",
                "team": "run_experiment_cloud_tests",
            },
            docker_config=self.docker_config,
        )
        run_experiment_kwargs = dict(
            model_dir=os.path.join(_BASE_PATH, "tpu", "saved_model"),
            **self.run_experiment_kwargs,
        )
        return models.run_experiment_cloud(run_experiment_kwargs,
                                           run_kwargs)

    def multi_mirror_strategy(self):
        run_kwargs = dict(
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            worker_count=1,
            worker_config=tfc.COMMON_MACHINE_CONFIGS["P100_1X"],
            requirements_txt=self.requirements_txt,
            job_labels={
                "job": "multi_mirror_strategy",
                "team": "run_experiment_cloud_tests",
            },
            docker_config=self.docker_config,
        )
        run_experiment_kwargs = dict(
            model_dir=os.path.join(_BASE_PATH, "multi_mirror", "saved_model"),
            **self.run_experiment_kwargs,
        )
        return models.run_experiment_cloud(run_experiment_kwargs,
                                           run_kwargs)

    def mirror_strategy(self):
        run_kwargs = dict(
            chief_config=tfc.COMMON_MACHINE_CONFIGS["P100_4X"],
            requirements_txt=self.requirements_txt,
            job_labels={
                "job": "mirror",
                "team": "run_experiment_cloud_tests",
            },
            docker_config=self.docker_config,
        )
        run_experiment_kwargs = dict(
            model_dir=os.path.join(_BASE_PATH, "mirror", "saved_model"),
            **self.run_experiment_kwargs,
        )
        return models.run_experiment_cloud(run_experiment_kwargs,
                                           run_kwargs)

    def one_device_strategy(self):
        run_kwargs = dict(
            requirements_txt=self.requirements_txt,
            job_labels={
                "job": "one_device",
                "team": "run_experiment_cloud_tests",
            },
            docker_config=self.docker_config,
        )
        run_experiment_kwargs = dict(
            model_dir=os.path.join(_BASE_PATH, "one_device", "saved_model"),
            **self.run_experiment_kwargs,
        )
        # Using the default T4 GPU for this test.
        return models.run_experiment_cloud(run_experiment_kwargs,
                                           run_kwargs)

    def test_run_experiment_cloud(self):
        track_status = {
            "one_device_strategy": self.one_device_strategy(),
            "mirror_strategy": self.mirror_strategy(),
            # TODO(b/148619319) Enable when bug is solved
            # "multi_mirror_strategy": self.multi_mirror_strategy(),
            # TODO(b/194857231) Enable when bug is solved
            # "tpu_strategy": self.tpu_strategy(),
        }

        for test_name, ret_val in track_status.items():
            self.assertTrue(
                google_api_client.wait_for_aip_training_job_completion(
                    ret_val["job_id"], _PROJECT_ID),
                "Job {} generated from the test: {} has failed".format(
                    ret_val["job_id"], test_name))

if __name__ == "__main__":
    tf.test.main()
