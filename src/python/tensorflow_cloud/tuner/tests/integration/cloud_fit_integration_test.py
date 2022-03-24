# Copyright 2019 Google LLC. All Rights Reserved.
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
"""End to end integration test for cloud_fit."""

import os
from typing import Text
import uuid
from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow_cloud.tuner import cloud_fit_client as client
from tensorflow_cloud.tuner import cloud_fit_utils as utils
from tensorflow_cloud.utils import google_api_client

# Can only export Datasets which were created executing eagerly
utils.enable_eager_for_tf_1()

MIRRORED_STRATEGY_NAME = utils.MIRRORED_STRATEGY_NAME
MULTI_WORKER_MIRRORED_STRATEGY_NAME = utils.MULTI_WORKER_MIRRORED_STRATEGY_NAME

# The staging bucket to use to copy the model and data for remote run.
_REMOTE_DIR = os.path.join("gs://", os.environ["TEST_BUCKET"])

# The project id to use to run tests.
_PROJECT_ID = os.environ["PROJECT_ID"]

# The GCP region in which the end-to-end test is run.
_REGION = os.environ["REGION"]

# The base docker image to use for remote environment.
_DOCKER_IMAGE = os.environ["DOCKER_IMAGE"]

# Using the build ID for testing
_BUILD_ID = os.environ["BUILD_ID"]


# TODO(b/169583124) add integration coverage for callback using pickle.
class CloudFitIntegrationTest(tf.test.TestCase):

    def setUp(self):
        super(CloudFitIntegrationTest, self).setUp()
        self._image_uri = _DOCKER_IMAGE
        self._project_id = _PROJECT_ID
        self._remote_dir = _REMOTE_DIR
        self._region = _REGION
        self._test_folders = []

    def tearDown(self):
        super(CloudFitIntegrationTest, self).tearDown()
        for folder in self._test_folders:
            self._delete_dir(folder)

    def _delete_dir(self, path: Text) -> None:
        """Deletes a directory if exists."""
        if tf.io.gfile.isdir(path):
            tf.io.gfile.rmtree(path)

    def _model(self):
        inputs = tf.keras.layers.Input(shape=(3,))
        outputs = tf.keras.layers.Dense(2)(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])
        return model

    def test_client_with_tf_1x_raises_error(self):
        # This test is only applicable to TF 1.x
        if not utils.is_tf_v1():
           return

        x = np.random.random((2, 3))
        y = np.random.randint(0, 2, (2, 2))

        # TF 1.x is not supported, verify proper error is raised for TF 1.x.
        with self.assertRaises(RuntimeError):
            client.cloud_fit(
                self._model(),
                x=x,
                y=y,
                remote_dir="gs://some_test_dir",
                region=self._region,
                project_id=self._project_id,
                image_uri=self._image_uri,
                epochs=2,
            )

    def test_in_memory_data(self):
        # This test should only run in tf 2.x
        if utils.is_tf_v1():
            return

        # Create a folder under remote dir for this test's data
        tmp_folder = str(uuid.uuid4())
        remote_dir = os.path.join(self._remote_dir, tmp_folder)

        # Keep track of test folders created for final clean up
        self._test_folders.append(remote_dir)

        x = np.random.random((2, 3))
        y = np.random.randint(0, 2, (2, 2))

        job_id = client.cloud_fit(
            self._model(),
            x=x,
            y=y,
            remote_dir=remote_dir,
            region=self._region,
            project_id=self._project_id,
            image_uri=self._image_uri,
            job_id="cloud_fit_e2e_test_{}_{}".format(
                _BUILD_ID.replace("-", "_"), "test_in_memory_data"
            ),
            epochs=2,
        )
        logging.info("test_in_memory_data submitted with job id: %s", job_id)

        # Wait for AIP Training job to finish successfully
        self.assertTrue(
            google_api_client.wait_for_aip_training_job_completion(
                job_id, self._project_id))

        # load model from remote dir
        trained_model = tf.keras.models.load_model(os.path.join(
            remote_dir, "checkpoint"))
        eval_results = trained_model.evaluate(x, y)

        # Accuracy should be better than zero
        self.assertListEqual(trained_model.metrics_names, ["loss", "accuracy"])
        self.assertGreater(eval_results[1], 0)

if __name__ == "__main__":
    tf.test.main()
