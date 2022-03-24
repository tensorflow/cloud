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
"""Integration tests for Distributing Cloud Tuner."""

import contextlib
import io
import os
import re
import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow_cloud.tuner import vizier_client
from tensorflow_cloud.tuner.tuner import DistributingCloudTuner

# If input dataset is created outside tuner.search(),
# it requires eager execution even in TF 1.x.
if tf.version.VERSION.split(".")[0] == "1":
    tf.compat.v1.enable_eager_execution()

# The project id to use to run tests.
_PROJECT_ID = os.environ["PROJECT_ID"]

# The GCP region in which the end-to-end test is run.
_REGION = os.environ["REGION"]

# Study ID for testing
_STUDY_ID_BASE = "dct_{}".format((os.environ["BUILD_ID"]).replace("-", "_"))

# The base docker image to use for the remote environment.
_DOCKER_IMAGE = os.environ["DOCKER_IMAGE"]

# The staging bucket to use to copy the model and data for the remote run.
_REMOTE_DIR = os.path.join("gs://", os.environ["TEST_BUCKET"], _STUDY_ID_BASE)

# The search space for hyperparameters
_HPS = keras_tuner.HyperParameters()
_HPS.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
_HPS.Int("num_layers", 2, 10)


def _load_data(dir_path=None):
    """Loads and prepares data."""

    mnist_file_path = None
    if dir_path:
        mnist_file_path = os.path.join(dir_path, "mnist.npz")

    (x, y), (val_x, val_y) = keras.datasets.mnist.load_data(mnist_file_path)
    x = x.astype("float32") / 255.0
    val_x = val_x.astype("float32") / 255.0

    return ((x[:10000], y[:10000]), (val_x, val_y))


def _build_model(hparams):
    # Note that CloudTuner does not support adding hyperparameters in
    # the model building function. Instead, the search space is configured
    # by passing a hyperparameters argument when instantiating (constructing)
    # the tuner.
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Build the model with number of layers from the hyperparameters
    for _ in range(hparams.get("num_layers")):
        model.add(keras.layers.Dense(units=64, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    # Compile the model with learning rate from the hyperparameters
    model.compile(
        optimizer=keras.optimizers.Adam(lr=hparams.get("learning_rate")),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model


class _DistributingCloudTunerIntegrationTestBase(tf.test.TestCase):

    def setUp(self):
        super(_DistributingCloudTunerIntegrationTestBase, self).setUp()
        self._study_id = None

    def _assert_output(self, fn, regex_str):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            fn()
        output = stdout.getvalue()
        self.assertRegex(output, re.compile(regex_str, re.DOTALL))

    def _assert_results_summary(self, fn):
        self._assert_output(
            fn, ".*Results summary.*Trial summary.*Hyperparameters.*")

    def _delete_dir(self, path) -> None:
        """Deletes a directory if exists."""
        if tf.io.gfile.isdir(path):
            tf.io.gfile.rmtree(path)

    def tearDown(self):
        super(_DistributingCloudTunerIntegrationTestBase, self).tearDown()

        # Delete the study used in the test, if present
        if self._study_id:
            service = vizier_client.create_or_load_study(
                _PROJECT_ID, _REGION, self._study_id, None)
            service.delete_study()

        tf.keras.backend.clear_session()

        # Delete log files, saved_models and other training assets
        self._delete_dir(_REMOTE_DIR)


class DistributingCloudTunerIntegrationTest(
        _DistributingCloudTunerIntegrationTestBase):

    def setUp(self):
        super(DistributingCloudTunerIntegrationTest, self).setUp()
        (self._x, self._y), (self._val_x, self._val_y) = _load_data(
            self.get_temp_dir())

    def testCloudTunerHyperparameters(self):
        """Test case to configure Distributing Tuner with HyperParameters."""
        study_id = "{}_hyperparameters".format(_STUDY_ID_BASE)
        self._study_id = study_id

        tuner = DistributingCloudTuner(
            _build_model,
            project_id=_PROJECT_ID,
            region=_REGION,
            objective="acc",
            hyperparameters=_HPS,
            max_trials=2,
            study_id=study_id,
            directory=_REMOTE_DIR,
            container_uri=_DOCKER_IMAGE
        )

        tuner.search(
            x=self._x,
            y=self._y,
            epochs=2,
            validation_data=(self._val_x, self._val_y),
        )

        self._assert_results_summary(tuner.results_summary)

if __name__ == "__main__":
    tf.test.main()
