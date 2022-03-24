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
"""Integration tests for Cloud Keras Tuner."""

import contextlib
import io
import multiprocessing
import os
import re
import time
import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow_cloud import CloudTuner
from tensorflow_cloud.tuner import vizier_client

# If input dataset is created outside tuner.search(),
# it requires eager execution even in TF 1.x.
if tf.version.VERSION.split(".")[0] == "1":
    tf.compat.v1.enable_eager_execution()

# The project id to use to run tests.
_PROJECT_ID = os.environ["PROJECT_ID"]

# The GCP region in which the end-to-end test is run.
_REGION = os.environ["REGION"]

# Study ID for testing
_STUDY_ID_BASE = os.environ["BUILD_ID"]

# The search space for hyperparameters
_HPS = keras_tuner.HyperParameters()
_HPS.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
_HPS.Int("num_layers", 2, 10)

# Number of search loops we would like to run in parallel for distributed tuning
_NUM_PARALLEL_TRIALS = 4


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


def _dist_search_fn(temp_dir, study_id, tuner_id):
    """Multi-process safe tuner instantiation and tuner.search()."""

    # Jitter instantiation so as to avoid contention on study
    # creation and dataset download.
    time.sleep(int(tuner_id[5:]) * 20)  # tuner_id is formatted as 'tuner%d'

    # Dataset must be loaded independently in sub-process.
    (x, y), (val_x, val_y) = _load_data(temp_dir)

    tuner = CloudTuner(
        _build_model,
        project_id=_PROJECT_ID,
        region=_REGION,
        objective="acc",
        hyperparameters=_HPS,
        max_trials=5,
        study_id=study_id,
        directory=os.path.join(temp_dir, study_id, tuner_id),
    )
    tuner.tuner_id = tuner_id

    tuner.search(
        x=x,
        y=y,
        epochs=2,
        steps_per_epoch=20,
        validation_steps=10,
        validation_data=(val_x, val_y),
    )
    return tuner


def _dist_search_fn_wrapper(args):
    return _dist_search_fn(*args)


class _CloudTunerIntegrationTestBase(tf.test.TestCase):

    def setUp(self):
        super(_CloudTunerIntegrationTestBase, self).setUp()
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

    def tearDown(self):
        super(_CloudTunerIntegrationTestBase, self).tearDown()

        # Delete the study used in the test, if present
        if self._study_id:
            service = vizier_client.create_or_load_study(
                _PROJECT_ID, _REGION, self._study_id, None)
            service.delete_study()

        tf.keras.backend.clear_session()


class CloudTunerIntegrationTest(_CloudTunerIntegrationTestBase):

    def setUp(self):
        super(CloudTunerIntegrationTest, self).setUp()
        (self._x, self._y), (self._val_x, self._val_y) = _load_data(
            self.get_temp_dir())

    def testCloudTunerHyperparameters(self):
        """Test case to configure Tuner with HyperParameters object."""
        study_id = "{}_hyperparameters".format(_STUDY_ID_BASE)
        self._study_id = study_id

        tuner = CloudTuner(
            _build_model,
            project_id=_PROJECT_ID,
            region=_REGION,
            objective="acc",
            hyperparameters=_HPS,
            max_trials=5,
            study_id=study_id,
            directory=os.path.join(self.get_temp_dir(), study_id),
        )

        # "Search space summary" comes first, but the order of
        # "learning_rate (Float)" and "num_layers (Int)" is not deterministic,
        # hence they are wrapped as look-ahead assertions in the regex.
        self._assert_output(
            tuner.search_space_summary,
            r".*Search space summary(?=.*learning_rate \(Float\))"
            r"(?=.*num_layers \(Int\).*)",
        )

        tuner.search(
            x=self._x,
            y=self._y,
            epochs=2,
            steps_per_epoch=20,
            validation_steps=10,
            validation_data=(self._val_x, self._val_y),
        )

        self._assert_results_summary(tuner.results_summary)

    def testCloudTunerDatasets(self):
        """Test case to configure Tuner with tf.data.Dataset as input data."""
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((self._x, self._y))
            .batch(128)
            .cache()
            .prefetch(1000)
        )
        eval_dataset = (
            tf.data.Dataset.from_tensor_slices((self._val_x, self._val_y))
            .batch(128)
            .cache()
            .prefetch(1000)
        )

        study_id = "{}_dataset".format(_STUDY_ID_BASE)
        self._study_id = study_id

        tuner = CloudTuner(
            _build_model,
            project_id=_PROJECT_ID,
            region=_REGION,
            objective="acc",
            hyperparameters=_HPS,
            study_id=study_id,
            max_trials=5,
            directory=os.path.join(self.get_temp_dir(), study_id),
        )

        self._assert_output(
            tuner.search_space_summary,
            r".*Search space summary(?=.*learning_rate \(Float\))"
            r"(?=.*num_layers \(Int\).*)",
        )

        tuner.search(
            x=train_dataset,
            epochs=2,
            steps_per_epoch=20,
            validation_steps=10,
            validation_data=eval_dataset,
        )

        self._assert_results_summary(tuner.results_summary)

    def testCloudTunerStudyConfig(self):
        """Test case to configure Tuner with StudyConfig object."""
        # Configure the search space. Specification:
        # https://cloud.google.com/ai-platform/optimizer/docs/reference/rest/v1/projects.locations.studies#StudyConfig  # pylint: disable=line-too-long
        study_config = {
            "metrics": [{"goal": "MAXIMIZE", "metric": "acc"}],
            "parameters": [
                {
                    "discrete_value_spec": {"values": [0.0001, 0.001, 0.01]},
                    "parameter": "learning_rate",
                    "type": "DISCRETE",
                },
                {
                    "integer_value_spec": {"max_value": 10, "min_value": 2},
                    "parameter": "num_layers",
                    "type": "INTEGER",
                },
                {
                    "discrete_value_spec": {"values": [32, 64, 96, 128]},
                    "parameter": "units",
                    "type": "DISCRETE",
                },
            ],
            "algorithm": "ALGORITHM_UNSPECIFIED",
            "automatedStoppingConfig": {
                "decayCurveStoppingConfig": {"useElapsedTime": True}
            },
        }

        study_id = "{}_study_config".format(_STUDY_ID_BASE)
        self._study_id = study_id

        tuner = CloudTuner(
            _build_model,
            project_id=_PROJECT_ID,
            region=_REGION,
            study_config=study_config,
            study_id=study_id,
            max_trials=5,
            directory=os.path.join(self.get_temp_dir(), study_id),
        )

        self._assert_output(
            tuner.search_space_summary,
            r".*Search space summary(?=.*learning_rate \(Choice\))"
            r"(?=.*num_layers \(Int\))(?=.*units \(Int\))",
        )

        tuner.search(
            x=self._x,
            y=self._y,
            epochs=2,
            steps_per_epoch=20,
            validation_steps=10,
            validation_data=(self._val_x, self._val_y),
        )

        self._assert_results_summary(tuner.results_summary)


class CloudTunerInDistributedIntegrationTest(_CloudTunerIntegrationTestBase):

    def testCloudTunerInProcessDistributedTuning(self):
        """Test case to simulate multiple parallel tuning workers."""
        study_id = "{}_dist".format(_STUDY_ID_BASE)
        self._study_id = study_id

        with multiprocessing.Pool(processes=_NUM_PARALLEL_TRIALS) as pool:
            results = pool.map(
                _dist_search_fn_wrapper,
                [
                    (self.get_temp_dir(), study_id, "tuner{}".format(i))
                    for i in range(_NUM_PARALLEL_TRIALS)
                ],
            )

        self._assert_results_summary(results[0].results_summary)

    def testCloudTunerAIPlatformTrainingDistributedTuning(self):
        """Test case of parallel tuning using CAIP Training as flock manager."""
        # TODO(b/169697464): Implement test for tuning with CAIP Training
        study_id = "{}_caip_dist".format(_STUDY_ID_BASE)
        del study_id


if __name__ == "__main__":
    tf.test.main()
