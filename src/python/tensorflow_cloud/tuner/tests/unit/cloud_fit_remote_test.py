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
"""Tests for cloud_fit.remote."""

import json
import os
import tempfile
from unittest import mock
from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow_cloud.tuner import cloud_fit_client as client
from tensorflow_cloud.tuner import cloud_fit_remote as remote
from tensorflow_cloud.tuner import cloud_fit_utils as utils

# Can only export Datasets which were created executing eagerly
utils.enable_eager_for_tf_1()

MULTI_WORKER_MIRRORED_STRATEGY_NAME = utils.MULTI_WORKER_MIRRORED_STRATEGY_NAME
MIRRORED_STRATEGY_NAME = utils.MIRRORED_STRATEGY_NAME

FLAGS = flags.FLAGS
FLAGS.remote_dir = "test_remote_dir"
FLAGS.distribution_strategy = MULTI_WORKER_MIRRORED_STRATEGY_NAME


class _MockCallable(object):

    @classmethod
    def reset(cls):
        cls.mock_callable = mock.Mock()

    @classmethod
    def mock_call(cls):
        cls.mock_callable()


class CustomCallbackExample(tf.keras.callbacks.Callback):

    def on_train_begin(self, unused_logs):
        _MockCallable.mock_call()


class CloudFitRemoteTest(tf.test.TestCase):

    def setUp(self):
        super(CloudFitRemoteTest, self).setUp()
        self._image_uri = "gcr.io/some_test_image:latest"
        self._project_id = "test_project_id"
        self._remote_dir = tempfile.mkdtemp()
        self._output_dir = os.path.join(self._remote_dir, "checkpoint")
        self._x = np.random.random(10)
        self._y = np.random.random(10)
        self._model = self._model()
        self._logs_dir = os.path.join(self._remote_dir, "logs")
        self._fit_kwargs = {
            "x": self._x,
            "y": self._y,
            "verbose": 2,
            "batch_size": 2,
            "epochs": 10,
            "callbacks": [tf.keras.callbacks.TensorBoard(
                log_dir=self._logs_dir)],
        }
        client._serialize_assets(
            self._remote_dir, self._model, **self._fit_kwargs)
        os.environ["TF_CONFIG"] = json.dumps(
            {
                "cluster": {"worker": ["localhost:9999", "localhost:9999"]},
                "task": {"type": "worker", "index": 0},
            }
        )

    def _model(self):
        inputs = tf.keras.layers.Input(shape=(1,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        return model

    def test_run(self):
        # TF 1.x is not supported
        if utils.is_tf_v1():
            return

        remote.run(self._remote_dir, MIRRORED_STRATEGY_NAME)
        self.assertGreaterEqual(len(tf.io.gfile.listdir(self._output_dir)), 1)
        self.assertGreaterEqual(len(tf.io.gfile.listdir(self._logs_dir)), 1)

        model = tf.keras.models.load_model(self._output_dir)

        # Test saved model load and works properly
        self.assertGreater(
            model.evaluate(
                self._x, self._y)[0], np.array([0.0], dtype=np.float32))

    def test_custom_callback(self):
        # TF 1.x is not supported
        if utils.is_tf_v1():
            return

        # Setting up custom callback with mock calls
        _MockCallable.reset()

        self._fit_kwargs["callbacks"] = [CustomCallbackExample()]
        client._serialize_assets(
            self._remote_dir, self._model, **self._fit_kwargs)

        # Verify callback function has not been called yet.
        _MockCallable.mock_callable.assert_not_called()

        remote.run(self._remote_dir, MIRRORED_STRATEGY_NAME)
        # Verifying callback functions triggered properly
        _MockCallable.mock_callable.assert_called_once_with()


if __name__ == "__main__":
    tf.test.main()
