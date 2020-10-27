# Lint as: python3
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
"""Utilities for cloud_fit."""

import tensorflow as tf

MULTI_WORKER_MIRRORED_STRATEGY_NAME = (
    tf.distribute.experimental.MultiWorkerMirroredStrategy.__name__
)
MIRRORED_STRATEGY_NAME = tf.distribute.MirroredStrategy.__name__

SUPPORTED_DISTRIBUTION_STRATEGIES = {
    MULTI_WORKER_MIRRORED_STRATEGY_NAME:
        tf.distribute.experimental.MultiWorkerMirroredStrategy,
    MIRRORED_STRATEGY_NAME: tf.distribute.MirroredStrategy,
}


def is_tf_v1():
    """Returns true if Tensorflow is 1.x."""
    return tf.version.VERSION.split(".")[0] == "1"


def enable_eager_for_tf_1():
    """Enables eager execution mode for Tensorflow 1.x."""
    if is_tf_v1():
        tf.compat.v1.enable_eager_execution()
