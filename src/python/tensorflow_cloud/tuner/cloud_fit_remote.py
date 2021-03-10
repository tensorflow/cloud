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
"""Module that deserializes and runs the provided Keras Model and Dataset.

This is the server side execution module for model & data set serializer and
deserializer that is intended for remote execution of in memory models in AI
Platform training.
"""

import os
import pickle
from typing import Text
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_cloud.tuner import cloud_fit_utils

MULTI_WORKER_MIRRORED_STRATEGY_NAME = cloud_fit_utils.MULTI_WORKER_MIRRORED_STRATEGY_NAME
MIRRORED_STRATEGY_NAME = cloud_fit_utils.MIRRORED_STRATEGY_NAME
SUPPORTED_DISTRIBUTION_STRATEGIES = cloud_fit_utils.SUPPORTED_DISTRIBUTION_STRATEGIES

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "distribution_strategy",
    None,
    "String representing distribution strategy. "
    "Must match DistributionStrategy values",
)

flags.DEFINE_string(
    "remote_dir",
    None,
    "Temporary cloud storage folder for transferring model and dataset",
)


def main(unused_argv):
    logging.set_verbosity(logging.INFO)
    if FLAGS.distribution_strategy not in SUPPORTED_DISTRIBUTION_STRATEGIES:
        raise ValueError(
            "{} is not supported. Supported Strategies are {}".format(
                FLAGS.distribution_strategy,
                list(SUPPORTED_DISTRIBUTION_STRATEGIES.keys()),
            )
        )

    run(FLAGS.remote_dir, FLAGS.distribution_strategy)


def run(
    remote_dir: Text,
    distribution_strategy_text: Text
) -> None:
    """deserializes Model and Dataset and runs them.

    Args:
        remote_dir: Temporary cloud storage folder that contains model and
            Dataset graph. This folder is also used for job output.
        distribution_strategy_text: Specifies the distribution strategy for
            remote execution when a jobspec is provided. Accepted values are
            strategy names as specified by 'tf.distribute.<strategy>.__name__'.
    """
    logging.info("Setting distribution strategy to %s",
                 distribution_strategy_text)

    distribution_strategy = SUPPORTED_DISTRIBUTION_STRATEGIES[
        distribution_strategy_text
    ]()

    with distribution_strategy.scope():
        if cloud_fit_utils.is_tf_v1():
            training_assets_graph = tf.compat.v2.saved_model.load(
                export_dir=os.path.join(remote_dir, "training_assets"),
                tags=None)
        else:
            training_assets_graph = tf.saved_model.load(
                os.path.join(remote_dir, "training_assets")
            )

        fit_kwargs = {}
        if hasattr(training_assets_graph, "fit_kwargs_fn"):
            fit_kwargs = tfds.as_numpy(training_assets_graph.fit_kwargs_fn())
            logging.info("fit_kwargs were loaded successfully.")

        if hasattr(training_assets_graph, "x_fn"):
            fit_kwargs["x"] = training_assets_graph.x_fn()
            logging.info("x was loaded successfully.")

        if hasattr(training_assets_graph, "y_fn"):
            fit_kwargs["y"] = training_assets_graph.y_fn()
            logging.info("y was loaded successfully.")

        if hasattr(training_assets_graph, "validation_data_fn"):
            fit_kwargs["validation_data"] = (
                training_assets_graph.validation_data_fn())

        if hasattr(training_assets_graph, "callbacks_fn"):
            pickled_callbacks = tfds.as_numpy(
                training_assets_graph.callbacks_fn())
            fit_kwargs["callbacks"] = pickle.loads(pickled_callbacks)
            logging.info("callbacks were loaded successfully.")

        model = tf.keras.models.load_model(os.path.join(remote_dir, "model"))
        logging.info(
            "Model was loaded from %s successfully.",
            os.path.join(remote_dir, "model")
        )
        model.fit(**fit_kwargs)
        # Model needs to be saved via ModelCheckpoint callback due to issues
        # with the save model in MWMS. See b/148619319 for details.
        # TODO(b/176829535) Evaluate using save model once b/148619319 is done.


if __name__ == "__main__":
    flags.mark_flag_as_required("remote_dir")
    flags.mark_flag_as_required("distribution_strategy")
    app.run(main)
