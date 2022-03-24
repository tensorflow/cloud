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
from typing import List, Text
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_transform as tft

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


def _transformed_name(key):
    return key + "_xf"


# TODO(b/183734637) Consiger using TFXIO to ingest data
def _gzip_reader_fn(filenames: List[Text]):
    """Small utility returning a record reader that can read gzip'ed files.

    Args:
        filenames: List of paths or patterns of input tfrecord files.
    Returns:
        A reader function to read upstream ExampleGen artifacts from GCS and by
        default they are gzip'ed TF.Records files.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              label_key: str,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        tf_transform_output: A TFTransformOutput.
        label_key: label key.
        batch_size: representing the number of consecutive elements of returned
          dataset to combine in a single batch

    Returns:
        A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_transformed_name(label_key))

    # If the input dataset is file-based but the number of files is less than
    # the number of workers, an error will be raised. Turning off auto shard
    # policy here so that Dataset will sharded by data instead of by file.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA)
    dataset = dataset.with_options(options)

    return dataset


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
            # Specific fit_kwargs required for TFX tuner_fn.
            train_files = None
            eval_files = None
            transform_graph = None
            label_key = None
            train_batch_size = None
            eval_batch_size = None
            if "label_key" in training_assets_graph.fit_kwargs_fn():
                label_key_byte = tfds.as_numpy(
                    training_assets_graph.fit_kwargs_fn()["label_key"])
                label_key = label_key_byte.decode("ASCII")
            if "transform_graph_path" in training_assets_graph.fit_kwargs_fn():
                transform_graph_path = tfds.as_numpy(
                    training_assets_graph.fit_kwargs_fn(
                        )["transform_graph_path"])
                # Decode the path from byte to string object.
                transform_graph = tft.TFTransformOutput(
                    transform_graph_path.decode("ASCII"))
                logging.info("transform_graph was loaded successfully.")
            if "train_files" in training_assets_graph.fit_kwargs_fn():
                train_files_byte = tfds.as_numpy(
                    training_assets_graph.fit_kwargs_fn()["train_files"])
                train_files = [x.decode("ASCII") for x in train_files_byte]
            if "eval_files" in training_assets_graph.fit_kwargs_fn():
                eval_files_byte = tfds.as_numpy(
                    training_assets_graph.fit_kwargs_fn()["eval_files"])
                eval_files = [x.decode("ASCII") for x in eval_files_byte]

            if "train_batch_size" in training_assets_graph.fit_kwargs_fn():
                train_batch_size = tfds.as_numpy(
                    training_assets_graph.fit_kwargs_fn()["train_batch_size"])
            if "eval_batch_size" in training_assets_graph.fit_kwargs_fn():
                eval_batch_size = tfds.as_numpy(
                    training_assets_graph.fit_kwargs_fn()["eval_batch_size"])

            if train_files and transform_graph and label_key and train_batch_size:  # pylint: disable=line-too-long
                fit_kwargs["x"] = _input_fn(
                    train_files,
                    transform_graph,
                    label_key,
                    batch_size=train_batch_size)
                logging.info("x was loaded successfully.")

            if eval_files and transform_graph and label_key and eval_batch_size:
                fit_kwargs["validation_data"] = _input_fn(
                    eval_files,
                    transform_graph,
                    label_key,
                    batch_size=eval_batch_size)
                logging.info("validation data was loaded successfully.")

            for k in training_assets_graph.fit_kwargs_fn().keys():
                # Specific fit_kwargs for TFX AIP Tuner component.
                tfx_fit_kwargs = ["train_files", "eval_files", "label_key",
                                  "transform_graph_path", "train_batch_size",
                                  "eval_batch_size"]
                # deserialize the rest of the fit_kwargs
                if k not in tfx_fit_kwargs:
                    fit_kwargs[k] = tfds.as_numpy(
                        training_assets_graph.fit_kwargs_fn()[k])
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
