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
"""Test module that exercises model save and load."""

# Example from TF tutorials
# https://www.tensorflow.org/tutorials/distribute/save_and_load
# Expects keras model path to be passed in the command line

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

parser = argparse.ArgumentParser(description="A tutorial of argparse!")
parser.add_argument("--path",
                    required=True, type=str, help="Keras model save path")
args = parser.parse_args()
model_save_path = args.path

# Prepare the data and model
mirrored_strategy = tf.distribute.MirroredStrategy()


def get_data():
    """Creates datasets."""
    datasets, _ = tfds.load(name="mnist",
                            with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets["train"], datasets["test"]

    buffer_size = 10000

    batch_size_per_replica = 64
    num_replicas = mirrored_strategy.num_replicas_in_sync
    batch_size = batch_size_per_replica * num_replicas

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset_ = mnist_train.map(scale).cache()
    train_dataset_ = train_dataset_.shuffle(buffer_size).batch(batch_size)
    eval_dataset_ = mnist_test.map(scale).batch(batch_size)

    return train_dataset_, eval_dataset_


def get_model():
    """Constracts a model."""
    with mirrored_strategy.scope():
        model_ = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, 3, activation="relu", input_shape=(28, 28, 1)
                ),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

    model_.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model_


print("Initial training + saving model weights")
model = get_model()
train_dataset, eval_dataset = get_data()
checkpoint_path = "{}/cp.ckpt".format(model_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)
model.fit(train_dataset, epochs=2, callbacks=[cp_callback])

print("Creating new model instance")
model = get_model()

print("Evaluating untrained model")
loss, acc = model.evaluate(eval_dataset, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

print("Loading model weights")
another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
with another_strategy.scope():
    model.load_weights(checkpoint_path)

print("Evaluating model with loaded weights")
loss, acc = model.evaluate(eval_dataset, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

print("Saving model")
model.save(model_save_path)  # save() should be called out of strategy scope

print("Restore model and train without dist strat")
restored_keras_model = tf.keras.models.load_model(model_save_path)
restored_keras_model.fit(train_dataset, epochs=2)

print("Restore model and training using a different strategy")
another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
with another_strategy.scope():
    restored_keras_model_ds = tf.keras.models.load_model(model_save_path)
    restored_keras_model_ds.fit(train_dataset, epochs=2)
