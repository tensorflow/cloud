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
"""Test module that instantiates custom training loop."""

# From TF tutorials
# https://www.tensorflow.org/tutorials/distribute/custom_training

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

print(tf.__version__)


# Down fashion mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

mnist_data = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = mnist_data

# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]
test_images = test_images[..., None]

# Getting the images in [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

# Create strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


# Setup input pipeline
BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 2

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(BUFFER_SIZE)
    .batch(GLOBAL_BATCH_SIZE)
)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


def create_model():
    """Constructs a model."""
    model_ = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    return model_


# Define loss function
with strategy.scope():
    # Set reduction to `none` so we can do the reduction afterwards and
    # divide by global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE
        )


# Define the metrics to track loss and accuracy
with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name="test_loss")

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="test_accuracy")

# Training loop# model and optimizer must be created under `strategy.scope`.
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()

with strategy.scope():

    def train_step(inputs):
        """Defines a custom training step."""
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        return loss

    def test_step(inputs):
        """Defines a custom test step."""
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)


with strategy.scope():

    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        """Defines a custom distributed training step."""
        per_replica_losses = strategy.run(
            train_step, args=(dataset_inputs,)
        )
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    @tf.function
    def distributed_test_step(dataset_inputs):
        """Defines a custom distributed test step."""
        return strategy.run(test_step, args=(dataset_inputs,))

    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)

        template = (
            "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
            "Test Accuracy: {}"
        )
        print(
            template.format(
                epoch + 1,
                train_loss,
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
