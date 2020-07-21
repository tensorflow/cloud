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

# Example from TF tutorials
# https://www.tensorflow.org/tutorials/distribute/keras

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_cloud as tfc

tfds.disable_progress_bar()

print(tf.__version__)

# Calling `run` from within a script with contains the Keras model.
tfc.run(
    entry_point=None,
    distribution_strategy="auto",
    requirements_txt="tests/testdata/requirements.txt",
    chief_config=tfc.MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
        accelerator_count=2,
    ),
    worker_count=0,
)

# Download the dataset
datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets["train"], datasets["test"]

# Setup input pipeline
num_train_examples = info.splits["train"].num_examples
num_test_examples = info.splits["test"].num_examples

BUFFER_SIZE = 10000
BATCH_SIZE = 64


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label


train_dataset = mnist_train.map(scale).cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

# Create the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)


# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            "\nLearning rate for epoch {} is {}".format(
                epoch + 1, model.optimizer.lr.numpy()
            )
        )


callbacks = [tf.keras.callbacks.LearningRateScheduler(decay), PrintLR()]

model.fit(train_dataset, epochs=12, callbacks=callbacks)
