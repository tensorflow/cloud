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

import os

import tensorflow as tf
import tensorflow_cloud as tfc
import tensorflow_datasets as tfds

from create_model import create_keras_model

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

model = create_keras_model()

if tfc.remote():
    epochs = 10
else:
    epochs = 1

model.fit(train_dataset, epochs=epochs)
