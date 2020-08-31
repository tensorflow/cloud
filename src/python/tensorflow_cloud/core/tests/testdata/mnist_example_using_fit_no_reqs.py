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
"""Test module that calls model.fit() that does not require requirements.txt."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28 * 28))
x_train = x_train.astype("float32") / 255
y_train = y_train.astype("int32")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(512, activation="relu", input_shape=(28 * 28,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=2, batch_size=32)
