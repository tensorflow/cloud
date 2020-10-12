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

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_cloud as tfc
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Set Cloud GCP bucket name
GCP_BUCKET = "your-bucket-name"
MODEL_PATH = "resnet-dogs"

# Setup dataset
(ds_train, ds_test), metadata = tfds.load(
    "stanford_dogs",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)

NUM_CLASSES = metadata.features["label"].num_classes
print("Number of training samples: %d" % tf.data.experimental.cardinality(ds_train))
print("Number of test samples: %d" % tf.data.experimental.cardinality(ds_test))
print("Number of classes: %d" % NUM_CLASSES)

# Visualize dataset
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")

# Preprocess dataset
IMG_SIZE = 224
BATCH_SIZE = 64
BUFFER_SIZE = 2

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))


def input_preprocess(image, label):
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)

# Build the model
inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_tensor=inputs
)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES)(x)

model = tf.keras.Model(inputs, outputs)

# Freeze base model
base_model.trainable = False

# Setup callbacks
checkpoint_path = os.path.join("gs://", GCP_BUCKET, MODEL_PATH, "save_at_{epoch}")
tensorboard_path = os.path.join(
    "gs://", GCP_BUCKET, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
    tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
]

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model.
if tfc.remote():
    epochs = 500
    train_data = ds_train
    test_data = ds_test
else:
    epochs = 1
    train_data = ds_train.take(5)
    test_data = ds_test.take(5)
    callbacks = None

model.fit(
    train_data, epochs=epochs, callbacks=callbacks, validation_data=test_data, verbose=2
)

# Calling `tfc.run` with `auto` distribution strategy with multi-gpu
# chief_config. This will automate TensorFlow Mirrored distribution
# strategy when training this model.
# Tip: Move this call to the top of this file if you do not want to
# train your model locally first.
tfc.run(
    requirements_txt="tests/testdata/requirements.txt",
    chief_config=tfc.MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
        accelerator_count=2,
    ),
    docker_config=tfc.DockerConfig(image_build_bucket=GCP_BUCKET),
)

# Save, load and evaluate the model
if tfc.remote():
    SAVE_PATH = os.path.join("gs://", GCP_BUCKET, MODEL_PATH)
    model.save(SAVE_PATH)
    model = tf.keras.models.load_model(SAVE_PATH)
model.evaluate(test_data)
