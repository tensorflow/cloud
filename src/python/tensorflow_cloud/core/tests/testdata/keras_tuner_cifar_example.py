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
"""Test module that instantiates KerasTuner."""

import argparse

import keras_tuner
import tensorflow as tf


def build_model(hp):
    """Constructs a model."""
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomWidth(0.1),
            tf.keras.layers.experimental.preprocessing.RandomHeight(0.1),
        ]
    )
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation=hp.Choice(
            "conv_activation_0", values=["relu", "elu"], default="relu",
        ),
    )(x)

    for i in range(hp.Int("num_blocks", 1, 3)):
        for j in range(hp.Int("num_conv_{}".format(str(i)), 1, 3)):
            x = tf.keras.layers.Conv2D(
                hp.Int("filter_{}_{}".format(str(i), str(j)), 16, 256, step=16),
                (3, 3),
                padding="same",
                activation=hp.Choice(
                    "conv_activation_{}_{}".format(str(i), str(j)),
                    values=["relu", "elu"],
                    default="relu",
                ),
            )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(
        rate=hp.Float(
            "dropout_1", min_value=0.0, max_value=0.9, default=0.25, step=0.05,
        )
    )(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.Choice("initial_learning_rate",
                                        [1e-1, 1e-2, 1e-3]),
        decay_steps=100000,
        decay_rate=hp.Choice("decay_rate", [0.5, 0.75, 0.95]),
        staircase=True,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule),
        metrics=["sparse_categorical_accuracy"],
    )
    return model


tuner = keras_tuner.RandomSearch(
    build_model,
    objective="val_sparse_categorical_accuracy",
    max_trials=5,
    executions_per_trial=3,
    directory="test_dir",
)

tuner.search_space_summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
BUFFER_SIZE = 10000
BATCH_SIZE = 64


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


train_dataset = train_dataset.map(scale).cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.map(scale).batch(BATCH_SIZE)

tuner.search(
    train_dataset,
    epochs=2,
    validation_data=test_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min"),
    ],
)
print("Tuner results summary")
tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
scores = best_model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

parser = argparse.ArgumentParser(description="Keras model save path")
parser.add_argument("--path",
                    required=True, type=str, help="Keras model save path")
args = parser.parse_args()
model_save_path = args.path

print("Saving best model")
best_model.save(model_save_path)
