import argparse
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255


def build_model(hp):
    model = keras.Sequential()
    model.add(
        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu", input_shape=x_train.shape[1:]
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Dropout(
            rate=hp.Float(
                "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
            )
        )
    )

    model.add(
        layers.Conv2D(
            hp.Choice("num_filters", values=[32, 64], default=64,),
            (3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Dropout(
            rate=hp.Float(
                "dropout_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
            )
        )
    )

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(
        layers.Dropout(
            rate=hp.Float(
                "dropout_3", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
            )
        )
    )
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.RMSprop(
            learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4]), decay=1e-6
        ),
        metrics=["accuracy"],
    )
    return model


datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(x_train)

tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=3,
    directory="test_dir",
)

tuner.search_space_summary()
tuner.search(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(x_test, y_test),
)
tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
scores = best_model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

parser = argparse.ArgumentParser(description="Keras model save path")
parser.add_argument("--path", required=True, type=str, help="Keras model save path")
args = parser.parse_args()
model_save_path = args.path
best_model.save(model_save_path)
