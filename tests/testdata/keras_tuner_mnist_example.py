from tensorflow import keras
from tensorflow.keras import layers


from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters


(x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
x = x.astype("float32") / 255.0
val_x = val_x.astype("float32") / 255.0

x = x[:10000]
y = y[:10000]


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.Int("num_layers", 2, 20)):
        model.add(
            layers.Dense(
                units=hp.Int("units_" + str(i), 32, 512, 32), activation="relu"
            )
        )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=3,
    directory="test_dir",
)

tuner.search_space_summary()

tuner.search(x=x, y=y, epochs=3, validation_data=(val_x, val_y))

tuner.results_summary()
