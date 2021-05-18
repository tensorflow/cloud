# TensorFlow Cloud Tuner Guide

## What is this module?

`tuner` is a module which is part of the broader `tensorflow_cloud`. This module
is an implementation of a library for hyperparameter tuning that is built upon
the [KerasTuner](https://github.com/keras-team/keras-tuner) and creates a
seamless integration with
[Cloud AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs)
as a backend to get suggestions of hyperparameters and run trials.

The `tuner` module creates a seamless integration with
[Cloud AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs)
as a backend to get suggestions of hyperparameters and run trials.

```python
from tensorflow_cloud import CloudTuner
import kerastuner
import tensorflow as tf

(x, y), (val_x, val_y) = tf.keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for _ in range(hp.get('num_layers')):
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=hp.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

# Configure the search space
HPS = kerastuner.engine.hyperparameters.HyperParameters()
HPS.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
HPS.Int('num_layers', 2, 10)

# Instantiate CloudTuner
hptuner = CloudTuner(
    build_model,
    project_id=PROJECT_ID,
    region=REGION,
    objective='accuracy',
    hyperparameters=HPS,
    max_trials=5,
    directory='tmp_dir/1')

# Execute our search for the optimization study
hptuner.search(x=x, y=y, epochs=10, validation_data=(val_x, val_y))

# Get a summary of the trials from this optimization study
hptuner.results_summary()
```

See
[this runnable notebook](../tutorials/hp_tuning_cifar10_using_google_cloud.ipynb)
for a more complete example.
