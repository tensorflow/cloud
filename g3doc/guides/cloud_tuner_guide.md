# TensorFlow Cloud Tuner Guide

## What is this module?

`tuner` is a module which is part of the broader `tensorflow_cloud`. This module
is an implementation of a library for hyperparameter tuning that is built upon
the [KerasTuner](https://github.com/keras-team/keras-tuner) and creates a
seamless integration with
[Cloud AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs)
as a backend to get suggestions of hyperparameters and run trials.

## Installation

### Requirements

-   Python >= 3.6
-   Tensorflow >= 2.0
-   [Set up your Google Cloud project](https://cloud.google.com/ai-platform/docs/getting-started-keras#set_up_your_project)
-   [Authenticate your GCP account](https://cloud.google.com/ai-platform/docs/getting-started-keras#authenticate_your_gcp_account)
-   Enable [Google AI platform](https://cloud.google.com/ai-platform/) APIs on
    your GCP project.

For detailed end to end setup instructions, please see
[Setup instructions](#setup-instructions) section.

## High level overview

`tuner` module creates a seamless integration with
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

## Setup instructions

End to end instructions to help set up your environment for `tuner`. If you are
using
[Google Cloud Hosted Notebooks](https://cloud.google.com/ai-platform-notebooks)
you can skip the setup and authentication steps and start from step 8.

1.  Create a new local directory

    ```shell
    mkdir tuner
    cd tuner
    ```

1.  Make sure you have `python >= 3.6`

    ```shell
    python -V
    ```

1.  Setup virtual environment

    ```shell
    virtualenv venv --python=python3
    source venv/bin/activate
    ```

1.  [Set up your Google Cloud project](https://cloud.google.com/ai-platform/docs/getting-started-keras#set_up_your_project)

    Verify that gcloud sdk is installed.

    ```shell
    which gcloud
    ```

    Set default gcloud project

    ```shell
    export PROJECT_ID=<your-project-id>
    gcloud config set project $PROJECT_ID
    ```

1.  [Authenticate your GCP account](https://cloud.google.com/ai-platform/docs/getting-started-keras#authenticate_your_gcp_account)
    Follow the prompts after executing the command. This is not required in
    hosted notebooks.

    ```shell
    gcloud auth application-default
    ```

1.  Authenticate Docker to access Google Cloud Container Registry (gcr)

    To use local [Docker] with google
    [Cloud Container Registry](https://cloud.google.com/container-registry/docs/advanced-authentication)
    for docker build and publish

    ```shell
    gcloud auth configure-docker
    ```

1.  Create a Bucket for Data and Model transfer
    [Create a cloud storage bucket](https://cloud.google.com/ai-platform/docs/getting-started-keras#create_a_bucket)
    for `tuner` temporary storage.

    ```shell
    export BUCKET_NAME="your-bucket-name"
    export REGION="us-central1"
    gsutil mb -l $REGION gs://$BUCKET_NAME
    ```

1.  Build and install latest release

    ```shell
    git clone https://github.com/tensorflow/cloud.git
    cd cloud
    pip install src/python/.
    ```

You are all set! You can now follow our
[examples](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tests/examples/ai_platform_vizier_tuner.ipynb)
to try out `tuner`.

## License

[Apache License 2.0](https://github.com/tensorflow/cloud/blob/master/LICENSE)
