# Cloud Fit

## What is this module?

`cloud_fit` is a module which is part of the broader
`tensorflow_cloud`. This module provides an API that enables training
on [Google Cloud AI Platform](https://cloud.google.com/ai-platform). `cloud_fit`
serializes the model, datasets, and callback functions and submits them for
remote execution on AI Platform. `cloud_fit` is intended to function in the same
manner as `model.fit()`. This module is designed to be used within a pipeline or
an automated process such as from Tuner however it can be used directly as
well.

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

`cloud_fit` module provides an API for training your models on GCP. It can
simply be used in the same manner as `model.fit()` with a few additional
parameters to enable remote execution on AI Platform.

```python
import tensorflow as tf
from tensorflow_cloud.tuner import cloud_fit_client as client

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28 * 28))
x_train = x_train.astype('float32') / 255

model = tf.keras.Sequential([
  tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Run model locally for validation
model.fit(x_train, y_train, epochs=10, batch_size=128)

# Submit for remote execution with cloud_fit
client.cloud_fit(model=model, remote_dir = GCS_REMOTE_DIR, region =REGION , x=x_train, y= y_train, epochs=100, batch_size=128)

# After training is completed you can retrieve the model from GCS_REMOTE_DIR/output
model = tf.keras.models.load_model(os.path.join(GCS_REMOTE_DIR, 'output'))
```

## Setup instructions

End to end instructions to help set up your environment for `cloud_fit`. If you
are using
[Google Cloud Hosted Notebooks](https://cloud.google.com/ai-platform-notebooks)
you can skip the setup and authentication steps and start from step 8.

1.  Create a new local directory

    ```shell
    mkdir cloud_fit
    cd cloud_fit
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
    for cloud_fit temporary storage.

    ```shell
    export BUCKET_NAME="your-bucket-name"
    export REGION="us-central1"
    gsutil mb -l $REGION gs://$BUCKET_NAME
    ```

1.  Build and install latest release

    ```shell
    git clone https://github.com/tensorflow/cloud.git
    cd cloud/src/python && python3 setup.py -q bdist_wheel
    pip install -U cloud/src/python/dist/tensorflow_cloud-*.whl --quiet
    ```

1.  Create a docker image as the base image for remote training

    Create a dockerfile as follows:

    ```shell
    # Using DLVM base image. For GPU training use
    # gcr.io/deeplearning-platform-release/tf2-gpu instead.
    FROM gcr.io/deeplearning-platform-release/tf2-cpu
    WORKDIR /root

    # Path configuration
    ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

    # Make sure gsutil will use the default service account
    RUN echo '[Google Compute]\nservice_account = default' > /etc/boto.cfg

    # Copy and install tensorflow_cloud wheel file
    ADD cloud/src/python/dist/tensorflow_cloud-*.whl /tmp/
    RUN pip3 install --upgrade /tmp/tensorflow_cloud-*.whl --quiet

    # Sets up the entry point to invoke cloud_fit.
    ENTRYPOINT ["python3","-m","tensorflow_cloud.tuner.cloud_fit_remote"]
    ```

    Build and push a docker image using the dockerfile above, where IMAGE_URI
    follows the format `gcr.io/{PROJECT_ID}/[name-for-docker-image]:latest`.

    ```shell
    export IMAGE_NAME=[name-for-docker-image]
    export IMAGE_URI=gcr.io/$PROJECT_ID$/$IMAGE_NAME:latest
    docker build -t $IMAGE_URI -f Dockerfile . -q && docker push $IMAGE_URI
    ```

    You are all set! You can now follow our
    [examples](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tests/examples/cloud_fit.ipynb)
    to try out `cloud_fit`.

## License

[Apache License 2.0](https://github.com/tensorflow/cloud/blob/master/LICENSE)

