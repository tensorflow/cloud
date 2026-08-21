# TensorFlow Cloud

The TensorFlow Cloud repository provides APIs that will allow to easily go from
debugging, training, tuning your Keras and TensorFlow code in a local
environment to distributed training/tuning on Cloud.

## Introduction

-   [TensorFlow Cloud `run` API](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/README.md)

-   [TensorFlow Cloud Tuner](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/README.md)

## TensorFlow Cloud `run` API for GCP training/tuning

### Installation

#### Requirements

-   Python >= 3.6
-   [A Google Cloud project](https://cloud.google.com/ai-platform/docs/getting-started-keras#set_up_your_project)
-   An
    [authenticated GCP account](https://cloud.google.com/ai-platform/docs/getting-started-keras#authenticate_your_gcp_account)
-   [Google AI platform](https://cloud.google.com/ai-platform/) APIs enabled for
    your GCP account. We use the AI platform for deploying docker images on GCP.
-   Either a functioning version of
    [docker](https://docs.docker.com/engine/install/) if you want to use a local
    docker process for your build, or
    [create a cloud storage bucket](https://cloud.google.com/ai-platform/docs/getting-started-keras#create_a_bucket)
    to use with [Google Cloud build](https://cloud.google.com/cloud-build) for
    docker image build and publishing.

-   [Authenticate to your Docker Container Registry](https://cloud.google.com/container-registry/docs/advanced-authentication#gcloud-helper)

-   (optional) [nbconvert](https://nbconvert.readthedocs.io/en/latest/) if you
    are using a notebook file as `entry_point` as shown in
    [usage guide #4](#usage-guide).

For detailed end to end setup instructions, please see
[Setup instructions](#setup-instructions).

#### Install latest release

```shell
pip install -U tensorflow-cloud
```

#### Install from source

```shell
git clone https://github.com/tensorflow/cloud.git
cd cloud
pip install src/python/.
```

### High level overview

TensorFlow Cloud package provides the `run` API for training your models on GCP.
To start, let's walk through a simple workflow using this API.

1.  Let's begin with a Keras model training code such as the following, saved as
    `mnist_example.py`.

    ```python
    import tensorflow as tf

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

    model.fit(x_train, y_train, epochs=10, batch_size=128)
    ```

1.  After you have tested this model on your local environment for a few epochs,
    probably with a small dataset, you can train the model on Google Cloud by
    writing the following simple script `scale_mnist.py`.

    ```python
    import tensorflow_cloud as tfc
    tfc.run(entry_point='mnist_example.py')
    ```

    Running `scale_mnist.py` will automatically apply TensorFlow
    [one device strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy)
    and train your model at scale on Google Cloud Platform. Please see the
    [usage guide](#usage-guide) section for detailed instructions and additional
    API parameters.

1.  You will see an output similar to the following on your console. This
    information can be used to track the training job status.

    ```shell
    user@desktop$ python scale_mnist.py
    Job submitted successfully.
    Your job ID is:  tf_cloud_train_519ec89c_a876_49a9_b578_4fe300f8865e
    Please access your job logs at the following URL:
    https://console.cloud.google.com/mlengine/jobs/tf_cloud_train_519ec89c_a876_49a9_b578_4fe300f8865e?project=prod-123
    ```

### Setup instructions

End to end instructions to help set up your environment for Tensorflow Cloud.
You use one of the following notebooks to setup your project or follow the
instructions below.

<table align="left">
    <td>
        <a href="https://colab.research.google.com/github/tensorflow/cloud/blob/master/examples/google_cloud_project_setup_instructions.ipynb">
            <img width="50" src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Colab logo">Run in Colab
        </a>
    </td>
    <td>
        <a href="https://github.com/tensorflow/cloud/blob/master/examples/google_cloud_project_setup_instructions.ipynb">
            <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo">View on GitHub
        </a>
     </td>
    <td>
        <a href="https://www.kaggle.com/nitric/google-cloud-project-setup-instructions">
            <img width="90" src="https://www.kaggle.com/static/images/site-logo.png" alt="Kaggle logo">Run in Kaggle
        </a>
     </td>
</table>

1.  Create a new local directory

    ```shell
    mkdir tensorflow_cloud
    cd tensorflow_cloud
    ```

1.  Make sure you have `python >= 3.6`

    ```shell
    python -V
    ```

1.  Set up virtual environment

    ```shell
    virtualenv tfcloud --python=python3
    source tfcloud/bin/activate
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

    Create a service account.

    ```shell
    export SA_NAME=<your-sa-name>
    gcloud iam service-accounts create $SA_NAME
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com \
        --role 'roles/editor'
    ```

    Create a key for your service account.

    ```shell
    gcloud iam service-accounts keys create ~/key.json --iam-account $SA_NAME@$PROJECT_ID.iam.gserviceaccount.com
    ```

    Create the GOOGLE_APPLICATION_CREDENTIALS environment variable.

    ```shell
    export GOOGLE_APPLICATION_CREDENTIALS=~/key.json
    ```

1.  [Create a Cloud Storage bucket](https://cloud.google.com/ai-platform/docs/getting-started-keras#create_a_bucket).
    Using [Google Cloud build](https://cloud.google.com/cloud-build) is the
    recommended method for building and publishing docker images, although we
    optionally allow for local
    [docker daemon process](https://docs.docker.com/config/daemon/#start-the-daemon-manually)
    depending on your specific needs.

    ```shell
    BUCKET_NAME="your-bucket-name"
    REGION="us-central1"
    gcloud auth login
    gsutil mb -l $REGION gs://$BUCKET_NAME
    ```

    (optional for local docker setup) `shell sudo dockerd`

1.  Authenticate access to Google Cloud registry.

    ```shell
    gcloud auth configure-docker
    ```

1.  Install [nbconvert](https://nbconvert.readthedocs.io/en/latest/) if you plan
    to use a notebook file `entry_point` as shown in
    [usage guide #4](#usage-guide).

    ```shell
    pip install nbconvert
    ```

1.  Install latest release of tensorflow-cloud

    ```shell
    pip install tensorflow-cloud
    ```

### Usage guide

As described in the [high level overview](#high-level-overview), the `run` API
allows you to train your models at scale on GCP. The
[`run`](https://github.com/tensorflow/cloud/blob/master/src/python/core/run.py#L31)
API can be used in four different ways. This is defined by where you are running
the API (Terminal vs IPython notebook), and your `entry_point` parameter.
`entry_point` is an optional Python script or notebook file path to the file
that contains your TensorFlow Keras training code. This is the most important
parameter in the API.

```python
run(entry_point=None,
    requirements_txt=None,
    distribution_strategy='auto',
    docker_config='auto',
    chief_config='auto',
    worker_config='auto',
    worker_count=0,
    entry_point_args=None,
    stream_logs=False,
    job_labels=None,
    **kwargs)
```

1.  **Using a python file as `entry_point`.**

    If you have your `tf.keras` model in a python file (`mnist_example.py`),
    then you can write the following simple script (`scale_mnist.py`) to scale
    your model on GCP.

    ```python
    import tensorflow_cloud as tfc
    tfc.run(entry_point='mnist_example.py')
    ```

    Please note that all the files in the same directory tree as `entry_point`
    will be packaged in the docker image created, along with the `entry_point`
    file. It's recommended to create a new directory to house each cloud project
    which includes necessary files and nothing else, to optimize image build
    times.

1.  **Using a notebook file as `entry_point`.**

    If you have your `tf.keras` model in a notebook file
    (`mnist_example.ipynb`), then you can write the following simple script
    (`scale_mnist.py`) to scale your model on GCP.

    ```python
    import tensorflow_cloud as tfc
    tfc.run(entry_point='mnist_example.ipynb')
    ```

    Please note that all the files in the same directory tree as `entry_point`
    will be packaged in the docker image created, along with the `entry_point`
    file. Like the python script `entry_point` above, we recommended creating a
    new directory to house each cloud project which includes necessary files and
    nothing else, to optimize image build times.

1.  **Using `run` within a python script that contains the `tf.keras` model.**

    You can use the `run` API from within your python file that contains the
    `tf.keras` model (`mnist_scale.py`). In this use case, `entry_point` should
    be `None`. The `run` API can be called anywhere and the entire file will be
    executed remotely. The API can be called at the end to run the script
    locally for debugging purposes (possibly with fewer epochs and other flags).

    ```python
    import tensorflow_datasets as tfds
    import tensorflow as tf
    import tensorflow_cloud as tfc

    tfc.run(
        entry_point=None,
        distribution_strategy='auto',
        requirements_txt='requirements.txt',
        chief_config=tfc.MachineConfig(
                cpu_cores=8,
                memory=30,
                accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
                accelerator_count=2),
        worker_count=0)

    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']

    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    train_dataset = mnist_train.map(scale).cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(
            28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(train_dataset, epochs=12)
    ```

    Please note that all the files in the same directory tree as the python
    script will be packaged in the docker image created, along with the python
    file. It's recommended to create a new directory to house each cloud project
    which includes necessary files and nothing else, to optimize image build
    times.

1.  **Using `run` within a notebook script that contains the `tf.keras` model.**

    ![Image of colab](https://github.com/tensorflow/cloud/blob/master/images/colab.png)

    In this use case, `entry_point` should be `None` and
    `docker_config.image_build_bucket` must be specified, to ensure the build
    can be stored and published.

    ### Cluster and distribution strategy configuration

    By default, `run` API takes care of wrapping your model code in a TensorFlow
    distribution strategy based on the cluster configuration you have provided.

    ***No distribution***

    CPU chief config and no additional workers

    ```python
    tfc.run(entry_point='mnist_example.py',
            chief_config=tfc.COMMON_MACHINE_CONFIGS['CPU'])
    ```

    ***OneDeviceStrategy***

    1 GPU on chief (defaults to `AcceleratorType.NVIDIA_TESLA_T4`) and no
    additional workers.

    ```python
    tfc.run(entry_point='mnist_example.py')
    ```

    ***MirroredStrategy***

    Chief config with multiple GPUS (`AcceleratorType.NVIDIA_TESLA_V100`).

    ```python
    tfc.run(entry_point='mnist_example.py',
            chief_config=tfc.COMMON_MACHINE_CONFIGS['V100_4X'])
    ```

    ***MultiWorkerMirroredStrategy***

    Chief config with 1 GPU and 2 workers each with 8 GPUs
    (`AcceleratorType.NVIDIA_TESLA_V100`).

    ```python
    tfc.run(entry_point='mnist_example.py',
            chief_config=tfc.COMMON_MACHINE_CONFIGS['V100_1X'],
            worker_count=2,
            worker_config=tfc.COMMON_MACHINE_CONFIGS['V100_8X'])
    ```

    ***TPUStrategy***

    Chief config with 1 CPU and 1 worker with TPU.

    ```python
    tfc.run(entry_point="mnist_example.py",
            chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
            worker_count=1,
            worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"])
    ```

    Please note that TPUStrategy with TensorFlow Cloud works only with TF
    version 2.1 as this is the latest version supported by
    [AI Platform cloud TPU](https://cloud.google.com/ai-platform/training/docs/runtime-version-list#tpu-support)

    ***Custom distribution strategy***

    If you would like to take care of specifying distribution strategy in your
    model code and do not want `run` API to create a strategy, then set
    `distribution_stategy` as `None`. This will be required for example when you
    are using `strategy.experimental_distribute_dataset`.

    ```python
    tfc.run(entry_point='mnist_example.py',
            distribution_strategy=None,
            worker_count=2)
    ```

#### What happens when you call run?

The API call will encompass the following:

1.  Making code entities such as a Keras script/notebook, **cloud and
    distribution ready**.
1.  Converting this distribution entity into a **docker container** with the
    required dependencies.
1.  **Deploy** this container at scale and train using TensorFlow distribution
    strategies.
1.  **Stream logs** and monitor them on hosted TensorBoard, manage checkpoint
    storage.

By default, we will use local docker daemon for building and publishing docker
images to Google container registry. Images are published to
`gcr.io/your-gcp-project-id`. If you specify `docker_config.image_build_bucket`,
then we will use [Google Cloud build](https://cloud.google.com/cloud-build) to
build and publish docker images.

We use [Google AI platform](https://cloud.google.com/ai-platform/) for deploying
docker images on GCP.

Please note that, when `entry_point` argument is specified, all the files in the
same directory tree as `entry_point` will be packaged in the docker image
created, along with the `entry_point` file.

Please see `run` API documentation for detailed information on the parameters
and how you can modify the above processes to suit your needs.

### End to end examples

```shell
cd src/python/tensorflow_cloud/core
python tests/examples/call_run_on_script_with_keras_fit.py
```

-   [Using a python file as `entry_point` (Keras fit API)](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/examples/call_run_on_script_with_keras_fit.py).
-   [Using a python file as `entry_point` (Keras custom training loop)](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/examples/call_run_on_script_with_keras_ctl.py).
-   [Using a python file as `entry_point` (Keras save and load)](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/examples/call_run_on_script_with_keras_save_and_load.py).
-   [Using a notebook file as `entry_point`](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/examples/call_run_on_notebook_with_keras_fit.py).
-   [Using `run` within a python script that contains the `tf.keras` model](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/examples/call_run_within_script_with_keras_fit.py).
-   [Using cloud build instead of local docker](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/examples/call_run_on_script_with_keras_fit_cloud_build.py).
-   [Run AutoKeras with TensorFlow Cloud](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/examples/call_run_within_script_with_autokeras.py).

### Running unit tests

```shell
pytest src/python/tensorflow_cloud/core/tests/unit/
```

### Local vs remote training

Things to keep in mind when running your jobs remotely:

[Coming soon]

### Debugging workflow

Here are some tips for fixing unexpected issues.

#### Operation disallowed within distribution strategy scope

**Error like**: Creating a generator within a strategy scope is disallowed,
because there is ambiguity on how to replicate a generator (e.g. should it be
copied so that each replica gets the same random numbers, or 'split' so that
each replica gets different random numbers).

**Solution**: Passing `distribution_strategy='auto'` to `run` API wraps all of
your script in a TF distribution strategy based on the cluster configuration
provided. You will see the above error or something similar to it, if for some
reason an operation is not allowed inside distribution strategy scope. To fix
the error, please pass `None` to the `distribution_strategy` param and create a
strategy instance as part of your training code as shown in
[this](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/testdata/save_and_load.py)
example.

#### Docker image build timeout

**Error like**: requests.exceptions.ConnectionError: ('Connection aborted.',
timeout('The write operation timed out'))

**Solution**: The directory being used as an entry point likely has too much
data for the image to successfully build, and there may be extraneous data
included in the build. Reformat your directory structure such that the folder
which contains the entry point only includes files necessary for the current
project.

#### Version not supported for TPU training

**Error like**: There was an error submitting the job.Field: tpu_tf_version
Error: The specified runtime version '2.3' is not supported for TPU training.
Please specify a different runtime version.

**Solution**: Please use TF version 2.1. See TPU Strategy in
[Cluster and distribution strategy configuration section](#cluster-and-distribution-strategy-configuration).

#### TF nightly build.

**Warning like**: Docker parent image '2.4.0.dev20200720' does not exist. Using
the latest TF nightly build.

**Solution**: If you do not provide `docker_config.parent_image` param, then by
default we use pre-built TF docker images as parent image. If you do not have TF
installed on the environment where `run` is called, then TF docker image for the
`latest` stable release will be used. Otherwise, the version of the docker image
will match the locally installed TF version. However, pre-built TF docker images
aren't available for TF nightlies except for the latest. So, if your local TF is
an older nightly version, we upgrade to the latest nightly automatically and
raise this warning.

#### Mixing distribution strategy objects.

**Error like**: RuntimeError: Mixing different tf.distribute.Strategy objects.

**Solution**: Please provide `distribution_strategy=None` when you already have
a distribution strategy defined in your model code. Specifying
`distribution_strategy'='auto'`, will wrap your code in a TensorFlow
distribution strategy. This will cause the above error, if there is a strategy
object already used in your code.

### Coming up

-   Distributed Keras tuner support.

## Contributing

We welcome community contributions, see [CONTRIBUTING.md](CONTRIBUTING.md) and,
for style help,
[Writing TensorFlow documentation](https://www.tensorflow.org/community/contribute/docs)
guide.

## License

[Apache License 2.0](LICENSE)

## Privacy Notice

This application reports technical and operational details of your usage of
Cloud Services in accordance with Google privacy policy, for more information
please refer to https://policies.google.com/privacy. If you wish to opt-out, you
may do so by running
tensorflow_cloud.utils.google_api_client.optout_metrics_reporting().
