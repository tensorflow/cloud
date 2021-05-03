# TensorFlow Cloud

The TensorFlow Cloud repository provides APIs that will allow to easily go from
debugging, training, tuning your Keras and TensorFlow code in a local
environment to distributed training/tuning on Cloud.

## TensorFlow Cloud `run` API for GCP training/tuning

TensorFlow Cloud provides the `run` API for training your models on GCP. To
start, let's walk through a simple workflow using this API.

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
    and train your model at scale on Google Cloud Platform.

1.  You will see an output similar to the following on your console. This
    information can be used to track the training job status.

    ```shell
    user@desktop$ python scale_mnist.py
    Job submitted successfully.
    Your job ID is:  tf_cloud_train_519ec89c_a876_49a9_b578_4fe300f8865e
    Please access your job logs at the following URL:
    https://console.cloud.google.com/mlengine/jobs/tf_cloud_train_519ec89c_a876_49a9_b578_4fe300f8865e?project=prod-123
    ```

## Setup instructions

Follow these instructions in your local environment, or in the
[project setup notebook](../cloud/tutorials/google_cloud_project_setup_instructions.ipynb).

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

## Usage guide

The `run` API allows you to train your models at scale on GCP.

The
[`run`](https://github.com/tensorflow/cloud/blob/master/src/python/core/run.py#L31)
API can be used in four different ways. This is defined by where you are running
the API (Python script vs Python notebook), and your `entry_point` parameter:

*   Python file as `entry_point`.
*   Notebook file as `entry_point`.
*   `run` within a Python script that contains the `tf.keras` model.
*   `run` within a notebook script that contains the `tf.keras` model.

The `entry_point` is a (path to a) Python script or notebook file, or `None`. If
`None`, the entire current File is sent to Google Cloud.

### Using a Python file as `entry_point`.

If you have your `tf.keras` model in a python file (`mnist_example.py`), then
you can write the following simple script (`scale_mnist.py`) to scale your model
on GCP.

```python
import tensorflow_cloud as tfc
tfc.run(entry_point='mnist_example.py')
```

Please note that all the files in the same directory tree as `entry_point` will
be packaged in the docker image created, along with the `entry_point` file. It's
recommended to create a new directory to house each cloud project which includes
necessary files and nothing else, to optimize image build times.

### Using a notebook file as `entry_point`.

If you have your `tf.keras` model in a notebook file (`mnist_example.ipynb`),
then you can write the following simple script (`scale_mnist.py`) to scale your
model on GCP.

```python
import tensorflow_cloud as tfc
tfc.run(entry_point='mnist_example.ipynb')
```

Please note that all the files in the same directory tree as `entry_point` will
be packaged in the docker image created, along with the `entry_point` file. Like
the python script `entry_point` above, we recommended creating a new directory
to house each cloud project which includes necessary files and nothing else, to
optimize image build times.

### Using `run` within a Python script that contains the `tf.keras` model.

You can use the `run` API from within your python file that contains the
`tf.keras` model (`mnist_scale.py`). In this use case, `entry_point` should be
`None`. The `run` API can be called anywhere and the entire file will be
executed remotely. The API can be called at the end to run the script locally
for debugging purposes (possibly with fewer epochs and other flags).

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

Please note that all the files in the same directory tree as the python script
will be packaged in the docker image created, along with the python file. It's
recommended to create a new directory to house each cloud project which includes
necessary files and nothing else, to optimize image build times.

### Using `run` within a notebook script that contains the `tf.keras` model.

![Image of colab](https://github.com/tensorflow/cloud/blob/master/images/colab.png)

In this use case, `entry_point` should be `None` and
`docker_config.image_build_bucket` must be specified, to ensure the build can be
stored and published.

## Cluster and distribution strategy configuration

By default, `run` API takes care of wrapping your model code in a TensorFlow
distribution strategy based on the cluster configuration you have provided.

### No distribution

CPU chief config and no additional workers

```python
tfc.run(entry_point='mnist_example.py',
        chief_config=tfc.COMMON_MACHINE_CONFIGS['CPU'])
```

### `OneDeviceStrategy`

1 GPU on chief (defaults to `AcceleratorType.NVIDIA_TESLA_T4`) and no additional
workers.

```python
tfc.run(entry_point='mnist_example.py')
```

### `MirroredStrategy`

Chief config with multiple GPUs (`AcceleratorType.NVIDIA_TESLA_V100`).

```python
tfc.run(entry_point='mnist_example.py',
        chief_config=tfc.COMMON_MACHINE_CONFIGS['V100_4X'])
```

### `MultiWorkerMirroredStrategy`

Chief config with 1 GPU and 2 workers each with 8 GPUs
(`AcceleratorType.NVIDIA_TESLA_V100`).

```python
tfc.run(entry_point='mnist_example.py',
        chief_config=tfc.COMMON_MACHINE_CONFIGS['V100_1X'],
        worker_count=2,
        worker_config=tfc.COMMON_MACHINE_CONFIGS['V100_8X'])
```

### `TPUStrategy`

Chief config with 1 CPU and 1 worker with TPU.

```python
tfc.run(entry_point="mnist_example.py",
        chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
        worker_count=1,
        worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"])
```

Please note that TPUStrategy with TensorFlow Cloud works only with TF version
2.1 as this is the latest version supported by
[AI Platform cloud TPU](https://cloud.google.com/ai-platform/training/docs/runtime-version-list#tpu-support)

### Custom distribution strategy

If you would like to take care of specifying a distribution strategy in your
model code and do not want `run` API to create a strategy, then set
`distribution_stategy` as `None`. This will be required, for example, when you
are using `strategy.experimental_distribute_dataset`.

```python
tfc.run(entry_point='mnist_example.py',
        distribution_strategy=None,
        worker_count=2)
```

## What happens when you call `run`?

The API call will accomplish the following:

1.  Making code entities such as a Keras script/notebook, **cloud and
    distribution ready**.
1.  Converting this distribution entity into a **docker container** with the
    required dependencies.
1.  **Deploy** this container at scale and train using TensorFlow distribution
    strategies.
1.  **Stream logs** and monitor them on hosted TensorBoard, manage checkpoint
    storage.

By default, the local Docker daemon for building and publishing Docker images to
Google container registry. Images are published to `gcr.io/your-gcp-project-id`.
If you specify `docker_config.image_build_bucket`, then we will use
[Google Cloud build](https://cloud.google.com/cloud-build) to build and publish
docker images.

[Google AI platform](https://cloud.google.com/ai-platform/) is used for for
deploying Docker images on Google Cloud.

Note: When `entry_point` is a file path, all the files in the same directory
tree as `entry_point` will be packaged in the Docker image created along with
the `entry_point` file.
