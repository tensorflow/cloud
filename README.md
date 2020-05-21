# TensorFlow Cloud

## What is this repo?

The Tensorflow cloud repository provides APIs that will allow to easily go from debugging and training your Keras and TensorFlow code in a local environment to distributed training in the cloud.

## Installation

### Requirements

- Python >= 3.5
- [Set up your Google Cloud project](https://cloud.google.com/ai-platform/docs/getting-started-keras#set_up_your_project)
- [Authenticate your GCP account](https://cloud.google.com/ai-platform/docs/getting-started-keras#authenticate_your_gcp_account)
- We use [Google AI platform](https://cloud.google.com/ai-platform/) for deploying docker images on GCP. Please make sure you have AI platform APIs enabled on your GCP project.
- Please make sure `docker` is installed and running if you want to use local docker process for docker build, otherwise [create a cloud storage bucket](https://cloud.google.com/ai-platform/docs/getting-started-keras#create_a_bucket) for using [Google cloud build](https://cloud.google.com/cloud-build) for docker image build and publish.
- Install [nbconvert](https://nbconvert.readthedocs.io/en/latest/) if you are using a notebook file as `entry_point` as shown in [usage guide #4](#detailed-usage-guide).

### Install latest release

```console
pip install -U tensorflow-cloud
```

### Install from source

```console
git clone https://github.com/tensorflow/cloud.git
cd cloud
pip install .
```

## High level overview 

Tensorflow cloud package provides the `run` API for training your models on GCP. Before we get into the details of the API, let's see how a simple workflow will look like using this API.

1. Let's say you have a Keras model training code, such as the following, saved as `mnist_example.py`.

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
mnist_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

train_dataset = mnist_train.map(scale).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(train_dataset, epochs=12)
```

2. After you have tested this model on your local environment for a few epochs, probably with a small dataset, you can train the model on Google cloud by writing the following simple script `scale_mnist.py`.

```python
import tensorflow_cloud as tfc
tfc.run(entry_point='mnist_example.py')
```

Running this script will automatically apply Tensorflow [Mirrored distribution strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) and train your model at scale on Google Cloud Platform. Please see the [usage guide](#usage-guide) section for detailed instructions on how to use the API.

3. You will see an output similar to the following on your console. The information from the output can be used to track the training job status. 

```console
usr@desktop$ python scale_mnist.py
Job submitted successfully.
Your job ID is:  tf_cloud_train_519ec89c_a876_49a9_b578_4fe300f8865e
Please access your job logs at the following URL:
https://console.cloud.google.com/mlengine/jobs/tf_cloud_train_519ec89c_a876_49a9_b578_4fe300f8865e?project=prod-123
```

## Detailed usage guide

As described in the [high level overview](#high-level-overview) section, the `run` API allows you to train your models at scale on GCP. The [`run`](https://github.com/tensorflow/cloud/blob/master/tensorflow_cloud/run.py#L31) API can be used in four different ways. This is defined by where you are running the API (Terminal vs IPython notebook) and what the `entry_point` parameter value is. `entry_point` is an optional Python script or notebook file path to the file that contains the TensorFlow Keras training code. This is the most important parameter in the API.


```python
run(entry_point=None,
    requirements_txt=None,
    distribution_strategy='auto',
    docker_base_image=None,
    chief_config='auto',
    worker_config='auto',
    worker_count=0,
    entry_point_args=None,
    stream_logs=False,
    docker_image_bucket_name=None,
    **kwargs)
```

**1. Using a python file as `entry_point`.**

If you have your `tf.keras` model in a python file (`mnist_example.py`), then you can write the following simple script (`scale_mnist.py`) to scale your model on GCP.

```python
import tensorflow_cloud as tfc
tfc.run(entry_point='mnist_example.py')
```

**2. Using a notebook file as `entry_point`.**

If you have your `tf.keras` model in a notebook file (`mnist_example.ipynb`), then you can write the following simple script (`sclae_mnist.py`) to scale your model on GCP.

```python
import tensorflow_cloud as tfc
tfc.run(entry_point='mnist_example.ipynb')
```

**3. Using `run` within a python script that contains the `tf.keras` model.**

You can use the `run` API from within your python file that contains the `tf.keras` model (`mnist_scale.py`). 

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_cloud as tfc

tfc.run(
    entry_point=None,
    distribution_strategy='auto',
    requirements_txt='tests/testdata/requirements.txt',
    chief_config=tfc.MachineConfig(
            cpu_cores=8,
            memory=30,
            accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_P100,
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

In this use case, `entry_point` should be `None`. The `run` API can be called anywhere and the entire file will be executed remotely. The API can be called at the end to run the script locally once for debugging purposes (possibly with different #epochs and other flags).

**4. Using `run` within a notebook script that contains the `tf.keras` model.**

![Image of colab](https://github.com/tensorflow/cloud/blob/master/images/colab.png)

In this use case, `entry_point` should be `None` and `docker_image_bucket_name` must be provided.

### What happens when you call run?

The API call will encompass the following:
1. Making code entities such as a Keras script/notebook, **cloud and distribution ready**.
2. Converting this distribution entity into a **docker container** with all the required dependencies.
3. **Deploy** this container at scale and train using Tensorflow distribution strategies.
4. **Stream logs** and monitor them on hosted TensorBoard, manage checkpoint storage.

By default, we will use local docker daemon for building and publishing docker images to Google container registry. Images are published to `gcr.io/your-gcp-project-id`. If you specify `docker_image_bucket_name`, then we will use [Google cloud build](https://cloud.google.com/cloud-build) to build and publish docker images. 

**Note** If you are using `run` within a notebook script that contains the `tf.keras` model, `docker_image_bucket_name` must be specified.

We use [Google AI platform](https://cloud.google.com/ai-platform/) for deploying docker images on GCP.

Please see `run` API documentation for detailed information on the parameters and how you can control the above processes. 

## End to end examples

- [Using a python file as `entry_point` (Keras fit API)](https://github.com/tensorflow/cloud/blob/master/tests/integration/call_run_on_script_with_keras_fit.py).
- [Using a python file as `entry_point` (Keras custom training loop)](https://github.com/tensorflow/cloud/blob/master/tests/integration/call_run_on_script_with_keras_ctl.py).
- [Using a python file as `entry_point` (Keras save and load)](https://github.com/tensorflow/cloud/blob/master/tests/integration/call_run_on_script_with_keras_save_and_load.py).
- [Using a notebook file as `entry_point`](https://github.com/tensorflow/cloud/blob/master/tests/integration/call_run_on_notebook_with_keras_fit.py).
- [Using `run` within a python script that contains the `tf.keras` model](https://github.com/tensorflow/cloud/blob/master/tests/integration/call_run_within_script_with_keras_fit.py).
- [Using cloud build instead of local docker](https://github.com/tensorflow/cloud/blob/master/tests/integration/call_run_on_script_with_keras_fit_cloud_build.py).

## Coming up

- Keras tuner support.
- TPU support.

## Contributing

We welcome community contributions, see [CONTRIBUTING.md](CONTRIBUTING.md) and, for style help,
[Writing TensorFlow documentation](https://www.tensorflow.org/community/documentation)
guide.

## License

[Apache License 2.0](LICENSE)
