# Usage guide

The `tfc.run` API allows you to train your models at scale on GCP.

The `tfc.run` API can be used in four different ways. This is defined by where
you are running the API (Python script vs Python notebook), and your
`entry_point` parameter:

*   Python file as `entry_point`.
*   Notebook file as `entry_point`.
*   `run` within a Python script that contains the `tf.keras` model.
*   `run` within a notebook script that contains the `tf.keras` model. The most
    common way is to use `run` within a notebook.

The `entry_point` is a (path to a) Python script or notebook file, or `None`. If
`None`, the entire current File is sent to Google Cloud.

## Using a Python file as `entry_point`.

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

## Using a notebook file as `entry_point`.

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

## Using `run` within a Python script that contains the `tf.keras` model.

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

## Using `run` within a notebook script that contains the `tf.keras` model.

![Image of colab](https://github.com/tensorflow/cloud/blob/master/images/colab.png)

In this use case, `entry_point` should be `None` and
`docker_config.image_build_bucket` must be specified, to ensure the build can be
stored and published.

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
