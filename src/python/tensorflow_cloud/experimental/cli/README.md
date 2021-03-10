# Tensorflow cloud CLI

Is a feature of tfcloud that let you run tfcloud from your terminal using `tfc` command.


### How to use it

1. [Follow the configuration and installation instructions.](https://github.com/tensorflow/cloud#tensorflow-cloud-run-api-for-gcp-trainingtuning)
2. Create a file with all that you need to train your model or a jupyter notebook. for example `train.py`:

``` python
import tensorflow_datasets as tfds
import tensorflow as tf

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

3. Train your model on the cloud with:

``` bash
tfc run train.py
```

### Contribution guidelines

We use typer to build the CLI interface for tfcloud, if you want to contributed to this feature is important to follow typer ideology of write type annotations in the code. [Typer documentation.](https://typer.tiangolo.com/)
