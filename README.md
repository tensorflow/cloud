# TensorFlow Cloud

## What is this repo?

This repository provides APIs that will allow to easily go from debugging and training your Keras and TensorFlow code in a local environment to distributed training in the cloud.

## Installation

### Requirements:

- Python >= 3.5
- [Set up your Google Cloud project](https://cloud.google.com/ai-platform/docs/getting-started-keras#set_up_your_project)
- [Authenticate your GCP account](https://cloud.google.com/ai-platform/docs/getting-started-keras#authenticate_your_gcp_account)


### Install latest release:

```
pip install -U tensorflow-cloud
```

### Install from source:

```
git clone https://github.com/tensorflow/cloud.git
cd cloud
pip install .
```

## Usage examples

- [Usage with `tf.keras` script that trains using `model.fit`](tests/integration/call_run_on_script_with_keras_fit.py).
- [Usage with `tf.keras` script that trains using a custom training loop](tests/integration/call_run_on_script_with_keras_ctl.py).

## Contributing

We welcome community contributions, see [CONTRIBUTING.md](CONTRIBUTING.md) and, for style help,
[Writing TensorFlow documentation](https://www.tensorflow.org/community/documentation)
guide.

## License

[Apache License 2.0](LICENSE)
