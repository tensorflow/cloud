# Current Version (Still in Development)

*   Add notes for next release here.

# Release 0.1.10

*   Py 3.5 support removed.
*   Small bug fixes.

# Release 0.1.9

*   Added Kaggle integration.

# Release 0.1.8

*   `cloud_fit` now moved to a sub-module under `Tuner`
*   HParams plugin integration with DistributingCloudTuner
*   Added integration tests
*   Small bug fixes.

# Release 0.1.7

*   `cloud_fit` uses pickle instead of cloudpickle.
*   Better integration tests checking for job status.
*   Small bug fixes.

# Release 0.1.6

*   New module CloudTuner - Implementation of a library for hyperparameter
    tuning that is built into the KerasTuner and creates a seamless integration
    with
    [Cloud AI Platform Optimizer](https://cloud.google.com/ai-platform/optimizer/docs/overview)
    as a backend to get suggestions of hyperparameters and run trials.
*   New application Monitoring - TensorFlow extension that exports its metrics
    to Stackdriver backend, allowing users to monitor the training and inference
    jobs in real time.
*   New experimental project cloud_fit - an experimental module that enables
    training keras models on
    [Cloud AI Platform Training](https://cloud.google.com/ai-platform/training/docs/overview)
    by serializing the model, and datasets for remote execution.
*   Small bug fixes.

# Release 0.1.5

*   Restructuring of source code for new projects
*   Multi-file code example
*   Integration test example
*   Small bug fixes

# Release 0.1.4

*   New API remote() to detect if currently in a remote cloud env.
*   CI using Github Action.
*   Updated README.
*   Some minor bug fixes.

# Release 0.1.3

## New features

*   Support for single node Keras tuner workflow.
*   Support for TPU training.

## Fixes

*   Fixed docker build decode errors.
*   Default to Py3 for TF docker images.

## Others

*   New colab notebook example.
*   New Auto Keras example.
*   Improved ReadMedocs.
*   Improved error messages.

# Release 0.1.2

*   Support for passing colab notebook as entry_point.
*   Support for cloud docker build and colab workflow.
*   Support for log streaming in colab

# Release 0.1.1

*   Detailed README with setup instructions and examples.
*   Support for running `run` API from within a python script which contains a
    Keras model.

# Release 0.1.0

## First release

*   Initial release with support for running a python script on GCP.
*   Examples for basic workflows in Keras.
