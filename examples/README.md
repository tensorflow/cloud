# TensorFlow Cloud Examples

If you have not set up a Google Cloud Project yet start by running
[Google Cloud Project Setup Instructions](https://github.com/tensorflow/cloud/blob/master/examples/google_cloud_project_setup_instructions.ipynb).
This guide will help you setup a Google Cloud Project, and configure it for
running tensorflow-cloud projects. Once you have set up your Google Cloud
Project move on to one of the samples below.

*   **[Google Cloud Project Setup Instructions](https://github.com/tensorflow/cloud/blob/master/examples/google_cloud_project_setup_instructions.ipynb)**

    This guide is to help first time users set up a Google Cloud Platform
    account specifically with the intention to use
    [tensorflow_cloud](https://github.com/tensorflow/cloud) to easily run
    training at scale on Google Cloud AI Platform.
    [tensorflow_cloud](https://github.com/tensorflow/cloud) provides APIs that
    allow users to easily go from debugging, training, tuning Keras and
    TensorFlow code in a local or kaggle environment to distributed
    training/tuning on Cloud.

*   **[Distributed training NasNet with tensorflow_cloud and Google Cloud](https://github.com/tensorflow/cloud/blob/master/examples/distributed_training_nasnet_with_tensorflow_cloud.ipynb)**

    This example is based on
    [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
    to demonstrate how to train a
    [NasNetMobile](https://keras.io/api/applications/nasnet/#nasnetmobile-function)
    model using [tensorflow_cloud](https://github.com/tensorflow/cloud) and
    Google Cloud Platform at scale using distributed training.

*   **[HP Tuning CIFAR10 on Google Cloud with tensorflow_cloud and CloudTuner](https://github.com/tensorflow/cloud/blob/master/examples/hp_tuning_cifar10_using_google_cloud.ipynb)**

    This example is based on
    [Keras-Tuner CIFAR10 sample](https://github.com/keras-team/keras-tuner/blob/master/examples/cifar10.py)
    to demonstrate how to run HP tuning jobs using
    [tensorflow_cloud](https://github.com/tensorflow/cloud) and Google Cloud
    Platform at scale.

*   **[Tuning a wide and Deep model using Google Cloud](https://github.com/tensorflow/cloud/blob/master/examples/hp_tuning_wide_and_deep_model.ipynb)**

    In this example we will use CloudTuner and Google Cloud to Tune a
    [Wide and Deep Model](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
    based on the tunable model introduced in
    [structured data learning with Wide, Deep, and Cross networks](https://keras.io/examples/structured_data/wide_deep_cross_networks/).
    In this example we will use the data set from
    [CAIIS Dogfood Day](https://www.kaggle.com/c/caiis-dogfood-day-2020/overview)
