# Installation

## Requirements

-   Python >= 3.6
-   [A Google Cloud project](https://cloud.google.com/ai-platform/docs/getting-started-keras#set_up_your_project)
-   An
    [authenticated GCP account](https://cloud.google.com/ai-platform/docs/getting-started-keras#authenticate_your_gcp_account)
-   [Google AI platform](https://cloud.google.com/ai-platform/) APIs enabled for
    your GCP account. We use the AI platform for deploying docker images on GCP.
-   If you are using `tensorflow_cloud` in a local development environment (not
    a Kaggle or Colab notebook):

    -   Either a functioning version of
        [docker](https://docs.docker.com/engine/install/) if you want to use a
        local docker process for your build, or
        [create a cloud storage bucket](https://cloud.google.com/ai-platform/docs/getting-started-keras#create_a_bucket)
        to use with [Google Cloud build](https://cloud.google.com/cloud-build)
        for docker image build and publishing.

    -   [Authenticate to your Docker Container Registry](https://cloud.google.com/container-registry/docs/advanced-authentication#gcloud-helper)

For detailed end to end setup instructions, please see
[Setup instructions](#setup-instructions).

## Install latest release

```shell
pip install -U tensorflow-cloud
```

## Install from source

```shell
git clone https://github.com/tensorflow/cloud.git
cd cloud
pip install src/python/.
```
