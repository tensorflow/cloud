# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module that contains the `run` API for scaling Keras/TensorFlow jobs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from . import containerize
from . import deploy
from . import docker_config as docker_config_module
from . import machine_config
from . import preprocess
from . import validate


def remote():
  """True when code is run in a remote cloud environment by TF Cloud."""
  return bool(os.environ.get("TF_KERAS_RUNNING_REMOTELY"))


def run(
    entry_point=None,
    requirements_txt=None,
    docker_config="auto",
    distribution_strategy="auto",
    chief_config="auto",
    worker_config="auto",
    worker_count=0,
    entry_point_args=None,
    stream_logs=False,
    job_labels=None,
    **kwargs
):
    """Runs your Tensorflow code in Google Cloud Platform.

    Args:
        entry_point: Optional string. File path to the python file or iPython
            notebook that contains the TensorFlow code.
            Note this path must be in the current working directory tree.
            Example - 'train.py', 'training/mnist.py', 'mnist.ipynb'
            If `entry_point` is not provided, then
            - If you are in an iPython notebook environment, then the
                current notebook is taken as the `entry_point`.
            - Otherwise, the current python script is taken as the
                `entry_point`.
        requirements_txt: Optional string. File path to requirements.txt file
            containing additional pip dependencies if any. ie. a file with a
            list of pip dependency package names.
            Note this path must be in the current working directory tree.
            Example - 'requirements.txt', 'deps/reqs.txt'
        docker_config: Optional `DockerConfig`. Represents Docker related
            configuration for the `run` API.
            - image: Optional Docker image URI for the Docker image being built.
            - parent_image: Optional parent Docker image to use.
            - cache_from: Optional Docker image URI to be used as a cache when
                building the new Docker image.
            - image_build_bucket: Optional GCS bucket name to be used for
                building a Docker image via
                [Google Cloud Build](https://cloud.google.com/cloud-build/).
            Defaults to 'auto'. 'auto' maps to a default `tfc.DockerConfig`
            instance.
        distribution_strategy: 'auto' or None. Defaults to 'auto'.
            'auto' means we will take care of creating a Tensorflow
            distribution strategy instance based on the machine configurations
            you have provided using the `chief_config`, `worker_config` and
            `worker_count` params.
            - If the number of workers > 0, we will use
                `tf.distribute.experimental.MultiWorkerMirroredStrategy` or
                `tf.distribute.experimental.TPUStrategy` based on the
                accelerator type.
            - If number of GPUs > 0, we will use
                `tf.distribute.MirroredStrategy`
            - Otherwise, we will use `tf.distribute.OneDeviceStrategy`
            If you have created a distribution strategy instance in your script
            already, please set `distribution_strategy` as None here.
            For example, if you are using `tf.keras` custom training loops,
            you will need to create a strategy in the script for distributing
            the dataset.
        chief_config: Optional `MachineConfig` that represents the
            configuration for the chief worker in a distribution cluster.
            Defaults to 'auto'. 'auto' maps to a standard gpu config such as
            `COMMON_MACHINE_CONFIGS.T4_1X` (8 cpu cores, 30GB memory,
            1 Nvidia Tesla T4).
            For TPU strategy, `chief_config` refers to the config of the host
            that controls the TPU workers.
        worker_config: Optional `MachineConfig` that represents the
            configuration for the general workers in a distribution cluster.
            Defaults to 'auto'. 'auto' maps to a standard gpu config such as
            `COMMON_MACHINE_CONFIGS.T4_1X` (8 cpu cores, 30GB memory,
            1 Nvidia Tesla T4).
            For TPU strategy, `worker_config` should be a TPU config with
            8 TPU cores (eg. `COMMON_MACHINE_CONFIGS.TPU`).
        worker_count: Optional integer that represents the number of general
            workers in a distribution cluster. Defaults to 0. This count does
            not include the chief worker.
            For TPU strategy, `worker_count` should be set to 1.
        entry_point_args: Optional list of strings. Defaults to None.
            Command line arguments to pass to the `entry_point` program.
        stream_logs: Boolean flag which when enabled streams logs back from
            the cloud job.
        job_labels: Dict of str: str. Labels to organize jobs. You can specify
            up to 64 key-value pairs in lowercase letters and numbers, where
            the first character must be lowercase letter. For more details see
            https://cloud.google.com/ai-platform/training/docs/resource-labels.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict.
        ```
        {
          'job_id': training job id,
          'docker_image': Docker image generated for the training job,
        }
        ```
    """
    # If code is triggered in a cloud environment, do nothing.
    # This is required for the use case when `run` is invoked from within
    # a python script.
    if remote():
        return

    docker_base_image = kwargs.pop("docker_base_image", None)
    docker_image_bucket_name = kwargs.pop("docker_image_bucket_name", None)

    if kwargs:
        # We are using kwargs for forward compatibility in the cloud. For eg.,
        # if a new param is added to `run` API, this will not exist in the
        # latest tensorflow-cloud package installed in the cloud Docker envs.
        # So, if `run` is used inside a python script or notebook, this python
        # code will fail to run in the cloud even before we can check
        # `TF_KERAS_RUNNING_REMOTELY` env var because of an additional unknown
        # param.
        raise TypeError("Unknown keyword arguments: %s" % (kwargs.keys(),))

    # Get defaults values for input param

    # If `entry_point` is not provided it means that the `run` API call
    # is embedded in the script/notebook that contains the Keras module.
    # For this to run successfully in the cloud env, `tensorflow-cloud` pip
    # package is required to be installed in addition to the user provided
    # packages.
    if chief_config == "auto":
        chief_config = machine_config.COMMON_MACHINE_CONFIGS["T4_1X"]
    if worker_config == "auto":
        worker_config = machine_config.COMMON_MACHINE_CONFIGS["T4_1X"]
    if docker_config == "auto":
        docker_config = docker_config_module.DockerConfig()
    docker_config.parent_image = (docker_config.parent_image or
                                  docker_base_image)
    docker_config.image_build_bucket = (docker_config.image_build_bucket or
                                        docker_image_bucket_name)

    # Working directory in the Docker container filesystem.
    destination_dir = "/app/"
    if not isinstance(worker_count, int):
        worker_count = int(worker_count)
    called_from_notebook = _called_from_notebook()

    # Run validations.
    validate.validate(
        entry_point,
        requirements_txt,
        distribution_strategy,
        chief_config,
        worker_config,
        worker_count,
        entry_point_args,
        stream_logs,
        docker_config.image_build_bucket,
        called_from_notebook,
        job_labels=job_labels or {},
        docker_parent_image=docker_config.parent_image,
    )

    # Make the `entry_point` cloud and distribution ready.
    # A temporary script called `preprocessed_entry_point` is created.
    # This contains the `entry_point` wrapped in a distribution strategy.
    preprocessed_entry_point = None
    if (distribution_strategy == "auto"
        or entry_point.endswith("ipynb")
        or entry_point is None):
        preprocessed_entry_point = preprocess.get_preprocessed_entry_point(
            entry_point,
            chief_config,
            worker_config,
            worker_count,
            distribution_strategy,
            called_from_notebook=called_from_notebook,
        )

    # Create Docker file, generate a tarball, build and push Docker
    # image using the tarball.
    cb_args = (
        entry_point,
        preprocessed_entry_point,
        chief_config,
        worker_config,
    )
    cb_kwargs = {
        "requirements_txt": requirements_txt,
        "destination_dir": destination_dir,
        "docker_config": docker_config,
        "called_from_notebook": called_from_notebook,
    }
    if docker_config.image_build_bucket is None:
        container_builder = containerize.LocalContainerBuilder(
            *cb_args, **cb_kwargs)
    else:
        container_builder = containerize.CloudContainerBuilder(
            *cb_args, **cb_kwargs)
    docker_img_uri = container_builder.get_docker_image()

    # Delete all the temporary files we created.
    if preprocessed_entry_point is not None:
        os.remove(preprocessed_entry_point)
    for f in container_builder.get_generated_files():
        os.remove(f)

    # Deploy Docker image on the cloud.
    job_id = deploy.deploy_job(
        docker_img_uri,
        chief_config,
        worker_count,
        worker_config,
        entry_point_args,
        stream_logs,
        job_labels=job_labels,
    )

    # Call `exit` to prevent training the Keras model in the local env.
    # To stop execution after encountering a `run` API call in local env.
    if not remote() and entry_point is None and not called_from_notebook:
        sys.exit(0)
    return {
        "job_id": job_id,
        "docker_image": docker_img_uri,
    }


def _called_from_notebook():
    """Detects if we are currently executing in a notebook environment."""
    try:
        import IPython  # pylint: disable=g-import-not-at-top
    except ImportError:
        return False

    try:
        shell = IPython.get_ipython().__class__.__name__
        if "Shell" in shell:
            return True
        else:
            return False
    except NameError:
        return False
