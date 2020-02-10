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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from . import containerize
from . import deploy
from . import gcp
from . import machine_config
from . import package
from . import preprocess
from . import validate


# Flag which indicates whether current process is running in a cloud
# environment created by the `cloud.run` API.
_IS_RUNNING_REMOTELY = False


def _is_running_remotely():
    return _IS_RUNNING_REMOTELY


def _set_running_remotely(value):
    global _IS_RUNNING_REMOTELY
    _IS_RUNNING_REMOTELY = value


def run(
    entry_point,
    requirements_txt=None,
    distribution_strategy='auto',
    docker_base_image=None,
    chief_config='auto',
    worker_config='auto',
    worker_count=0,
    region=None,
    entry_point_args=None,
    stream_logs=False,
):
    """Runs your Tensorflow code in Google Cloud Platform.

    # Arguments:
        entry_point: String. Python file path to the file that contains the
            TensorFlow code.
            Note: This path must be in the current working directory tree.
            Example: 'train.py', 'training/mnist.py'
        requirements_txt: Optional string. File path to requirements.txt file
            containing aditionally pip dependencies if any.
            Note: This path must be in the current working directory tree.
            Example: 'requirements.txt', 'deps/reqs.txt'
        distribution_strategy: 'auto' or None. Defaults to 'auto'.
            'auto' means we will take care of creating a Tensorflow
            distribution strategy instance based on the machine configurations
            you have provided using the `chief_config`, `worker_config` and
            `worker_count` params.
            - If the number of workers > 0, we will use
                `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
            - If number of GPUs > 0, we will use
                `tf.distribute.MirroredStrategy`
            If you have created a distribution strategy instance in your script
            already, please set `distribution_stratgey` as None here.
            For example, if you are using `tf.keras` custom training loops,
            you will need to create a strategy in the script for distributing
            the dataset.
        docker_base_image: Optional base docker image to use. Defaults to None.
            Example: 'gcr.io/my_gcp_project/deep_learning:v2'
            If a base docker image is not provided here, we will use a
            Tensorflow docker image (https://www.tensorflow.org/install/docker)
            as the base image. The version of TensorFlow and Python in that
            case will match your local environment.
        chief_config: Optional `MachineConfig` that represents the
            configuration for the chief worker in a distribution cluster.
            Defaults to 'auto'. 'auto' maps to a standard gpu config such as
            `COMMON_MACHINE_CONFIGS.P100_1X` (8 cpu cores, 30GB memory,
            1 Nvidia Tesla P100).
        worker_config: Optional `MachineConfig` that represents the
            configuration for the general workers in a distribution cluster.
            Defaults to 'auto'. 'auto' maps to a standard gpu config such as
            `COMMON_MACHINE_CONFIGS.P100_1X` (8 cpu cores, 30GB memory,
            1 Nvidia Tesla P100).
        worker_count: Optional integer that represents the number of general
            workers in a distribution cluster. Defaults to 0. This count does
            not include the chief worker.
        region: Optional string. Cloud region in which to submit the
            job. Defaults to 'us-central1' for GCP.
        entry_point_args: Optional list of strings. Defaults to None.
            Command line arguments to pass to the `entry_point` program.
        stream_logs: Boolean flag which when enabled streams logs back from
            the cloud job.
    """
    # If code is triggered in a cloud environment, do nothing.
    if _is_running_remotely():
        return
    _set_running_remotely(True)

    # Get defaults.
    if chief_config == 'auto':
        chief_config = machine_config.COMMON_MACHINE_CONFIGS['P100_1X']
    if worker_config == 'auto':
        worker_config = machine_config.COMMON_MACHINE_CONFIGS['P100_1X']
    region = region or gcp.get_region()
    dst_path_prefix = '/app/'
    docker_registry = 'gcr.io/{}'.format(gcp.get_project_name())

    # Run validations.
    validate.validate(
        entry_point, distribution_strategy, requirements_txt,
        chief_config, worker_config, worker_count, region,
        entry_point_args, stream_logs)

    # Create the script to run (starter_script).
    # Make the `entry_point` cloud and distribution ready.
    startup_script = preprocess.get_startup_script(
        entry_point, chief_config, worker_count, distribution_strategy)

    # Get all the files, that we need to package, mapped to the dst location.
    # This will include the startup script, requirements_txt, dockerfile,
    # files in the entry_point dir.
    dockerfile, file_map = containerize.get_file_map(
        entry_point, startup_script, chief_config, requirements_txt,
        dst_path_prefix, docker_base_image)

    # Create a tarball with the files.
    tarball = package.get_tarball(file_map)

    # Create docker image.
    docker_img = containerize.get_docker_image(docker_registry, tarball)

    # Delete all the temporary files we created.
    os.remove(startup_script)
    os.remove(dockerfile)
    os.remove(tarball)

    # Deploy docker image on the cloud.
    job_name = deploy.deploy_job(
        region, docker_img, chief_config, worker_count, worker_config,
        entry_point_args, stream_logs)
