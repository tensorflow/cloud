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
"""Module that contains the `run` API for scaling Keras/Tensorflow jobs."""

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


def run(entry_point=None,
        requirements_txt=None,
        distribution_strategy='auto',
        docker_base_image=None,
        chief_config='auto',
        worker_config='auto',
        worker_count=0,
        entry_point_args=None,
        stream_logs=False):
    """Runs your Tensorflow code in Google Cloud Platform.

    Args:
        entry_point: Optional string. Python file path to the file that
            contains the TensorFlow code.
            Note: This path must be in the current working directory tree.
            Example: 'train.py', 'training/mnist.py'
            If `entry_point` is not provided, then the current python script is
            assumed to be the `entry_point`.
        requirements_txt: Optional string. File path to requirements.txt file
            containing aditional pip dependencies if any. ie. a file with a
            list of pip dependency package names.
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
            - Otherwise, we will use `tf.distribute.OneDeviceStrategy`
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
            If both docker_base_image and a local TF installation are not
            available, the latest TF docker image will be used.
            For example: 'tensorflow/tensorflow:latest-gpu'
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
        entry_point_args: Optional list of strings. Defaults to None.
            Command line arguments to pass to the `entry_point` program.
        stream_logs: Boolean flag which when enabled streams logs back from
            the cloud job.
    """
    # If code is triggered in a cloud environment, do nothing.
    print('ENVIRONMENT VARIABLE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', os.environ.get('TF_KERAS_RUNNING_REMOTELY'))
    if os.environ.get('TF_KERAS_RUNNING_REMOTELY'):
        return

    # Get defaults values for input param

    # If `entry_point` is not provided it means that the `run` API call
    # is embedded in the script that contains the Keras module. For this
    # script to run successfully in the cloud env, `tensorflow-cloud` pip
    # package is required to be installed in addition to the user provided
    # packages.
    install_tf_cloud = False
    if entry_point is None:
        entry_point = sys.argv[0]
        install_tf_cloud = True
    if chief_config == 'auto':
        chief_config = machine_config.COMMON_MACHINE_CONFIGS['P100_1X']
    if worker_config == 'auto':
        worker_config = machine_config.COMMON_MACHINE_CONFIGS['P100_1X']
    region = gcp.get_region()
    # Working directory in the docker container filesystem.
    destination_dir = '/app/'
    if not isinstance(worker_count, int):
        worker_count = int(worker_count)
    # Default location to which the docker image that is created is pushed.
    docker_registry = 'gcr.io/{}'.format(gcp.get_project_name())

    # Run validations.
    validate.validate(
        entry_point, requirements_txt, distribution_strategy,
        chief_config, worker_config, worker_count, region,
        entry_point_args, stream_logs)

    # Make the `entry_point` cloud and distribution ready.
    # A temporary script called `wrapped_entry_point` is created.
    # This contains the `entry_point` wrapped in distribution strategy.
    wrapped_entry_point = None
    if (distribution_strategy == 'auto' and
        chief_config.accelerator_type !=
            machine_config.AcceleratorType.NO_ACCELERATOR):
        wrapped_entry_point = preprocess.get_wrapped_entry_point(
            entry_point, chief_config, worker_count)

    # Create docker file.
    docker_entry_point = wrapped_entry_point or entry_point
    docker_file = containerize.create_docker_file(
        docker_entry_point,
        chief_config,
        requirements_txt=requirements_txt,
        destination_dir=destination_dir,
        docker_base_image=docker_base_image,
        install_tf_cloud=install_tf_cloud)

    # Get all the files, that we need to package, mapped to the destination
    # location. This will include the wrapped_entry_point, requirements_txt,
    # dockerfile and the entry_point directory tree.
    file_map = containerize.get_file_path_map(
        entry_point,
        docker_file,
        wrapped_entry_point=wrapped_entry_point,
        requirements_txt=requirements_txt,
        destination_dir=destination_dir)

    # Create a tarball with the files.
    tar_file_path = package.get_tar_file_path(file_map)

    # Build and push docker image.
    docker_img_uri = containerize.get_docker_image(
        docker_registry, tar_file_path)

    # Delete all the temporary files we created.
    if wrapped_entry_point is not None:
        os.remove(wrapped_entry_point)
    os.remove(docker_file)
    os.remove(tar_file_path)

    # Deploy docker image on the cloud.
    job_name = deploy.deploy_job(
        region,
        docker_img_uri,
        chief_config,
        worker_count,
        worker_config,
        entry_point_args,
        stream_logs)

    # Call `exit` to prevent training the Keras model in the local env.
    # To stop execution after encountering a `run` API call in local env.
    sys.exit(0)
