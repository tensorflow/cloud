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
"""Docker related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import tempfile
import uuid

from . import machine_config

from docker import APIClient


try:
    from tensorflow.python.framework.versions import VERSION
except ImportError:
    # Use the latest TF docker image if a local installation is not available.
    VERSION = 'latest'


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_docker_file(docker_entry_point,
                       chief_config,
                       requirements_txt=None,
                       destination_dir='/app/',
                       docker_base_image=None):
    """Creates a Dockerfile.

    Args:
        docker_entry_point: The python program that will be run at the startup
            of the docker container.
        chief_config: `MachineConfig` that represents the configuration for
            the chief worker in a distribution cluster.
        requirements_txt: Optional string. File path to requirements.txt file
            containing aditionally pip dependencies, if any.
        destination_dir: Optional working directory in the docker container
            filesystem.
        docker_base_image: Optional base docker image to use. Defaults to None.

    Returns:
        The generated docker file path.
    """
    _, output_file = tempfile.mkstemp()

    if docker_base_image is None:
        # Get the TF docker base image to use based on the current TF version.
        docker_base_image = 'tensorflow/tensorflow:{}'.format(VERSION)
        if (chief_config.accelerator_type !=
                machine_config.AcceleratorType.NO_ACCELERATOR):
            docker_base_image += '-gpu'

    lines = ['FROM {}'.format(docker_base_image),
             'WORKDIR {}'.format(destination_dir)]

    # Copies the files from the `destination_dir` in docker daemon location
    # to the `destination_dir` in docker container filesystem.
    lines.append('COPY {} {}'.format(destination_dir, destination_dir))

    if requirements_txt is not None:
        _, requirements_txt_name = os.path.split(requirements_txt)
        dst_requirements_txt = os.path.join(requirements_txt_name)
        # install pip requirements from requirements_txt if it exists.
        lines.append('RUN if [ -e {} ]; '
                     'then pip install --no-cache -r {}; '
                     'fi'.format(dst_requirements_txt, dst_requirements_txt))

    _, docker_entry_point_file_name = os.path.split(docker_entry_point)

    # Using `ENTRYPOINT` here instead of `CMD` specifically because we want to
    # support passing user code flags.
    lines.extend([
        'ENTRYPOINT ["python", "{}"]'.format(docker_entry_point_file_name)
    ])

    content = '\n'.join(lines)
    with open(output_file, 'w') as f:
        f.write(content)
    return output_file


def get_file_path_map(entry_point,
                      docker_file_path,
                      wrapped_entry_point=None,
                      requirements_txt=None,
                      destination_dir='/app/'):
    """Maps local file paths to the docker daemon process location.

    Args:
        entry_point: String. Python file path to the file that contains the
            TensorFlow code.
        docker_file_path: The local Docker file path.
        wrapped_entry_point: Optional `wrapped_entry_point` file path. This is
            the `entry_point` wrapped in distribution strategy.
        requirements_txt: Optional string. File path to requirements.txt file
            containing aditionally pip dependencies, if any.
        destination_dir: Optional working directory in the docker container
            filesystem.

    Returns:
        A file path map.
    """
    location_map = {}
    # Map entry_point directory to the dst directory.
    entry_point_dir, _ = os.path.split(entry_point)
    if entry_point_dir == '':  # Current directory
        entry_point_dir = '.'
    location_map[entry_point_dir] = destination_dir

    # Place wrapped_entry_point in the dst directory.
    if wrapped_entry_point is not None:
        _, wrapped_entry_point_file_name = os.path.split(wrapped_entry_point)
        location_map[wrapped_entry_point] = os.path.join(
            destination_dir, wrapped_entry_point_file_name)

    # Place requirements_txt in the dst directory.
    if requirements_txt is not None:
        _, requirements_txt_name = os.path.split(
            requirements_txt)
        location_map[requirements_txt] = os.path.join(
            destination_dir, requirements_txt_name)

    # Place docker file in the root directory.
    location_map[docker_file_path] = 'Dockerfile'
    return location_map


def get_docker_image(docker_registry, tar_file_path):
    """Builds, publishes and returns a docker image.

    Args:
        docker_registry: The docker registry name.
        tar_file_path: Tarball file path that will be used as the docker
            build `fileobj`.

    Returns:
        Docker image tag.
    """
    docker_client = APIClient(version='auto')
    # create docker image from tarball
    image_uri = _build_docker_image(
        docker_registry, tar_file_path, docker_client)
    # push to the registry
    _publish_docker_image(image_uri, docker_client)
    return image_uri


def _build_docker_image(docker_registry, tar_file_path, docker_client):
    """Builds docker image.

    Args:
        docker_registry: The docker registry name.
        tar_file_path: Tarball file path that will be used as the docker
            build `fileobj`.
        docker_client: Docker API client.

    Returns:
        Docker image tag.
    """
    image_uri = _generate_name(docker_registry)
    logger.info(' Building docker image: {}'.format(image_uri))

    # `fileobj` is generally set to the Dockerfile file path. If a tar file
    # is used for docker build context (ones that includes a Dockerfile) then
    # `custom_context` should be enabled.
    with open(tar_file_path, 'rb') as fileobj:
        bld_logs_generator = docker_client.build(
            path='.',
            custom_context=True,
            fileobj=fileobj,
            tag=image_uri,
            encoding='utf-8')
    _get_logs(bld_logs_generator, 'build')
    return image_uri


def _publish_docker_image(image_uri, docker_client):
    """Publishes docker image.

    Args:
        image_uri: String, the registry name and tag.
        docker_client: Docker API client.
    """
    logger.info(' Publishing docker image: {}'.format(image_uri))
    pb_logs_generator = docker_client.push(image_uri, stream=True)
    _get_logs(pb_logs_generator, 'publish')


def _generate_name(docker_registry):
    """Returns unique name+tag for the docker image."""
    unique_tag = uuid.uuid4()
    return '{}/{}:{}'.format(
        docker_registry, 'tensorflow-train', unique_tag)


def _get_logs(logs_generator, name):
    """Decodes logs from docker and generates user friendly logs.

    Args:
        logs_generator: Generator returned from docker build/push APIs.
        name: String, 'build' or 'publish' used to identify where the generator
            came from.

    Raises:
        RuntimeError, if there are any errors when building or publishing a
        docker image.
    """
    for line in logs_generator:
        try:
            unicode_line = line.decode('utf-8').strip()
        except UnicodeError:
            logger.warn('Unable to decode logs.')
        line = json.loads(unicode_line)
        if line.get('error'):
            raise RuntimeError(
                'Docker image {} failed: {}'.format(
                    name, str(line.get('error'))))
