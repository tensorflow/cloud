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

import json
import logging
import os
import random
import string
import sys
import tempfile

from . import machine_config

from docker import APIClient
from tensorflow.python.framework.versions import VERSION


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_file_map(entry_point, startup_script, chief_config,
                 requirements_txt=None, dst_dir='/app/',
                 docker_base_image=None):
    location_map = {}
    # Map entry_point directory to the dst directory.
    entry_point_dir, _ = os.path.split(entry_point)
    if entry_point_dir == '':  # Current directory
        entry_point_dir = '.'
    location_map[entry_point_dir] = dst_dir

    # Place startup_script in the dst directory.
    _, startup_file_name = os.path.split(startup_script)
    location_map[startup_script] = os.path.join(dst_dir, startup_file_name)

    # Place requirements_txt in the dst directory.
    if requirements_txt is not None:
        _, requirements_txt_name = os.path.split(
            requirements_txt)
        location_map[requirements_txt] = os.path.join(
            dst_dir, requirements_txt_name)

    # Place docker file in the root directory.
    docker_file = _create_docker_file(
        startup_script, chief_config, requirements_txt,
        dst_dir, docker_base_image)
    location_map[docker_file] = 'Dockerfile'
    return docker_file, location_map


def get_docker_image(docker_registry, tar_file):
    docker_client = APIClient(version='auto')
    # create docker image from tarball
    image_tag = _build_docker_image(docker_registry, tar_file, docker_client)
    # push to the registry
    _publish_docker_image(image_tag, docker_client)
    return image_tag


def _build_docker_image(docker_registry, tar_file, docker_client):
    image_tag = _generate_name(docker_registry)
    logger.info(' Building docker image: {}'.format(image_tag))
    with open(tar_file, 'rb') as fileobj:
        bld_logs_generator = docker_client.build(
            path='.',
            custom_context=True,
            fileobj=fileobj,
            tag=image_tag,
            encoding='utf-8')
    _get_logs(bld_logs_generator, 'build')
    return image_tag


def _publish_docker_image(image_tag, docker_client):
    logger.info(' Publishing docker image: {}'.format(image_tag))
    pb_logs_generator = docker_client.push(image_tag, stream=True)
    _get_logs(pb_logs_generator, 'publish')


def _create_docker_file(startup_script, chief_config, requirements_txt,
                        dst_dir, docker_base_image):
    # Create a Dockerfile.
    _, output_file = tempfile.mkstemp()

    if docker_base_image is None:
        # Get the TF docker base image to use based on the current TF
        # and python version.
        docker_base_image = 'tensorflow/tensorflow:{}'.format(VERSION)
        if (chief_config.accelerator_type !=
                machine_config.AcceleratorType.NO_ACCELERATOR):
            docker_base_image += '-gpu'

        if sys.version_info[0] == 3:
            docker_base_image += '-py3'

    lines = ['FROM {}'.format(docker_base_image), 'WORKDIR {}'.format(dst_dir)]
    lines.append('COPY {} {}'.format(dst_dir, dst_dir))

    if requirements_txt is not None:
        _, requirements_txt_name = os.path.split(requirements_txt)
        dst_requirements_txt = os.path.join(requirements_txt_name)
        # install pip requirements from requirements_txt if it exists.
        lines.append('RUN if [ -e {} ]; '
                     'then pip install --no-cache -r {}; '
                     'fi'.format(dst_requirements_txt, dst_requirements_txt))

    _, startup_file_name = os.path.split(startup_script)
    # Using `ENTRYPOINT` here instead of `CMD` specifically because we want to
    # support passing user code flags.
    lines.extend([
        'ENTRYPOINT ["python", "{}"]'.format(startup_file_name)
    ])

    content = '\n'.join(lines)
    with open(output_file, 'w') as f:
        f.write(content)
    return output_file


def _generate_name(docker_registry):
    unique_tag = ''.join(random.choice(
        string.ascii_lowercase + string.digits) for _ in range(32))
    return '{}/{}:{}'.format(
        docker_registry, 'tensorflow-train', unique_tag)


def _get_logs(logs_generator, name):
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
