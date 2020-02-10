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
"""Tests for the cloud docker containerization module."""

import os
import unittest

from tensorflow_cloud import containerize
from tensorflow_cloud import machine_config
from tensorflow_cloud import package

from mock import call, patch
from tensorflow.python.framework.versions import VERSION


class TestContainerize(unittest.TestCase):

    def setup(self):
        self.entry_point = 'tests/testdata/mnist_example_using_fit.py'
        self.chief_config = machine_config.COMMON_MACHINE_CONFIGS['K80_1X']

    def cleanup(self):
        os.remove(self.docker_file)

    def assert_docker_file(self, expected_lines):
        with open(self.docker_file, 'r') as f:
            actual_lines = f.readlines()
            self.assertListEqual(expected_lines, actual_lines)

    def test_get_file_map_defaults(self):
        self.setup()
        self.docker_file, file_map = containerize.get_file_map(
            self.entry_point, self.entry_point, self.chief_config)

        expected_docker_file_lines = [
          'FROM tensorflow/tensorflow:{}-gpu\n'.format(VERSION),
          'WORKDIR /app/\n',
          'COPY /app/ /app/\n',
          'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines)

        entry_point_dir, _ = os.path.split(self.entry_point)
        self.assertDictEqual(file_map, {
            self.docker_file: 'Dockerfile',
            entry_point_dir: '/app/',
            self.entry_point: '/app/mnist_example_using_fit.py'})

        self.cleanup()

    def test_get_file_map_with_requirements(self):
        self.setup()
        req_file = 'requirements.txt'
        with open(req_file, 'w') as f:
            f.writelines(['tensorflow-datasets'])

        self.docker_file, file_map = containerize.get_file_map(
            self.entry_point, self.entry_point, self.chief_config,
            requirements_txt=req_file)

        expected_docker_file_lines = [
          'FROM tensorflow/tensorflow:{}-gpu\n'.format(VERSION),
          'WORKDIR /app/\n',
          'COPY /app/ /app/\n',
          'RUN if [ -e requirements.txt ]; '
          'then pip install --no-cache -r requirements.txt; fi\n',
          'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines)

        entry_point_dir, _ = os.path.split(self.entry_point)
        self.assertDictEqual(file_map, {
            self.docker_file: 'Dockerfile',
            req_file: '/app/requirements.txt',
            entry_point_dir: '/app/',
            self.entry_point: '/app/mnist_example_using_fit.py'})

        os.remove(req_file)
        self.cleanup()

    def test_get_file_map_with_dst_dir(self):
        self.setup()
        self.docker_file, file_map = containerize.get_file_map(
            self.entry_point, self.entry_point, self.chief_config,
            dst_dir='/my_app/temp/')

        expected_docker_file_lines = [
          'FROM tensorflow/tensorflow:{}-gpu\n'.format(VERSION),
          'WORKDIR /my_app/temp/\n',
          'COPY /my_app/temp/ /my_app/temp/\n',
          'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines)

        entry_point_dir, _ = os.path.split(self.entry_point)
        self.assertDictEqual(file_map, {
            self.docker_file: 'Dockerfile',
            entry_point_dir: '/my_app/temp/',
            self.entry_point: '/my_app/temp/mnist_example_using_fit.py'})

        self.cleanup()

    def test_get_file_map_with_docker_base_image(self):
        self.setup()
        self.docker_file, file_map = containerize.get_file_map(
            self.entry_point, self.entry_point, self.chief_config,
            docker_base_image='tensorflow/tensorflow:latest')

        expected_docker_file_lines = [
          'FROM tensorflow/tensorflow:latest\n',
          'WORKDIR /app/\n',
          'COPY /app/ /app/\n',
          'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines)

        entry_point_dir, _ = os.path.split(self.entry_point)
        self.assertDictEqual(file_map, {
            self.docker_file: 'Dockerfile',
            entry_point_dir: '/app/',
            self.entry_point: '/app/mnist_example_using_fit.py'})

        self.cleanup()

    def test_get_file_map_with_gpu_config(self):
        self.setup()
        self.docker_file, file_map = containerize.get_file_map(
            self.entry_point,  # mocking startup_script
            self.entry_point,
            machine_config.COMMON_MACHINE_CONFIGS['CPU'])

        expected_docker_file_lines = [
          'FROM tensorflow/tensorflow:{}\n'.format(VERSION),
          'WORKDIR /app/\n',
          'COPY /app/ /app/\n',
          'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines)

        entry_point_dir, _ = os.path.split(self.entry_point)
        self.assertDictEqual(file_map, {
            self.docker_file: 'Dockerfile',
            entry_point_dir: '/app/',
            self.entry_point: '/app/mnist_example_using_fit.py'})

        self.cleanup()

    @patch('tensorflow_cloud.containerize.logger')
    @patch('tensorflow_cloud.containerize.APIClient')
    def test_get_docker_image(self, MockAPIClient, MockLogger):
        self.setup()
        mock_registry = 'gcr.io/my-project'
        mock_img_tag = mock_registry + '/tensorflow-train:abcde'

        # Verify mocking is correct and mock img tag.
        assert MockAPIClient is containerize.APIClient
        assert MockLogger is containerize.logger
        docker_client = MockAPIClient.return_value

        def _mock_generate_name(docker_registry):
            return mock_img_tag
        containerize._generate_name = _mock_generate_name

        entry_point_dir = os.path.dirname(os.path.abspath(self.entry_point))
        self.docker_file, file_map = containerize.get_file_map(
            self.entry_point, self.entry_point, self.chief_config)
        tarball = package.get_tarball(file_map)
        img_tag = containerize.get_docker_image(mock_registry, tarball)

        self.assertEqual(img_tag, mock_img_tag)

        # Verify docker APIClient is invoked as expected.
        self.assertEqual(MockAPIClient.call_count, 1)
        _, kwargs = MockAPIClient.call_args
        self.assertDictEqual(kwargs, {'version': 'auto'})

        # Verify APIClient().build is invoked as expected.
        self.assertEqual(docker_client.build.call_count, 1)
        _, kwargs = docker_client.build.call_args
        expected = {
            'path': '.',
            'custom_context': True,
            'encoding': 'utf-8',
            'tag': img_tag
        }
        self.assertTrue(set(expected.items()).issubset(set(kwargs.items())))

        # Verify APIClient().push is invoked as expected.
        self.assertEqual(docker_client.push.call_count, 1)
        args, kwargs = docker_client.push.call_args
        self.assertListEqual(list(args), [img_tag])
        self.assertDictEqual(kwargs, {'stream': True})

        # Verify logger info calls.
        self.assertEqual(MockLogger.info.call_count, 2)
        MockLogger.info.assert_has_calls([
            call(r' Building docker image: ' + img_tag),
            call(r' Publishing docker image: ' + img_tag)])
        self.cleanup()
