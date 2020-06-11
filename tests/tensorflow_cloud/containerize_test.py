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
import tarfile
import unittest

from tensorflow_cloud import containerize
from tensorflow_cloud import machine_config

from mock import call, patch

try:
    from tensorflow.python.framework.versions import VERSION
except ImportError:
    # Use the latest TF docker image if a local installation is not available.
    VERSION = "latest"


class TestContainerize(unittest.TestCase):
    def setup(self):
        self.entry_point = "tests/testdata/mnist_example_using_fit.py"
        self.chief_config = machine_config.COMMON_MACHINE_CONFIGS["K80_1X"]
        self.entry_point_dir, _ = os.path.split(self.entry_point)
        self.mock_registry = "gcr.io/my-project"
        self.project_id = "my-project"

    def cleanup(self, docker_file):
        os.remove(docker_file)

    def assert_docker_file(self, expected_lines, docker_file):
        with open(docker_file, "r") as f:
            actual_lines = f.readlines()
            self.assertListEqual(expected_lines, actual_lines)

    def test_create_docker_file_defaults(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.mock_registry,
            self.project_id,
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}-gpu\n".format(VERSION),
            "WORKDIR /app/\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines, lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_with_requirements(self):
        self.setup()
        req_file = "requirements.txt"
        with open(req_file, "w") as f:
            f.writelines(["tensorflow-datasets"])

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.mock_registry,
            self.project_id,
            requirements_txt=req_file,
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}-gpu\n".format(VERSION),
            "WORKDIR /app/\n",
            "COPY /app/ /app/\n",
            "RUN if [ -e requirements.txt ]; "
            "then pip install --no-cache -r requirements.txt; fi\n",
            'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines, lcb.docker_file_path)

        os.remove(req_file)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_file_with_destination_dir(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.mock_registry,
            self.project_id,
            destination_dir="/my_app/temp/",
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}-gpu\n".format(VERSION),
            "WORKDIR /my_app/temp/\n",
            "COPY /my_app/temp/ /my_app/temp/\n",
            'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines, lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_file_with_docker_base_image(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.mock_registry,
            self.project_id,
            docker_base_image="tensorflow/tensorflow:latest",
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:latest\n",
            "WORKDIR /app/\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines, lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_file_with_cpu_config(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            machine_config.COMMON_MACHINE_CONFIGS["CPU"],
            self.mock_registry,
            self.project_id,
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}\n".format(VERSION),
            "WORKDIR /app/\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "mnist_example_using_fit.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines, lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_get_file_path_map_defaults(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.mock_registry,
            self.project_id,
        )
        lcb._create_docker_file()
        file_map = lcb._get_file_path_map()

        self.assertDictEqual(
            file_map,
            {lcb.docker_file_path: "Dockerfile", self.entry_point_dir: "/app/"},
        )

        self.cleanup(lcb.docker_file_path)

    def test_get_file_path_map_with_requirements(self):
        self.setup()
        req_file = "requirements.txt"
        with open(req_file, "w") as f:
            f.writelines(["tensorflow-datasets"])

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.mock_registry,
            self.project_id,
            requirements_txt=req_file,
        )
        lcb._create_docker_file()
        file_map = lcb._get_file_path_map()

        self.assertDictEqual(
            file_map,
            {
                lcb.docker_file_path: "Dockerfile",
                req_file: "/app/requirements.txt",
                self.entry_point_dir: "/app/",
            },
        )

        os.remove(req_file)
        self.cleanup(lcb.docker_file_path)

    def test_get_file_path_map_with_destination_dir(self):
        self.setup()

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.mock_registry,
            self.project_id,
            destination_dir="/my_app/temp/",
        )
        lcb._create_docker_file()
        file_map = lcb._get_file_path_map()

        self.assertDictEqual(
            file_map,
            {lcb.docker_file_path: "Dockerfile", self.entry_point_dir: "/my_app/temp/"},
        )

        self.cleanup(lcb.docker_file_path)

    def test_get_file_path_map_with_wrapped_entry_point(self):
        self.setup()

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.mock_registry,
            self.project_id,
            destination_dir="/my_app/temp/",
        )
        lcb._create_docker_file()
        file_map = lcb._get_file_path_map()

        self.assertDictEqual(
            file_map,
            {
                lcb.docker_file_path: "Dockerfile",
                self.entry_point_dir: "/my_app/temp/",
                self.entry_point: "/my_app/temp/mnist_example_using_fit.py",
            },
        )

        self.cleanup(lcb.docker_file_path)

    def test_get_tar_file_path(self):
        self.setup()
        req_file = "requirements.txt"
        with open(req_file, "w") as f:
            f.writelines(["tensorflow-datasets"])

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.mock_registry,
            self.project_id,
            requirements_txt=req_file,
        )

        lcb._get_tar_file_path()
        assert tarfile.is_tarfile(lcb.tar_file_path)

        tar_file = tarfile.open(lcb.tar_file_path)
        tar_file_names = [m.name for m in tar_file.getmembers()]
        self.assertIn("app/mnist_example_using_fit.py", tar_file_names)
        self.assertIn("app/requirements.txt", tar_file_names)
        self.assertIn("Dockerfile", tar_file_names)

        os.remove(req_file)
        self.cleanup(lcb.docker_file_path)

    @patch("tensorflow_cloud.containerize.logger")
    @patch("tensorflow_cloud.containerize.APIClient")
    def test_get_docker_image(self, MockAPIClient, MockLogger):
        self.setup()
        mock_registry = "gcr.io/my-project"
        mock_img_tag = mock_registry + "/tensorflow-train:abcde"

        # Verify mocking is correct and mock img tag.
        assert MockAPIClient is containerize.APIClient
        assert MockLogger is containerize.logger
        docker_client = MockAPIClient.return_value

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.mock_registry,
            self.project_id,
            destination_dir="/my_app/temp/",
        )

        def _mock_generate_name():
            return mock_img_tag

        lcb._generate_name = _mock_generate_name
        img_tag = lcb.get_docker_image()

        self.assertEqual(img_tag, mock_img_tag)

        # Verify docker APIClient is invoked as expected.
        self.assertEqual(MockAPIClient.call_count, 1)
        _, kwargs = MockAPIClient.call_args
        self.assertDictEqual(kwargs, {"version": "auto"})

        # Verify APIClient().build is invoked as expected.
        self.assertEqual(docker_client.build.call_count, 1)
        _, kwargs = docker_client.build.call_args
        expected = {
            "path": ".",
            "custom_context": True,
            "encoding": "utf-8",
            "tag": img_tag,
        }
        self.assertTrue(set(expected.items()).issubset(set(kwargs.items())))

        # Verify APIClient().push is invoked as expected.
        self.assertEqual(docker_client.push.call_count, 1)
        args, kwargs = docker_client.push.call_args
        self.assertListEqual(list(args), [img_tag])
        self.assertDictEqual(kwargs, {"stream": True})

        # Verify logger info calls.
        self.assertEqual(MockLogger.info.call_count, 2)
        MockLogger.info.assert_has_calls(
            [
                call(r"Building docker image: " + img_tag),
                call(r"Publishing docker image: " + img_tag),
            ]
        )
        self.cleanup(lcb.docker_file_path)
