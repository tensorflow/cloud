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
import tempfile
import uuid

from absl.testing import absltest
import docker
import mock

from tensorflow_cloud.core import containerize
from tensorflow_cloud.core import docker_config
from tensorflow_cloud.core import gcp
from tensorflow_cloud.core import machine_config
from tensorflow_cloud.utils import google_api_client
from tensorflow_cloud.utils import tf_utils

_TF_VERSION = tf_utils.get_version()


class TestContainerize(absltest.TestCase):

    def setup(self, requests_get_return_value=True):
        self.entry_point = "sample.py"
        self.chief_config = machine_config.COMMON_MACHINE_CONFIGS["K80_1X"]
        self.worker_config = machine_config.COMMON_MACHINE_CONFIGS["K80_1X"]
        self.entry_point_dir = "."
        self.project_id = "my-project"

        self._mock_request_get = mock.patch("requests.get").start()
        self._mock_request_get.return_value = mock.Mock()
        self._mock_request_get.return_value.ok = requests_get_return_value

        mock.patch.object(
            gcp,
            "get_project_name",
            autospec=True,
            return_value="my-project",
        ).start()

        mock.patch.object(
            uuid,
            "uuid4",
            autospec=True,
            return_value="abcde",
        ).start()

    def cleanup(self, docker_file):
        mock.patch.stopall()
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
            self.worker_config,
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}-gpu\n".format(_TF_VERSION),
            "WORKDIR /app/\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "sample.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines,
                                lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_with_requirements(self):
        self.setup()
        req_file = os.path.join(tempfile.mkdtemp(), "requirements.txt")
        with open(req_file, "w") as f:
            f.writelines(["tensorflow-datasets"])

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.worker_config,
            requirements_txt=req_file,
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}-gpu\n".format(_TF_VERSION),
            "WORKDIR /app/\n",
            "COPY /app/requirements.txt /app/requirements.txt\n",
            "RUN if [ -e requirements.txt ]; "
            "then pip install --no-cache -r requirements.txt; fi\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "sample.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines,
                                lcb.docker_file_path)

        os.remove(req_file)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_file_with_destination_dir(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.worker_config,
            destination_dir="/my_app/temp/",
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}-gpu\n".format(_TF_VERSION),
            "WORKDIR /my_app/temp/\n",
            "COPY /my_app/temp/ /my_app/temp/\n",
            'ENTRYPOINT ["python", "sample.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines,
                                lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_file_with_docker_parent_image(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.worker_config,
            docker_config=docker_config.DockerConfig(
                parent_image="tensorflow/tensorflow:latest"),
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:latest\n",
            "WORKDIR /app/\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "sample.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines,
                                lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_file_with_kaggle_base_image(self):
        self.setup()
        os.environ["KAGGLE_DOCKER_IMAGE"] = "gcr.io/kaggle-images/python"
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.worker_config,
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM gcr.io/kaggle-images/python\n",
            "WORKDIR /app/\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "sample.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines,
                                lcb.docker_file_path)
        del os.environ["KAGGLE_DOCKER_IMAGE"]
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_file_with_cpu_config(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            machine_config.COMMON_MACHINE_CONFIGS["CPU"],
            self.worker_config,
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}\n".format(_TF_VERSION),
            "WORKDIR /app/\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "sample.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines,
                                lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_create_docker_file_with_tpu_config(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            machine_config.COMMON_MACHINE_CONFIGS["CPU"],
            machine_config.COMMON_MACHINE_CONFIGS["TPU"],
        )
        lcb._create_docker_file()

        expected_docker_file_lines = [
            "FROM tensorflow/tensorflow:{}\n".format(_TF_VERSION),
            "WORKDIR /app/\n",
            "RUN pip install cloud-tpu-client\n",
            "COPY /app/ /app/\n",
            'ENTRYPOINT ["python", "sample.py"]',
        ]
        self.assert_docker_file(expected_docker_file_lines,
                                lcb.docker_file_path)
        self.cleanup(lcb.docker_file_path)

    def test_get_file_path_map_defaults(self):
        self.setup()
        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.worker_config,
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
        req_file = os.path.join(tempfile.mkdtemp(), "requirements.txt")
        with open(req_file, "w") as f:
            f.writelines(["tensorflow-datasets"])

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            None,
            self.chief_config,
            self.worker_config,
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
            self.worker_config,
            destination_dir="/my_app/temp/",
        )
        lcb._create_docker_file()
        file_map = lcb._get_file_path_map()

        self.assertDictEqual(
            file_map,
            {lcb.docker_file_path: "Dockerfile",
             self.entry_point_dir: "/my_app/temp/"},
        )

        self.cleanup(lcb.docker_file_path)

    def test_get_file_path_map_with_wrapped_entry_point(self):
        self.setup()

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.worker_config,
            destination_dir="/my_app/temp/",
        )
        lcb._create_docker_file()
        file_map = lcb._get_file_path_map()

        self.assertDictEqual(
            file_map,
            {
                lcb.docker_file_path: "Dockerfile",
                self.entry_point_dir: "/my_app/temp/",
                self.entry_point: "/my_app/temp/sample.py",
            },
        )

        self.cleanup(lcb.docker_file_path)

    @mock.patch("tensorflow_cloud.core.containerize.logger")  # pylint: disable=line-too-long
    @mock.patch("docker.APIClient")
    def test_get_docker_image(self, mock_api_client, mock_logger):
        self.setup()

        # Verify mocking is correct and mock img tag.
        assert mock_api_client is docker.APIClient
        assert mock_logger is containerize.logger
        docker_client = mock_api_client.return_value

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.worker_config,
            destination_dir="/my_app/temp/",
        )

        lcb._get_file_path_map = mock.Mock()
        lcb._get_file_path_map.return_value = {}
        lcb.tar_file_path = ""
        with mock.patch("builtins.open", mock.mock_open(read_data="")):
            img_tag = lcb.get_docker_image()

        self.assertEqual(img_tag, "gcr.io/my-project/tf_cloud_train:abcde")

        # Verify docker APIClient is invoked as expected.
        self.assertEqual(mock_api_client.call_count, 1)
        _, kwargs = mock_api_client.call_args
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
        self.assertDictEqual(kwargs, {"decode": True, "stream": True})

        # Verify logger info calls.
        self.assertEqual(mock_logger.info.call_count, 2)
        mock_logger.info.assert_has_calls(
            [
                mock.call("Building Docker image: %s", img_tag),
                mock.call("Publishing Docker image: %s", img_tag),
            ]
        )
        self.cleanup(lcb.docker_file_path)

    @mock.patch("googleapiclient.discovery.build")
    @mock.patch("google.cloud.storage.Client")
    def test_get_docker_image_cloud_build(
        self, mock_gcs_client, mock_discovery_build
    ):
        self.setup()

        lcb = containerize.CloudContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.worker_config,
            destination_dir="/my_app/temp/",
            docker_config=docker_config.DockerConfig(
                image_build_bucket="test_gcs_bucket"),
        )

        # Mock tar file generation
        lcb._get_file_path_map = mock.Mock()
        lcb._get_file_path_map.return_value = {}

        # Mock cloud build return value
        proj_ret_val = mock_discovery_build.return_value.projects.return_value
        builds_ret_val = proj_ret_val.builds.return_value
        create_ret_val = builds_ret_val.create.return_value
        create_ret_val.execute.return_value = {
            "metadata": {"build": {"id": "test_build_id"}}
        }

        get_ret_val = builds_ret_val.get.return_value
        get_ret_val.execute.return_value = {"status": "SUCCESS"}

        # Get docker image
        img_tag = lcb.get_docker_image()
        self.assertEqual(img_tag, "gcr.io/my-project/tf_cloud_train:abcde")

        # Verify gcs get_bucket is invoked
        client_ret_val = mock_gcs_client.return_value
        self.assertEqual(client_ret_val.get_bucket.call_count, 1)
        args, _ = client_ret_val.get_bucket.call_args
        self.assertListEqual(list(args), ["test_gcs_bucket"])

        # Verify that a new blob is created in the bucket
        get_bucket_ret_val = client_ret_val.get_bucket.return_value
        self.assertEqual(get_bucket_ret_val.blob.call_count, 1)
        args, _ = get_bucket_ret_val.blob.call_args
        self.assertLen(args, 1)
        storage_object_name = args[0]
        self.assertTrue(storage_object_name, "tf_cloud_train_tar_")

        # Verify that tarfile is added to the blob
        blob_ret_val = get_bucket_ret_val.blob.return_value
        self.assertEqual(blob_ret_val.upload_from_filename.call_count, 1)

        # Verify discovery API is invoked as expected
        self.assertEqual(mock_discovery_build.call_count, 1)
        args, kwargs = mock_discovery_build.call_args
        self.assertListEqual(list(args), ["cloudbuild", "v1"])
        self.assertDictEqual(
            kwargs,
            {
                "cache_discovery": False,
                "requestBuilder": google_api_client.TFCloudHttpRequest,
            },
        )

        # Verify cloud build create is invoked as expected
        self.assertEqual(builds_ret_val.create.call_count, 1)
        _, kwargs = builds_ret_val.create.call_args

        request_dict = {}
        request_dict["projectId"] = self.project_id
        request_dict["timeout"] = "1200s"
        request_dict["images"] = [["gcr.io/my-project/tf_cloud_train:abcde"]]
        request_dict["steps"] = [{
            "name": "gcr.io/cloud-builders/docker",
            "args": [
                "build",
                "-t",
                "gcr.io/my-project/tf_cloud_train:abcde",
                "."],
        }]
        request_dict["source"] = {
            "storageSource": {
                "bucket": "test_gcs_bucket",
                "object": storage_object_name,
            }
        }
        self.assertDictEqual(
            kwargs,
            {
                "projectId": self.project_id,
                "body": request_dict,
            },
        )

        # Verify cloud build get is invoked as expected
        self.assertEqual(builds_ret_val.get.call_count, 1)
        _, kwargs = builds_ret_val.get.call_args
        self.assertDictEqual(
            kwargs,
            {
                "projectId": self.project_id,
                "id": "test_build_id",
            },
        )

        self.cleanup(lcb.docker_file_path)

    @mock.patch("tensorflow_cloud.core.containerize.logger")  # pylint: disable=line-too-long
    @mock.patch("docker.APIClient")
    def test_get_docker_image_with_custom_image_uri(
        self, mock_api_client, mock_logger):
        self.setup()

        # Verify mocking is correct and mock img tag.
        assert mock_api_client is docker.APIClient
        assert mock_logger is containerize.logger
        docker_client = mock_api_client.return_value

        lcb = containerize.LocalContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.worker_config,
            destination_dir="/my_app/temp/",
            docker_config=docker_config.DockerConfig(image="gcr.io/test-name"),
        )

        lcb._get_file_path_map = mock.Mock()
        lcb._get_file_path_map.return_value = {}
        lcb.tar_file_path = ""
        with mock.patch("builtins.open", mock.mock_open(read_data="")):
            img_tag = lcb.get_docker_image()

        self.assertEqual(img_tag, "gcr.io/test-name")

        # Verify docker APIClient is invoked as expected.
        self.assertEqual(mock_api_client.call_count, 1)
        _, kwargs = mock_api_client.call_args
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
        self.assertDictEqual(kwargs, {"decode": True, "stream": True})

        # Verify logger info calls.
        self.assertEqual(mock_logger.info.call_count, 2)
        mock_logger.info.assert_has_calls(
            [
                mock.call("Building Docker image: %s", img_tag),
                mock.call("Publishing Docker image: %s", img_tag),
            ]
        )
        self.cleanup(lcb.docker_file_path)

    @mock.patch("googleapiclient.discovery.build")
    @mock.patch("google.cloud.storage.Client")
    def test_get_docker_image_cloud_build_with_custom_image_uri(
        self, mock_gcs_client, mock_discovery_build
    ):
        self.setup()

        lcb = containerize.CloudContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.worker_config,
            destination_dir="/my_app/temp/",
            docker_config=docker_config.DockerConfig(
                image="gcr.io/test-name", image_build_bucket="test_gcs_bucket"),
        )

        # Mock tar file generation
        lcb._get_file_path_map = mock.Mock()
        lcb._get_file_path_map.return_value = {}

        # Mock cloud build return value
        proj_ret_val = mock_discovery_build.return_value.projects.return_value
        builds_ret_val = proj_ret_val.builds.return_value
        create_ret_val = builds_ret_val.create.return_value
        create_ret_val.execute.return_value = {
            "metadata": {"build": {"id": "test_build_id"}}
        }

        get_ret_val = builds_ret_val.get.return_value
        get_ret_val.execute.return_value = {"status": "SUCCESS"}

        # Get docker image
        img_tag = lcb.get_docker_image()
        self.assertEqual(img_tag, "gcr.io/test-name")

        # Verify gcs get_bucket is invoked
        client_ret_val = mock_gcs_client.return_value
        self.assertEqual(client_ret_val.get_bucket.call_count, 1)
        args, _ = client_ret_val.get_bucket.call_args
        self.assertListEqual(list(args), ["test_gcs_bucket"])

        # Verify that a new blob is created in the bucket
        get_bucket_ret_val = client_ret_val.get_bucket.return_value
        self.assertEqual(get_bucket_ret_val.blob.call_count, 1)
        args, _ = get_bucket_ret_val.blob.call_args
        self.assertLen(args, 1)
        storage_object_name = args[0]
        self.assertTrue(storage_object_name, "tf_cloud_train_tar_")

        # Verify that tarfile is added to the blob
        blob_ret_val = get_bucket_ret_val.blob.return_value
        self.assertEqual(blob_ret_val.upload_from_filename.call_count, 1)

        # Verify discovery API is invoked as expected
        self.assertEqual(mock_discovery_build.call_count, 1)
        args, kwargs = mock_discovery_build.call_args
        self.assertListEqual(list(args), ["cloudbuild", "v1"])
        self.assertDictEqual(
            kwargs,
            {
                "cache_discovery": False,
                "requestBuilder": google_api_client.TFCloudHttpRequest,
            },
        )

        # Verify cloud build create is invoked as expected
        self.assertEqual(builds_ret_val.create.call_count, 1)
        _, kwargs = builds_ret_val.create.call_args

        request_dict = {}
        request_dict["projectId"] = self.project_id
        request_dict["timeout"] = "1200s"
        request_dict["images"] = [[img_tag]]
        request_dict["steps"] = [{
            "name": "gcr.io/cloud-builders/docker",
            "entrypoint": "bash",
            "args": [
                "-c",
                "docker pull gcr.io/test-name || exit 0",
            ],
        }, {
            "name": "gcr.io/cloud-builders/docker",
            "args": [
                "build",
                "-t",
                img_tag,
                "--cache-from",
                img_tag,
                "."],
        }]
        request_dict["source"] = {
            "storageSource": {
                "bucket": "test_gcs_bucket",
                "object": storage_object_name,
            }
        }
        self.assertDictEqual(
            kwargs,
            {
                "projectId": self.project_id,
                "body": request_dict,
            },
        )

        # Verify cloud build get is invoked as expected
        self.assertEqual(builds_ret_val.get.call_count, 1)
        _, kwargs = builds_ret_val.get.call_args
        self.assertDictEqual(
            kwargs,
            {
                "projectId": self.project_id,
                "id": "test_build_id",
            },
        )

        self.cleanup(lcb.docker_file_path)

    @mock.patch("googleapiclient.discovery.build")
    @mock.patch("google.cloud.storage.Client")
    def test_get_docker_image_cloud_build_with_cache(
        self, mock_gcs_client, mock_discovery_build
    ):
        self.setup()

        lcb = containerize.CloudContainerBuilder(
            self.entry_point,
            self.entry_point,
            self.chief_config,
            self.worker_config,
            destination_dir="/my_app/temp/",
            docker_config=docker_config.DockerConfig(
                image="gcr.io/test-name:1",
                parent_image="tensorflow/tensorflow:latest",
                cache_from="gcr.io/test-name:0",
                image_build_bucket="test_gcs_bucket"),
        )

        # Mock tar file generation
        lcb._get_file_path_map = mock.Mock()
        lcb._get_file_path_map.return_value = {}

        # Mock cloud build return value
        proj_ret_val = mock_discovery_build.return_value.projects.return_value
        builds_ret_val = proj_ret_val.builds.return_value
        create_ret_val = builds_ret_val.create.return_value
        create_ret_val.execute.return_value = {
            "metadata": {"build": {"id": "test_build_id"}}
        }

        get_ret_val = builds_ret_val.get.return_value
        get_ret_val.execute.return_value = {"status": "SUCCESS"}

        # Get docker image
        img_tag = lcb.get_docker_image()
        self.assertEqual(img_tag, "gcr.io/test-name:1")

        # Verify gcs get_bucket is invoked
        client_ret_val = mock_gcs_client.return_value
        self.assertEqual(client_ret_val.get_bucket.call_count, 1)
        args, _ = client_ret_val.get_bucket.call_args
        self.assertListEqual(list(args), ["test_gcs_bucket"])

        # Verify that a new blob is created in the bucket
        get_bucket_ret_val = client_ret_val.get_bucket.return_value
        self.assertEqual(get_bucket_ret_val.blob.call_count, 1)
        args, _ = get_bucket_ret_val.blob.call_args
        self.assertLen(args, 1)
        storage_object_name = args[0]
        self.assertTrue(storage_object_name, "tf_cloud_train_tar_")

        # Verify that tarfile is added to the blob
        blob_ret_val = get_bucket_ret_val.blob.return_value
        self.assertEqual(blob_ret_val.upload_from_filename.call_count, 1)

        # Verify discovery API is invoked as expected
        self.assertEqual(mock_discovery_build.call_count, 1)
        args, kwargs = mock_discovery_build.call_args
        self.assertListEqual(list(args), ["cloudbuild", "v1"])
        self.assertDictEqual(
            kwargs,
            {
                "cache_discovery": False,
                "requestBuilder": google_api_client.TFCloudHttpRequest,
            },
        )

        # Verify cloud build create is invoked as expected
        self.assertEqual(builds_ret_val.create.call_count, 1)
        _, kwargs = builds_ret_val.create.call_args

        request_dict = {}
        request_dict["projectId"] = self.project_id
        request_dict["timeout"] = "1200s"
        request_dict["images"] = [[img_tag]]
        request_dict["steps"] = [{
            "name": "gcr.io/cloud-builders/docker",
            "entrypoint": "bash",
            "args": [
                "-c",
                "docker pull gcr.io/test-name:0 || exit 0",
            ],
        }, {
            "name": "gcr.io/cloud-builders/docker",
            "args": [
                "build",
                "-t",
                "gcr.io/test-name:1",
                "--cache-from",
                "gcr.io/test-name:0",
                "."],
        }]
        request_dict["source"] = {
            "storageSource": {
                "bucket": "test_gcs_bucket",
                "object": storage_object_name,
            }
        }
        self.assertDictEqual(
            kwargs,
            {
                "projectId": self.project_id,
                "body": request_dict,
            },
        )

        # Verify cloud build get is invoked as expected
        self.assertEqual(builds_ret_val.get.call_count, 1)
        _, kwargs = builds_ret_val.get.call_args
        self.assertDictEqual(
            kwargs,
            {
                "projectId": self.project_id,
                "id": "test_build_id",
            },
        )

        self.cleanup(lcb.docker_file_path)


if __name__ == "__main__":
    absltest.main()
