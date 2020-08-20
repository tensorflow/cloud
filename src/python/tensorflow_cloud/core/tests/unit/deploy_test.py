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
"""Tests for the cloud deploy module."""

import io
import mock
import os
import shutil
import subprocess
import sys
import tarfile
import unittest

from googleapiclient import errors

from tensorflow_cloud import version
from tensorflow_cloud.core import deploy
from tensorflow_cloud.core import machine_config
from tensorflow_cloud.utils import google_api_client

import mock


class TestDeploy(unittest.TestCase):
    def setup(self, mock_discovery):
        self.mock_job_id = "tf-train-abcde"
        self.mock_project_name = "my-gcp-project"
        self.entry_point = "sample_compile_fit.py"
        self.chief_config = machine_config.COMMON_MACHINE_CONFIGS["K80_4X"]
        self.worker_count = 2
        self.worker_config = machine_config.COMMON_MACHINE_CONFIGS["K80_1X"]
        self.region = "us-central-a"
        self.docker_img = "custom-image-tag"
        self.entry_point_args = ["1000"]
        self.stream_logs = False

        self.expected_request_dict = {
            "jobId": self.mock_job_id,
            "trainingInput": {
                "use_chief_in_tf_config": True,
                "scaleTier": "custom",
                "region": self.region,
                "args": self.entry_point_args,
                "masterType": "n1-standard-16",
                "workerType": "n1-standard-8",
                "workerCount": str(self.worker_count),
                "workerConfig": {
                    "acceleratorConfig": {"count": "1", "type": "NVIDIA_TESLA_K80"},
                    "imageUri": self.docker_img,
                },
                "masterConfig": {
                    "acceleratorConfig": {"count": "4", "type": "NVIDIA_TESLA_K80"},
                    "imageUri": self.docker_img,
                },
            },
        }

        # Verify mocking is correct and setup method mocks.
        assert mock_discovery is deploy.discovery

        def _mock_generate_job_id():
            return self.mock_job_id

        deploy._generate_job_id = _mock_generate_job_id

        def _mock_get_project_name():
            return self.mock_project_name

        deploy.gcp.get_project_name = _mock_get_project_name

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    @mock.patch("tensorflow_cloud.core.deploy.discovery")
    def test_deploy_job(self, mock_discovery, mock_stdout):
        self.setup(mock_discovery)

        job_name = deploy.deploy_job(
            self.region,
            self.docker_img,
            self.chief_config,
            self.worker_count,
            self.worker_config,
            self.entry_point_args,
            self.stream_logs,
        )

        self.assertEqual(job_name, self.mock_job_id)

        # Verify discovery API is invoked as expected.
        self.assertEqual(mock_discovery.build.call_count, 1)
        args, kwargs = mock_discovery.build.call_args
        self.assertListEqual(list(args), ["ml", "v1"])
        self.assertDictEqual(
            kwargs,
            {
                "cache_discovery": False,
                "requestBuilder": google_api_client.TFCloudHttpRequest,
            },
        )

        # Verify job is created as expected
        build_ret_val = mock_discovery.build.return_value
        self.assertEqual(build_ret_val.projects.call_count, 1)
        proj_ret_val = build_ret_val.projects.return_value
        self.assertEqual(proj_ret_val.jobs.call_count, 1)
        jobs_ret_val = proj_ret_val.jobs.return_value
        self.assertEqual(jobs_ret_val.create.call_count, 1)

        # Verify job creation args
        _, kwargs = jobs_ret_val.create.call_args
        self.assertDictEqual(
            kwargs,
            {
                "parent": "projects/" + self.mock_project_name,
                "body": self.expected_request_dict,
            },
        )

        # Verify print statement
        self.assertEqual(
            mock_stdout.getvalue(),
            "Job submitted successfully.\nYour job ID is:  {}\nPlease access "
            "your job logs at the following URL:\nhttps://"
            "console.cloud.google.com/mlengine/jobs/{}?project={}\n".format(
                self.mock_job_id, self.mock_job_id, self.mock_project_name
            ),
        )

    @mock.patch("tensorflow_cloud.core.deploy.discovery")
    def test_request_dict_without_workers(self, mock_discovery):
        self.setup(mock_discovery)
        worker_count = 0

        job_name = deploy.deploy_job(
            self.region,
            self.docker_img,
            self.chief_config,
            worker_count,
            None,
            self.entry_point_args,
            self.stream_logs,
        )
        build_ret_val = mock_discovery.build.return_value
        proj_ret_val = build_ret_val.projects.return_value
        jobs_ret_val = proj_ret_val.jobs.return_value

        self.expected_request_dict["trainingInput"]["workerCount"] = str(worker_count)
        del self.expected_request_dict["trainingInput"]["workerType"]
        del self.expected_request_dict["trainingInput"]["workerConfig"]

        # Verify job creation args
        _, kwargs = jobs_ret_val.create.call_args
        self.assertDictEqual(
            kwargs,
            {
                "parent": "projects/" + self.mock_project_name,
                "body": self.expected_request_dict,
            },
        )

    @mock.patch("tensorflow_cloud.core.deploy.discovery")
    def test_request_dict_without_user_args(self, mock_discovery):
        self.setup(mock_discovery)
        job_name = deploy.deploy_job(
            self.region,
            self.docker_img,
            self.chief_config,
            self.worker_count,
            self.worker_config,
            None,
            self.stream_logs,
        )
        build_ret_val = mock_discovery.build.return_value
        proj_ret_val = build_ret_val.projects.return_value
        jobs_ret_val = proj_ret_val.jobs.return_value

        del self.expected_request_dict["trainingInput"]["args"]

        # Verify job creation args
        _, kwargs = jobs_ret_val.create.call_args
        self.assertDictEqual(
            kwargs,
            {
                "parent": "projects/" + self.mock_project_name,
                "body": self.expected_request_dict,
            },
        )

    @mock.patch("tensorflow_cloud.core.deploy.discovery")
    def test_request_dict_with_TPU_worker(self, mock_discovery):
        self.setup(mock_discovery)
        chief_config = machine_config.COMMON_MACHINE_CONFIGS["CPU"]
        worker_config = machine_config.COMMON_MACHINE_CONFIGS["TPU"]
        worker_count = 1

        job_name = deploy.deploy_job(
            self.region,
            self.docker_img,
            chief_config,
            worker_count,
            worker_config,
            self.entry_point_args,
            self.stream_logs,
        )
        build_ret_val = mock_discovery.build.return_value
        proj_ret_val = build_ret_val.projects.return_value
        jobs_ret_val = proj_ret_val.jobs.return_value

        self.expected_request_dict["trainingInput"]["workerCount"] = "1"
        self.expected_request_dict["trainingInput"]["workerType"] = "cloud_tpu"
        self.expected_request_dict["trainingInput"]["masterType"] = "n1-standard-4"
        self.expected_request_dict["trainingInput"]["workerConfig"][
            "acceleratorConfig"
        ]["type"] = "TPU_V3"
        self.expected_request_dict["trainingInput"]["workerConfig"][
            "acceleratorConfig"
        ]["count"] = "8"
        v = deploy.VERSION.split(".")
        self.expected_request_dict["trainingInput"]["workerConfig"]["tpuTfVersion"] = (
            v[0] + "." + v[1]
        )
        self.expected_request_dict["trainingInput"]["masterConfig"][
            "acceleratorConfig"
        ]["type"] = "ACCELERATOR_TYPE_UNSPECIFIED"
        self.expected_request_dict["trainingInput"]["masterConfig"][
            "acceleratorConfig"
        ]["count"] = "0"

        # Verify job creation args
        _, kwargs = jobs_ret_val.create.call_args
        self.assertDictEqual(
            kwargs,
            {
                "parent": "projects/" + self.mock_project_name,
                "body": self.expected_request_dict,
            },
        )

    @mock.patch("tensorflow_cloud.core.deploy.discovery")
    def test_deploy_job_error(self, mock_discovery):
        self.setup(mock_discovery)
        chief_config = machine_config.COMMON_MACHINE_CONFIGS["CPU"]
        worker_config = machine_config.COMMON_MACHINE_CONFIGS["TPU"]
        worker_count = 1

        build_ret_val = mock_discovery.build.return_value
        build_ret_val.projects.side_effect = errors.HttpError(
            mock.Mock(status=404), b"not found"
        )

        with self.assertRaises(errors.HttpError):
            deploy.deploy_job(
                self.region,
                self.docker_img,
                chief_config,
                worker_count,
                worker_config,
                self.entry_point_args,
                self.stream_logs,
            )

    @mock.patch("tensorflow_cloud.core.deploy.subprocess")
    @mock.patch("tensorflow_cloud.core.deploy.discovery")
    def test_logs_streaming_error(self, mock_discovery, mock_subprocess):
        self.setup(mock_discovery)
        chief_config = machine_config.COMMON_MACHINE_CONFIGS["CPU"]
        worker_config = machine_config.COMMON_MACHINE_CONFIGS["TPU"]
        worker_count = 1

        mock_subprocess.Popen.side_effect = ValueError("error")
        self.stream_logs = True

        with self.assertRaises(ValueError):
            deploy.deploy_job(
                self.region,
                self.docker_img,
                chief_config,
                worker_count,
                worker_config,
                self.entry_point_args,
                self.stream_logs,
            )


if __name__ == "__main__":
    unittest.main()
