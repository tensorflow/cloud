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

from absl.testing import absltest
from googleapiclient import discovery
from googleapiclient import errors
import mock

from tensorflow_cloud.core import deploy
from tensorflow_cloud.core import gcp
from tensorflow_cloud.core import machine_config
from tensorflow_cloud.utils import google_api_client


class TestDeploy(absltest.TestCase):

    def setUp(self):
        super(TestDeploy, self).setUp()

        self._mock_discovery_build = mock.patch.object(
            discovery, "build", autospec=True
        ).start()

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
        self.service_account = "test_sa@test-project.iam.gserviceaccount.com"

        self.expected_request_dict = {
            "jobId": self.mock_job_id,
            "trainingInput": {
                "use_chief_in_tf_config": True,
                "scaleTier": "custom",
                "region": self.region,
                "args": self.entry_point_args,
                "serviceAccount": self.service_account,
                "masterType": "n1-standard-16",
                "workerType": "n1-standard-8",
                "workerCount": str(self.worker_count),
                "workerConfig": {
                    "acceleratorConfig": {
                        "count": "1", "type": "NVIDIA_TESLA_K80"},
                    "imageUri": self.docker_img,
                },
                "masterConfig": {
                    "acceleratorConfig": {
                        "count": "4", "type": "NVIDIA_TESLA_K80"},
                    "imageUri": self.docker_img,
                },
            },
        }

        mock.patch.object(
            deploy,
            "_generate_job_id",
            autospec=True,
            return_value=self.mock_job_id,
        ).start()

        mock.patch.object(
            gcp,
            "get_project_name",
            autospec=True,
            return_value=self.mock_project_name,
        ).start()

        mock.patch.object(
            gcp,
            "get_region",
            autospec=True,
            return_value=self.region,
        ).start()

    def tearDown(self):
        mock.patch.stopall()
        super(TestDeploy, self).tearDown()

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_deploy_job(self, mock_stdout):
        job_name = deploy.deploy_job(
            self.docker_img,
            self.chief_config,
            self.worker_count,
            self.worker_config,
            self.entry_point_args,
            self.stream_logs,
            service_account=self.service_account
        )

        self.assertEqual(job_name, self.mock_job_id)

        # Verify discovery API is invoked as expected.
        self.assertEqual(self._mock_discovery_build.call_count, 1)
        args, kwargs = self._mock_discovery_build.call_args
        self.assertListEqual(list(args), ["ml", "v1"])
        self.assertDictEqual(
            kwargs,
            {
                "cache_discovery": False,
                "requestBuilder": google_api_client.TFCloudHttpRequest,
            },
        )

        # Verify job is created as expected
        build_ret_val = self._mock_discovery_build.return_value
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
            "\nJob submitted successfully."
            "\nYour job ID is:  {}\n"
            "\nPlease access your training job information here:\nhttps://"
            "console.cloud.google.com/mlengine/jobs/{}?project={}\n"
            "\nPlease access your training job logs here: "
            "https://console.cloud.google.com/logs/viewer?resource=ml_job%2F"
            "job_id%2F{}&interval=NO_LIMIT&project={}\n\n".format(
                self.mock_job_id, self.mock_job_id, self.mock_project_name,
                self.mock_job_id, self.mock_project_name
            ),
        )

    def test_deploy_job_with_default_service_account_has_no_serviceaccount_key(
        self):
        # If user does not provide a service account (i.e. service_account=None,
        # the service account key should not be included in the request dict as
        # AI Platform will treat None as the name of the service account.
        _ = deploy.deploy_job(
            self.docker_img,
            self.chief_config,
            self.worker_count,
            self.worker_config,
            self.entry_point_args,
            self.stream_logs,
        )
        build_ret_val = self._mock_discovery_build.return_value
        proj_ret_val = build_ret_val.projects.return_value
        jobs_ret_val = proj_ret_val.jobs.return_value

        del self.expected_request_dict["trainingInput"]["serviceAccount"]

        # Verify job creation args
        _, kwargs = jobs_ret_val.create.call_args
        self.assertDictEqual(
            kwargs,
            {
                "parent": "projects/" + self.mock_project_name,
                "body": self.expected_request_dict,
            },
        )

    def test_request_dict_without_workers(self):
        worker_count = 0

        _ = deploy.deploy_job(
            self.docker_img,
            self.chief_config,
            worker_count,
            None,
            self.entry_point_args,
            self.stream_logs,
            service_account=self.service_account
        )
        build_ret_val = self._mock_discovery_build.return_value
        proj_ret_val = build_ret_val.projects.return_value
        jobs_ret_val = proj_ret_val.jobs.return_value

        self.expected_request_dict["trainingInput"]["workerCount"] = str(
            worker_count)
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

    def test_request_dict_without_user_args(self):
        _ = deploy.deploy_job(
            self.docker_img,
            self.chief_config,
            self.worker_count,
            self.worker_config,
            None,
            self.stream_logs,
            service_account=self.service_account
        )
        build_ret_val = self._mock_discovery_build.return_value
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

    def test_request_dict_with_tpu_worker(self):
        chief_config = machine_config.COMMON_MACHINE_CONFIGS["CPU"]
        worker_config = machine_config.COMMON_MACHINE_CONFIGS["TPU"]
        worker_count = 1

        _ = deploy.deploy_job(
            self.docker_img,
            chief_config,
            worker_count,
            worker_config,
            self.entry_point_args,
            self.stream_logs,
            service_account=self.service_account
        )
        build_ret_val = self._mock_discovery_build.return_value
        proj_ret_val = build_ret_val.projects.return_value
        jobs_ret_val = proj_ret_val.jobs.return_value

        self.expected_request_dict["trainingInput"]["workerCount"] = "1"
        self.expected_request_dict["trainingInput"]["workerType"] = "cloud_tpu"
        self.expected_request_dict["trainingInput"]["masterType"] = (
            "n1-standard-4")
        self.expected_request_dict["trainingInput"]["workerConfig"][
            "acceleratorConfig"
        ]["type"] = "TPU_V3"
        self.expected_request_dict["trainingInput"]["workerConfig"][
            "acceleratorConfig"
        ]["count"] = "8"
        self.expected_request_dict["trainingInput"]["workerConfig"][
            "tpuTfVersion"
        ] = "2.1"
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

    def test_deploy_job_error(self):
        chief_config = machine_config.COMMON_MACHINE_CONFIGS["CPU"]
        worker_config = machine_config.COMMON_MACHINE_CONFIGS["TPU"]
        worker_count = 1

        build_ret_val = self._mock_discovery_build.return_value
        build_ret_val.projects.side_effect = errors.HttpError(
            mock.Mock(status=404), b"not found"
        )

        with self.assertRaises(errors.HttpError):
            deploy.deploy_job(
                self.docker_img,
                chief_config,
                worker_count,
                worker_config,
                self.entry_point_args,
                self.stream_logs,
            )

    @mock.patch("subprocess.Popen")
    def test_logs_streaming_error(self, mock_subprocess_popen):
        chief_config = machine_config.COMMON_MACHINE_CONFIGS["CPU"]
        worker_config = machine_config.COMMON_MACHINE_CONFIGS["TPU"]
        worker_count = 1

        mock_subprocess_popen.side_effect = ValueError("error")
        self.stream_logs = True

        with self.assertRaises(ValueError):
            deploy.deploy_job(
                self.docker_img,
                chief_config,
                worker_count,
                worker_config,
                self.entry_point_args,
                self.stream_logs,
            )


if __name__ == "__main__":
    absltest.main()
