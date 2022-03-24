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
"""Unit test for utilities module."""

import json
import os
from googleapiclient import discovery
from googleapiclient import errors
from googleapiclient import http as googleapiclient_http
import httplib2
import mock
import tensorflow as tf
from tensorflow_cloud import version
from tensorflow_cloud.utils import google_api_client


class GoogleApiClientTest(tf.test.TestCase):

    def setUp(self):
        super(GoogleApiClientTest, self).setUp()
        self.addCleanup(mock.patch.stopall)

        # Setting wait time to 1 sec to speed up the tests execution.
        google_api_client._POLL_INTERVAL_IN_SECONDS = 1
        self._project_id = "project-a"
        self._job_id = "job_id"

        self.mock_discovery_build = mock.patch.object(
            discovery, "build", autospec=True
        ).start()
        self.mock_apiclient = mock.Mock()
        self.mock_discovery_build.return_value = self.mock_apiclient
        self.mock_request = mock.Mock()
        self.mock_apiclient.projects().jobs(
                ).get.return_value = self.mock_request
        self.mock_apiclient.projects().jobs(
                ).cancel.return_value = self.mock_request
        self._local_config_path = os.path.join(
            self.get_temp_dir(), "config.json")
        google_api_client._LOCAL_CONFIG_PATH = self._local_config_path

    # TODO(b/177023448) Remove mock on logging.error here and below.
    def test_wait_for_aip_training_job_completion_non_blocking_success(self):
        self.mock_request.execute.return_value = {
            "state": "SUCCEEDED",
        }
        status = google_api_client.wait_for_aip_training_job_completion(
            self._job_id, self._project_id)
        self.assertTrue(status)
        self.mock_request.execute.assert_called_once()
        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().get.assert_called_with(
            name=job_name)

    def test_wait_for_aip_training_job_completion_non_blocking_cancelled(self):
        self.mock_request.execute.return_value = {
            "state": "CANCELLED",
        }
        status = google_api_client.wait_for_aip_training_job_completion(
            self._job_id, self._project_id)
        self.assertTrue(status)
        self.mock_request.execute.assert_called_once()
        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().get.assert_called_with(
            name=job_name)

    def test_wait_for_aip_training_job_completion_non_blocking_failed(self):
        self.mock_request.execute.return_value = {
            "state": "FAILED", "errorMessage": "test_error_message"}
        status = google_api_client.wait_for_aip_training_job_completion(
            self._job_id, self._project_id)
        self.assertFalse(status)
        self.mock_request.execute.assert_called_once()

    def test_wait_for_aip_training_job_completion_multiple_checks_success(self):
        self.mock_request.execute.side_effect = [
            {"state": "PREPARING"},
            {"state": "RUNNING"},
            {"state": "SUCCEEDED"}
        ]
        status = google_api_client.wait_for_aip_training_job_completion(
            self._job_id, self._project_id)
        self.assertTrue(status)
        self.assertEqual(3, self.mock_request.execute.call_count)

    def test_wait_for_aip_training_job_completion_multiple_checks_failed(self):
        self.mock_request.execute.side_effect = [
            {"state": "PREPARING"},
            {"state": "RUNNING"},
            {"state": "FAILED", "errorMessage": "test_error_message"}]
        status = google_api_client.wait_for_aip_training_job_completion(
            self._job_id, self._project_id)
        self.assertFalse(status)
        self.assertEqual(3, self.mock_request.execute.call_count)

    def test_is_aip_training_job_running_with_completed_job(self):
        self.mock_request.execute.side_effect = [
            {"state": "SUCCEEDED"},
            {"state": "CANCELLED"},
            {"state": "FAILED", "errorMessage": "test_error_message"}]
        succeeded_status = google_api_client.is_aip_training_job_running(
            self._job_id, self._project_id)
        self.assertFalse(succeeded_status)
        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().get.assert_called_with(
            name=job_name)
        cancelled_status = google_api_client.is_aip_training_job_running(
            self._job_id, self._project_id)
        self.assertFalse(cancelled_status)
        failed_status = google_api_client.is_aip_training_job_running(
            self._job_id, self._project_id)
        self.assertFalse(failed_status)
        self.assertEqual(3, self.mock_request.execute.call_count)

    def test_is_aip_training_job_running_with_running_job(self):
        self.mock_request.execute.side_effect = [
            {"state": "QUEUED"},
            {"state": "PREPARING"},
            {"state": "RUNNING"},
            {"state": "CANCELLING"}]
        queued_status = google_api_client.is_aip_training_job_running(
            self._job_id, self._project_id)
        self.assertTrue(queued_status)
        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().get.assert_called_with(
            name=job_name)
        preparing_status = google_api_client.is_aip_training_job_running(
            self._job_id, self._project_id)
        self.assertTrue(preparing_status)
        running_status = google_api_client.is_aip_training_job_running(
            self._job_id, self._project_id)
        self.assertTrue(running_status)
        canceling_status = google_api_client.is_aip_training_job_running(
            self._job_id, self._project_id)
        self.assertTrue(canceling_status)
        self.assertEqual(4, self.mock_request.execute.call_count)

    def test_stop_aip_training_job_with_running_job(self):
        self.mock_request.execute.return_value = {}
        google_api_client.stop_aip_training_job(self._job_id, self._project_id)

        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().cancel.assert_called_with(
            name=job_name)

    def test_stop_aip_training_job_with_completed_job(self):
        self.mock_request.execute.side_effect = errors.HttpError(
            httplib2.Response(info={"status": 400}), b""
        )
        google_api_client.stop_aip_training_job(self._job_id, self._project_id)

        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().cancel.assert_called_with(
            name=job_name)

    def test_stop_aip_training_job_with_failing_request(self):
        self.mock_request.execute.side_effect = errors.HttpError(
            httplib2.Response(info={"status": 404}), b"")

        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        with self.assertRaises(errors.HttpError):
            google_api_client.stop_aip_training_job(
                self._job_id, self._project_id)
        self.mock_apiclient.projects().jobs().cancel.assert_called_with(
            name=job_name)

    def test_get_client_environment_name_with_kaggle(self):
        os.environ["KAGGLE_CONTAINER_NAME"] = "test_container_name"
        self.assertEqual(
            google_api_client.get_client_environment_name(),
            google_api_client.ClientEnvironment.KAGGLE_NOTEBOOK.name)

    def test_get_client_environment_name_with_hosted_notebook(self):
        os.environ["DL_PATH"] = "test_dl_path"
        os.environ["USER"] = "jupyter"
        self.assertEqual(
            google_api_client.get_client_environment_name(),
            google_api_client.ClientEnvironment.HOSTED_NOTEBOOK.name)

    def test_get_client_environment_name_with_hosted_dlvm(self):
        os.environ["DL_PATH"] = "test_dl_path"
        self.assertEqual(
            google_api_client.get_client_environment_name(),
            google_api_client.ClientEnvironment.DLVM.name)

    @mock.patch.object(google_api_client, "_is_module_present", autospec=True)
    @mock.patch.object(google_api_client, "_get_env_variable", autospec=True)
    def test_get_client_environment_name_with_hosted_unknown(
        self, mock_getenv, mock_modules):
        mock_getenv.return_value = None
        mock_modules.return_value = {}
        self.assertEqual(
            google_api_client.get_client_environment_name(),
            google_api_client.ClientEnvironment.UNKNOWN.name)

    @mock.patch.object(google_api_client, "_is_module_present", autospec=True)
    @mock.patch.object(google_api_client, "_get_env_variable", autospec=True)
    def test_get_client_environment_name_with_hosted_colab(
        self, mock_getenv, mock_modules):
        mock_getenv.return_value = None
        mock_modules.return_value = True
        self.assertEqual(
            google_api_client.get_client_environment_name(),
            google_api_client.ClientEnvironment.COLAB.name)

    @mock.patch.object(google_api_client, "_is_module_present", autospec=True)
    @mock.patch.object(google_api_client, "_get_env_variable", autospec=True)
    def test_get_client_environment_name_with_hosted_dl_container(
        self, mock_getenv, mock_modules):
        mock_getenv.return_value = None
        mock_modules.side_effect = [False, True]
        self.assertEqual(
            google_api_client.get_client_environment_name(),
            google_api_client.ClientEnvironment.DL_CONTAINER.name)

    def test_get_or_set_consent_status_rejected(self):
        config_data = {}
        config_data["telemetry_rejected"] = True

        # Create the config path if it does not already exist
        os.makedirs(os.path.dirname(self._local_config_path), exist_ok=True)

        with open(self._local_config_path, "w") as config_json:
            json.dump(config_data, config_json)

        self.assertFalse(google_api_client.get_or_set_consent_status())

    def test_get_or_set_consent_status_verified(self):
        config_data = {}
        config_data["notification_version"] = version.__version__

        # Create the config path if it does not already exist
        os.makedirs(os.path.dirname(self._local_config_path), exist_ok=True)

        with open(self._local_config_path, "w") as config_json:
            json.dump(config_data, config_json)

        self.assertTrue(google_api_client.get_or_set_consent_status())

    def test_get_or_set_consent_status_notify_user(self):
        if os.path.exists(self._local_config_path):
            os.remove(self._local_config_path)

        self.assertTrue(google_api_client.get_or_set_consent_status())

        with open(self._local_config_path) as config_json:
            config_data = json.load(config_json)
            self.assertDictContainsSubset(
                config_data, {"notification_version": version.__version__})

    @mock.patch.object(google_api_client,
                       "get_or_set_consent_status", autospec=True)
    def test_TFCloudHttpRequest_with_rejected_consent(
        self, mock_consent_status):

        mock_consent_status.return_value = False
        http_request = google_api_client.TFCloudHttpRequest(
            googleapiclient_http.HttpMockSequence([({"status": "200"}, "{}")]),
            object(),
            "fake_uri",
        )
        self.assertIsInstance(http_request, googleapiclient_http.HttpRequest)
        self.assertIn("user-agent", http_request.headers)
        self.assertDictEqual(
            {"user-agent": f"tf-cloud/{version.__version__} ()"},
            http_request.headers)

    @mock.patch.object(google_api_client,
                       "get_or_set_consent_status", autospec=True)
    @mock.patch.object(google_api_client,
                       "get_client_environment_name", autospec=True)
    def test_TFCloudHttpRequest_with_consent(
        self, mock_get_env_name, mock_consent_status):

        mock_consent_status.return_value = True
        mock_get_env_name.return_value = "TEST_ENV"
        google_api_client.TFCloudHttpRequest.set_telemetry_dict({})
        http_request = google_api_client.TFCloudHttpRequest(
            googleapiclient_http.HttpMockSequence([({"status": "200"}, "{}")]),
            object(),
            "fake_uri",
        )
        self.assertIsInstance(http_request, googleapiclient_http.HttpRequest)
        self.assertIn("user-agent", http_request.headers)

        header_comment = "client_environment:TEST_ENV;"
        full_header = f"tf-cloud/{version.__version__} ({header_comment})"

        self.assertDictEqual({"user-agent": full_header}, http_request.headers)

    @mock.patch.object(google_api_client,
                       "get_or_set_consent_status", autospec=True)
    @mock.patch.object(google_api_client,
                       "get_client_environment_name", autospec=True)
    def test_TFCloudHttpRequest_with_additional_metrics(
        self, mock_get_env_name, mock_consent_status):

        google_api_client.TFCloudHttpRequest.set_telemetry_dict(
            {"TEST_KEY1": "TEST_VALUE1"})
        mock_consent_status.return_value = True
        mock_get_env_name.return_value = "TEST_ENV"
        http_request = google_api_client.TFCloudHttpRequest(
            googleapiclient_http.HttpMockSequence([({"status": "200"}, "{}")]),
            object(),
            "fake_uri",
        )
        self.assertIsInstance(http_request, googleapiclient_http.HttpRequest)
        self.assertIn("user-agent", http_request.headers)

        header_comment = "TEST_KEY1:TEST_VALUE1;client_environment:TEST_ENV;"
        full_header = f"tf-cloud/{version.__version__} ({header_comment})"

        self.assertDictEqual({"user-agent": full_header}, http_request.headers)

        # Verify when telemetry dict is refreshed it is used in new http request
        google_api_client.TFCloudHttpRequest.set_telemetry_dict(
            {"TEST_KEY2": "TEST_VALUE2"})
        mock_consent_status.return_value = True
        mock_get_env_name.return_value = "TEST_ENV"
        http_request = google_api_client.TFCloudHttpRequest(
            googleapiclient_http.HttpMockSequence([({"status": "200"}, "{}")]),
            object(),
            "fake_uri",
        )

        header_comment = "TEST_KEY2:TEST_VALUE2;client_environment:TEST_ENV;"
        full_header = f"tf-cloud/{version.__version__} ({header_comment})"

        self.assertDictEqual({"user-agent": full_header}, http_request.headers)


if __name__ == "__main__":
    tf.test.main()
