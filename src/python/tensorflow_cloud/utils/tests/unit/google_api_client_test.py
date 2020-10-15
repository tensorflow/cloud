# Lint as: python3
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

from absl import logging
from googleapiclient import discovery
from googleapiclient import errors
import httplib2
import mock
import tensorflow as tf
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

    @mock.patch.object(logging, "error", auto_spec=True)
    def test_wait_for_api_training_job_completion_non_blocking_success(
        self, mock_log_error):
        self.mock_request.execute.return_value = {
            "state": "SUCCEEDED",
        }
        status = google_api_client.wait_for_api_training_job_completion(
            self._job_id, self._project_id)
        self.assertTrue(status)
        self.mock_request.execute.assert_called_once()
        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().get.assert_called_with(
            name=job_name)
        mock_log_error.assert_not_called()

    @mock.patch.object(logging, "error", auto_spec=True)
    def test_wait_for_api_training_job_completion_non_blocking_cancelled(
        self, mock_log_error):
        self.mock_request.execute.return_value = {
            "state": "CANCELLED",
        }
        status = google_api_client.wait_for_api_training_job_completion(
            self._job_id, self._project_id)
        self.assertTrue(status)
        self.mock_request.execute.assert_called_once()
        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().get.assert_called_with(
            name=job_name)
        mock_log_error.assert_not_called()

    @mock.patch.object(logging, "error", auto_spec=True)
    def test_wait_for_api_training_job_completion_non_blocking_failed(
        self, mock_log_error):
        self.mock_request.execute.return_value = {
            "state": "FAILED", "errorMessage": "test_error_message"}
        status = google_api_client.wait_for_api_training_job_completion(
            self._job_id, self._project_id)
        self.assertFalse(status)
        self.mock_request.execute.assert_called_once()
        mock_log_error.assert_called_once_with(
            "AIP Training job %s failed with error %s.",
            self._job_id, "test_error_message")

    def test_wait_for_api_training_job_completion_multiple_checks_success(self):
        self.mock_request.execute.side_effect = [
            {"state": "PREPARING"},
            {"state": "RUNNING"},
            {"state": "SUCCEEDED"}
        ]
        status = google_api_client.wait_for_api_training_job_completion(
            self._job_id, self._project_id)
        self.assertTrue(status)
        self.assertEqual(3, self.mock_request.execute.call_count)

    @mock.patch.object(logging, "error", auto_spec=True)
    def test_wait_for_api_training_job_completion_multiple_checks_failed(
        self, mock_log_error):
        self.mock_request.execute.side_effect = [
            {"state": "PREPARING"},
            {"state": "RUNNING"},
            {"state": "FAILED", "errorMessage": "test_error_message"}]
        status = google_api_client.wait_for_api_training_job_completion(
            self._job_id, self._project_id)
        self.assertFalse(status)
        self.assertEqual(3, self.mock_request.execute.call_count)
        mock_log_error.assert_called_once_with(
            "AIP Training job %s failed with error %s.",
            self._job_id, "test_error_message")

    def test_is_api_training_job_running_with_completed_job(self):
        self.mock_request.execute.side_effect = [
            {"state": "SUCCEEDED"},
            {"state": "CANCELLED"},
            {"state": "FAILED", "errorMessage": "test_error_message"}]
        succeeded_status = google_api_client.is_api_training_job_running(
            self._job_id, self._project_id)
        self.assertFalse(succeeded_status)
        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().get.assert_called_with(
            name=job_name)
        cancelled_status = google_api_client.is_api_training_job_running(
            self._job_id, self._project_id)
        self.assertFalse(cancelled_status)
        failed_status = google_api_client.is_api_training_job_running(
            self._job_id, self._project_id)
        self.assertFalse(failed_status)
        self.assertEqual(3, self.mock_request.execute.call_count)

    def test_is_api_training_job_running_with_running_job(self):
        self.mock_request.execute.side_effect = [
            {"state": "QUEUED"},
            {"state": "PREPARING"},
            {"state": "RUNNING"},
            {"state": "CANCELLING"}]
        queued_status = google_api_client.is_api_training_job_running(
            self._job_id, self._project_id)
        self.assertTrue(queued_status)
        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().get.assert_called_with(
            name=job_name)
        preparing_status = google_api_client.is_api_training_job_running(
            self._job_id, self._project_id)
        self.assertTrue(preparing_status)
        running_status = google_api_client.is_api_training_job_running(
            self._job_id, self._project_id)
        self.assertTrue(running_status)
        canceling_status = google_api_client.is_api_training_job_running(
            self._job_id, self._project_id)
        self.assertTrue(canceling_status)
        self.assertEqual(4, self.mock_request.execute.call_count)

    def test_stop_aip_training_job_with_running_job(self):
        self.mock_request.execute.return_value = {}
        google_api_client.stop_aip_training_job(self._job_id, self._project_id)

        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().cancel.assert_called_with(
            name=job_name)

    @mock.patch.object(logging, "info", auto_spec=True)
    def test_stop_aip_training_job_with_completed_job(self, mock_logs):
        self.mock_request.execute.side_effect = errors.HttpError(
            httplib2.Response(info={"status": 400}), b""
        )
        google_api_client.stop_aip_training_job(self._job_id, self._project_id)

        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        self.mock_apiclient.projects().jobs().cancel.assert_called_with(
            name=job_name)
        self.assertEqual(2, mock_logs.call_count)

    @mock.patch.object(logging, "error", auto_spec=True)
    def test_stop_aip_training_job_with_failing_request(self, mock_logs):
        self.mock_request.execute.side_effect = errors.HttpError(
            httplib2.Response(info={"status": 404}), b"")

        job_name = "projects/{}/jobs/{}".format(self._project_id, self._job_id)
        with self.assertRaises(errors.HttpError):
            google_api_client.stop_aip_training_job(
                self._job_id, self._project_id)
        self.mock_apiclient.projects().jobs().cancel.assert_called_with(
            name=job_name)
        mock_logs.assert_called_once()

if __name__ == "__main__":
    tf.test.main()
