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
"""Utilities for Google API client."""

import time
from typing import Text
from .. import version
from absl import logging
from googleapiclient import discovery
from googleapiclient import http as googleapiclient_http

_USER_AGENT_FOR_TF_CLOUD_TRACKING = "tf-cloud/" + version.__version__
_POLL_INTERVAL_IN_SECONDS = 30


class TFCloudHttpRequest(googleapiclient_http.HttpRequest):
    """HttpRequest builder that sets a customized useragent header for TF Cloud.

    This is used to track the usage of the TF Cloud.
    """

    def __init__(self, *args, **kwargs):
        """Construct a HttpRequest.

        Args:
            *args: Positional arguments to pass to the base class constructor.
            **kwargs: Keyword arguments to pass to the base class constructor.
        """
        headers = kwargs.setdefault("headers", {})
        headers["user-agent"] = _USER_AGENT_FOR_TF_CLOUD_TRACKING
        super(TFCloudHttpRequest, self).__init__(*args, **kwargs)


def wait_for_api_training_job_success(job_id: Text, project_id: Text)->bool:
    """Blocks until the AIP Training job is completed and returns the status.

    Args:
        job_id: ID for AIP training job.
        project_id: Project under which the AIP Training job is running.
    Returns:
        True if job succeeded, False if job failed.
    """
    # Wait for AIP Training job to finish
    job_name = "projects/{}/jobs/{}".format(project_id, job_id)
    api_client = discovery.build("ml", "v1")

    request = api_client.projects().jobs().get(name=job_name)

    response = request.execute()

    counter = 0
    logging.info(
        "Waiting for job to complete, polling status every %s sec.",
        _POLL_INTERVAL_IN_SECONDS)
    while response["state"] not in ("SUCCEEDED", "FAILED"):
        logging.info("Attempt number %s to retrieve job status.", counter)
        counter += 1
        time.sleep(_POLL_INTERVAL_IN_SECONDS)
        response = request.execute()

    if response["state"] == "FAILED":
        logging.error("AIP Training job %s failed with error %s.",
                      job_id, response["errorMessage"])
        return False

    logging.info("AIP Training job %s succeeded.", job_id)
    return True

