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

import enum
import json
import os
import sys
import time
from typing import Dict, Text, Union
from .. import version
from absl import logging
from googleapiclient import discovery
from googleapiclient import errors
from googleapiclient import http as googleapiclient_http

_TF_CLOUD_USER_AGENT_HEADER = "tf-cloud/" + version.__version__
_POLL_INTERVAL_IN_SECONDS = 30
_LOCAL_CONFIG_PATH = os.path.expanduser(
        "~/.config/tf_cloud/tf_cloud_config.json")
_PRIVACY_NOTICE = """
This application reports technical and operational details of your usage of
Cloud Services in accordance with Google privacy policy, for more information
please refer to https://policies.google.com/privacy. If you wish
to opt-out, you may do so by running
tensorflow_cloud.utils.google_api_client.optout_metrics_reporting().
"""

_TELEMETRY_REJECTED_CONFIG = "telemetry_rejected"
_TELEMETRY_VERSION_CONFIG = "notification_version"


_KAGGLE_ENV_VARIABLE = "KAGGLE_CONTAINER_NAME"
_DL_ENV_PATH_VARIABLE = "DL_PATH"


class ClientEnvironment(enum.Enum):
    """Types of client environment for telemetry reporting."""
    UNKNOWN = 0
    KAGGLE_NOTEBOOK = 1
    HOSTED_NOTEBOOK = 2
    DLVM = 3
    DL_CONTAINER = 4
    COLAB = 5


class TFCloudHttpRequest(googleapiclient_http.HttpRequest):
    """HttpRequest builder that sets a customized useragent header for TF Cloud.

    This is used to track the usage of the TF Cloud.
    """

    # Class property for passing additional telemetry fields to constructor.
    _telemetry_dict = {}

    def __init__(self, *args, **kwargs):
        """Construct a HttpRequest.

        Args:
            *args: Positional arguments to pass to the base class constructor.
            **kwargs: Keyword arguments to pass to the base class constructor.
        """
        headers = kwargs.setdefault("headers", {})

        comments = {}
        if get_or_set_consent_status():
            comments = self._telemetry_dict

            # Add the local environment to the user agent header comment field.
            comments["client_environment"] = get_client_environment_name()

        # construct comment string using comments dict
        user_agent_text = f"{_TF_CLOUD_USER_AGENT_HEADER} ("
        for key, value in comments.items():
            user_agent_text = f"{user_agent_text}{key}:{value};"
        user_agent_text = f"{user_agent_text})"

        headers["user-agent"] = user_agent_text
        super(TFCloudHttpRequest, self).__init__(*args, **kwargs)

    # @classmethod @property chain is only supported in python 3.9+, see
    # https://docs.python.org/3/howto/descriptor.html#id27. Using class
    # getter and setter instead.
    @classmethod
    def get_telemetry_dict(cls):
        telemetry_dict = cls._telemetry_dict.copy()
        return telemetry_dict

    @classmethod
    def set_telemetry_dict(cls, telemetry_dict: Dict[Text, Text]):
        cls._telemetry_dict = telemetry_dict.copy()


# TODO(b/176097105) Use get_client_environment_name in tfc.run and cloud_fit
def get_client_environment_name() -> Text:
    """Identifies the local environment where tensorflow_cloud is running.

    Returns:
        ClientEnvironment Enum representing the environment type.
    """
    if _get_env_variable(_KAGGLE_ENV_VARIABLE):
        logging.info("Kaggle client environment detected.")
        return ClientEnvironment.KAGGLE_NOTEBOOK.name

    if _is_module_present("google.colab"):
        logging.info("Detected running in COLAB environment.")
        return ClientEnvironment.COLAB.name

    if _get_env_variable(_DL_ENV_PATH_VARIABLE):
        # TODO(b/171720710) Update logic based resolution of the issue.
        if _get_env_variable("USER") == "jupyter":
            logging.info("Detected running in HOSTED_NOTEBOOK environment.")
            return ClientEnvironment.HOSTED_NOTEBOOK.name

        # TODO(b/175815580) Update logic based resolution of the issue.
        logging.info("Detected running in DLVM environment.")
        return ClientEnvironment.DLVM.name

    # TODO(b/175815580) Update logic based resolution of the issue.
    if  _is_module_present("google"):
        logging.info("Detected running in DL_CONTAINER environment.")
        return ClientEnvironment.DL_CONTAINER.name

    logging.info("Detected running in UNKNOWN environment.")
    return ClientEnvironment.UNKNOWN.name


def _is_module_present(module_name: Text) -> bool:
    """Checks if module_name is present in sys.modules.

    Args:
        module_name: Name of the module to look up in the system modules.
    Returns:
        True if module exists, False otherwise.
    """
    return module_name in sys.modules


def _get_env_variable(variable_name: Text) -> Union[Text, None]:
    """Looks up the value of environment varialbe variable_name.

    Args:
        variable_name: Name of the variable to look up in the environment.
    Returns:
        A string representing the varialbe value or None if varialbe is not
        defined in the environment.
    """
    return os.getenv(variable_name)


def get_or_set_consent_status()-> bool:
    """Gets or sets the user consent status for telemetry collection.

    Returns:
        If the user has rejected client side telemetry collection returns
        False, otherwise it returns true, if a consent flag is not found the
        user is notified of telemetry collection and a flag is set.
    """
    # Verify if user consent exists and if it is valid for current version of
    # tensorflow_cloud
    if os.path.exists(_LOCAL_CONFIG_PATH):
        with open(_LOCAL_CONFIG_PATH) as config_json:
            config_data = json.load(config_json)
            if config_data.get(_TELEMETRY_REJECTED_CONFIG):
                logging.info("User has opt-out of telemetry reporting.")
                return False
            if config_data.get(
                _TELEMETRY_VERSION_CONFIG) == version.__version__:
                return True

    # Either user has not been notified of telemetry collection or a different
    # version of the tensorflow_cloud has been installed since the last
    # notification. Notify the user and update the configuration.
    logging.info(_PRIVACY_NOTICE)
    print(_PRIVACY_NOTICE)

    config_data = {}
    config_data[_TELEMETRY_VERSION_CONFIG] = version.__version__

    # Create the config path if it does not already exist
    os.makedirs(os.path.dirname(_LOCAL_CONFIG_PATH), exist_ok=True)

    with open(_LOCAL_CONFIG_PATH, "w") as config_json:
        json.dump(config_data, config_json)
    return True


def optout_metrics_reporting():
    """Set configuration to opt-out of client side metric reporting."""

    config_data = {}
    config_data["telemetry_rejected"] = True

    # Create the config path if it does not already exist
    os.makedirs(os.path.dirname(_LOCAL_CONFIG_PATH), exist_ok=True)

    with open(_LOCAL_CONFIG_PATH, "w") as config_json:
        json.dump(config_data, config_json)

    logging.info("Client side metrics reporting has been disabled.")


def wait_for_aip_training_job_completion(job_id: Text, project_id: Text)->bool:
    """Blocks until the AIP Training job is completed and returns the status.

    Args:
        job_id: ID for AIP training job.
        project_id: Project under which the AIP Training job is running.
    Returns:
        True if the job succeeded or it was cancelled, False if the job failed.
    """
    # Wait for AIP Training job to finish
    job_name = "projects/{}/jobs/{}".format(project_id, job_id)
    # Disable cache_discovery to remove excessive info logs see:
    # https://github.com/googleapis/google-api-python-client/issues/299
    api_client = discovery.build("ml", "v1", cache_discovery=False)

    request = api_client.projects().jobs().get(name=job_name)

    response = request.execute()

    counter = 0
    logging.info(
        "Waiting for job to complete, polling status every %s sec.",
        _POLL_INTERVAL_IN_SECONDS)
    while response["state"] not in ("SUCCEEDED", "FAILED", "CANCELLED"):
        logging.info("Attempt number %s to retrieve job status.", counter)
        counter += 1
        time.sleep(_POLL_INTERVAL_IN_SECONDS)
        response = request.execute()

    if response["state"] == "FAILED":
        logging.error("AIP Training job %s failed with error %s.",
                      job_id, response["errorMessage"])
        return False

    # Both CANCELLED and SUCCEEDED status count as successful completion.
    logging.info("AIP Training job %s completed with status %s.",
                 job_id, response["state"])
    return True


def is_aip_training_job_running(job_id: Text, project_id: Text)->bool:
    """Non-blocking call that checks if AIP Training job is running.

    Args:
        job_id: ID for AIP training job.
        project_id: Project under which the AIP Training job is running.
    Returns:
        True if the job is running, False if it has succeeded, failed, or it was
        cancelled.
    """
    job_name = "projects/{}/jobs/{}".format(project_id, job_id)
    # Disable cache_discovery to remove excessive info logs see:
    # https://github.com/googleapis/google-api-python-client/issues/299
    api_client = discovery.build("ml", "v1", cache_discovery=False)

    logging.info("Retrieving status for job %s.", job_name)

    request = api_client.projects().jobs().get(name=job_name)
    response = request.execute()

    return response["state"] not in ("SUCCEEDED", "FAILED", "CANCELLED")


def stop_aip_training_job(job_id: Text, project_id: Text):
    """Cancels a running AIP Training job.

    Args:
        job_id: ID for AIP training job.
        project_id: Project under which the AIP Training job is running.
    """
    job_name = "projects/{}/jobs/{}".format(project_id, job_id)
    # Disable cache_discovery to remove excessive info logs see:
    # https://github.com/googleapis/google-api-python-client/issues/299
    api_client = discovery.build("ml", "v1", cache_discovery=False)

    logging.info("Canceling the job %s.", job_name)

    request = api_client.projects().jobs().cancel(name=job_name)

    try:
        request.execute()
    except errors.HttpError as e:
        if e.resp.status == 400:
            logging.info(
                # If job is already completed, the request will result in a 400
                # error with similar to 'description': 'Cannot cancel an already
                # completed job.' In this case we will absorb the error.
                "Job %s has already completed.", job_id)
            return
        logging.error("Cancel Request for job %s failed.", job_name)
        raise e
