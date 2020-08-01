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
"""A thin client for the Cloud AI Platform Optimizer Service."""

import datetime
import json
import os
import time
from typing import Any, Dict, List, Mapping, Text, Union

from googleapiclient import discovery
from googleapiclient import errors
from googleapiclient import http as googleapiclient_http
import tensorflow as tf
import tensorflow_cloud as tfc

_OPTIMIZER_API_DOCUMENT_FILE = "api/ml_public_google_rest_v1.json"

# By default, the Tuner worker(s) always requests one trial at a time because
# we would parallelize the tuning loop themselves as opposed to getting multiple
# trial suggestions in one tuning loop.
_SUGGESTION_COUNT_PER_REQUEST = 1

# Number of tries to retry getting study if it was already created
_NUM_TRIES_FOR_STUDIES = 3

_USER_AGENT_FOR_CLOUD_TUNER_TRACKING = "cloud-tuner/" + tfc.__version__


class SuggestionInactiveError(Exception):
    """Indicates that GetSuggestion was called on an inactive study."""


class _OptimizerClient(object):
    """A wrapper class that allows for easy interaction with a Study."""

    def __init__(self, service_client, project_id, region, study_id=None):
        """Create an OptimizerClient object.

    Use this constructor when you know the study_id, and when the Study
    already exists.  Otherwise, you'll probably want to use
    create_or_load_study() instead of constructing the
    OptimizerClient class directly.

    Args:
      service_client: An API client of CAIP Optimizer service.
      project_id: A GCP project id.
      region: A GCP region. e.g. 'us-central1'.
      study_id: An identifier of the study. The full study name will be
        projects/{project_id}/locations/{region}/studies/{study_id}. And the
        full trial name will be {study name}/trials/{trial_id}.
    """
        self.service_client = service_client
        self.project_id = project_id
        self.region = region
        if not study_id:
            raise ValueError(
                "Use create_or_load_study() instead of constructing the"
                "OptimizerClient class directly"
            )
        self.study_id = study_id

    def get_suggestions(self, client_id):
        """Gets a list of suggested Trials.

    Arguments:
      client_id: An ID that identifies the `Tuner` requesting a `Trial`.
        `Tuners` that should run the same trial (for instance, when running a
        multi-worker model) should have the same ID. If multiple
        suggestTrialsRequests have the same tuner_id, the service will return
        the identical suggested trial if the trial is PENDING, and provide a new
        trial if the last suggest trial was completed.

    Returns:
      A list of Trials, This may be an empty list in case that a finite search
      space has been exhausted, if max_num_trials = 1000 has been reached,
      or if there are no longer any trials that match a supplied Context.

    Raises:
      SuggestionInactiveError: Indicates that a suggestion was requested from an
        inactive study. Note that this is NOT raised when a finite Study runs
        out of suggestions. In such a case, an empty list is returned.
    """
        # Requests a trial.
        try:
            resp = (
                self.service_client.projects()
                .locations()
                .studies()
                .trials()
                .suggest(
                    parent=self._make_study_name(),
                    body={
                        "client_id": client_id,
                        "suggestion_count": _SUGGESTION_COUNT_PER_REQUEST,
                    },
                )
                .execute()
            )
        except errors.HttpError as e:
            if e.resp.status == 429:
                # Status 429 'RESOURCE_EXAUSTED' is raised when trials more than the
                # maximum limit (1000) of the Optimizer service for a study are
                # requested, or the number of finite search space. For distributed
                # tuning, a tuner worker may request the 1001th trial, while the other
                # tuner worker has not completed training the 1000th trial, and triggers
                # this error.
                tf.get_logger().info("Reached max number of trials.")
                return {}
            else:
                tf.get_logger().info("SuggestTrial failed.")
                raise e

        # Polls the suggestion of long-running operations.
        tf.get_logger().info("CreateTrial: polls the suggestions.")
        operation = self._obtain_long_running_operation(resp)

        suggestions = operation["response"]

        if "trials" not in suggestions:
            if operation["response"]["studyState"] == "INACTIVE":
                raise SuggestionInactiveError(
                    "The study is stopped due to an internal error."
                )
        return suggestions

    def report_intermediate_objective_value(
        self, step, elapsed_secs, metric_list, trial_id
    ):
        """Calls AddMeasurementToTrial with the provided objective_value.

    Args:
      step: The number of steps the model has trained for.
      elapsed_secs: The number of seconds since Trial execution began.
      metric_list: A list of dictionary from metric names (strings) to values
        (doubles) for additional metrics to record.
      trial_id: trial_id.
    """
        measurement = {
            "stepCount": step,
            "elapsedTime": {"seconds": int(elapsed_secs)},
            "metrics": metric_list,
        }
        try:
            self.service_client.projects().locations().studies().trials().addMeasurement(
                name=self._make_trial_name(trial_id), body={"measurement": measurement}
            ).execute()
        except errors.HttpError as e:
            tf.get_logger().info("AddMeasurement failed.")
            raise e

    def should_trial_stop(self, trial_id):
        """"Returns whether trial should stop early.

    Args:
      trial_id: trial_id.
    Returns:
      Whether it is recommended to stop the trial early.
    """
        trial_name = self._make_trial_name(trial_id)
        try:
            resp = (
                self.service_client.projects()
                .locations()
                .studies()
                .trials()
                .checkEarlyStoppingState(name=trial_name)
                .execute()
            )
        except errors.HttpError as e:
            tf.get_logger().info("CheckEarlyStoppingState failed.")
            raise e
        # Polls the stop decision of long-running operations.
        operation = self._obtain_long_running_operation(resp)

        tf.get_logger().info("CheckEarlyStoppingStateResponse")
        if operation["response"].get("shouldStop"):
            # Stops a trial.
            try:
                tf.get_logger().info("Stop the Trial.")
                self.service_client.projects().locations().studies().trials().stop(
                    name=trial_name
                ).execute()
            except errors.HttpError as e:
                tf.get_logger().info("StopTrial failed.")
                raise e
            return True
        return False

    def complete_trial(self, trial_id, trial_infeasible, infeasibility_reason=None):
        """Marks the trial as COMPLETED and sets the final measurement.

    Args:
      trial_id: trial_id.
      trial_infeasible: If True, the parameter setting is not feasible.
      infeasibility_reason: The reason the Trial was infeasible. Should only be
        non-empty if trial_infeasible==True.

    Returns:
      The Completed Optimizer trials.
    """
        try:
            optimizer_trial = (
                self.service_client.projects()
                .locations()
                .studies()
                .trials()
                .complete(
                    name=self._make_trial_name(trial_id),
                    body={
                        "trial_infeasible": trial_infeasible,
                        "infeasible_reason": infeasibility_reason,
                    },
                )
                .execute()
            )
        except errors.HttpError as e:
            tf.get_logger().info("CompleteTrial failed.")
            raise e
        return optimizer_trial

    def list_trials(self):
        """List trials."""
        study_name = self._make_study_name()
        try:
            resp = (
                self.service_client.projects()
                .locations()
                .studies()
                .trials()
                .list(parent=study_name)
                .execute()
            )
        except errors.HttpError as e:
            tf.get_logger().info("ListTrials failed.")
            raise e
        return resp.get("trials", [])

    def _obtain_long_running_operation(self, resp):
        """Obtain the long-running operation."""
        op_id = resp["name"].split("/")[-1]
        operation_name = "projects/{}/locations/{}/operations/{}".format(
            self.project_id, self.region, op_id
        )
        try:
            get_op = (
                self.service_client.projects()
                .locations()
                .operations()
                .get(name=operation_name)
            )
            operation = get_op.execute()
        except errors.HttpError as e:
            tf.get_logger().info("GetLongRunningOperations failed.")
            raise e

        polling_secs = 1
        num_attempts = 0
        while not operation.get("done"):
            sleep_time = self._polling_delay(num_attempts, polling_secs)
            num_attempts += 1
            tf.get_logger().info(
                "Waiting for operation; attempt {}; sleeping for {} seconds".format(
                    num_attempts, sleep_time
                )
            )
            time.sleep(sleep_time.total_seconds())
            if num_attempts > 30:  # about 10 minutes
                raise RuntimeError("GetLongRunningOperations timeout.")
            operation = get_op.execute()
        return operation

    def _polling_delay(self, num_attempts, time_scale):
        """Computes a delay to the next attempt to poll the Optimizer service.

    This does bounded exponential backoff, starting with $time_scale.
    If $time_scale == 0, it starts with a small time interval, less than
    1 second.

    Args:
      num_attempts: The number of times have we polled and found that the
        desired result was not yet available.
      time_scale: The shortest polling interval, in seconds, or zero. Zero is
        treated as a small interval, less than 1 second.

    Returns:
      A recommended delay interval, in seconds.
    """
        small_interval = 0.3  # Seconds
        interval = max(time_scale, small_interval) * 1.41 ** min(num_attempts, 9)
        return datetime.timedelta(seconds=interval)

    def _make_study_name(self):
        return "projects/{}/locations/{}/studies/{}".format(
            self.project_id, self.region, self.study_id
        )

    def _make_trial_name(self, trial_id):
        return "projects/{}/locations/{}/studies/{}/trials/{}".format(
            self.project_id, self.region, self.study_id, trial_id
        )


def create_or_load_study(project_id, region, study_id, study_config):
    """Factory method for creating or loading a CAIP Optimizer client.

  Given an Optimizer study_config, this will either create or open the specified
  study. It will create it if it doesn't already exist, and open it if someone
  has already created it.

  Note that once a study is created, you CANNOT modify it with this function.

  This function is designed for use in a distributed system, where many jobs
  call create_or_load_study() nearly simultaneously with the same $study_config.
  In that situation, all clients will end up pointing nicely to the same study.

  Args:
    project_id: A GCP project id.
    region: A GCP region. e.g. 'us-central1'.
    study_id: An identifier of the study. If not supplied, system-determined
      unique ID is given. The full study name will be
      projects/{project_id}/locations/{region}/studies/{study_id}. And the full
      trial name will be {study name}/trials/{trial_id}.
    study_config: Study configuration for CAIP Optimizer service.

  Returns:
    An _OptimizerClient object with the specified study created or loaded..
  """
    # Build the API client
    # Note that Optimizer service is exposed as a regional endpoint. As such,
    # an API client needs to be created separately from the default.
    apidoc_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(apidoc_path, _OPTIMIZER_API_DOCUMENT_FILE)) as f:
        service_client = discovery.build_from_document(
            service=json.load(f), requestBuilder=TunerHttpRequest
        )

    # Creates or loads a study.
    study_parent = "projects/{}/locations/{}".format(project_id, region)

    request = (
        service_client.projects()
        .locations()
        .studies()
        .create(
            body={"study_config": study_config}, parent=study_parent, studyId=study_id
        )
    )
    try:
        tf.get_logger().info(request.execute())
    except errors.HttpError as e:
        if e.resp.status != 409:  # 409 implies study exists, will be handled below
            raise e

        tf.get_logger().info("Study already existed. Load existing study...")
        # Get study
        study_name = "{}/studies/{}".format(study_parent, study_id)
        x = 1
        while True:
            try:
                service_client.projects().locations().studies().get(
                    name=study_name
                ).execute()
            except errors.HttpError as err:
                if x >= _NUM_TRIES_FOR_STUDIES:
                    raise RuntimeError(
                        "GetStudy wasn't successful after {0} tries: {1!s}".format(
                            _NUM_TRIES_FOR_STUDIES, err
                        )
                    )
                x += 1
                time.sleep(1)  # wait 1 second before trying to get the study again
            else:
                break

    return _OptimizerClient(service_client, project_id, region, study_id)


class TunerHttpRequest(googleapiclient_http.HttpRequest):
    """HttpRequest builder that sets a customized user-agent header to Cloud Tuner.

  This is used to track the usage of the Cloud Tuner.
  """

    def __init__(self, *args, **kwargs):
        """Construct a HttpRequest.

    Args:
      *args: Positional arguments to pass to the base class constructor.
      **kwargs: Keyword arguments to pass to the base class constructor.
    """
        headers = kwargs.setdefault("headers", {})
        headers["user-agent"] = _USER_AGENT_FOR_CLOUD_TUNER_TRACKING
        super(TunerHttpRequest, self).__init__(*args, **kwargs)
