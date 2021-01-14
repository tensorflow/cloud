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
import http
import json
import time
from typing import Any, Dict, List, Mapping, Optional, Text, Union

from googleapiclient import discovery
from googleapiclient import errors
import tensorflow as tf

from tensorflow_cloud.tuner import vizier_client_interface
from tensorflow_cloud.tuner import constants
from tensorflow_cloud.utils import google_api_client


class SuggestionInactiveError(Exception):
    """Indicates that GetSuggestion was called on an inactive study."""


class _OptimizerClient(vizier_client_interface.VizierClientInterface):
    """A wrapper class that allows for easy interaction with a Study."""

    def __init__(self,
                 service_client: discovery.Resource,
                 project_id: Text,
                 region: Text,
                 study_id: Text = None):
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
                `projects/{project_id}/locations/{region}/studies/{study_id}`.
                The full trial name will be `{study name}/trials/{trial_id}`.
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

    def get_suggestions(
        self,
        client_id: Text,
        suggestion_count: int = constants.SUGGESTION_COUNT_PER_REQUEST
    ) -> List[Dict[Text, Any]]:
        """Gets a list of suggested Trials.

        Args:
            client_id: An ID that identifies the `Tuner` requesting a `Trial`.
                `Tuners` that should run the same trial (for instance, when
                running a multi-worker model) should have the same ID. If
                multiple suggestTrialsRequests have the same tuner_id, the
                service will return the identical suggested trial if the trial
                is PENDING, and provide a new trial if the last suggest trial
                was completed.
            suggestion_count: The number of suggestions to request.

        Returns:
          A list of Trials (represented by JSON dicts). This may be an empty
          list if:
          1. A finite search space has been exhausted.
          2. If max_num_trials = 1000 has been reached.
          3. Or if there are no longer any trials that match a supplied Context.

        Raises:
            SuggestionInactiveError: Indicates that a suggestion was requested
                from an inactive study. Note that this is NOT raised when a
                finite Study runs out of suggestions. In such a case, an empty
                list is returned.
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
                        "suggestion_count": suggestion_count,
                    },
                )
                .execute()
            )
        except errors.HttpError as e:
            if e.resp.status == 429:
                # Status 429 'RESOURCE_EXAUSTED' is raised when trials more than
                # the maximum limit (1000) of the Optimizer service for a study
                # are requested, or the number of finite search space.
                # For distributed tuning, a tuner worker may request the 1001th
                # trial, while the other tuner worker has not completed training
                # the 1000th trial, and triggers this error.
                tf.get_logger().info("Reached max number of trials.")
                return []
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
        return suggestions["trials"]

    def report_intermediate_objective_value(
        self,
        step: int,
        elapsed_secs: float,
        metric_list: List[Mapping[Text, Union[int, float]]],
        trial_id: Text,
    ) -> None:
        """Calls AddMeasurementToTrial with the provided objective_value.

        Args:
            step: The number of steps the model has trained for.
            elapsed_secs: The number of seconds since Trial execution began.
            metric_list: A list of dictionary from metric names (strings) to
                values (doubles) for additional metrics to record.
            trial_id: trial_id.
        """
        measurement = {
            "stepCount": step,
            "elapsedTime": {"seconds": int(elapsed_secs)},
            "metrics": metric_list,
        }
        try:
            self.service_client.projects().locations().studies().trials(
                ).addMeasurement(
                    name=self._make_trial_name(trial_id),
                    body={"measurement": measurement}).execute()
        except errors.HttpError as e:
            tf.get_logger().info("AddMeasurement failed.")
            raise e

    def should_trial_stop(self, trial_id: Text) -> bool:
        """Returns whether trial should stop early.

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
                self.service_client.projects().locations().studies().trials(
                    ).stop(name=trial_name).execute()
            except errors.HttpError as e:
                tf.get_logger().info("StopTrial failed.")
                raise e
            return True
        return False

    def complete_trial(self,
                       trial_id: Text,
                       trial_infeasible: bool,
                       infeasibility_reason: Text = None)  -> Dict[Text, Any]:
        """Marks the trial as COMPLETED and sets the final measurement.

        Args:
            trial_id: trial_id.
            trial_infeasible: If True, the parameter setting is not feasible.
            infeasibility_reason: The reason the Trial was infeasible. Should
                only be non-empty if trial_infeasible==True.

        Returns:
            The Completed Optimizer trial, represented as a JSON Dictionary.
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

    def get_trial(self, trial_id: Text) -> Dict[Text, Any]:
        """Return the Optimizer trial for the given trial_id."""
        try:
            trial = (
                self.service_client.projects()
                .locations()
                .studies()
                .trials()
                .get(name=self._make_trial_name(trial_id))
                .execute()
            )
        except errors.HttpError:
            tf.get_logger().info("GetTrial failed.")
            raise
        return trial

    def list_trials(self) -> List[Dict[Text, Any]]:
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

    def list_studies(self) -> List[Dict[Text, Any]]:
        """List all studies under the current project and region.

        Returns:
            The list of studies.
        """
        parent_name = self._make_parent_name()
        try:
            resp = self.service_client.projects().locations().studies().list(
                parent=parent_name).execute()
        except errors.HttpError:
            tf.get_logger().info("ListStudies failed.")
            raise
        return resp.get("studies", [])

    def delete_study(self, study_name: Text = None) -> None:
        """Deletes the study.

        Args:
            study_name: Name of the study.

        Raises:
            ValueError: Indicates that the study_name does not exist.
            HttpError: Indicates a HTTP error from calling the discovery API.
        """
        if study_name is None:
            study_name = self._make_study_name()
        try:
            self.service_client.projects().locations().studies().delete(
                name=study_name).execute()
        except errors.HttpError as e:
            if e.resp.status == http.HTTPStatus.NOT_FOUND.value:
                raise ValueError(
                    "DeleteStudy failed. Study not found: {}."
                    .format(study_name))
            tf.get_logger().info("DeleteStudy failed.")
            raise
        tf.get_logger().info("Study deleted: {}.".format(study_name))

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
                "Waiting for operation; attempt {}; "
                "sleeping for {} seconds".format(
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
            time_scale: The shortest polling interval, in seconds, or zero.
                Zero is treated as a small interval, less than 1 second.

        Returns:
            A recommended delay interval, in seconds.
        """
        small_interval = 0.3  # Seconds
        interval = max(
            time_scale, small_interval) * 1.41 ** min(num_attempts, 9)
        return datetime.timedelta(seconds=interval)

    def _make_study_name(self):
        return "projects/{}/locations/{}/studies/{}".format(
            self.project_id, self.region, self.study_id
        )

    def _make_trial_name(self, trial_id):
        return "projects/{}/locations/{}/studies/{}/trials/{}".format(
            self.project_id, self.region, self.study_id, trial_id
        )

    def _make_parent_name(self):
        return "projects/{}/locations/{}".format(self.project_id, self.region)


def create_or_load_study(
    project_id: Text,
    region: Text,
    study_id: Text,
    study_config: Optional[Dict[Text, Any]] = None,
) -> _OptimizerClient:
    """Factory method for creating or loading a CAIP Optimizer client.

    Given an Optimizer study_config, this will either create or open the
    specified study. It will create it if it doesn't already exist, and open
    it if someone has already created it.

    Note that once a study is created, you CANNOT modify it with this function.

    This function is designed for use in a distributed system, where many jobs
    call create_or_load_study() nearly simultaneously with the same
    `study_config`. In that situation, all clients will end up pointing nicely
    to the same study.

    Args:
        project_id: A GCP project id.
        region: A GCP region. e.g. 'us-central1'.
        study_id: An identifier of the study. If not supplied, system-determined
            unique ID is given. The full study name will be
            projects/{project_id}/locations/{region}/studies/{study_id}.
            And the full trial name will be {study name}/trials/{trial_id}.
        study_config: Study configuration for CAIP Optimizer service. If not
            supplied, it will be assumed that the study with the given study_id
            already exists, and will try to retrieve that study.

    Returns:
        An _OptimizerClient object with the specified study created or loaded.

    Raises:
        RuntimeError: Indicates that study_config is supplied but CreateStudy
            failed and GetStudy did not succeed after
            constants.MAX_NUM_TRIES_FOR_STUDIES tries.
        ValueError: Indicates that study_config is not supplied and the study
            with the given study_id does not exist.
    """
    # Build the API client
    # Note that Optimizer service is exposed as a regional endpoint. As such,
    # an API client needs to be created separately from the default.
    with open(constants.OPTIMIZER_API_DOCUMENT_FILE) as f:
        service_client = discovery.build_from_document(
            service=json.load(f),
            requestBuilder=google_api_client.TFCloudHttpRequest,
        )

    # Creates or loads a study.
    study_parent = "projects/{}/locations/{}".format(project_id, region)

    if study_config is None:
        # If study config is unspecified, assume that the study already exists.
        _get_study(
            service_client=service_client,
            study_parent=study_parent,
            study_id=study_id,
            study_should_exist=True,
        )

    else:
        request = (
            service_client.projects()
            .locations()
            .studies()
            .create(
                body={"study_config": study_config},
                parent=study_parent,
                studyId=study_id,
            )
        )
        try:
            tf.get_logger().info(request.execute())
        except errors.HttpError as e:
            if e.resp.status != 409:  # 409 implies study exists, handled below
                raise

            _get_study(
                service_client=service_client,
                study_parent=study_parent,
                study_id=study_id,
            )

    return _OptimizerClient(service_client, project_id, region, study_id)


def _get_study(
    service_client: discovery.Resource,
    study_parent: Text,
    study_id: Text,
    study_should_exist: bool = False,
):
    """Method for loading a study.

    Given the study_parent and the study_id, this method will load the specified
    study, up to constants.MAX_NUM_TRIES_FOR_STUDIES tries.

    Args:
        service_client: An API client of CAIP Optimizer service.
        study_parent: Prefix of the study name. The full study name will be
            {study_parent}/studies/{study_id}.
        study_id: An identifier of the study.
        study_should_exist: Indicates whether it should be assumed that the
            study with the given study_id exists.
    """
    study_name = "{}/studies/{}".format(study_parent, study_id)
    tf.get_logger().info(
        "Study already exists: {}.\nLoad existing study...".format(study_name))
    num_tries = 0
    while True:
        try:
            service_client.projects().locations().studies().get(
                name=study_name
            ).execute()
        except errors.HttpError as err:
            num_tries += 1
            if num_tries >= constants.MAX_NUM_TRIES_FOR_STUDIES:
                if (
                    study_should_exist
                    and err.resp.status == http.HTTPStatus.NOT_FOUND.value
                ):
                    raise ValueError(
                        "GetStudy failed. Study not found: {}.".format(study_id)
                    )
                else:
                    raise RuntimeError(
                        "GetStudy failed. Max retries reached: {0!s}".format(
                            err
                        )
                    )
            time.sleep(1)  # wait 1 second before trying to get the study again
        else:
            break
