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
"""An abstract class for the client used in both OSS Vizier and Cloud AI Platform Optimizer Service."""
import abc
from typing import List, Mapping, Text, Union, Dict, Any


class VizierClientInterface(abc.ABC):
  """Abstract class for clients to interact with a Study."""

  @abc.abstractmethod
  def get_suggestions(self, client_id: Text,
                      suggestion_count: int) -> List[Dict[Text, Any]]:
    """Gets a list of suggested Trials.

    Args:
      client_id: An ID that identifies the `Tuner` requesting a `Trial`.
        `Tuners` that should run the same trial (for instance, when running a
        multi-worker model) should have the same ID. If multiple
        suggestTrialsRequests have the same tuner_id, the service will return
        the identical suggested trial if the trial is PENDING, and provide a new
        trial if the last suggest trial was completed.
      suggestion_count: The number of suggestions to request.

    Returns:
      A list of Trials (represented by JSON dicts). This may be an empty list
      if:
      1. A finite search space has been exhausted.
      2. If max_num_trials = 1000 has been reached.
      3. Or if there are no longer any trials that match a supplied Context.

    Raises:
      SuggestionInactiveError: Indicates that a suggestion was requested from an
      inactive study. Note that this is NOT raised when a finite Study runs out
      of suggestions. In such a case, an empty list is returned.
    """

  @abc.abstractmethod
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
      metric_list: A list of dictionary from metric names (strings) to values
        (doubles) for additional metrics to record.
      trial_id: trial_id.
    """

  @abc.abstractmethod
  def should_trial_stop(self, trial_id: Text) -> bool:
    """Returns whether trial should stop early.

    Args:
      trial_id: trial_id.

    Returns:
      Whether it is recommended to stop the trial early.
    """

  @abc.abstractmethod
  def complete_trial(self,
                     trial_id: Text,
                     trial_infeasible: bool,
                     infeasibility_reason: Text = None) -> Dict[Text, Any]:
    """Marks the trial as COMPLETED and sets the final measurement.

    Args:
      trial_id: trial_id.
      trial_infeasible: If True, the parameter setting is not feasible.
      infeasibility_reason: The reason the Trial was infeasible. Should only be
        non-empty if trial_infeasible==True.

    Returns:
      The Completed Vizier trial, represented as a JSON Dictionary.
    """

  @abc.abstractmethod
  def get_trial(self, trial_id: Text) -> Dict[Text, Any]:
    """Return the Optimizer trial for the given trial_id."""

  @abc.abstractmethod
  def list_trials(self) -> List[Dict[Text, Any]]:
    """List trials."""

  @abc.abstractmethod
  def list_studies(self) -> List[Dict[Text, Any]]:
    """List all studies under the current project and region.

    Returns:
      The list of studies.
    """

  @abc.abstractmethod
  def delete_study(self, study_name: Text = None) -> None:
    """Deletes the study.

    Args:
      study_name: Name of the study.

    Raises:
      ValueError: Indicates that the study_name does not exist.
    """
