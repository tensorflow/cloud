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
"""Utilities for CAIP Optimizer and KerasTuner integration."""

from typing import Any, Dict, List, Optional, Text, Union

from kerastuner.engine import hyperparameters as hp_module
from kerastuner.engine import metrics_tracking
from kerastuner.engine import oracle as oracle_module
from kerastuner.engine import trial as trial_module
import numpy as np
from tensorboard.plugins.hparams import api as hparams_api

# CAIP Optimizer constants.
_DISCRETE = "DISCRETE"
_CATEGORICAL = "CATEGORICAL"
_DOUBLE = "DOUBLE"
_INTEGER = "INTEGER"
_PARAMETER_TYPE_UNSPECIFIED = "PARAMETER_TYPE_UNSPECIFIED"

_SCALE_TYPE_UNSPECIFIED = "SCALE_TYPE_UNSPECIFIED"
_LINEAR_SCALE = "UNIT_LINEAR_SCALE"
_LOG_SCALE = "UNIT_LOG_SCALE"
_REVERSE_LOG_SCALE = "UNIT_REVERSE_LOG_SCALE"

_GOAL_TYPE_UNSPECIFIED = "GOAL_TYPE_UNSPECIFIED"
_GOAL_MAXIMIZE = "MAXIMIZE"
_GOAL_MINIMIZE = "MINIMIZE"

# KerasTuner constants.
_DIRECTION_MAX = "max"
_DIRECTION_MIN = "min"
_SAMPLING_LINEAR = "linear"
_SAMPLING_LOG = "log"
_SAMPLING_REVERSE_LOG = "reverse_log"


def make_study_config(
    objective: Union[Text, oracle_module.Objective],
    hyperparams: hp_module.HyperParameters) -> Dict[Text, Any]:
    """Generates Optimizer study_config from kerastuner configurations.

    Arguments:
        objective: String or `oracle_module.Objective`. If a string,
            the direction of the optimization (min or max) will be inferred.
        hyperparams: HyperParameters class instance. Can be used to override (or
            register in advance) hyperparameters in the search space.

    Returns:
        A dict that holds the study configuration.
    """
    study_config = {}
    # The default algorithm used by the CAIP Optimizer.
    study_config["algorithm"] = "ALGORITHM_UNSPECIFIED"
    # If no implementation_config is set, automated early stopping will not be
    # run.
    study_config["automatedStoppingConfig"] = {
        "decayCurveStoppingConfig": {"useElapsedTime": True}
    }

    # Converts oracle_module.Objective to metrics.
    study_config["metrics"] = []
    for obj in format_objective(objective):
        study_config["metrics"].append(
            {"metric": obj.name, "goal": format_goal(obj.direction)}
        )

    # Converts hp_module.HyperParameters to parameters.
    study_config["parameters"] = _convert_hyperparams_to_optimizer_params(
        hyperparams)

    return study_config


def convert_study_config_to_objective(
    study_config: Dict[Text, Any]) -> List[oracle_module.Objective]:
    """Converts Optimizer study_config to a list of oracle_module.Objective."""
    if not study_config.get("metrics"):
        raise ValueError('"metrics" not found in study_config {}'.format(
            study_config))
    if not isinstance(study_config["metrics"], list):
        raise ValueError(
            'study_config["metrics"] should be a list of {"metric": ...}')
    if not study_config["metrics"][0].get("metric"):
        raise ValueError('"metric" not found in study_config["metrics"][0]')
    return [
        format_objective(m["metric"], format_goal(m["goal"]))[0]
        for m in study_config["metrics"]
    ]


def convert_study_config_to_hps(
    study_config: Dict[Text, Any]) -> hp_module.HyperParameters:
    """Converts CAIP Optimizer study_config to HyperParameters."""
    if not study_config.get("parameters"):
        raise ValueError("Parameters are not found in the study_config: ",
                         study_config)
    if not isinstance(study_config["parameters"], list):
        raise ValueError(
            "Parameters should be a list of parameter with at least 1 "
            "parameter, found ", study_config["parameters"],
        )

    hps = hp_module.HyperParameters()
    for param in study_config["parameters"]:
        _is_parameter_valid(param)
        name = param["parameter"]
        if param["type"] == _DISCRETE:
            values = param["discrete_value_spec"]["values"]
            is_numeric = True
            for v in values:
                if not isinstance(v, (int, float)):
                    is_numeric = False
            if (
                is_numeric and len(values) > 2 and
                np.all(np.diff(values, 2) == 0)
            ):
                # If the numeric sequence is an arithmetic sequence, use
                # Int/Float with step
                is_int = True
                for v in values:
                    if not isinstance(v, int):
                        is_int = False
                hps_type = hps.Int if is_int else hps.Float

                if (
                    param.get("scale_type")
                    and param["scale_type"] != _SCALE_TYPE_UNSPECIFIED
                ):
                    hps_type(
                        name,
                        min_value=values[0],
                        max_value=values[-1],
                        step=values[1] - values[0],
                        sampling=_format_sampling(param["scale_type"]),
                    )
                else:
                    hps_type(
                        name,
                        min_value=values[0],
                        max_value=values[-1],
                        step=values[1] - values[0],
                    )
            else:
                hps.Choice(name, values)
        elif param["type"] == _CATEGORICAL:
            hps.Choice(name, param["categorical_value_spec"]["values"])
        elif param["type"] == _DOUBLE:
            if (
                param.get("scale_type")
                and param["scale_type"] != _SCALE_TYPE_UNSPECIFIED
            ):
                hps.Float(
                    name,
                    min_value=param["double_value_spec"]["min_value"],
                    max_value=param["double_value_spec"]["max_value"],
                    sampling=_format_sampling(param["scale_type"]),
                )
            else:
                hps.Float(
                    name,
                    min_value=param["double_value_spec"]["min_value"],
                    max_value=param["double_value_spec"]["max_value"],
                )
        elif param["type"] == _INTEGER:
            if (
                param.get("scale_type")
                and param["scale_type"] != _SCALE_TYPE_UNSPECIFIED
            ):
                hps.Int(
                    name,
                    min_value=param["integer_value_spec"]["min_value"],
                    max_value=param["integer_value_spec"]["max_value"],
                    sampling=_format_sampling(param["scale_type"]),
                )
            else:
                hps.Int(
                    name,
                    min_value=param["integer_value_spec"]["min_value"],
                    max_value=param["integer_value_spec"]["max_value"],
                )
        else:
            raise ValueError(
                "Unknown parameter type: {}.".format(param["type"]))
    return hps


def _is_parameter_valid(param: Dict[Text, Any]):
    """Checks if study_config parameter is valid."""
    if not param.get("parameter"):
        raise ValueError('"parameter" (name) is not specified.')
    if not param.get("type"):
        raise ValueError("Parameter {} type is not specified.".format(param))
    if param["type"] == _DISCRETE:
        if not param.get("discrete_value_spec"):
            raise ValueError(
                "Parameter {} is missing discrete_value_spec.".format(param)
            )
        if not isinstance(param["discrete_value_spec"].get("values"), list):
            raise ValueError(
                'Parameter spec {} is missing "values".'.format(
                    param["discrete_value_spec"]
                )
            )
    elif param["type"] == _CATEGORICAL:
        if not param.get("categorical_value_spec"):
            raise ValueError(
                "Parameter {} is missing categorical_value_spec.".format(param)
            )
        if not isinstance(param["categorical_value_spec"].get("values"), list):
            raise ValueError(
                'Parameter spec {} is missing "values".'.format(
                    param["categorical_value_spec"]
                )
            )
    elif param["type"] == _DOUBLE:
        if not param.get("double_value_spec"):
            raise ValueError(
                "Parameter {} is missing double_value_spec.".format(param))
        spec = param["double_value_spec"]
        if not (
            isinstance(spec.get("min_value"), float)
            and isinstance(spec.get("max_value"), float)
        ):
            raise ValueError(
                'Parameter spec {} requires both "min_value" and '
                '"max_value".'.format(spec)
            )
    elif param["type"] == _INTEGER:
        if not param.get("integer_value_spec"):
            raise ValueError(
                "Parameter {} is missing integer_value_spec.".format(param)
            )
        spec = param["integer_value_spec"]
        if not (
            isinstance(spec.get("min_value"), int)
            and isinstance(spec.get("max_value"), int)
        ):
            raise ValueError(
                'Parameter spec {} requires both "min_value" and '
                '"max_value".'.format(spec)
            )
    else:
        raise ValueError("Unknown parameter type: {}.".format(param["type"]))


def _convert_hyperparams_to_optimizer_params(
    hyperparams: hp_module.HyperParameters) -> List[Any]:
    """Converts HyperParameters to a list of ParameterSpec in study_config."""
    param_type = []
    for hp in hyperparams.space:
        param = {}
        param["parameter"] = hp.name
        if isinstance(hp, hp_module.Choice):
            values = hp.values
            if isinstance(values[0], str):
                param["type"] = _CATEGORICAL
                param["categorical_value_spec"] = {"values": values}
            else:
                param["type"] = _DISCRETE
                param["discrete_value_spec"] = {"values": values}
        elif isinstance(hp, hp_module.Int):
            if hp.step is None or hp.step == 1:
                param["type"] = _INTEGER
                param["integer_value_spec"] = {
                    "min_value": hp.min_value,
                    "max_value": hp.max_value,
                }
                if hp.sampling is not None:
                    param.update(_get_scale_type(hp.sampling))
            else:
                # Note: hp.max_value is inclusive, while the end index of
                # range() is exclusive, hence the +1
                values = list(range(hp.min_value, hp.max_value + 1, hp.step))
                param["type"] = _DISCRETE
                param["discrete_value_spec"] = {"values": values}
        elif isinstance(hp, hp_module.Float):
            if hp.step is None:
                param["type"] = _DOUBLE
                param["double_value_spec"] = {
                    "min_value": hp.min_value,
                    "max_value": hp.max_value,
                }
                if hp.sampling is not None:
                    param.update(_get_scale_type(hp.sampling))
            else:
                # Match how KerasTuner generates the range
                values = np.arange(
                    hp.min_value, hp.max_value + 1e-7, step=hp.step).tolist()
                param["type"] = _DISCRETE
                param["discrete_value_spec"] = {"values": values}
        elif isinstance(hp, hp_module.Boolean):
            param["type"] = _CATEGORICAL
            param["categorical_value_spec"] = {"values": ["True", "False"]}
        elif isinstance(hp, hp_module.Fixed):
            if isinstance(hp.value, (str, bool)):
                param["type"] = _CATEGORICAL
                param["categorical_value_spec"] = {"values": [str(hp.value)]}
            else:
                param["type"] = _DISCRETE
                param["discrete_value_spec"] = {"values": [float(hp.value)]}
        else:
            raise ValueError(
                "`HyperParameter` type not recognized: {}".format(hp))

        param_type.append(param)

    return param_type


def convert_hyperparams_to_hparams(
    hyperparams: hp_module.HyperParameters) -> Dict[hparams_api.HParam, Any]:
    """Converts KerasTuner HyperParameters to TensorBoard HParams.

    Args:
        hyperparams: A KerasTuner HyperParameters instance

    Returns:
        A dict that maps TensorBoard HParams to current values.
    """
    hparams = {}
    for hp in hyperparams.space:
        hparams_value = {}
        try:
            hparams_value = hyperparams.get(hp.name)
        except ValueError:
            continue

        hparams_domain = {}
        if isinstance(hp, hp_module.Choice):
            hparams_domain = hparams_api.Discrete(hp.values)
        elif isinstance(hp, hp_module.Int):
            if hp.step is None or hp.step == 1:
                hparams_domain = hparams_api.IntInterval(
                    hp.min_value, hp.max_value)
            else:
                # Note: `hp.max_value` is inclusive, unlike the end index
                # of Python `range()`, which is exclusive
                values = list(
                    range(hp.min_value, hp.max_value + 1, hp.step))
                hparams_domain = hparams_api.Discrete(values)
        elif isinstance(hp, hp_module.Float):
            if hp.step is None:
                hparams_domain = hparams_api.RealInterval(
                    hp.min_value, hp.max_value)
            else:
                # Note: `hp.max_value` is inclusive, which is also
                # the default for Numpy's linspace
                num_samples = int((hp.max_value - hp.min_value) / hp.step)
                end_value = hp.min_value + (num_samples * hp.step)
                values = np.linspace(
                    hp.min_value, end_value, num_samples + 1).tolist()
                hparams_domain = hparams_api.Discrete(values)
        elif isinstance(hp, hp_module.Boolean):
            hparams_domain = hparams_api.Discrete([True, False])
        elif isinstance(hp, hp_module.Fixed):
            hparams_domain = hparams_api.Discrete([hp.value])
        else:
            raise ValueError(
                "`HyperParameter` type not recognized: {}".format(hp))

        hparams_key = hparams_api.HParam(hp.name, hparams_domain)
        hparams[hparams_key] = hparams_value

    return hparams


def format_objective(
    objective: Union[Text, oracle_module.Objective,
                     List[Union[Text, oracle_module.Objective]]],
    direction: Text = None) -> List[oracle_module.Objective]:
    """Formats objective to a list of oracle_module.Objective.

    Arguments:
        objective: If a string, the direction of the optimization (min or max)
            will be inferred.
        direction: Optional. e.g. 'min' or 'max'.

    Returns:
        A list of oracle_module.Objective.

    Raises:
        TypeError: indicates wrong objective format.
    """
    if isinstance(objective, oracle_module.Objective):
        return [objective]
    if isinstance(objective, str):
        if direction:
            return [oracle_module.Objective(objective, direction)]
        return [
            oracle_module.Objective(
                objective, metrics_tracking.infer_metric_direction(objective)
            )
        ]
    if isinstance(objective, list):
        if isinstance(objective[0], oracle_module.Objective):
            return objective
        if isinstance(objective[0], str):
            return [
                oracle_module.Objective(
                    m, metrics_tracking.infer_metric_direction(m))
                for m in objective
            ]
    raise TypeError(
        "Objective should be either string or oracle_module.Objective, "
        "found {}".format(objective)
    )


def format_goal(metric_direction: Text) -> Text:
    """Converts oracle_module.Objective 'direction' to study_config 'goal'.

    Args:
        metric_direction: If oracle_module.Objective 'direction' is supplied,
            returns 'goal' in CAIP Optimizer study_config. If 'goal' in CAIP
            Optimizer 'study_config' is supplied, returns 'direction' in
            oracle_module.Objective.

    Returns:
        'goal' or 'direction'.
    """
    if metric_direction == _DIRECTION_MAX:
        return _GOAL_MAXIMIZE
    if metric_direction == _DIRECTION_MIN:
        return _GOAL_MINIMIZE
    if metric_direction == _GOAL_MAXIMIZE:
        return _DIRECTION_MAX
    if metric_direction == _GOAL_MINIMIZE:
        return _DIRECTION_MIN
    return _GOAL_TYPE_UNSPECIFIED


def _get_scale_type(sampling):
    """Returns scale_type in CAIP Optimizer study_config."""
    if sampling == _SAMPLING_LINEAR:
        return {"scale_type": _LINEAR_SCALE}
    if sampling == _SAMPLING_LOG:
        return {"scale_type": _LOG_SCALE}
    if sampling == _SAMPLING_REVERSE_LOG:
        return {"scale_type": _REVERSE_LOG_SCALE}
    return {"scale_type": _SCALE_TYPE_UNSPECIFIED}


def get_trial_id(optimizer_trial: Dict[Text, Any]) -> Text:
    r"""Gets trial_id from a CAIP Optimizer Trial.

    Arguments:
        optimizer_trial: A CAIP Optimizer Trial instance.

    Returns:
        trial_id. Note that a trial name follows the following format
        `projects/{project_id}/locations/{region}/studies/{study_id}/trials/\
            {trial_id}`
    """
    return optimizer_trial["name"].split("/")[-1]


def convert_optimizer_trial_to_dict(
    optimizer_trial: Dict[Text, Any]
) -> Dict[Text, Any]:
    """Converts Optimizer Trial parameters into a Python dict.

    Arguments:
        optimizer_trial: A CAIP Optimizer Trial instance.

    Returns:
        A dict that maps the Optimizer Trial parameters' names to their
        respective values.
    """
    result = {}
    for param in optimizer_trial["parameters"]:
        if "floatValue" in param:
            result[param["parameter"]] = float(param["floatValue"])
        if "intValue" in param:
            result[param["parameter"]] = int(param["intValue"])
        if "stringValue" in param:
            result[param["parameter"]] = str(param["stringValue"])
    return result


def convert_optimizer_trial_to_hps(
    hps: hp_module.HyperParameters,
    optimizer_trial: Dict[Text, Any]
) -> hp_module.HyperParameters:
    """Converts Optimizer Trial parameters into KerasTuner HyperParameters.

    Arguments:
        hps: Sample KerasTuner HyperParameters object for config initialization
        optimizer_trial: A CAIP Optimizer Trial instance.

    Returns:
        A KerasTuner HyperParameters object that holds the Optimizer Trial
        parameters.
    """
    hps = hp_module.HyperParameters.from_config(hps.get_config())
    hps.values = convert_optimizer_trial_to_dict(optimizer_trial)
    return hps


def convert_completed_optimizer_trial_to_keras_trial(
    optimizer_trial: Dict[Text, Any],
    hyperparameter_space: hp_module.HyperParameters,
) -> trial_module.Trial:
    """Converts completed Optimizer Trial into KerasTuner Trial.

    Arguments:
        optimizer_trial: A CAIP Optimizer Trial Instance.
        hyperparameter_space: Mandatory and must include definitions for all
            hyperparameters used during the search.

    Returns:
        A KerasTuner Trial.
    """
    kerastuner_trial = trial_module.Trial(
        hyperparameters=convert_optimizer_trial_to_hps(
            hyperparameter_space, optimizer_trial
        ),
        trial_id=get_trial_id(optimizer_trial),
        status=trial_module.TrialStatus.COMPLETED,
    )
    # If trial had ended before having intermediate metric reporting,
    # set stepCount = 0.
    final_measurement = optimizer_trial.get("finalMeasurement")
    if not final_measurement:
        raise ValueError('"finalMeasurement" not found in this trial {}'
                         .format(optimizer_trial))

    kerastuner_trial.best_step = int(final_measurement.get("stepCount", 0))
    kerastuner_trial.score = final_measurement["metrics"][0].get("value")
    return kerastuner_trial


def _format_sampling(scale_type: Text) -> Optional[Text]:
    """Format CAIP Optimizer scale_type for HyperParameter.sampling."""
    if scale_type == _LINEAR_SCALE:
        return _SAMPLING_LINEAR
    if scale_type == _LOG_SCALE:
        return _SAMPLING_LOG
    if scale_type == _REVERSE_LOG_SCALE:
        return _SAMPLING_REVERSE_LOG
    return None
