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
"""Tests for utils."""

import copy
from absl.testing import parameterized
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
import tensorflow as tf
from tensorboard.plugins.hparams import api as hparams_api
from tensorflow_cloud.tuner.tuner import utils

STUDY_CONFIG_DISCRETE = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "val_accuracy"}],
    "parameters": [
        {
            "discrete_value_spec": {"values": [1e-4, 1e-3, 1e-2]},
            "parameter": "learning_rate",
            "type": "DISCRETE",
        }
    ],
}
STUDY_CONFIG_CATEGORICAL = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "categorical_value_spec": {"values": ["LINEAR", "WIDE_AND_DEEP"]},
            "parameter": "model_type",
            "type": "CATEGORICAL",
        }
    ],
}
STUDY_CONFIG_INT = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "parameter": "units",
            "type": "INTEGER",
            "integer_value_spec": {"min_value": 1, "max_value": 4},
            "scale_type": "UNIT_LINEAR_SCALE",
        }
    ],
}
STUDY_CONFIG_INT_STEP = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "parameter": "units",
            "type": "DISCRETE",
            "discrete_value_spec": {"values": [32, 64, 96, 128]},
        }
    ],
}
STUDY_CONFIG_FLOAT = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "double_value_spec": {"max_value": 0.5, "min_value": 0.1},
            "parameter": "learning_rate",
            "type": "DOUBLE",
            "scale_type": "UNIT_LINEAR_SCALE",
        }
    ],
}
STUDY_CONFIG_FLOAT_STEP = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "discrete_value_spec": {"values": [1, 1.25, 1.5, 1.75, 2]},
            "parameter": "learning_rate",
            "type": "DISCRETE",
        }
    ],
}
STUDY_CONFIG_FLOAT_LINEAR_SCALE = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "double_value_spec": {"max_value": 0.8, "min_value": 0.1},
            "parameter": "learning_rate",
            "scale_type": "UNIT_LINEAR_SCALE",
            "type": "DOUBLE",
        }
    ],
}
STUDY_CONFIG_FLOAT_LOG_SCALE = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "double_value_spec": {"max_value": 0.1, "min_value": 0.0001},
            "parameter": "learning_rate",
            "scale_type": "UNIT_LOG_SCALE",
            "type": "DOUBLE",
        }
    ],
}
STUDY_CONFIG_FLOAT_REVERSE_LOG_SCALE = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "double_value_spec": {"max_value": 0.1, "min_value": 0.0001},
            "parameter": "learning_rate",
            "scale_type": "UNIT_REVERSE_LOG_SCALE",
            "type": "DOUBLE",
        }
    ],
}
STUDY_CONFIG_MULTI_FLOAT = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "parameter": "theta",
            "type": "DOUBLE",
            "double_value_spec": {"min_value": 0.0, "max_value": 1.57},
            "scale_type": "UNIT_LINEAR_SCALE",
        },
        {
            "parameter": "r",
            "type": "DOUBLE",
            "double_value_spec": {"min_value": 0.0, "max_value": 1.0},
            "scale_type": "UNIT_LINEAR_SCALE",
        },
    ],
}
STUDY_CONFIG_BOOL = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "categorical_value_spec": {"values": ["True", "False"]},
            "parameter": "has_beta",
            "type": "CATEGORICAL",
        }
    ],
}
STUDY_CONFIG_FIXED_FLOAT = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "discrete_value_spec": {"values": [0.1]},
            "parameter": "beta",
            "type": "DISCRETE",
        }
    ],
}
STUDY_CONFIG_FIXED_CATEGORICAL = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "categorical_value_spec": {"values": ["WIDE_AND_DEEP"]},
            "parameter": "type",
            "type": "CATEGORICAL",
        }
    ],
}
STUDY_CONFIG_FIXED_BOOLEAN = {
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "categorical_value_spec": {"values": ["True"]},
            "parameter": "condition",
            "type": "CATEGORICAL",
        }
    ],
}
OPTIMIZER_TRIAL = {
    "name": "projects/project/locations/region/studies/study/trials/trial_1",
    "state": "ACTIVE",
    "parameters": [
        {"parameter": "learning_rate", "floatValue": 0.0001},
        {"parameter": "num_layers", "intValue": "2"},
        {"parameter": "units_0", "floatValue": 96},
        {"parameter": "units_1", "floatValue": 352},
        {"parameter": "type", "stringValue": "WIDE_AND_DEEP"},
    ],
}
COMPLETED_OPTIMIZER_TRIAL = {
    "name": "projects/project/locations/region/studies/study/trials/trial_1",
    "state": "ACTIVE",
    "parameters": [
        {"parameter": "learning_rate", "floatValue": 0.0001},
    ],
    "finalMeasurement": {
        "stepCount": "1",
        "metrics": [{"value": 0.9}],
    },
}
EXPECTED_TRIAL_HPS = {
    "learning_rate": 0.0001,
    "num_layers": 2,
    "units_0": 96.0,
    "units_1": 352.0,
    "type": "WIDE_AND_DEEP",
}


class CloudTunerUtilsTest(tf.test.TestCase, parameterized.TestCase):

    def test_convert_study_config_discrete(self):
        hps = hp_module.HyperParameters()
        hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
        study_config = utils.make_study_config(
            objective=oracle_module.Objective("val_accuracy", "max"),
            hyperparams=hps
        )
        self._assert_study_config_equal(study_config, STUDY_CONFIG_DISCRETE)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self._assert_hps_equal(actual_hps, hps)

    def test_convert_study_config_categorical(self):
        hps = hp_module.HyperParameters()
        hps.Choice("model_type", ["LINEAR", "WIDE_AND_DEEP"])
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self._assert_study_config_equal(study_config, STUDY_CONFIG_CATEGORICAL)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self._assert_hps_equal(actual_hps, hps)

    @parameterized.parameters(
        (1, 4, None, STUDY_CONFIG_INT),
        (1, 4, 1, STUDY_CONFIG_INT),
        (32, 128, 32, STUDY_CONFIG_INT_STEP))
    def test_convert_study_config_int(self, min_value, max_value, step,
                                      expected_config):
        hps = hp_module.HyperParameters()
        if step:
            hps.Int(
                "units", min_value=min_value, max_value=max_value, step=step)
        else:
            hps.Int("units", min_value=min_value, max_value=max_value)
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self._assert_study_config_equal(study_config, expected_config)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self._assert_hps_equal(actual_hps, hps)

    @parameterized.parameters(
        (0.1, 0.5, None, "linear", STUDY_CONFIG_FLOAT),
        (1, 2, 0.25, "linear", STUDY_CONFIG_FLOAT_STEP),
        (0.1, 0.8, None, "linear", STUDY_CONFIG_FLOAT_LINEAR_SCALE),
        (1e-4, 1e-1, None, "log", STUDY_CONFIG_FLOAT_LOG_SCALE),
        (1e-4, 1e-1, None, "reverse_log", STUDY_CONFIG_FLOAT_REVERSE_LOG_SCALE))
    def test_convert_study_config_float(self, min_value, max_value, step,
                                        sampling, expected_config):
        hps = hp_module.HyperParameters()
        hps.Float("learning_rate", min_value=min_value, max_value=max_value,
                  step=step, sampling=sampling)
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self._assert_study_config_equal(study_config, expected_config)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self._assert_hps_equal(actual_hps, hps)

    def test_convert_study_config_multi_float(self):
        hps = hp_module.HyperParameters()
        hps.Float("theta", min_value=0.0, max_value=1.57)
        hps.Float("r", min_value=0.0, max_value=1.0)
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self._assert_study_config_equal(study_config, STUDY_CONFIG_MULTI_FLOAT)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self._assert_hps_equal(actual_hps, hps)

    def test_convert_study_config_bool(self):
        hps = hp_module.HyperParameters()
        hps.Boolean("has_beta")
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self._assert_study_config_equal(study_config, STUDY_CONFIG_BOOL)

    @parameterized.parameters(
        ("beta", 0.1, STUDY_CONFIG_FIXED_FLOAT),
        ("type", "WIDE_AND_DEEP", STUDY_CONFIG_FIXED_CATEGORICAL),
        ("condition", True, STUDY_CONFIG_FIXED_BOOLEAN))
    def test_convert_study_config_fixed(self, name, value, expected_config):
        hps = hp_module.HyperParameters()
        hps.Fixed(name, value)
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps
        )
        self._assert_study_config_equal(study_config, expected_config)

    def test_convert_vizier_trial_to_dict(self):
        hps = hp_module.HyperParameters()
        hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
        params = utils.convert_vizier_trial_to_dict(OPTIMIZER_TRIAL)
        self.assertDictEqual(params, EXPECTED_TRIAL_HPS)

    def test_convert_vizier_trial_to_hps(self):
        hps = hp_module.HyperParameters()
        hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
        trial_hps = utils.convert_vizier_trial_to_hps(hps, OPTIMIZER_TRIAL)
        self.assertDictEqual(trial_hps.values, EXPECTED_TRIAL_HPS)

    def test_convert_completed_vizier_trial_to_keras_trial(self):
        hps = hp_module.HyperParameters()
        hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
        trial = utils.convert_completed_vizier_trial_to_keras_trial(
            COMPLETED_OPTIMIZER_TRIAL, hps)
        self.assertEqual(trial.trial_id, "trial_1")
        self.assertEqual(trial.score, 0.9)
        self.assertEqual(trial.best_step, 1)
        self.assertEqual(trial.status, trial_module.TrialStatus.COMPLETED)
        self.assertEqual(
            trial.hyperparameters.values, {"learning_rate": 0.0001})

    def test_convert_hyperparams_to_hparams_choice(self):
        hps = hp_module.HyperParameters()
        hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
        hparams = utils.convert_hyperparams_to_hparams(hps)
        expected_hparams = {
            hparams_api.HParam("learning_rate",
                               hparams_api.Discrete([1e-4, 1e-3, 1e-2])): 1e-4,
        }
        self.assertEqual(repr(hparams), repr(expected_hparams))

    @parameterized.parameters(
        ("units", 2, 16, None, hparams_api.IntInterval(2, 16), 2),
        ("units", 32, 128, 32, hparams_api.Discrete([32, 64, 96, 128]), 32))
    def test_convert_hyperparams_to_hparams_int(self, name, min_value,
                                                max_value, step,
                                                expected_domain,
                                                expected_value):
        hps = hp_module.HyperParameters()
        if step:
            hps.Int(name, min_value=min_value, max_value=max_value, step=step)
        else:
            hps.Int(name, min_value=min_value, max_value=max_value)
        hparams = utils.convert_hyperparams_to_hparams(hps)
        expected_hparams = {
            hparams_api.HParam(name, expected_domain): expected_value,
        }
        self.assertEqual(repr(hparams), repr(expected_hparams))

    @parameterized.parameters(
        ("learning_rate", 0.5, 1.5, 0.25,
         hparams_api.Discrete([0.5, 0.75, 1.0, 1.25, 1.5]), 0.5),
        ("learning_rate", 1e-4, 1e-1, None,
         hparams_api.RealInterval(1e-4, 1e-1), 1e-4))
    def test_convert_hyperparams_to_hparams_float(self, name, min_value,
                                                  max_value, step,
                                                  expected_domain,
                                                  expected_value):
        hps = hp_module.HyperParameters()
        hps.Float(name, min_value=min_value, max_value=max_value, step=step)
        hparams = utils.convert_hyperparams_to_hparams(hps)
        expected_hparams = {
            hparams_api.HParam(name, expected_domain): expected_value,
        }
        self.assertEqual(repr(hparams), repr(expected_hparams))

    def test_convert_hyperparams_to_hparams_multi_float(self):
        hps = hp_module.HyperParameters()
        hps.Float("theta", min_value=0.0, max_value=1.57)
        hps.Float("r", min_value=0.0, max_value=1.0)
        hparams = utils.convert_hyperparams_to_hparams(hps)
        expected_hparams = {
            hparams_api.HParam("r", hparams_api.RealInterval(0.0, 1.0)): 0.0,
            hparams_api.HParam("theta",
                               hparams_api.RealInterval(0.0, 1.57)): 0.0,
        }
        hparams_repr_list = [repr(hparams[x]) for x in hparams.keys()]
        expected_hparams_repr_list = [
            repr(expected_hparams[x]) for x in expected_hparams.keys()
        ]
        self.assertCountEqual(hparams_repr_list, expected_hparams_repr_list)

    def test_convert_hyperparams_to_hparams_boolean(self):
        hps = hp_module.HyperParameters()
        hps.Boolean("has_beta")
        hparams = utils.convert_hyperparams_to_hparams(hps)
        expected_hparams = {
            hparams_api.HParam("has_beta", hparams_api.Discrete([True, False])):
                False,
        }
        self.assertEqual(repr(hparams), repr(expected_hparams))

    @parameterized.parameters(
        ("beta", 0.1),
        ("type", "WIDE_AND_DEEP"),
        ("num_layers", 2))
    def test_convert_hyperparams_to_hparams_fixed(self, name, value):
        hps = hp_module.HyperParameters()
        hps.Fixed(name, value)
        hparams = utils.convert_hyperparams_to_hparams(hps)
        expected_hparams = {
            hparams_api.HParam(name, hparams_api.Discrete([value])): value,
        }
        self.assertEqual(repr(hparams), repr(expected_hparams))

    def test_convert_hyperparams_to_hparams_fixed_bool(self):
        hps = hp_module.HyperParameters()
        hps.Fixed("condition", True)
        hparams = utils.convert_hyperparams_to_hparams(hps)
        expected_hparams = {
            hparams_api.HParam("condition", hparams_api.Discrete([True])): True,
        }
        self.assertEqual(repr(hparams), repr(expected_hparams))

    @parameterized.parameters(
        ("val_loss", "min",
         [oracle_module.Objective(name="val_loss", direction="min")]),
        (oracle_module.Objective(name="val_acc", direction="max"), None,
         [oracle_module.Objective(name="val_acc", direction="max")]),
        ("accuracy", None,
         [oracle_module.Objective(name="accuracy", direction="max")]),
        (["val_acc", "val_loss"], None, [
            oracle_module.Objective(name="val_acc", direction="max"),
            oracle_module.Objective(name="val_loss", direction="min"),
        ]))
    def test_format_objective(self, objective, direction, expected_oracle_obj):
        formatted_objective = utils.format_objective(objective, direction)
        self.assertEqual(formatted_objective, expected_oracle_obj)

    @parameterized.parameters(
        ("max", "MAXIMIZE"),
        ("min", "MINIMIZE"),
        ("MAXIMIZE", "max"),
        ("MINIMIZE", "min"))
    def test_format_goal(self, metric_direction, expected_goal):
        goal = utils.format_goal(metric_direction)
        self.assertEqual(goal, expected_goal)

    def test_get_trial_id(self):
        trial_id = utils.get_trial_id(OPTIMIZER_TRIAL)
        self.assertEqual(trial_id, "trial_1")

    def _assert_hps_equal(self, hps1, hps2):
        self.assertEqual(len(hps1.space), len(hps2.space))
        hps1_repr_list = [repr(x) for x in hps1.space]
        hps2_repr_list = [repr(x) for x in hps2.space]
        self.assertCountEqual(hps1_repr_list, hps2_repr_list)

    def _assert_study_config_equal(
        self, test_study_config, expected_study_config
    ):
        study_config = copy.deepcopy(test_study_config)
        expected_config = copy.deepcopy(expected_study_config)

        algo = study_config.pop("algorithm")
        self.assertEqual(algo, "ALGORITHM_UNSPECIFIED")

        stopping_config = study_config.pop("automatedStoppingConfig")
        self.assertDictEqual(stopping_config, {
            "decayCurveStoppingConfig": {
                "useElapsedTime": True
            }
        })

        params = study_config.pop("parameters")
        expected_params = expected_config.pop("parameters")
        self.assertCountEqual(params, expected_params)

        # Check the rest of the study config
        self.assertDictEqual(study_config, expected_config)


if __name__ == "__main__":
    tf.test.main()
