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
"""Tests for utils."""

from kerastuner.engine import hyperparameters as hp_module
from kerastuner.engine import oracle as oracle_module
import tensorflow as tf
from tensorflow_cloud.tuner.tuner import utils

STUDY_CONFIG_DISCRETE = {
    "algorithm": "ALGORITHM_UNSPECIFIED",
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
    "algorithm": "ALGORITHM_UNSPECIFIED",
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
    "algorithm": "ALGORITHM_UNSPECIFIED",
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "integer_value_spec": {"max_value": 4, "min_value": 1},
            "parameter": "units",
            "type": "INTEGER",
        }
    ],
}
STUDY_CONFIG_INT_STEP = {
    "algorithm": "ALGORITHM_UNSPECIFIED",
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "discrete_value_spec": {"values": [32, 64, 96]},
            "parameter": "units",
            "type": "DISCRETE",
        }
    ],
}
STUDY_CONFIG_FLOAT_LOG_SCALE = {
    "algorithm": "ALGORITHM_UNSPECIFIED",
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
STUDY_CONFIG_MULTI_FLOAT = {
    "algorithm": "ALGORITHM_UNSPECIFIED",
    "metrics": [{"goal": "MAXIMIZE", "metric": "accuracy"}],
    "parameters": [
        {
            "double_value_spec": {"max_value": 1.57, "min_value": 0.0},
            "parameter": "theta",
            "type": "DOUBLE",
        },
        {
            "double_value_spec": {"max_value": 1.0, "min_value": 0.0},
            "parameter": "r",
            "type": "DOUBLE",
        },
    ],
}
STUDY_CONFIG_BOOL = {
    "algorithm": "ALGORITHM_UNSPECIFIED",
    "metrics": [{"goal": "MAXIMIZE", "metric": "val_accuracy"}],
    "parameters": [
        {
            "categorical_value_spec": {"values": ["True", "False"],},
            "parameter": "has_beta",
            "type": "CATEGORICAL",
        }
    ],
}
STUDY_CONFIG_FIXED_FLOAT = {
    "algorithm": "ALGORITHM_UNSPECIFIED",
    "metrics": [{"goal": "MAXIMIZE", "metric": "val_accuracy"}],
    "parameters": [
        {
            "discrete_value_spec": {"values": [1.0],},
            "parameter": "beta",
            "type": "DISCRETE",
        }
    ],
}
STUDY_CONFIG_FIXED_CATEGORICAL = {
    "algorithm": "ALGORITHM_UNSPECIFIED",
    "metrics": [{"goal": "MAXIMIZE", "metric": "val_accuracy"}],
    "parameters": [
        {
            "categorical_value_spec": {"values": ["WIDE_AND_DEEP"]},
            "parameter": "type",
            "type": "CATEGORICAL",
        }
    ],
}
EXPECTE_TRIAL_HPS = {
    "learning_rate": 0.0001,
    "num_layers": 2,
    "units_0": 96.0,
    "units_1": 352.0,
}


class CloudTunerUtilsTest(tf.test.TestCase):

    def convert_study_config_discrete(self):
        hps = hp_module.HyperParameters()
        hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
        study_config = utils.make_study_config(
            objective=oracle_module.Objective("val_accuracy", "max"),
            hyperparams=hps
        )
        self.assertEqual(study_config, STUDY_CONFIG_DISCRETE)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self.assertEqual(actual_hps.space, hps.space)

    def convert_study_config_categorical(self):
        hps = hp_module.HyperParameters()
        hps.Choice("model_type", ["LINEAR", "WIDE_AND_DEEP"])
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self.assertEqual(study_config, STUDY_CONFIG_CATEGORICAL)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self.assertEqual(actual_hps.space, hps.space)

    def convert_study_config_int(self):
        hps = hp_module.HyperParameters()
        hps.Int("units", min_value=1, max_value=4)
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self.assertEqual(study_config, STUDY_CONFIG_INT)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self.assertEqual(actual_hps.space, hps.space)

    def convert_study_config_int_step(self):
        hps = hp_module.HyperParameters()
        hps.Int("units", min_value=32, max_value=128, step=32)
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self.assertEqual(study_config, STUDY_CONFIG_INT_STEP)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self.assertEqual(actual_hps.space, hps.space)

    def convert_study_config_float_log_scale(self):
        hps = hp_module.HyperParameters()
        hps.Float("learning_rate", min_value=1e-4, max_value=1e-1,
                  sampling="log")
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self.assertEqual(study_config, STUDY_CONFIG_FLOAT_LOG_SCALE)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self.assertEqual(actual_hps.space, hps.space)

    def convert_study_config_multi_float(self):
        hps = hp_module.HyperParameters()
        hps.Float("theta", min_value=0.0, max_value=1.57)
        hps.Float("r", min_value=0.0, max_value=1.0)
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self.assertEqual(study_config, STUDY_CONFIG_MULTI_FLOAT)

        actual_hps = utils.convert_study_config_to_hps(study_config)
        self.assertEqual(actual_hps.space, hps.space)

    def convert_study_config_bool(self):
        hps = hp_module.HyperParameters()
        hps.Boolean("has_beta")
        study_config = utils.make_study_config(
            objective="accuracy", hyperparams=hps)
        self.assertEqual(study_config, STUDY_CONFIG_BOOL)

    def convert_study_config_fixed(self):
        hps = hp_module.HyperParameters()
        hps.Fixed("beta", 0.1)
        study_config_float = utils.make_study_config(
            objective="accuracy", hyperparams=hps
        )
        self.assertEqual(study_config_float, STUDY_CONFIG_FIXED_FLOAT)

        hps = hp_module.HyperParameters()
        hps.Fixed("type", "WIDE_AND_DEEP")
        study_config_categorical = utils.make_study_config(
            objective="accuracy", hyperparams=hps
        )
        self.assertEqual(study_config_categorical,
                         STUDY_CONFIG_FIXED_CATEGORICAL)

    def test_convert_optimizer_trial_to_hps(self):
        hps = hp_module.HyperParameters()
        hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
        optimizer_trial = {
            "name": "trial_name",
            "state": "ACTIVE",
            "parameters": [
                {"parameter": "learning_rate", "floatValue": 0.0001},
                {"parameter": "num_layers", "intValue": "2"},
                {"parameter": "units_0", "floatValue": 96},
                {"parameter": "units_1", "floatValue": 352},
            ],
        }
        trial_hps = utils.convert_optimizer_trial_to_hps(hps, optimizer_trial)
        self.assertEqual(trial_hps.values, EXPECTE_TRIAL_HPS)

    def test_format_objective(self):
        objective = utils.format_objective(
            oracle_module.Objective(name="val_acc", direction="max")
        )
        self.assertEqual(
            objective,
            [oracle_module.Objective(name="val_acc", direction="max")]
        )

        objective = utils.format_objective("accuracy")
        self.assertEqual(
            objective,
            [oracle_module.Objective(name="accuracy", direction="max")]
        )

        objective = utils.format_objective(["val_acc", "val_loss"])
        self.assertEqual(
            objective,
            [
                oracle_module.Objective(name="val_acc", direction="max"),
                oracle_module.Objective(name="val_loss", direction="min"),
            ],
        )


if __name__ == "__main__":
    tf.test.main()
