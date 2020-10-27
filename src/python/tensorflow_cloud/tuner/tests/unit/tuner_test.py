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
"""Tests for Cloud Keras Tuner."""

import os
import time
from kerastuner.engine import hypermodel as hypermodel_module
from kerastuner.engine import hyperparameters as hp_module
from kerastuner.engine import oracle as oracle_module
from kerastuner.engine import trial as trial_module
from kerastuner.engine import tuner as super_tuner

import mock

import tensorflow as tf
from tensorflow_cloud.core import deploy
from tensorflow_cloud.core import machine_config
from tensorflow_cloud.core import validate
from tensorflow_cloud.tuner import cloud_fit_client
from tensorflow_cloud.tuner import tuner
from tensorflow_cloud.tuner.tuner import optimizer_client
from tensorflow_cloud.utils import google_api_client
from tensorflow_cloud.utils import tf_utils


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", [0.0001, 0.001, 0.01])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CloudTunerTest(tf.test.TestCase):

    def setUp(self):
        super(CloudTunerTest, self).setUp()
        self.addCleanup(mock.patch.stopall)

        self._study_id = "study-a"
        self._region = "us-central1"
        self._remote_dir = "gs://remote_dir"
        self._project_id = "project-a"
        # TODO(b/170687807) Switch from using "{}".format() to f-string
        self._trial_parent = "projects/{}/locations/{}/studies/{}".format(
            self._project_id, self._region, self._study_id
        )
        self._container_uri = "test_container_uri",
        hps = hp_module.HyperParameters()
        hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
        self._test_hyperparameters = hps

        self._study_config = {
            "algorithm": "ALGORITHM_UNSPECIFIED",
            "metrics": [{"metric": "val_acc", "goal": "MAXIMIZE"}],
            "parameters": [
                {
                    "parameter": "learning_rate",
                    "discrete_value_spec": {"values": [0.0001, 0.001, 0.01]},
                    "type": "DISCRETE",
                }
            ],
            "automatedStoppingConfig": {
                "decayCurveStoppingConfig": {"useElapsedTime": True}
            },
        }

        self._test_trial = trial_module.Trial(
            hyperparameters=self._test_hyperparameters,
            trial_id="1",
            status=trial_module.TrialStatus,
        )
        # TODO(b/170687807) Switch from using "{}".format() to f-string
        self._job_id = "{}_{}".format(self._study_id, self._test_trial.trial_id)
        self.mock_optimizer_client_module = mock.patch.object(
            tuner, "optimizer_client", autospec=True
        ).start()

        self.mock_client = mock.create_autospec(
            optimizer_client._OptimizerClient)
        self.mock_optimizer_client_module.create_or_load_study.return_value = (
            self.mock_client
        )

    def _tuner_with_hparams(self):
        self.tuner = self._tuner(
            objective=oracle_module.Objective("val_acc", "max"),
            hyperparameters=self._test_hyperparameters,
            study_config=None,
        )

    def _tuner(self, objective, hyperparameters, study_config, max_trials=None):
        return tuner.CloudTuner(
            hypermodel=build_model,
            objective=objective,
            study_config=study_config,
            hyperparameters=hyperparameters,
            max_trials=max_trials,
            project_id=self._project_id,
            region=self._region,
            study_id=self._study_id,
            directory=self.get_temp_dir(),
        )

    def _remote_tuner(
                self,
                objective,
                hyperparameters,
                study_config,
                directory="gs://remote_dir",
                max_trials=None
                ):
        return tuner.DistributingCloudTuner(
            hypermodel=build_model,
            objective=objective,
            study_config=study_config,
            hyperparameters=hyperparameters,
            max_trials=max_trials,
            project_id=self._project_id,
            region=self._region,
            directory=directory,
            study_id=self._study_id,
            container_uri=self._container_uri
        )

    def test_tuner_initialization_with_hparams(self):
        self._tuner_with_hparams()
        (self.mock_optimizer_client_module.create_or_load_study
         .assert_called_with(self._project_id,
                             self._region,
                             self._study_id,
                             self._study_config))

    def test_tuner_initialization_with_study_config(self):
        self.tuner = self._tuner(None, None, self._study_config)
        (self.mock_optimizer_client_module.create_or_load_study
         .assert_called_with(self._project_id,
                             self._region,
                             self._study_id,
                             self._study_config))

    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    def test_remote_tuner_initialization_with_study_config(self, mock_super):
        self._remote_tuner(None, None, self._study_config)
        (self.mock_optimizer_client_module.create_or_load_study
         .assert_called_with(self._project_id,
                             self._region,
                             self._study_id,
                             self._study_config))

    def test_tuner_initialization_neither_hparam_nor_study_config(self):
        with self.assertRaises(ValueError):
            _ = self._tuner(None, None, None)

    def test_tuner_initialization_with_hparams_missing_objectives(self):
        with self.assertRaises(ValueError):
            _ = self._tuner(None, self._test_hyperparameters, None)

    def test_tuner_initialization_both_hparam_and_study_config(self):
        with self.assertRaises(ValueError):
            _ = self._tuner(
                oracle_module.Objective("value_acc", "max"),
                self._test_hyperparameters,
                self._study_config,
            )

    def test_tuner_initialization_with_study_config_and_max_trials(self):
        self.tuner = self._tuner(None, None, self._study_config, max_trials=100)
        (self.mock_optimizer_client_module.create_or_load_study
         .assert_called_with(self._project_id,
                             self._region,
                             self._study_id,
                             self._study_config))

    def test_create_trial_initially(self):
        self._tuner_with_hparams()
        self.mock_client.list_trials.return_value = []
        self.mock_client.get_suggestions.return_value = {
            "trials": [
                {
                    "name": "1",
                    "state": "ACTIVE",
                    "parameters":
                        [{"parameter": "learning_rate", "floatValue": 0.001}],
                }
            ]
        }
        trial = self.tuner.oracle.create_trial("tuner_0")
        self.mock_client.list_trials.assert_called_once()
        self.mock_client.get_suggestions.assert_called_with("tuner_0")
        self.assertEqual(trial.hyperparameters.values, {"learning_rate": 0.001})

    def test_create_trial_before_reaching_max_trials(self):
        self.tuner = self._tuner(None, None, self._study_config, max_trials=100)
        self.mock_client.list_trials.return_value = [
            {"name": "a", "state": "ACTIVE"}
        ] * 50
        self.mock_client.get_suggestions.return_value = {
            "trials": [
                {
                    "name": "1",
                    "state": "ACTIVE",
                    "parameters":
                        [{"parameter": "learning_rate", "floatValue": 0.001}],
                }
            ]
        }
        trial = self.tuner.oracle.create_trial("tuner_0")
        self.mock_client.list_trials.assert_called_once()
        self.mock_client.get_suggestions.assert_called_with("tuner_0")
        self.assertEqual(trial.hyperparameters.values, {"learning_rate": 0.001})

    def test_create_trial_reaching_max_trials(self):
        self.tuner = self._tuner(None, None, self._study_config, max_trials=100)
        self.mock_client.list_trials.return_value = [
            {"name": "a", "state": "ACTIVE"}
        ] * 100

        trial = self.tuner.oracle.create_trial("tuner_0")

        self.mock_client.list_trials.assert_called_once()
        self.assertIsNone(trial.hyperparameters.values)
        self.assertEqual(trial.status, trial_module.TrialStatus.STOPPED)

    def test_create_trial_after_early_stopping(self):
        self._tuner_with_hparams()
        self.mock_client.list_trials.return_value = [
            {"name": "a", "state": "STOPPING"}]

        trial = self.tuner.oracle.create_trial("tuner_0")

        self.mock_client.list_trials.assert_called_once()
        self.assertIsNone(trial.hyperparameters.values)
        self.assertEqual(trial.status, trial_module.TrialStatus.STOPPED)

    def test_update_trial(self):
        self._tuner_with_hparams()

        self.mock_client.should_trial_stop.return_value = True

        mock_time = mock.patch.object(time, "time", autospec=True).start()
        mock_time.return_value = 1000

        self.tuner.oracle._start_time = 10
        self.tuner.oracle.trials = {"1": self._test_trial}
        status = self.tuner.oracle.update_trial(
            trial_id="1", metrics={"val_acc": 0.8}, step=3
        )

        (self.mock_client.report_intermediate_objective_value
         .assert_called_once_with(
             3,  # step
             990,  # elapsed_secs
             [{"metric": "val_acc", "value": 0.8}],  # metrics_list
             "1",  # trial_id,
             )
        )
        self.mock_client.should_trial_stop.assert_called_once_with("1")
        self.assertEqual(status, trial_module.TrialStatus.STOPPED)

    def test_end_trial_success(self):
        self._tuner_with_hparams()
        self.mock_client.complete_trial.return_value = {
            "name": "1",
            "state": "COMPLETED",
            "parameters": [{"parameter": "learning_rate", "floatValue": 0.01}],
            "finalMeasurement": {
                "stepCount": 3,
                "metrics": [{"metric": "val_acc", "value": 0.7}],
            },
            "trial_infeasible": False,
            "infeasible_reason": None,
        }

        self.tuner.oracle.ongoing_trials = {"tuner_0": self._test_trial}
        self.tuner.oracle.end_trial(trial_id="1")
        self.mock_client.complete_trial.assert_called_once_with(
            "1", False, None)

    def test_end_trial_infeasible_trial(self):
        self._tuner_with_hparams()
        # Return value from complete_trial is irrelevant to this test case.
        self.mock_client.complete_trial.return_value = {"dummy": "trial"}

        self.tuner.oracle.ongoing_trials = {"tuner_0": self._test_trial}
        self.tuner.oracle.end_trial(trial_id="1", status="INVALID")
        self.mock_client.complete_trial.assert_called_once_with(
            "1", True, "INVALID")

    def test_end_trial_invalid_trial(self):
        self._tuner_with_hparams()
        self.tuner.oracle.ongoing_trials = {"tuner_0": self._test_trial}
        with self.assertRaises(ValueError):
            self.tuner.oracle.end_trial(trial_id="2")

    def test_end_trial_invalid_status(self):
        self._tuner_with_hparams()
        self.tuner.oracle.ongoing_trials = {"tuner_0": self._test_trial}
        with self.assertRaises(ValueError):
            self.tuner.oracle.end_trial(trial_id="1", status="FOO")

    def test_get_trial_success(self):
        self._tuner_with_hparams()
        self.mock_client.get_trial.return_value = {
            "name": "1",
            "state": "COMPLETED",
            "parameters": [{"parameter": "learning_rate", "floatValue": 0.01}],
            "finalMeasurement": {
                "stepCount": 3,
                "metrics": [{"metric": "val_acc", "value": 0.7}],
            },
            "trial_infeasible": False,
            "infeasible_reason": None,
        }
        trial = self.tuner.oracle.get_trial(trial_id="1")
        self.mock_client.get_trial.assert_called_once_with("1")
        self.assertEqual(trial.trial_id, "1")
        self.assertEqual(trial.score, 0.7)
        self.assertEqual(trial.status, trial_module.TrialStatus.COMPLETED)
        self.assertEqual(trial.hyperparameters.values, {"learning_rate": 0.01})

    def test_get_trial_failed(self):
        self._tuner_with_hparams()
        self.mock_client.get_trial.return_value = {
            "name": "1",
            "state": "FOO"
        }
        with self.assertRaises(ValueError):
            self.tuner.oracle.get_trial(trial_id="1")

    def test_get_best_trials(self):
        self._tuner_with_hparams()

        self.mock_client.list_trials.return_value = [
            {
                "name": "1",
                "state": "COMPLETED",
                "parameters":
                    [{"parameter": "learning_rate", "floatValue": 0.01}],
                "finalMeasurement": {
                    "stepCount": 3,
                    "metrics": [{"metric": "val_acc", "value": 0.7}],
                },
                "trial_infeasible": False,
                "infeasible_reason": None,
            },
            {
                "name": "2",
                "state": "COMPLETED",
                "parameters":
                    [{"parameter": "learning_rate", "floatValue": 0.001}],
                "finalMeasurement": {
                    "stepCount": 3,
                    "metrics": [{"metric": "val_acc", "value": 0.9}],
                },
                "trial_infeasible": False,
                "infeasible_reason": None,
            },
        ]
        trials = self.tuner.oracle.get_best_trials(num_trials=2)

        self.mock_client.list_trials.assert_called_once()

        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0].trial_id, "2")
        self.assertEqual(trials[1].trial_id, "1")
        self.assertEqual(trials[0].score, 0.9)
        self.assertEqual(trials[0].best_step, 3)

    def test_get_best_trials_multi_tuners(self):
        # Instantiate tuner_1
        tuner_1 = self._tuner(
            objective=oracle_module.Objective("val_acc", "max"),
            hyperparameters=self._test_hyperparameters,
            study_config=None,
        )
        tuner_1.tuner_id = "tuner_1"
        # tuner_1 has a completed trial
        trial_1 = trial_module.Trial(
            hyperparameters=self._test_hyperparameters,
            trial_id="1",
            status=trial_module.TrialStatus.COMPLETED,
        )
        tuner_1.oracle.trials = {"1": trial_1}

        # Instantiate tuner_2
        tuner_2 = self._tuner(
            objective=oracle_module.Objective("val_acc", "max"),
            hyperparameters=self._test_hyperparameters,
            study_config=None,
        )
        tuner_2.tuner_id = "tuner_2"
        # tuner_2 has a completed trial
        trial_2 = trial_module.Trial(
            hyperparameters=self._test_hyperparameters,
            trial_id="2",
            status=trial_module.TrialStatus.COMPLETED,
        )
        tuner_2.oracle.trials = {"2": trial_2}

        self.mock_client.list_trials.return_value = [
            {
                "name": "1",
                "state": "COMPLETED",
                "parameters":
                    [{"parameter": "learning_rate", "floatValue": 0.01}],
                "finalMeasurement": {
                    "stepCount": 3,
                    "metrics": [{"metric": "val_acc", "value": 0.7}],
                },
                "trial_infeasible": False,
                "infeasible_reason": None,
            },
            {
                "name": "2",
                "state": "COMPLETED",
                "parameters":
                    [{"parameter": "learning_rate", "floatValue": 0.001}],
                "finalMeasurement": {
                    "stepCount": 3,
                    "metrics": [{"metric": "val_acc", "value": 0.9}],
                },
                "trial_infeasible": False,
                "infeasible_reason": None,
            },
        ]

        # For any tuner worker who tries to get the best trials, all the top N
        # sorted trials will be returned.
        best_trials_1 = tuner_1.oracle.get_best_trials(num_trials=2)
        self.mock_client.list_trials.assert_called_once()

        best_trials_2 = tuner_2.oracle.get_best_trials(num_trials=2)

        self.assertEqual(len(best_trials_1), 2)
        self.assertEqual(best_trials_1[0].trial_id, best_trials_2[0].trial_id)
        self.assertEqual(best_trials_1[1].trial_id, best_trials_2[1].trial_id)
        self.assertEqual(best_trials_1[0].score, 0.9)
        self.assertEqual(best_trials_1[0].best_step, 3)

    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    def test_add_tensorboard_callback(self, mock_super_tuner):
        remote_tuner = self._remote_tuner(None, None, self._study_config)

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir="user_defined_path_1"),
            tf.keras.callbacks.TensorBoard(log_dir="user_defined_path_2")]

        trial_id = "test_trial_id"
        remote_tuner._add_tensorboard_callback(callbacks, trial_id)
        self.assertLen(callbacks, 1)
        self.assertEqual(
            callbacks[0].log_dir,
            os.path.join(remote_tuner.directory, trial_id, "logs"))

    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    def test_add_model_checkpoint_callback(self, mock_super_tuner):
        remote_tuner = self._remote_tuner(None, None, self._study_config)
        callbacks = []
        trial_id = "test_trial_id"
        remote_tuner._add_model_checkpoint_callback(callbacks, trial_id)
        self.assertLen(callbacks, 1)
        self.assertEqual(
            callbacks[0].filepath,
            os.path.join(remote_tuner.directory, trial_id, "checkpoint"))

    @mock.patch.object(cloud_fit_client, "cloud_fit", auto_spec=True)
    @mock.patch.object(google_api_client,
                       "wait_for_api_training_job_completion", auto_spec=True)
    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    @mock.patch.object(google_api_client, "is_api_training_job_running",
                       auto_spec=True)
    @mock.patch.object(tf_utils, "get_tensorboard_log_watcher_from_path",
                       auto_spec=True)
    def test_remote_run_trial_with_successful_job(
        self, mock_log_watcher, mock_is_running, mock_super_tuner,
        mock_job_status, mock_cloud_fit):
        remote_tuner = self._remote_tuner(
            None, None, self._study_config, max_trials=10)

        mock_is_running.side_effect = [True, False]

        remote_dir = os.path.join(
            remote_tuner.directory, str(self._test_trial.trial_id))
        mock_job_status.return_value = True
        remote_tuner._get_remote_training_metrics = mock.Mock()
        remote_tuner._get_remote_training_metrics.return_value = (
            tuner._TrainingMetrics([{"loss": 0.001}], {}))
        remote_tuner.oracle = mock.Mock()
        remote_tuner.oracle.update_trial = mock.Mock()
        remote_tuner.hypermodel = mock.Mock()
        remote_tuner.run_trial(
            self._test_trial, "fit_arg",
            callbacks=["test_call_back"], fit_kwarg=1)

        self.assertEqual(2, remote_tuner.oracle.update_trial.call_count)
        mock_cloud_fit.assert_called_with(
            "fit_arg",
            fit_kwarg=1,
            model=mock.ANY,
            callbacks=["test_call_back", mock.ANY, mock.ANY],
            remote_dir=remote_dir,
            job_spec=mock.ANY,
            region=self._region,
            project_id=self._project_id,
            image_uri=self._container_uri,
            job_id=self._job_id)

        log_path = remote_tuner._get_tensorboard_log_dir(
            self._test_trial.trial_id)
        mock_log_watcher.assert_called_with(log_path)
        self.assertEqual(
            2, remote_tuner._get_remote_training_metrics.call_count)

    @mock.patch.object(cloud_fit_client, "cloud_fit", auto_spec=True)
    @mock.patch.object(google_api_client,
                       "wait_for_api_training_job_completion", auto_spec=True)
    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    @mock.patch.object(google_api_client, "is_api_training_job_running",
                       auto_spec=True)
    def test_remote_run_trial_with_failed_job(
        self, mock_is_running, mock_super_tuner,
        mock_job_status, mock_cloud_fit):

        remote_tuner = self._remote_tuner(
            None, None, self._study_config, max_trials=10)

        mock_is_running.return_value = False

        remote_tuner.hypermodel = mock.Mock()
        mock_job_status.return_value = False
        with self.assertRaises(RuntimeError):
            remote_tuner.run_trial(
                self._test_trial, "fit_arg",
                callbacks=["test_call_back"], fit_kwarg=1)

    @mock.patch.object(google_api_client, "stop_aip_training_job",
                       auto_spec=True)
    @mock.patch.object(cloud_fit_client, "cloud_fit", auto_spec=True)
    @mock.patch.object(google_api_client,
                       "wait_for_api_training_job_completion", auto_spec=True)
    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    @mock.patch.object(google_api_client, "is_api_training_job_running",
                       auto_spec=True)
    def test_remote_run_trial_with_oracle_canceling_job(
        self, mock_is_running, mock_super_tuner,
        mock_job_status, mock_cloud_fit, mock_stop_job):

        remote_tuner = self._remote_tuner(
            None, None, self._study_config, max_trials=10)

        mock_is_running.side_effect = [True, False]
        mock_job_status.return_value = True
        remote_tuner._get_remote_training_metrics = mock.Mock()
        remote_tuner._get_remote_training_metrics.return_value = (
            tuner._TrainingMetrics([{"loss": 0.001}], {}))
        remote_tuner.oracle = mock.create_autospec(
            oracle_module.Oracle, instance=True, spec_set=True)
        remote_tuner.oracle.update_trial = mock.Mock()
        remote_tuner.oracle.update_trial.return_value = "STOPPED"
        remote_tuner.hypermodel = mock.create_autospec(
            hypermodel_module.HyperModel, instance=True, spec_set=True)
        remote_tuner.run_trial(
            self._test_trial, "fit_arg",
            callbacks=["test_call_back"], fit_kwarg=1)

        self.assertEqual(2, remote_tuner.oracle.update_trial.call_count)
        self.assertEqual(
            2, remote_tuner._get_remote_training_metrics.call_count)
        mock_stop_job.assert_called_once_with(self._job_id, self._project_id)

    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    def test_get_remote_training_metrics(self, mock_super_tuner):
        remote_tuner = self._remote_tuner(
            None, None, self._study_config, max_trials=10)

        remote_tuner.directory = self.get_temp_dir()
        log_dir = os.path.join(
            remote_tuner.directory, str(self._test_trial.trial_id), "logs")

        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar(name="epoch_loss", data=0.1, step=0)
            tf.summary.scalar(name="epoch_accuracy", data=0.2, step=0)
            tf.summary.scalar(name="epoch_loss", data=0.3, step=1)
            tf.summary.scalar(name="epoch_accuracy", data=0.4, step=1)
            tf.summary.scalar(name="epoch_loss", data=0.5, step=2)
            tf.summary.scalar(name="epoch_accuracy", data=0.6, step=2)

        log_reader = tf_utils.get_tensorboard_log_watcher_from_path(log_dir)
        results = remote_tuner._get_remote_training_metrics(log_reader, {})

        self.assertLen(results.completed_epoch_metrics, 3)
        self.assertIn("accuracy", results.completed_epoch_metrics[0])
        self.assertIn("loss", results.completed_epoch_metrics[0])
        self.assertEqual(
            results.completed_epoch_metrics[0].get("loss"), tf.constant(0.1))

    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    def test_remote_load_model(self, mock_super_tuner):
        remote_tuner = self._remote_tuner(
            None, None, self._study_config, max_trials=10)
        with self.assertRaises(NotImplementedError):
            remote_tuner.load_model(self._test_trial)

    @mock.patch.object(super_tuner.Tuner, "save_model", auto_spec=True)
    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    def test_remote_save_model(self, mock_super_tuner, mock_super_save_model):
        remote_tuner = self._remote_tuner(
            None, None, self._study_config, max_trials=10)
        remote_tuner.save_model(self._test_trial.trial_id, mock.Mock(), step=0)
        mock_super_save_model.assert_not_called()

    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    def test_init_with_non_gcs_directory_path(self, mock_super_tuner):
        with self.assertRaisesRegex(
            ValueError, "Directory must be a valid Google Cloud Storage path."):
            self._remote_tuner(
                None, None, self._study_config, max_trials=10,
                directory="local_path")

    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    @mock.patch.object(deploy, "_create_request_dict", auto_spec=True)
    @mock.patch.object(validate, "_validate_cluster_config", auto_spec=True)
    def test_get_job_spec_with_default_config(
        self, mock_validate, mock_create_request, mock_super_tuner):
        remote_tuner = self._remote_tuner(
            None, None, self._study_config)

        # Expected worker configuration based on replica setting
        worker_count = 0
        worker_config = None

        remote_tuner._get_job_spec_from_config(self._job_id)

        mock_validate.assert_called_with(
            chief_config=remote_tuner._replica_config,
            worker_count=worker_count,
            worker_config=worker_config,
            docker_parent_image=remote_tuner._container_uri)

        mock_create_request.assert_called_with(
            job_id=self._job_id,
            region=remote_tuner._region,
            image_uri=remote_tuner._container_uri,
            chief_config=remote_tuner._replica_config,
            worker_count=worker_count,
            worker_config=worker_config,
            entry_point_args=None,
            job_labels=None)

    @mock.patch.object(super_tuner.Tuner, "__init__", auto_spec=True)
    @mock.patch.object(deploy, "_create_request_dict", auto_spec=True)
    @mock.patch.object(validate, "_validate_cluster_config", auto_spec=True)
    def test_get_job_spec_with_default_with_custom_config(
        self, mock_validate, mock_create_request, mock_super_tuner):

        remote_tuner = self._remote_tuner(None, None, self._study_config)

        replica_config = machine_config.COMMON_MACHINE_CONFIGS["K80_1X"]
        replica_count = 2

        remote_tuner._replica_config = replica_config
        remote_tuner._replica_count = replica_count

        # Expected worker configuration based on replica setting
        worker_count = 1
        worker_config = replica_config

        remote_tuner._get_job_spec_from_config(self._job_id)

        mock_validate.assert_called_with(
            chief_config=replica_config,
            worker_count=worker_count,
            worker_config=worker_config,
            docker_parent_image=remote_tuner._container_uri)

        mock_create_request.assert_called_with(
            job_id=self._job_id,
            region=remote_tuner._region,
            image_uri=remote_tuner._container_uri,
            chief_config=replica_config,
            worker_count=worker_count,
            worker_config=replica_config,
            entry_point_args=None,
            job_labels=None)

if __name__ == "__main__":
    tf.test.main()
