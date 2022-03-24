# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for the models experimental module."""

from absl.testing import absltest
import mock
import tensorflow as tf

from tensorflow_cloud.core.experimental import models_entry_point
from official.core import base_task
from official.core import config_definitions
from official.core import train_lib


class ModelsTest(absltest.TestCase):

    def setUp(self):
        super(ModelsTest, self).setUp()
        config = mock.MagicMock(spec=config_definitions.ExperimentConfig)
        task = mock.MagicMock(spec=base_task.Task)
        self.run_experiment_kwargs = dict(task=task,
                                          mode='train_and_eval',
                                          params=config,
                                          model_dir='model_path',
                                          distribution_strategy='one_device')
        self.load_params = mock.patch.object(
            models_entry_point,
            'load_params',
            autospec=True,
            return_value=self.run_experiment_kwargs,
        ).start()

        self.strategy = mock.patch.object(
            tf.distribute,
            'OneDeviceStrategy',
            autospec=True,
            return_value='one_device_strategy',
        ).start()

        self.model = mock.MagicMock()
        self.run_experiment = mock.patch.object(
            train_lib,
            'run_experiment',
            autospec=True,
            return_value=(self.model, {})
        ).start()

    def tearDown(self):
        mock.patch.stopall()
        super(ModelsTest, self).tearDown()

    def test_main(self):
        models_entry_point.main()

        self.load_params.assert_called_with(models_entry_point.PARAMS_FILE_NAME)
        self.run_experiment_kwargs.update(dict(
            distribution_strategy='one_device_strategy'))
        self.run_experiment.assert_called_with(**self.run_experiment_kwargs)
        self.model.save.assert_called_with(
            self.run_experiment_kwargs['model_dir'])

if __name__ == '__main__':
    absltest.main()
