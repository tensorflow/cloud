# Lint as: python3
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
from tensorflow_cloud.core import run
from tensorflow_cloud.core.experimental import models
from official.vision.image_classification.efficientnet import efficientnet_model


class ModelsTest(absltest.TestCase):

    def setup_get_model(self):
        self.batch_size = 64
        self.num_classes = 100

    def setup_normalize_img_and_label(self):
        self.small_img = tf.random.uniform(
            shape=[100, 50, 3], maxval=255, dtype=tf.dtypes.int32)
        self.big_img = tf.random.uniform(
            shape=[300, 500, 3], maxval=255, dtype=tf.dtypes.int32)
        self.image_size = 224
        self.width_ratio = 2
        self.expected_img_shape = [self.image_size,
                                   self.image_size * self.width_ratio,
                                   3]
        self.label = tf.convert_to_tensor(4)

    def setup_run_models(self, run_return_value=None, remote=True):
        self.run = mock.patch.object(
            run,
            'run',
            autospec=True,
            return_value=run_return_value,
        ).start()

        self.remote = mock.patch.object(
            run,
            'remote',
            autospec=True,
            return_value=remote,
        ).start()

        self.classifier_trainer = mock.patch.object(
            models,
            'classifier_trainer',
            autospec=True,
        ).start()

    def cleanup_run_models(self):
        mock.patch.stopall()

    def test_get_model_resnet(self):
        self.setup_get_model()
        resnet = models.get_model('resnet', self.batch_size, self.num_classes)
        self.assertEqual('resnet50', resnet._name)
        self.assertEqual(self.batch_size, resnet.outputs[0].shape[0])
        self.assertEqual(self.num_classes, resnet.outputs[0].shape[1])

    def test_get_model_efficientnet_default(self):
        self.setup_get_model()
        efficientnet = models.get_model('efficientnet', self.batch_size,
                                        self.num_classes)
        default_config = efficientnet_model.MODEL_CONFIGS['efficientnet-b0']
        default_config.num_classes = self.num_classes
        self.assertEqual(efficientnet.config, default_config)

    def test_get_model_efficientnet(self):
        self.setup_get_model()
        efficientnet_version = 'efficientnet-b2'
        efficientnet = models.get_model(efficientnet_version, self.batch_size,
                                        self.num_classes)
        default_config = efficientnet_model.MODEL_CONFIGS[efficientnet_version]
        default_config.num_classes = self.num_classes
        self.assertEqual(efficientnet.config, default_config)

    def test_get_model_error(self):
        self.setup_get_model()
        self.assertRaises(TypeError, models.get_model, 'not_a_model',
                          self.batch_size, self.num_classes)

    def test_normalize_image_and_label_without_one_hot(self):
        self.setup_normalize_img_and_label()
        expected_label = self.label
        result_img, result_label = models.normalize_img_and_label(
            self.small_img, self.label, self.image_size, self.width_ratio)
        self.assertEqual(result_img.shape, self.expected_img_shape)
        self.assertEqual(result_label, expected_label)

    def test_normalize_image_and_label_with_one_hot(self):
        self.setup_normalize_img_and_label()
        num_classes = 10
        expected_label = tf.convert_to_tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                              dtype=tf.dtypes.int8)
        result_img, result_label = models.normalize_img_and_label(
            self.big_img, self.label, self.image_size, self.width_ratio,
            num_classes, True)
        self.assertEqual(result_img.shape, self.expected_img_shape)
        self.assertTrue((result_label == expected_label).numpy().all())

    def test_run_models_locally(self):
        run_return = {'job_id': 'job_id',
                      'docker_image': 'docker_image'}

        self.setup_run_models(run_return, remote=False)
        run_kwargs = {'entry_point': 'entry_point',
                      'requirements_txt': 'requirements_txt',
                      'worker_count': 5,}
        result = models.run_models('dataset_name', 'model_name', 'gcs_bucket',
                                   'train', **run_kwargs)
        self.remote.assert_called()
        self.run.assert_called_with(**run_kwargs)
        self.classifier_trainer.assert_not_called()
        return_keys = ['job_id', 'docker_image', 'tensorboard_logs',
                       'model_checkpoint', 'save_model']
        self.assertListEqual(list(result.keys()), return_keys)

        self.cleanup_run_models()

    def test_run_models_remote(self):
        self.setup_run_models()
        result = models.run_models('dataset_name', 'model_name', 'gcs_bucket',
                                   'train')
        self.remote.assert_called()
        self.run.assert_not_called()
        self.classifier_trainer.assert_called()

        self.assertIsNone(result)

        self.cleanup_run_models()

if __name__ == '__main__':
  absltest.main()
