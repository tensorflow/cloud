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
"""Module that contains the `run_models` wrapper for training models from TF Model Garden."""

import os
import pickle
from typing import Any, Dict, Optional
import uuid

from .. import machine_config
from .. import run
import tensorflow as tf
import tensorflow_datasets as tfds

from official.legacy.image_classification.efficientnet import efficientnet_model
from official.legacy.image_classification.resnet import resnet_model

# pylint: disable=g-import-not-at-top
try:
  import importlib.resources as pkg_resources
except ImportError:
  # Backported for python<3.7
  import importlib_resources as pkg_resources
# pylint: enable=g-import-not-at-top

_PARAMS_FILE_NAME_FORMAT = '{}_params'
_ENTRY_POINT_FORMAT = '{}.py'
_ENTRY_POINT_TEMPLATE = 'models_entry_point.py'


def run_models(dataset_name: str,
               model_name: str,
               gcs_bucket: str,
               train_split: str,
               validation_split: str,
               one_hot: Optional[bool] = True,
               epochs: Optional[int] = 100,
               batch_size: Optional[int] = 128,
               job_name: Optional[str] = '',
               **run_kwargs) -> Optional[Dict[str, str]]:
  """A wrapper for tfc.run that runs models from TF Model Garden on the Cloud.

  This method allows for running models from TF Model Garden using datasets
  from TFDS directly on the Cloud. Currently it only supports image
  classification models. Specifically ResNet and EfficientNet.

  Args:
    dataset_name: the registered name of the `DatasetBuilder` (the snake case
      version of the class name). This can be either `'dataset_name'` or
      `'dataset_name/config_name'` for datasets with `BuilderConfig`s.
    model_name: the name of the model. Currently it supports:
      -'resnet': For the resnet50 model.
      -'efficientnet': For the efficientnet-b0 model. -Any efficientnet
        configuration present in efficientnet_model.MODEL_CONFIGS. Use the key
        as it appears in the dictionary.
    gcs_bucket: The gcs bucket that is going to be used to build and store the
      training.
    train_split: Which split of the data to load for training. Available options
      depend on the dataset and can be found on the TFDS docs.
    validation_split: Which split of the data to load for validation. Available
      options depend on the dataset and can be found on the TFDS docs. If None
      is provided, then 0.2 of the training split will be used for validation.
    one_hot: If True it performs one hot encoding on the label, and it assumes
      the label is the index.
    epochs: The number of epochs that are going to be used for training.
    batch_size: The batch size to use during training.
    job_name: The name of the job in GCP.
    **run_kwargs: keyword arguments for `tfc.run()`.

  Returns:
    A dictionary with five keys.
      1. 'job_id': the training job id.
      2. 'docker_image': Docker image generated for the training job.
      3. 'tensorboard_logs': the path to the tensorboard logs registered
      during callbacks.
      4. 'model_checkpoint': the path to the model checkpoints registered
      during callbacks.
      5. 'saved_model': the path to the saved model.
  """
  model_dirs = get_model_dirs(gcs_bucket, job_name)

  if run.remote():
    classifier_trainer(dataset_name, model_name, batch_size, epochs,
                       train_split, validation_split, one_hot, model_dirs)
    return

  validate_input()
  if 'job_labels' in run_kwargs and job_name:
    run_kwargs['job_labels']['job'] = job_name
  elif job_name:
    run_kwargs['job_labels'] = {'job': job_name}

  run_results = run.run(**run_kwargs)
  run_results.update(model_dirs)
  return run_results


def get_model_dirs(gcs_bucket, job_name):
  gcs_base_path = f'gs://{gcs_bucket}/{job_name}'
  return {
      'tensorboard_logs': os.path.join(gcs_base_path, 'logs'),
      'model_checkpoint': os.path.join(gcs_base_path, 'checkpoints'),
      'saved_model': os.path.join(gcs_base_path, 'saved_model')
  }


# TODO(uribejuan): Write function to make sure the input is valid
def validate_input():
  pass


def classifier_trainer(dataset_name, model_name, batch_size, epochs,
                       train_split, validation_split, one_hot, model_dirs):
  """Training loop for image classifier from TF model Garden using TFDS."""
  builder = tfds.builder(dataset_name)

  num_classes = builder.info.features['label'].num_classes
  model = get_model(model_name, batch_size, num_classes)

  if model_name == 'resnet':
    image_size = 224
    width_ratio = 1
  else:  # Assumes model_name is an efficientnet version
    image_size = model.config.resolution
    width_ratio = model.config.width_coefficient

  train_ds, validation_ds = load_data_from_builder(builder, train_split,
                                                   validation_split, image_size,
                                                   width_ratio, batch_size,
                                                   one_hot, num_classes)
  callbacks = [
      tf.keras.callbacks.TensorBoard(log_dir=model_dirs['tensorboard_logs']),
      tf.keras.callbacks.ModelCheckpoint(
          model_dirs['model_checkpoint'], save_best_only=True),
      tf.keras.callbacks.EarlyStopping(
          monitor='loss', min_delta=0.001, patience=3)
  ]

  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=[tf.keras.metrics.CategoricalAccuracy(dtype=tf.float32)],
  )

  model.fit(
      train_ds,
      validation_data=validation_ds,
      epochs=epochs,
      callbacks=callbacks)

  model.save(model_dirs['saved_model'])


def load_data_from_builder(builder, train_split, validation_split, image_size,
                           width_ratio, batch_size, one_hot, num_classes):
  """Loads the train and validation dataset from a dataset builder."""
  builder.download_and_prepare()

  num_examples = builder.info.splits[train_split].num_examples
  train_ds = builder.as_dataset(
      train_split, shuffle_files=True, as_supervised=True)
  train_ds = data_pipeline(train_ds, image_size, width_ratio, batch_size,
                           num_classes, one_hot, num_examples)

  if validation_split is not None:
    validation_ds = builder.as_dataset(
        validation_split, shuffle_files=True, as_supervised=True)
    validation_ds = data_pipeline(validation_ds, image_size, width_ratio,
                                  batch_size, num_classes, one_hot,
                                  num_examples)
  else:
    validation_ds = None

  return train_ds, validation_ds


def get_model(model_name, batch_size, num_classes):
  """Gets model_name from TF Model Garden."""
  if model_name == 'resnet':
    return load_resnet(batch_size, num_classes)
  elif model_name == 'efficientnet':
    return load_efficientnet(num_classes)
  elif model_name in efficientnet_model.MODEL_CONFIGS:
    return load_efficientnet(num_classes, model_name)
  raise TypeError(f'Unknown model_name argument: {model_name}')


def load_resnet(batch_size, num_classes):
  """Loads the ResNet model from TF Model Garden."""
  return resnet_model.resnet50(batch_size=batch_size, num_classes=num_classes)


def load_efficientnet(num_classes, model_name='efficientnet-b0'):
  """Loads the EfficientNet from TF Model Garden."""
  overrides = {
      'num_classes': num_classes,
  }
  return efficientnet_model.EfficientNet.from_name(
      model_name, overrides=overrides)


def normalize_img_and_label(image, label, image_size, width_ratio=1,
                            num_classes=None, one_hot=False):
  """Normalizes the image and label according to the params specified."""
  if one_hot:
    label = tf.one_hot(label, num_classes, dtype=tf.dtypes.int8)
  image = tf.image.resize_with_pad(image, image_size,
                                   int(image_size * width_ratio))
  return image, label


def data_pipeline(original_ds, image_size, width_ratio, batch_size, num_classes,
                  one_hot, num_examples):
  """Pipeline for pre-processing the data."""
  norm_args = {
      'image_size': image_size,
      'width_ratio': width_ratio,
      'num_classes': num_classes,
      'one_hot': one_hot
  }
  ds = original_ds.map(
      lambda image, label: normalize_img_and_label(image, label, **norm_args),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  ds = ds.shuffle(min(num_examples, 1000))
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds


def run_experiment_cloud(run_experiment_kwargs: Dict[str, Any],
                         run_kwargs: Optional[Dict[str, Any]] = None,
                         ) -> Optional[Dict[str, str]]:
  """A wrapper for run API and tf-models-official run_experiment.

  This method takes a dictionary of the parameters for run and a dictionary
  of the parameters for run_experiment to run the experiment directly on GCP.

  Args:
    run_experiment_kwargs: keyword arguments for `train_lib.run_experiment`. The
      docs can be found at
    https://github.com/tensorflow/models/blob/master/official/core/train_lib.py
      The distribution_strategy param is ignored because the distribution
      strategy is selected based on run_kwargs.
    run_kwargs: keyword arguments for `tfc.run`. The docs can be found at
    https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/run.py
      The params entry_point and distribution_strategy are ignored.

  Returns:
    A dictionary with two keys.
      1. 'job_id': the training job id.
      2. 'docker_image': Docker image generated for the training job.
  """
  if run_kwargs is None:
    run_kwargs = dict()
  distribution_strategy = get_distribution_strategy_str(run_kwargs)
  run_experiment_kwargs.update(
      dict(distribution_strategy=distribution_strategy))
  file_id = str(uuid.uuid4())
  params_file = save_params(run_experiment_kwargs, file_id)
  entry_point = copy_entry_point(file_id, params_file)

  run_kwargs.update(dict(entry_point=entry_point, distribution_strategy=None))
  info = run.run(**run_kwargs)
  os.remove(entry_point)
  os.remove(params_file)
  return info


def copy_entry_point(file_id, params_file):
  """Copy models_entry_point and add params file name."""
  lines = get_original_lines()
  entry_point = _ENTRY_POINT_FORMAT.format(file_id)
  with open(entry_point, 'w') as entry_file:
    for line in lines:
      if line.startswith('PARAMS_FILE_NAME = '):
        entry_file.write(f"PARAMS_FILE_NAME = '{params_file}'\n")
      else:
        entry_file.write(line)
  return entry_point


def get_original_lines():
  """Gets the file lines of models_entry_point.py as a list of strings."""
  with pkg_resources.files(__package__).joinpath(
      _ENTRY_POINT_TEMPLATE
  ).open('rt') as file:
    lines = file.readlines()
  return lines


def get_distribution_strategy_str(run_kwargs):
  """Gets the name of a distribution strategy based on cloud run config."""
  if ('worker_count' in run_kwargs and run_kwargs['worker_count'] > 0):
    if ('worker_config' in run_kwargs and
        machine_config.is_tpu_config(run_kwargs['worker_config'])):
      return 'tpu'
    else:
      return 'multi_mirror'
  elif ('chief_config' in run_kwargs and
        run_kwargs['chief_config'].accelerator_count > 1):
    return 'mirror'
  else:
    return 'one_device'


def save_params(params, file_id):
  """Pickles the params object using the file_id as prefix."""
  file_name = _PARAMS_FILE_NAME_FORMAT.format(file_id)
  with open(file_name, 'xb') as f:
    pickle.dump(params, f)
  return file_name
