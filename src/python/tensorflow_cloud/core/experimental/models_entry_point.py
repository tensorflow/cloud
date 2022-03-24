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
"""Entry point template file for run_experiment_cloud."""

import pickle

import tensorflow as tf

from official.core import train_lib

# PARAMS_FILE_NAME provides the name of the file that cointains the
# run_experiment_kwargs used to call run_experiment. In models.py, when copying
# this file, the value of PARAMS_FILE_NAME is updated to contain the actual name
# of the file.
PARAMS_FILE_NAME = 'file_name'


def load_params(file_name):
    with open(file_name, 'rb') as f:
        params = pickle.load(f)
    return params


def get_tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)


def get_one_device():
    return tf.distribute.OneDeviceStrategy(device='/gpu:0')

_DISTRIBUTION_STRATEGIES = dict(
        # TODO(b/194857231) Dependency conflict for using TPUs
        tpu=get_tpu_strategy,
        # TODO(b/148619319) Saving model currently failing for multi_mirror
        multi_mirror=tf.distribute.MultiWorkerMirroredStrategy,
        mirror=tf.distribute.MirroredStrategy,
        one_device=get_one_device)


def main():
    run_experiment_kwargs = load_params(PARAMS_FILE_NAME)
    strategy_str = run_experiment_kwargs['distribution_strategy']
    strategy = _DISTRIBUTION_STRATEGIES[strategy_str]()
    run_experiment_kwargs.update(dict(
        distribution_strategy=strategy))
    model, _ = train_lib.run_experiment(**run_experiment_kwargs)
    model.save(run_experiment_kwargs['model_dir'])


if __name__ == '__main__':
    main()
