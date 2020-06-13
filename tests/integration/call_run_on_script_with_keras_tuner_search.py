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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_cloud as tfc

# Automated CPU strategy: CPU chief config and no workers
# tfc.run(
#     entry_point='tests/testdata/keras_tuner_mnist_example.py',
#     distribution_strategy='auto',
#     requirements_txt='tests/testdata/requirements.txt',
#     chief_config=tfc.COMMON_MACHINE_CONFIGS['CPU'],
#     stream_logs=True)


# Automated OneDeviceStrategy: default 1 GPU chief config and no workers
# tfc.run(
#     entry_point='tests/testdata/keras_tuner_mnist_example.py',
#     distribution_strategy='auto',
#     requirements_txt='tests/testdata/requirements.txt',
#     stream_logs=True)

# Automated MirroredStrategy: chief config with multiple GPUs
tfc.run(
    entry_point="tests/testdata/keras_tuner_mnist_example.py",
    distribution_strategy="auto",
    requirements_txt="tests/testdata/requirements.txt",
    chief_config=tfc.COMMON_MACHINE_CONFIGS["V100_4X"],
    worker_count=0,
    stream_logs=True,
)
