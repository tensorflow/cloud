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
"""Module that makes the `entry_point` distribution ready."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from .machine_config import AcceleratorType


def get_wrapped_entry_point(entry_point,
                            chief_config,
                            worker_count):
    """Makes the `entry_point` distribution ready.

    Use this utility only when `distribution_strategy` input to the `run` API
    is `auto`. Otherwise, `entry_point` can be directly used as the docker
    entry point python program.

    This utility creates a new script called `wrapped_entry_point` that has the
    user given `entry_point` code wrapped in a Tensorflow distribution
    strategy. This will now become the new docker entry point python program.

    The distribution strategy instance created is based on the machine
    configurations provided using the `chief_config`, `worker_count` params.
    - If the number of workers > 0, we will create a default instance of
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
    - If number of GPUs > 0, we will create a default instance of
        `tf.distribute.MirroredStrategy`
    - Otherwise, we will use `tf.distribute.OneDeviceStrategy`

    Args:
        entry_point: String. Python file path to the file that contains the
            TensorFlow code.
        chief_config: `MachineConfig` that represents the configuration
            for the chief worker in a distribution cluster.
        worker_config: `MachineConfig` that represents the configuration
            for the workers in a distribution cluster.

    Returns:
        The `wrapped_entry_point` file path.
    """
    _, entry_point_file_name = os.path.split(entry_point)

    if worker_count > 0:
        strategy = (
            'strategy = tf.distribute.experimental.'
            'MultiWorkerMirroredStrategy()\n')
    elif chief_config.accelerator_count > 1:
        strategy = 'strategy = tf.distribute.MirroredStrategy()\n'
    else:
        strategy = (
            'strategy = tf.distribute.OneDeviceStrategy('
            'device="/gpu:0")\n')
    script_lines = [
        'import tensorflow as tf\n',
        strategy,
        'tf.distribute.experimental_set_strategy(strategy)\n',
        # Add user's code.
        # We are using exec here to execute the user code object.
        # This will support use case where the user's program has a
        # main method.
        'exec(open("{}").read())\n'.format(entry_point_file_name)
    ]

    # Create a tmp wrapped entry point script file.
    _, output_file = tempfile.mkstemp(suffix='.py')
    with open(output_file, 'w') as f:
        f.writelines(script_lines)
    return output_file
