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

import os
import tempfile

from .machine_config import AcceleratorType


def get_startup_script(entry_point,
                       chief_config,
                       worker_count,
                       distribution_strategy):
    """Returns the startup script for the cloud."""
    script_lines = [
        'import tensorflow as tf\n',
    ]
    if (chief_config.accelerator_type != AcceleratorType.NO_ACCELERATOR and
            distribution_strategy == 'auto'):
        # If distribution_strategy is set as 'auto' get the strategy to
        # be used based on the machine configurations provided.
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
        script_lines.extend([
            strategy,
            'tf.distribute.experimental_set_strategy(strategy)\n',
        ])

    # Add user's code
    _, entry_point_file_name = os.path.split(entry_point)
    script_lines.append('exec(open("{}").read())\n'.format(
        entry_point_file_name))

    # Create a tmp startup script file.
    _, output_file = tempfile.mkstemp(suffix='.py')
    with open(output_file, 'w') as f:
        f.writelines(script_lines)
    return output_file
