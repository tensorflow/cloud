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

import google.auth


def get_project_name():
    # https://google-auth.readthedocs.io/en/latest/reference/google.auth.html
    _, project_id = google.auth.default()
    if project_id is None:
        raise Exception('Could not determine the GCP project id.')

    return project_id


def validate_machine_configuration(
        cpu_cores, memory, accelerator_type, accelerator_count):
    valid_configurations = _get_valid_machine_configurations()
    current_config = (
        cpu_cores, memory, accelerator_type.value, accelerator_count)
    if current_config not in valid_configurations:
        raise ValueError(
            'Invalid machine configuration: cpu_cores:{}, memory:{}, '
            'accelerator_type:{}, accelerator_count:{}. Please see the '
            'following AI platform comptibility table for all valid '
            'configurations: '
            'https://cloud.google.com/ml-engine/docs/using-gpus#'
            'compute-engine-machine-types-with-gpu'.format(
                cpu_cores, memory, str(accelerator_type), accelerator_count))


def get_region():
    return 'us-central1'


def get_accelerator_type(accl_type):
    if accl_type == 'CPU':
        return 'ACCELERATOR_TYPE_UNSPECIFIED'
    if accl_type == 'K80':
        return 'NVIDIA_TESLA_K80'
    if accl_type == 'P100':
        return 'NVIDIA_TESLA_P100'
    if accl_type == 'V100':
        return 'NVIDIA_TESLA_V100'
    if accl_type == 'P4':
        return 'NVIDIA_TESLA_P4'
    if accl_type == 'T4':
        return 'NVIDIA_TESLA_T4'
    else:
        raise ValueError('Invalid accelerator type.')


def get_machine_type(cpu_cores, memory):
    config = (cpu_cores, memory)
    if config == (4, 15):
        return 'n1-standard-4'
    if config == (8, 30):
        return 'n1-standard-8'
    if config == (16, 60):
        return 'n1-standard-16'
    if config == (32, 120):
        return 'n1-standard-32'
    if config == (64, 240):
        return 'n1-standard-64'
    if config == (96, 360):
        return 'n1-standard-96'
    if config == (2, 13):
        return 'n1-highmem-2'
    if config == (4, 26):
        return 'n1-highmem-4'
    if config == (8, 52):
        return 'n1-highmem-8'
    if config == (16, 104):
        return 'n1-highmem-16'
    if config == (32, 208):
        return 'n1-highmem-32'
    if config == (64, 416):
        return 'n1-highmem-64'
    if config == (96, 624):
        return 'n1-highmem-96'
    if config == (16, 14.4):
        return 'n1-highcpu-16'
    if config == (32, 28.8):
        return 'n1-highcpu-32'
    if config == (64, 57.6):
        return 'n1-highcpu-64'
    if config == (96, 86.4):
        return 'n1-highcpu-96'
    else:
        raise ValueError('Invalid machine type.')


def _get_valid_machine_configurations():
    valid_configs = []

    # Add CPU configurations
    cpu_memory = [
        (4, 15), (8, 30), (16, 60), (32, 120), (64, 240), (96, 360),
        (2, 13), (4, 26), (8, 52), (16, 104), (32, 208), (64, 416),
        (96, 624), (16, 14.4), (32, 28.8), (64, 57.6), (96, 86.4)]
    for (cpu, memory) in cpu_memory:
        valid_configs.append((cpu, memory, 'CPU', 0))

    # https://cloud.google.com/ml-engine/docs/using-gpus#compute-engine-machine-types-with-gpu
    valid_configs.extend([
        # 'n1-standard-4', 'K80'
        (4, 15, 'K80', 1), (4, 15, 'K80', 2),
        (4, 15, 'K80', 4), (4, 15, 'K80', 8),
        # 'n1-standard-4', 'P4'
        (4, 15, 'P4', 1), (4, 15, 'P4', 2), (4, 15, 'P4', 4),
        # 'n1-standard-4', 'P100'
        (4, 15, 'P100', 1), (4, 15, 'P100', 2), (4, 15, 'P100', 4),
        # 'n1-standard-4', 'T4'
        (4, 15, 'T4', 1), (4, 15, 'T4', 2), (4, 15, 'T4', 4),
        # 'n1-standard-4', 'V100'
        (4, 15, 'V100', 1), (4, 15, 'V100', 2),
        (4, 15, 'V100', 4), (4, 15, 'V100', 8),

        # 'n1-standard-8', 'K80'
        (8, 30, 'K80', 1), (8, 30, 'K80', 2),
        (8, 30, 'K80', 4), (8, 30, 'K80', 8),
        # 'n1-standard-8', 'P4'
        (8, 30, 'P4', 1), (8, 30, 'P4', 2), (8, 30, 'P4', 4),
        # 'n1-standard-8', 'P100'
        (8, 30, 'P100', 1), (8, 30, 'P100', 2), (8, 30, 'P100', 4),
        # 'n1-standard-8', 'T4'
        (8, 30, 'T4', 1), (8, 30, 'T4', 2), (8, 30, 'T4', 4),
        # 'n1-standard-8', 'V100'
        (8, 30, 'V100', 1), (8, 30, 'V100', 2),
        (8, 30, 'V100', 4), (8, 30, 'V100', 8),

        # 'n1-standard-16', 'K80'
        (16, 60, 'K80', 2), (16, 60, 'K80', 4), (16, 60, 'K80', 8),
        # 'n1-standard-16', 'P4'
        (16, 60, 'P4', 1), (16, 60, 'P4', 2), (16, 60, 'P4', 4),
        # 'n1-standard-16', 'P100'
        (16, 60, 'P100', 1), (16, 60, 'P100', 2), (16, 60, 'P100', 4),
        # 'n1-standard-16', 'T4'
        (16, 60, 'T4', 1), (16, 60, 'T4', 2), (16, 60, 'T4', 4),
        # 'n1-standard-16', 'V100'
        (16, 60, 'V100', 2), (16, 60, 'V100', 4), (16, 60, 'V100', 8),

        # 'n1-standard-32', 'K80'
        (32, 120, 'K80', 4), (32, 120, 'K80', 8),
        # 'n1-standard-32', 'P4'
        (32, 120, 'P4', 2), (32, 120, 'P4', 4),
        # 'n1-standard-32', 'P100'
        (32, 120, 'P100', 2), (32, 120, 'P100', 4),
        # 'n1-standard-32', 'T4'
        (32, 120, 'T4', 2), (32, 120, 'T4', 4),
        # 'n1-standard-32', 'V100'
        (32, 120, 'V100', 4), (32, 120, 'V100', 8),

        # 'n1-standard-64', 'P4'
        (64, 240, 'P4', 4),
        # 'n1-standard-64', 'T4'
        (64, 240, 'T4', 4),
        # 'n1-standard-64', 'V100'
        (64, 240, 'V100', 8),

        # 'n1-standard-96', 'P4'
        (96, 360, 'P4', 4),
        # 'n1-standard-96', 'T4'
        (96, 360, 'T4', 4),
        # 'n1-standard-96', 'V100'
        (96, 360, 'V100', 8),

        # 'n1-highmem-2', 'K80'
        (2, 13, 'K80', 1), (2, 13, 'K80', 2),
        (2, 13, 'K80', 4), (2, 13, 'K80', 8),
        # 'n1-highmem-2', 'P4'
        (2, 13, 'P4', 1), (2, 13, 'P4', 2), (2, 13, 'P4', 4),
        # 'n1-highmem-2', 'P100'
        (2, 13, 'P100', 1), (2, 13, 'P100', 2), (2, 13, 'P100', 4),
        # 'n1-highmem-2', 'T4'
        (2, 13, 'T4', 1), (2, 13, 'T4', 2), (2, 13, 'T4', 4),
        # 'n1-highmem-2', 'V100'
        (2, 13, 'V100', 1), (2, 13, 'V100', 2),
        (2, 13, 'V100', 4), (2, 13, 'V100', 8),

        # 'n1-highmem-4', 'K80'
        (4, 26, 'K80', 1), (4, 26, 'K80', 2),
        (4, 26, 'K80', 4), (4, 26, 'K80', 8),
        # 'n1-highmem-4', 'P4'
        (4, 26, 'P4', 1), (4, 26, 'P4', 2), (4, 26, 'P4', 4),
        # 'n1-highmem-4', 'P100'
        (4, 26, 'P100', 1), (4, 26, 'P100', 2), (4, 26, 'P100', 4),
        # 'n1-highmem-4', 'T4'
        (4, 26, 'T4', 1), (4, 26, 'T4', 2), (4, 26, 'T4', 4),
        # 'n1-highmem-4', 'V100'
        (4, 26, 'V100', 1), (4, 26, 'V100', 2),
        (4, 26, 'V100', 4), (4, 26, 'V100', 8),

        # 'n1-highmem-8', 'K80'
        (8, 52, 'K80', 1), (8, 52, 'K80', 2),
        (8, 52, 'K80', 4), (8, 52, 'K80', 8),
        # 'n1-highmem-8', 'P4'
        (8, 52, 'P4', 1), (8, 52, 'P4', 2), (8, 52, 'P4', 4),
        # 'n1-highmem-8', 'P100'
        (8, 52, 'P100', 1), (8, 52, 'P100', 2), (8, 52, 'P100', 4),
        # 'n1-highmem-8', 'T4'
        (8, 52, 'T4', 1), (8, 52, 'T4', 2), (8, 52, 'T4', 4),
        # 'n1-highmem-8', 'V100'
        (8, 52, 'V100', 1), (8, 52, 'V100', 2),
        (8, 52, 'V100', 4), (8, 52, 'V100', 8),

        # 'n1-highmem-16', 'K80'
        (16, 104, 'K80', 2), (16, 104, 'K80', 4), (16, 104, 'K80', 8),
        # 'n1-highmem-16', 'P4'
        (16, 104, 'P4', 1), (16, 104, 'P4', 2), (16, 104, 'P4', 4),
        # 'n1-highmem-16', 'P100'
        (16, 104, 'P100', 1), (16, 104, 'P100', 2), (16, 104, 'P100', 4),
        # 'n1-highmem-16', 'T4'
        (16, 104, 'T4', 1), (16, 104, 'T4', 2), (16, 104, 'T4', 4),
        # 'n1-highmem-16', 'V100'
        (16, 104, 'V100', 2), (16, 104, 'V100', 4), (16, 104, 'V100', 8),

        # 'n1-highmem-32', 'K80'
        (32, 208, 'K80', 4), (32, 208, 'K80', 8),
        # 'n1-highmem-32', 'P4'
        (32, 208, 'P4', 2), (32, 208, 'P4', 4),
        # 'n1-highmem-32', 'P100'
        (32, 208, 'P100', 2), (32, 208, 'P100', 4),
        # 'n1-highmem-32', 'T4'
        (32, 208, 'T4', 2), (32, 208, 'T4', 4),
        # 'n1-highmem-32', 'V100'
        (32, 208, 'V100', 4), (32, 208, 'V100', 8),

        # 'n1-highmem-64', 'P4'
        (64, 416, 'P4', 4),
        # 'n1-highmem-64', 'T4'
        (64, 416, 'T4', 4),
        # 'n1-highmem-64', 'V100'
        (64, 416, 'V100', 8),

        # 'n1-highmem-96', 'P4'
        (96, 624, 'P4', 4),
        # 'n1-highmem-96', 'T4'
        (96, 624, 'T4', 4),
        # 'n1-highmem-96', 'V100'
        (96, 624, 'V100', 8),

        # 'n1-highcpu-16', 'K80'
        (16, 14.4, 'K80', 2), (16, 14.4, 'K80', 4), (16, 14.4, 'K80', 8),
        # 'n1-highcpu-16', 'P4'
        (16, 14.4, 'P4', 1), (16, 14.4, 'P4', 2), (16, 14.4, 'P4', 4),
        # 'n1-highcpu-16', 'P100'
        (16, 14.4, 'P100', 1), (16, 14.4, 'P100', 2), (16, 14.4, 'P100', 4),
        # 'n1-highcpu-16', 'T4'
        (16, 14.4, 'T4', 1), (16, 14.4, 'T4', 2), (16, 14.4, 'T4', 4),
        # 'n1-highcpu-16', 'V100'
        (16, 14.4, 'V100', 2), (16, 14.4, 'V100', 4), (16, 14.4, 'V100', 8),

        # 'n1-highcpu-32', 'K80'
        (32, 28.8, 'K80', 4), (32, 28.8, 'K80', 8),
        # 'n1-highcpu-32', 'P4'
        (32, 28.8, 'P4', 2), (32, 28.8, 'P4', 4),
        # 'n1-highcpu-32', 'P100'
        (32, 28.8, 'P100', 2), (32, 28.8, 'P100', 4),
        # 'n1-highcpu-32', 'T4'
        (32, 28.8, 'T4', 2), (32, 28.8, 'T4', 4),
        # 'n1-highcpu-32', 'V100'
        (32, 28.8, 'V100', 4), (32, 28.8, 'V100', 8),

        # 'n1-highcpu-64', 'K80'
        (64, 57.6, 'K80', 8),
        # 'n1-highcpu-64', 'P4'
        (64, 57.6, 'P4', 4),
        # 'n1-highcpu-64', 'P100'
        (64, 57.6, 'P100', 4),
        # 'n1-highcpu-64', 'T4'
        (64, 57.6, 'T4', 4),
        # 'n1-highcpu-64', 'V100'
        (64, 57.6, 'V100', 8),

        # 'n1-highcpu-96', 'P4'
        (96, 86.4, 'P4', 4),
        # 'n1-highcpu-96', 'T4'
        (96, 86.4, 'T4', 4),
        # 'n1-highcpu-96', 'V100'
        (96, 86.4, 'V100', 8),
    ])
    return valid_configs
