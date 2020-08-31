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
"""Machine configuration annotations used by the `run` API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

from . import gcp


class AcceleratorType(enum.Enum):
    """Types of accelerators."""

    NO_ACCELERATOR = "CPU"
    NVIDIA_TESLA_K80 = "K80"
    NVIDIA_TESLA_P100 = "P100"
    NVIDIA_TESLA_V100 = "V100"
    NVIDIA_TESLA_P4 = "P4"
    NVIDIA_TESLA_T4 = "T4"
    TPU_V2 = "TPU_V2"
    TPU_V3 = "TPU_V3"
    # NOTE: When making changes here, please make sure to update the list of
    # supported `accelerator_type`s in `MachineConfig`.

    @classmethod
    def all(cls):
        return (
            cls.NO_ACCELERATOR,
            cls.NVIDIA_TESLA_K80,
            cls.NVIDIA_TESLA_P100,
            cls.NVIDIA_TESLA_V100,
            cls.NVIDIA_TESLA_P4,
            cls.NVIDIA_TESLA_T4,
            cls.TPU_V2,
            cls.TPU_V3,
        )

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError("Invalid accelerator key provided: %s." % key)


class MachineConfig(object):
    """Represents the configuration or type of machine to be used."""

    def __init__(self,
                 cpu_cores=8,
                 memory=30,
                 accelerator_type="auto",
                 accelerator_count=1):
        """Constructor.

        Args:
          cpu_cores: Number of virtual CPU cores. Defaults to 8.
          memory: Amount of memory in GB. Defaults to 30GB.
          accelerator_type: Type of the accelerator to be used
            ('K80', 'P100', 'V100', 'P4', 'T4', 'TPU_V2', 'TPU_V3') or 'CPU'
            for no accelerator. Defaults to 'auto', which maps to a standard
            gpu config such as 'P100'.
          accelerator_count: Number of accelerators. Defaults to 1.
        """
        self.cpu_cores = cpu_cores
        self.memory = memory
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count

        if self.accelerator_type == "auto":
            self.accelerator_type = AcceleratorType.NVIDIA_TESLA_P100

        self.validate()

    def validate(self):
        """Checks that the machine configuration created is valid for GCP."""
        AcceleratorType.validate(self.accelerator_type)
        gcp.validate_machine_configuration(self.cpu_cores,
                                           self.memory,
                                           self.accelerator_type,
                                           self.accelerator_count)


# Dictionary with common machine configurations.
COMMON_MACHINE_CONFIGS = {
    "CPU": MachineConfig(
        cpu_cores=4,
        memory=15,
        accelerator_type=AcceleratorType.NO_ACCELERATOR,
        accelerator_count=0,
    ),
    "K80_1X": MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_K80,
        accelerator_count=1,
    ),
    "K80_4X": MachineConfig(
        cpu_cores=16,
        memory=60,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_K80,
        accelerator_count=4,
    ),
    "K80_8X": MachineConfig(
        cpu_cores=32,
        memory=120,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_K80,
        accelerator_count=8,
    ),
    "P100_1X": MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_P100,
        accelerator_count=1,
    ),
    "P100_4X": MachineConfig(
        cpu_cores=16,
        memory=60,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_P100,
        accelerator_count=4,
    ),
    "P4_1X": MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_P4,
        accelerator_count=1,
    ),
    "P4_4X": MachineConfig(
        cpu_cores=16,
        memory=60,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_P4,
        accelerator_count=4,
    ),
    "V100_1X": MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_V100,
        accelerator_count=1,
    ),
    "V100_4X": MachineConfig(
        cpu_cores=16,
        memory=60,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_V100,
        accelerator_count=4,
    ),
    "T4_1X": MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_T4,
        accelerator_count=1,
    ),
    "T4_4X": MachineConfig(
        cpu_cores=16,
        memory=60,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_T4,
        accelerator_count=4,
    ),
    "TPU": MachineConfig(
        cpu_cores=None,
        memory=None,
        accelerator_type=AcceleratorType.TPU_V3,
        accelerator_count=8,
    ),
}


def is_tpu_config(config):
    if config:
        return (
            config.accelerator_type == AcceleratorType.TPU_V2
            or config.accelerator_type == AcceleratorType.TPU_V3
        )
    return False
