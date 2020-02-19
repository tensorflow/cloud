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
"""Tests for gcp module."""

import pytest
import unittest

from tensorflow_cloud import gcp
from tensorflow_cloud import machine_config


class TestGcp(unittest.TestCase):

    def test_get_region(self):
        assert gcp.get_region() == 'us-central1'

    def test_get_accelerator_type(self):
        assert (gcp.get_accelerator_type('CPU') ==
                'ACCELERATOR_TYPE_UNSPECIFIED')
        assert gcp.get_accelerator_type('K80') == 'NVIDIA_TESLA_K80'
        assert gcp.get_accelerator_type('P100') == 'NVIDIA_TESLA_P100'
        assert gcp.get_accelerator_type('V100') == 'NVIDIA_TESLA_V100'
        assert gcp.get_accelerator_type('P4') == 'NVIDIA_TESLA_P4'
        assert gcp.get_accelerator_type('T4') == 'NVIDIA_TESLA_T4'

    def test_get_machine_type(self):
        assert gcp.get_machine_type(4, 15) == 'n1-standard-4'
        assert gcp.get_machine_type(8, 30) == 'n1-standard-8'
        assert gcp.get_machine_type(16, 60) == 'n1-standard-16'
        assert gcp.get_machine_type(32, 120) == 'n1-standard-32'
        assert gcp.get_machine_type(64, 240) == 'n1-standard-64'
        assert gcp.get_machine_type(96, 360) == 'n1-standard-96'
        assert gcp.get_machine_type(2, 13) == 'n1-highmem-2'
        assert gcp.get_machine_type(4, 26) == 'n1-highmem-4'
        assert gcp.get_machine_type(8, 52) == 'n1-highmem-8'
        assert gcp.get_machine_type(16, 104) == 'n1-highmem-16'
        assert gcp.get_machine_type(32, 208) == 'n1-highmem-32'
        assert gcp.get_machine_type(64, 416) == 'n1-highmem-64'
        assert gcp.get_machine_type(96, 624) == 'n1-highmem-96'
        assert gcp.get_machine_type(16, 14.4) == 'n1-highcpu-16'
        assert gcp.get_machine_type(32, 28.8) == 'n1-highcpu-32'
        assert gcp.get_machine_type(64, 57.6) == 'n1-highcpu-64'
        assert gcp.get_machine_type(96, 86.4) == 'n1-highcpu-96'

    def test_validate_machine_configuration(self):
        # valid config
        gcp.validate_machine_configuration(
            4, 15, machine_config.AcceleratorType.NVIDIA_TESLA_K80, 4)

        # test invalid config
        with pytest.raises(ValueError, match=r'Invalid machine configuration'):
            gcp.validate_machine_configuration(
                1, 15, machine_config.AcceleratorType.NVIDIA_TESLA_K80, 4)
