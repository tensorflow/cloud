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
'''Tests for the validation module.'''

import pytest
import unittest

from tensorflow_cloud import machine_config
from tensorflow_cloud import validate


class TestValidate(unittest.TestCase):

    VALID_MACHINE_CONFIG = machine_config.MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=machine_config.AcceleratorType.NVIDIA_TESLA_K80,
        accelerator_count=1)

    def test_valid_args(self):
        validate.validate(
            entry_point='tests/testdata/mnist_example_using_fit.py',
            distribution_strategy='auto',
            requirements_txt='tests/testdata/requirements.txt',
            chief_config=TestValidate.VALID_MACHINE_CONFIG,
            worker_config=TestValidate.VALID_MACHINE_CONFIG,
            worker_count=1,
            region='us-central1',
            args=None,
            stream_logs=True)

        validate.validate(
            entry_point='tests/testdata/mnist_example_using_fit.py',
            distribution_strategy=None,
            requirements_txt='tests/testdata/requirements.txt',
            chief_config=TestValidate.VALID_MACHINE_CONFIG,
            worker_config=None,
            worker_count=0,
            region='us-central1',
            args=['1000'],
            stream_logs=False)

    def test_invalid_entry_point(self):
        with pytest.raises(ValueError, match=r'Invalid `entry_point`'):
            validate.validate(
                entry_point='/mnist_example_using_fit.py',
                distribution_strategy='auto',
                requirements_txt='tests/testdata/requirements.txt',
                chief_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_count=1,
                region='us-central1',
                args=None,
                stream_logs=True)

    def test_invalid_requirements_txt(self):
        with pytest.raises(ValueError, match=r'Invalid `requirements_txt`'):
            validate.validate(
                entry_point='tests/testdata/mnist_example_using_fit.py',
                distribution_strategy='auto',
                requirements_txt='temp.txt',
                chief_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_count=1,
                region='us-central1',
                args=None,
                stream_logs=True)

    def test_invalid_distribution_strategy(self):
        with pytest.raises(
                ValueError,
                match=r'Invalid `distribution_strategy`'):
            validate.validate(
                entry_point='tests/testdata/mnist_example_using_fit.py',
                distribution_strategy='MirroredStrategy',
                requirements_txt='tests/testdata/requirements.txt',
                chief_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_count=1,
                region='us-central1',
                args=None,
                stream_logs=True)

    def test_invalid_chief_config(self):
        with pytest.raises(ValueError, match=r'Invalid `chief_config`'):
            validate.validate(
                entry_point='tests/testdata/mnist_example_using_fit.py',
                distribution_strategy='auto',
                requirements_txt='tests/testdata/requirements.txt',
                chief_config=None,
                worker_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_count=1,
                region='us-central1',
                args=None,
                stream_logs=True)

    def test_invalid_worker_config(self):
        with pytest.raises(ValueError, match=r'Invalid `worker_config`'):
            validate.validate(
                entry_point='tests/testdata/mnist_example_using_fit.py',
                distribution_strategy='auto',
                requirements_txt='tests/testdata/requirements.txt',
                chief_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_config=None,
                worker_count=1,
                region='us-central1',
                args=None,
                stream_logs=True)

    def test_invalid_worker_count(self):
        with pytest.raises(ValueError, match=r'Invalid `worker_count`'):
            validate.validate(
                entry_point='tests/testdata/mnist_example_using_fit.py',
                distribution_strategy='auto',
                requirements_txt='tests/testdata/requirements.txt',
                chief_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_count=-1,
                region='us-central1',
                args=None,
                stream_logs=True)

    def test_invalid_region(self):
        with pytest.raises(ValueError, match=r'Invalid `region`'):
            validate.validate(
                entry_point='tests/testdata/mnist_example_using_fit.py',
                distribution_strategy='auto',
                requirements_txt='tests/testdata/requirements.txt',
                chief_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_count=1,
                region=['us-region-a'],
                args=None,
                stream_logs=True)

    def test_invalid_args(self):
        with pytest.raises(ValueError, match=r'Invalid `entry_point_args`'):
            validate.validate(
                entry_point='tests/testdata/mnist_example_using_fit.py',
                distribution_strategy='auto',
                requirements_txt='tests/testdata/requirements.txt',
                chief_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_count=1,
                region='us-central1',
                args='1000',
                stream_logs=True)

    def test_invalid_stream_logs(self):
        with pytest.raises(ValueError, match=r'Invalid `stream_logs`'):
            validate.validate(
                entry_point='tests/testdata/mnist_example_using_fit.py',
                distribution_strategy='auto',
                requirements_txt='tests/testdata/requirements.txt',
                chief_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_config=TestValidate.VALID_MACHINE_CONFIG,
                worker_count=1,
                region='us-central1',
                args=None,
                stream_logs='True')
