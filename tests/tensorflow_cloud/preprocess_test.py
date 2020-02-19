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
"""Tests for the cloud preprocessing module."""

import os
import unittest

from tensorflow_cloud import machine_config
from tensorflow_cloud import preprocess


class TestPreprocess(unittest.TestCase):

    def setup(self):
        self.entry_point = 'testdata/sample_compile_fit.py'
        _, self.entry_point_name = os.path.split(self.entry_point)

    def get_wrapped_entry_point(
            self, chief_config=machine_config.COMMON_MACHINE_CONFIGS['K80_1X'],
            worker_count=0):
        self.wrapped_entry_point = preprocess.get_wrapped_entry_point(
            self.entry_point, chief_config, worker_count)

        with open(self.wrapped_entry_point, 'r') as f:
            script_lines = f.readlines()
        return script_lines

    def assert_and_cleanup(self, expected_lines, script_lines):
        self.assertListEqual(expected_lines, script_lines)
        os.remove(self.wrapped_entry_point)

    def test_auto_one_device_strategy(self):
        self.setup()
        script_lines = self.get_wrapped_entry_point()
        expected_lines = [
            'import tensorflow as tf\n',
            'strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")\n',
            'tf.distribute.experimental_set_strategy(strategy)\n',
            'exec(open("{}").read())\n'.format(self.entry_point_name)]
        self.assert_and_cleanup(expected_lines, script_lines)

    def test_auto_mirrored_strategy(self):
        self.setup()
        chief_config = machine_config.COMMON_MACHINE_CONFIGS['K80_4X']
        script_lines = self.get_wrapped_entry_point(chief_config=chief_config)
        expected_lines = [
            'import tensorflow as tf\n',
            'strategy = tf.distribute.MirroredStrategy()\n',
            'tf.distribute.experimental_set_strategy(strategy)\n',
            'exec(open("{}").read())\n'.format(self.entry_point_name)]
        self.assert_and_cleanup(expected_lines, script_lines)

    def test_auto_multi_worker_strategy(self):
        self.setup()
        chief_config = machine_config.COMMON_MACHINE_CONFIGS['K80_4X']
        script_lines = self.get_wrapped_entry_point(
            chief_config=chief_config, worker_count=2)
        expected_lines = [
            'import tensorflow as tf\n',
            'strategy = tf.distribute.experimental.'
            'MultiWorkerMirroredStrategy()\n',
            'tf.distribute.experimental_set_strategy(strategy)\n',
            'exec(open("{}").read())\n'.format(self.entry_point_name)]
        self.assert_and_cleanup(expected_lines, script_lines)
