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
"""Tests for cloud tar packaging module."""

import os
import shutil
import tarfile
import tempfile
import unittest

from tensorflow_cloud import package


class TestPackage(unittest.TestCase):

    def test_get_tarball(self):
        _, temp_file1 = tempfile.mkstemp(suffix='.py')
        _, temp_file2 = tempfile.mkstemp(suffix='.txt')

        file_map = {
            temp_file1: 'data.csv',
            temp_file2: 'Dockerfile',
        }

        tarball = package.get_tarball(file_map)
        assert tarfile.is_tarfile(tarball)

        tar_file = tarfile.open(tarball)
        tar_file_names = [m.name for m in tar_file.getmembers()]
        self.assertIn('data.csv', tar_file_names)
        self.assertIn('Dockerfile', tar_file_names)
        os.remove(temp_file1)
        os.remove(temp_file2)
