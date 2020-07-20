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

import os
import sys
from typing import Text
from unittest import mock
import tensorflow as tf
import tensorflow_cloud as tfc

# The staging bucket.
REMOTE_DIR = os.environ["E2E_TEST_BUCKET"]

# Using the build ID for testing
BUILD_ID = os.environ["E2E_BUID_ID"]

# Path to the source code in test enviroment
TEST_DATA_PATH = os.path.dirname(os.path.abspath(__file__))

class TensorflowCloudOnScriptTest(tf.test.TestCase):

  def setUp(self):
    super(TensorflowCloudOnScriptTest, self).setUp()
    self.remote_dir = REMOTE_DIR
    self.save_model_path = os.path.join(self.remote_dir, BUILD_ID)
    self.test_folders = [self.save_model_path]

  def tearDown(self):
    super(TensorflowCloudOnScriptTest, self).tearDown()
    for folder in self.test_folders:
      self.delete_dir(folder)

  def delete_dir(self, path: Text) -> None:
    """Deletes a directory if exists."""
    if tf.io.gfile.isdir(path):
      tf.io.gfile.rmtree(path)

  @mock.patch.object(sys, "exit", autospec=True)
  def test_on_script_save_and_load(self, mock_exit):
    mock_exit.side_effect = RuntimeError("exit called")
    with self.assertRaises(RuntimeError):
      tfc.run(
          entry_point=os.path.join(
            TEST_DATA_PATH, '../testdata/mnist_example_using_fit_no_reqs.py'),
          entry_point_args=["--path", self.save_model_path],
      )
    mock_exit.assert_called_once_with(0)


if __name__ == "__main__":
  tf.test.main()
