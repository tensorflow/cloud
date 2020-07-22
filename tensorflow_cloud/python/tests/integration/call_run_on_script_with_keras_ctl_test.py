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
"""Integration tests for calling tfc.run on script with keras."""

import os
import sys
from unittest import mock
import tensorflow as tf
import tensorflow_cloud as tfc


class TensorflowCloudOnScriptTest(tf.test.TestCase):
    def setUp(self):
        super(TensorflowCloudOnScriptTest, self).setUp()
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../testdata/"
        )

    @mock.patch.object(sys, "exit", autospec=True)
    def test_MWMS_on_script(self, mock_exit):
        mock_exit.side_effect = RuntimeError("exit called")
        with self.assertRaises(RuntimeError):
            tfc.run(
                entry_point=os.path.join(
                    self.test_data_path, "mnist_example_using_ctl.py"
                ),
                distribution_strategy=None,
                worker_count=1,
                requirements_txt=os.path.join(
                    self.test_data_path, "requirements.txt"),
            )
        mock_exit.assert_called_once_with(0)


if __name__ == "__main__":
    tf.test.main()
