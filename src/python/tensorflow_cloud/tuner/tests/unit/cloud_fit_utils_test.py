# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Unit test for cloud_fit Utilities."""

from unittest import mock
import tensorflow as tf
from tensorflow_cloud.tuner import cloud_fit_utils as utils


class CloudFitUtilitiesTest(tf.test.TestCase):

    @mock.patch.object(utils, "is_tf_v1", autospec=True)
    @mock.patch.object(tf.compat.v1, "enable_eager_execution", autospec=True)
    def test_enable_eager_for_tf_1(self, mock_eager_execution, mock_is_tf_v1):
        mock_is_tf_v1.return_value = False
        utils.enable_eager_for_tf_1()
        # Work around as assert_not_called is not supported in py35
        self.assertFalse(mock_eager_execution.called)

        mock_is_tf_v1.return_value = True
        utils.enable_eager_for_tf_1()
        mock_eager_execution.assert_called_once_with()


if __name__ == "__main__":
    tf.test.main()
