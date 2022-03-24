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
"""Unit test for tf_utils module."""

import mock
import tensorflow as tf
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorflow_cloud.utils import tf_utils


class TFUtilsTest(tf.test.TestCase):

    def setUp(self):
        super(TFUtilsTest, self).setUp()
        self.addCleanup(mock.patch.stopall)
        self._file_path = "test_file.123.v2"
        self._dir_path = "dir_folder/dir_subfolder"

    @mock.patch.object(io_wrapper, "IsSummaryEventsFile", autospec=True)
    def test_get_tensorboard_log_watcher_from_path_with_no_path(
        self, mock_issummaryeventsfile):

        with self.assertRaises(ValueError):
            tf_utils.get_tensorboard_log_watcher_from_path(None)

    @mock.patch.object(io_wrapper, "IsSummaryEventsFile", autospec=True)
    @mock.patch.object(event_file_loader, "EventFileLoader", autospec=True)
    @mock.patch.object(directory_watcher, "DirectoryWatcher", autospec=True)
    def test_get_tensorboard_log_watcher_from_path_with_file_path(
        self,
        mock_directorywatcher,
        mock_eventfileloader,
        mock_issummaryeventsfile):

        mock_issummaryeventsfile.return_value = True
        tf_utils.get_tensorboard_log_watcher_from_path(self._file_path)
        mock_eventfileloader.assert_called_with(self._file_path)
        mock_directorywatcher.assert_not_called()

    @mock.patch.object(io_wrapper, "IsSummaryEventsFile", autospec=True)
    @mock.patch.object(event_file_loader, "EventFileLoader", autospec=True)
    @mock.patch.object(directory_watcher, "DirectoryWatcher", autospec=True)
    def test_get_tensorboard_log_watcher_from_path_with_dir_path(
        self,
        mock_directorywatcher,
        mock_eventfileloader,
        mock_issummaryeventsfile):

        mock_issummaryeventsfile.return_value = False
        tf_utils.get_tensorboard_log_watcher_from_path(self._dir_path)
        mock_eventfileloader.assert_not_called()
        mock_directorywatcher.assert_called_with(
            self._dir_path, mock_eventfileloader, mock_issummaryeventsfile)

if __name__ == "__main__":
    tf.test.main()
