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
"""Utility for packaging files into a tarball."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import tempfile


def get_tar_file_path(file_location_map):
    """Packages files into a tarball and returns the tarball file path.

    Args:
        file_location_map: Dictionary mapping file paths in the local
            file system to the paths in the docker daemon process location.
            The `key` or source is the path of the file that will be used when
            creating the archive. The `value` or destination is set as the
            `arcname` for the file at this time. When extracting files from
            the archive, they are extracted to the destination path.

    Returns:
        The tarball file path.
    """
    _, output_file = tempfile.mkstemp()
    with tarfile.open(output_file, 'w:gz', dereference=True) as tar:
        for source, destination in file_location_map.items():
            tar.add(source, arcname=destination)
    return output_file
