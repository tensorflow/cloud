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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import tempfile


def get_tarball(file_location_map):
    _, output_file = tempfile.mkstemp()
    with tarfile.open(output_file, 'w:gz', dereference=True) as tar:
        for src, dst in file_location_map.items():
            tar.add(src, arcname=dst)
    return output_file
