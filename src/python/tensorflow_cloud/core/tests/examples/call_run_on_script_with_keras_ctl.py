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

import tensorflow_cloud as tfc

# MultiWorkerMirroredStrategy
tfc.run(
    entry_point="tests/testdata/mnist_example_using_ctl.py",
    distribution_strategy=None,
    worker_count=1,
    requirements_txt="tests/testdata/requirements.txt",
    stream_logs=True,
)
