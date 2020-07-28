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

import argparse

import tensorflow_cloud as tfc

parser = argparse.ArgumentParser(description="Model save path arguments.")
parser.add_argument("--path", required=True, type=str, help="Keras model save path")
args = parser.parse_args()

tfc.run(
    entry_point="tests/testdata/save_and_load.py",
    distribution_strategy=None,
    requirements_txt="tests/testdata/requirements.txt",
    entry_point_args=["--path", args.path],
)
