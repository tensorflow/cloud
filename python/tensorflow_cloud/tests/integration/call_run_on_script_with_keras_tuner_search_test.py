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

# python call_run_on_script_with_keras_tuner_search.py --path gs://my-bucket/keras-cifar
parser = argparse.ArgumentParser(description="Model save path arguments.")
parser.add_argument("--path", required=True, type=str, help="Keras model save path")
args = parser.parse_args()

# Automated MirroredStrategy: chief config with multiple GPUs
tfc.run(
    entry_point="tests/testdata/keras_tuner_cifar_example.py",
    distribution_strategy="auto",
    requirements_txt="tests/testdata/requirements.txt",
    chief_config=tfc.COMMON_MACHINE_CONFIGS["V100_4X"],
    worker_count=0,
    entry_point_args=["--path", args.path],
)
