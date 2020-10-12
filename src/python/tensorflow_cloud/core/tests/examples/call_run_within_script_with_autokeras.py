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
"""
Search for a good model for the
[MNIST](https://keras.io/datasets/#mnist-database-of-handwritten-digits) dataset.
"""
import argparse
import os

import autokeras as ak
import tensorflow_cloud as tfc
from tensorflow.keras.datasets import mnist


parser = argparse.ArgumentParser(description="Model save path arguments.")
parser.add_argument("--path", required=True, type=str, help="Keras model save path")
args = parser.parse_args()

tfc.run(
    chief_config=tfc.COMMON_MACHINE_CONFIGS["V100_1X"],
    docker_config=tfc.DockerConfig(
        base_image="haifengjin/autokeras:1.0.3"),
)

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the ImageClassifier.
clf = ak.ImageClassifier(max_trials=2)
# Search for the best model.
clf.fit(x_train, y_train, epochs=10)
# Evaluate on the testing data.
print("Accuracy: {accuracy}".format(accuracy=clf.evaluate(x_test, y_test)[1]))

clf.export_model().save(os.path.join(args.path, "model.h5"))
