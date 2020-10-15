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
"""Tests for docker config module."""

from absl.testing import absltest

from tensorflow_cloud.core import docker_config


class TestDockerConfig(absltest.TestCase):

    def test_valid(self):
      docker_config.DockerConfig(
          parent_image="tensorflow/tensorflow:latest-gpu",
          image_build_bucket="test-bucket",
          image="gcr.io/test-project/tf_cloud_train:1",
          cache_from="gcr.io/test-project/tf_cloud_train:0")


if __name__ == "__main__":
    absltest.main()
