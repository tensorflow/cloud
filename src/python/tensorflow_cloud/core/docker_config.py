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
"""Docker configuration for the `run` API."""


class DockerConfig(object):
    """Represents Docker-related configuration for the `run` API.

    `run` API uses docker for containerizing your code and it's dependencies.

    Attributes:
      base_image: Optional base docker image to use.
        Defaults to None.
        Example value - 'gcr.io/my_gcp_project/deep_learning:v2'
        If a base docker image is not provided here, we will use a
        [TensorFlow docker image](https://www.tensorflow.org/install/docker)
        as the base image. The version of TensorFlow and Python in that
        case will match your local environment.
        If both `base_image` and a local TF installation are not
        available, the latest stable TF docker image will be used.
        Example - 'tensorflow/tensorflow:latest-gpu'
      image_build_bucket: GCS bucket name used for Google Cloud Build.
        This is an optional string. If it is not specified, then your
        local docker daemon will be used for docker containerization.
        If a GCS bucket name is provided, then we will upload your code as
        a tarfile to this bucket, which is then used by Google Cloud Build
        for remote docker containization.
        More info on [cloud build](https://cloud.google.com/cloud-build/).
        Note - This parameter is required when using `run` API from within
        an iPython notebook.
    """

    def __init__(self,
                 base_image=None,
                 image_build_bucket=None):
        self.base_image = base_image
        self.image_build_bucket = image_build_bucket
