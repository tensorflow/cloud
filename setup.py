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

"""Setup script."""

from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup

VERSION = "0.1.4"


setup(
    name="tensorflow-cloud",
    version=VERSION,
    description="The TensorFlow Cloud repository provides APIs that will allow "
    "to easily go from debugging and training your Keras and TensorFlow "
    "code in a local environment to distributed training in the cloud.",
    url="https://github.com/tensorflow/cloud",
    author="The tensorflow cloud authors",
    author_email="tensorflow-cloud@google.com",
    license="Apache License 2.0",
    install_requires=["docker", "google-api-python-client", "google-cloud-storage",],
    extras_require={"tests": ["pytest", "flake8", "mock"],},
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("tests",)),
)
