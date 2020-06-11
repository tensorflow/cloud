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
"""Module that performs validations on the inputs to the `run` API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from . import machine_config


def validate(
    entry_point,
    requirements_txt,
    distribution_strategy,
    chief_config,
    worker_config,
    worker_count,
    region,
    args,
    stream_logs,
    docker_image_bucket_name,
    called_from_notebook,
):
    """Validates the inputs.

    # Arguments:
        entry_point: Optional string. File path to the python file or iPython
            notebook that contains the TensorFlow code.
        requirements_txt: Optional string. File path to requirements.txt file
            containing aditionally pip dependencies, if any.
        distribution_strategy: 'auto' or None. Defaults to 'auto'.
            'auto' means we will take care of creating a Tensorflow
            distribution strategy instance based on the machine configurations
            provided using the `chief_config`, `worker_config` and
            `worker_count` params.
        chief_config: `MachineConfig` that represents the configuration
            for the chief worker in a distribution cluster.
        worker_config: `MachineConfig` that represents the configuration
            for the workers in a distribution cluster.
        worker_count: Optional integer that represents the number of workers in
            a distribution cluster. Defaults to 0. This count does not include
            the chief worker.
        region: String. Cloud region in which to submit the job.
        entry_point_args: Optional list of strings. Defaults to None.
            Command line arguments to pass to the `entry_point` program.
        stream_logs: Boolean flag which when enabled streams logs back from
            the cloud job.
        docker_image_bucket_name: Optional string. Cloud storage bucket name.
        called_from_notebook: Boolean. True if the API is run in a
            notebook environment.

    # Raises:
        ValueError: if any of the inputs is invalid.
    """
    _validate_files(entry_point, requirements_txt)
    _validate_distribution_strategy(distribution_strategy)
    _validate_cluster_config(chief_config, worker_count, worker_config)
    _validate_other_args(
        region, args, stream_logs, docker_image_bucket_name, called_from_notebook
    )


def _validate_files(entry_point, requirements_txt):
    """Validates all the file path params."""
    cwd = os.getcwd()
    if entry_point is not None and (not os.path.isfile(os.path.join(cwd, entry_point))):
        raise ValueError(
            "Invalid `entry_point`. "
            "Expected a relative path in the current directory tree. "
            "Received: {}".format(entry_point)
        )

    if requirements_txt is not None and (
        not os.path.isfile(os.path.join(cwd, requirements_txt))
    ):
        raise ValueError(
            "Invalid `requirements_txt`. "
            "Expected a relative path in the current directory tree. "
            "Received: {}".format(requirements_txt)
        )

    if entry_point is not None and (
        not (entry_point.endswith("py") or entry_point.endswith("ipynb"))
    ):
        raise ValueError(
            "Invalid `entry_point`. "
            "Expected a python file or an iPython notebook. "
            "Received: {}".format(entry_point)
        )


def _validate_distribution_strategy(distribution_strategy):
    """Validates distribution strategy param."""
    if distribution_strategy not in ["auto", None]:
        raise ValueError(
            "Invalid `distribution_strategy` input. "
            'Expected "auto" or None. '
            "Received {}.".format(distribution_strategy)
        )


def _validate_cluster_config(chief_config, worker_count, worker_config):
    """Validates cluster config params."""
    if not isinstance(chief_config, machine_config.MachineConfig):
        raise ValueError(
            "Invalid `chief_config` input. "
            'Expected "auto" or `MachineConfig` instance. '
            "Received {}.".format(chief_config)
        )

    if worker_count < 0:
        raise ValueError(
            "Invalid `worker_count` input. "
            "Expected a postive integer value. "
            "Received {}.".format(worker_count)
        )

    if worker_count > 0 and not isinstance(worker_config, machine_config.MachineConfig):
        raise ValueError(
            "Invalid `worker_config` input. "
            'Expected "auto" or `MachineConfig` instance. '
            "Received {}.".format(worker_config)
        )
    # TODO(psv): Incompatible chief and worker configs


def _validate_other_args(
    region, args, stream_logs, docker_image_bucket_name, called_from_notebook
):
    """Validates all non-file/distribution strategy args."""
    if not isinstance(region, str):
        raise ValueError(
            "Invalid `region` input. "
            "Expected None or a string value. "
            "Received {}.".format(str(region))
        )

    if args is not None and not isinstance(args, list):
        raise ValueError(
            "Invalid `entry_point_args` input. "
            "Expected None or a list. "
            "Received {}.".format(str(args))
        )

    if not isinstance(stream_logs, bool):
        raise ValueError(
            "Invalid `stream_logs` input. "
            "Expected a boolean. "
            "Received {}.".format(str(stream_logs))
        )

    if called_from_notebook and docker_image_bucket_name is None:
        raise ValueError(
            "Invalid `docker_image_bucket_name` input. "
            "When `run` API is used within a python notebook, "
            "`docker_image_bucket_name` is expected to be specifed. We will "
            "use the bucket name in Google Cloud Storage/Build services for "
            "docker containerization. Received {}.".format(
                str(docker_image_bucket_name)
            )
        )
