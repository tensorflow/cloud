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

import os

from . import gcp
from . import machine_config
from ..utils import tf_utils


def validate(
    entry_point,
    requirements_txt,
    distribution_strategy,
    chief_config,
    worker_config,
    worker_count,
    entry_point_args,
    stream_logs,
    docker_image_build_bucket,
    called_from_notebook,
    job_labels=None,
    service_account=None,
    docker_parent_image=None,
):
    """Validates the inputs.

    Args:
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
        entry_point_args: Optional list of strings. Defaults to None.
            Command line arguments to pass to the `entry_point` program.
        stream_logs: Boolean flag which when enabled streams logs back from
            the cloud job.
        docker_image_build_bucket: Optional string. Cloud storage bucket name.
        called_from_notebook: Boolean. True if the API is run in a
            notebook environment.
        job_labels: Dict of str: str. Labels to organize jobs. See
            [resource-labels](
            https://cloud.google.com/ai-platform/training/docs/resource-labels).
        service_account: The email address of a user-managed service account
            to be used for training instead of the service account that AI
            Platform Training uses by default. see [custom-service-account](
            https://cloud.google.com/ai-platform/training/docs/custom-service-account)
        docker_parent_image: Optional parent Docker image to use.
            Defaults to None.

    Raises:
        ValueError: if any of the inputs is invalid.
    """
    _validate_files(entry_point, requirements_txt)
    _validate_distribution_strategy(distribution_strategy)
    _validate_cluster_config(
        chief_config, worker_count, worker_config, docker_parent_image
    )
    _validate_job_labels(job_labels or {})
    _validate_other_args(
        entry_point_args,
        stream_logs,
        docker_image_build_bucket,
        called_from_notebook,
    )
    _validate_service_account(service_account)


def _validate_files(entry_point, requirements_txt):
    """Validates all the file path params."""
    cwd = os.getcwd()
    if entry_point is not None and (
        not os.path.isfile(os.path.join(cwd, entry_point))):
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


def _validate_cluster_config(
    chief_config, worker_count, worker_config, docker_parent_image
):
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

    if (worker_count > 0 and
        not isinstance(worker_config, machine_config.MachineConfig)):
        raise ValueError(
            "Invalid `worker_config` input. "
            'Expected "auto" or `MachineConfig` instance. '
            "Received {}.".format(worker_config)
        )

    if machine_config.is_tpu_config(chief_config):
        raise ValueError(
            "Invalid `chief_config` input. "
            "`chief_config` cannot be a TPU config. "
            "Received {}.".format(chief_config)
        )

    if machine_config.is_tpu_config(worker_config):
        if worker_count != 1:
            raise ValueError(
                "Invalid `worker_count` input. "
                "Expected worker_count=1 for TPU `worker_config`. "
                "Received {}.".format(worker_count)
            )
        elif docker_parent_image is None:
            # If the user has not provided a custom Docker image, then verify
            # that the TF version is compatible with Cloud TPU support.
            # https://cloud.google.com/ai-platform/training/docs/runtime-version-list#tpu-support  # pylint: disable=line-too-long
            version = tf_utils.get_version()
            if (version is not None and
                version not in gcp.get_cloud_tpu_supported_tf_versions()):
                raise NotImplementedError(
                    "TPUs are only supported for TF version <= 2.1.0"
                )


def _validate_job_labels(job_labels):
    """Validates job labels."""
    gcp.validate_job_labels(job_labels)


def _validate_service_account(service_account):
    """Validates service_account."""
    gcp.validate_service_account(service_account)


def _validate_other_args(
    args, stream_logs, docker_image_build_bucket, called_from_notebook
):
    """Validates all non-file/distribution strategy args."""

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

    if called_from_notebook and docker_image_build_bucket is None:
        raise ValueError(
            "Invalid `docker_config.image_build_bucket` input. "
            "When `run` API is used within a python notebook, "
            "`docker_config.image_build_bucket` is expected to be specifed. We "
            "will use the bucket name in Google Cloud Storage/Build services "
            "for Docker containerization. Received {}.".format(
                str(docker_image_build_bucket)
            )
        )
