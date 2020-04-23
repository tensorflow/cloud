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
"""Utilities for deploying jobs to GCP."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import uuid

from googleapiclient import discovery
from googleapiclient import errors

from . import gcp


def deploy_job(region,
               image_uri,
               chief_config,
               worker_count,
               worker_config,
               entry_point_args,
               enable_stream_logs):
    """Deploys job with the given parameters to Google Cloud.

    Args:
        region: GCP region name.
        image_uri: The docker image uri.
        chief_config: `MachineConfig` that represents the configuration for
            the chief worker in a distribution cluster.
        worker_count: Integer that represents the number of general workers
            in a distribution cluster. This count does not include the
            chief worker.
        worker_config: `MachineConfig` that represents the configuration for
            the general workers in a distribution cluster.
        entry_point_args: Command line arguments to pass to the
            `entry_point` program.
        enable_stream_logs: Boolean flag which when enabled streams logs
            back from the cloud job.

    Raises:
        RuntimeError, if there was an error submitting the job.
    """
    job_id = _generate_job_id()
    project_id = gcp.get_project_name()
    ml_apis = discovery.build('ml', 'v1', cache_discovery=False)

    request_dict = _create_request_dict(
        job_id, region, image_uri, chief_config, worker_count, worker_config,
        entry_point_args)
    try:
        response = ml_apis.projects().jobs().create(
            parent='projects/{}'.format(project_id),
            body=request_dict
        ).execute()
        _print_logs_info(job_id, project_id)
        if enable_stream_logs:
            _stream_logs(job_id)
    except errors.HttpError as err:
        raise RuntimeError(
            'There was an error submitting the job.' + err._get_reason())
    return job_id


def _create_request_dict(job_id,
                         region,
                         image_uri,
                         chief_config,
                         worker_count,
                         worker_config,
                         entry_point_args):
    """Creates request dictionary for the CAIP training service.

    Args:
        job_id: String, unique job id.
        region: GCP region name.
        image_uri: The docker image uri.
        chief_config: `MachineConfig` that represents the configuration for
            the chief worker in a distribution cluster.
        worker_count: Integer that represents the number of general workers
            in a distribution cluster. This count does not include the
            chief worker.
        worker_config: `MachineConfig` that represents the configuration for
            the general workers in a distribution cluster.
        entry_point_args: Command line arguments to pass to the
            `entry_point` program.

    Returns:
        The job request dictionary.
    """
    training_input = {}
    training_input['region'] = region
    training_input['scaleTier'] = 'custom'
    training_input['masterType'] = gcp.get_machine_type(
        chief_config.cpu_cores, chief_config.memory)

    # Set master config
    chief_machine_config = {}
    chief_machine_config['imageUri'] = image_uri
    chief_machine_config['acceleratorConfig'] = {}
    chief_machine_config['acceleratorConfig']['count'] = str(
        chief_config.accelerator_count)
    chief_machine_config['acceleratorConfig']['type'] = (
        gcp.get_accelerator_type(chief_config.accelerator_type.value))

    training_input['masterConfig'] = chief_machine_config
    training_input['workerCount'] = str(worker_count)

    if worker_count > 0:
        training_input['workerType'] = gcp.get_machine_type(
            worker_config.cpu_cores, worker_config.memory)

        worker_machine_config = {}
        worker_machine_config['imageUri'] = image_uri
        worker_machine_config['acceleratorConfig'] = {}
        worker_machine_config['acceleratorConfig']['count'] = str(
            worker_config.accelerator_count)
        worker_machine_config['acceleratorConfig']['type'] = (
            gcp.get_accelerator_type(worker_config.accelerator_type.value))
        training_input['workerConfig'] = worker_machine_config

    if entry_point_args is not None:
        training_input['args'] = entry_point_args

    # This is temporarily required so that the `TF_CONFIG` generated by
    # CAIP uses the keyword 'chief' instead of 'master'.
    training_input['use_chief_in_tf_config'] = True
    request_dict = {}
    request_dict['jobId'] = job_id
    request_dict['trainingInput'] = training_input
    return request_dict


def _print_logs_info(job_id, project_id):
    """Prints job id and access url.

    Args:
        job_id: The job id to print.
        project_id: The project id that is used to generate job access url.
    """
    print('Job submitted successfully.')
    print('Your job ID is: ', job_id)
    print('Please access your job logs at the following URL:')
    print('https://console.cloud.google.com/mlengine/jobs/{}?project={}'
          .format(job_id, project_id))


def _stream_logs(job_id):
    """Streams job logs to stdout.

    Args:
        job_id: The job id to stream logs from.

    Raises:
        RuntimeError: if there are any errors from the streaming subprocess.
    """
    try:
        print('Streaming job logs: ')
        subprocess.run(
            ["gcloud", "ai-platform", "jobs", "stream-logs", job_id])
    except subprocess.SubprocessError as err:
        raise RuntimeError(
            'There was an error streaming the job logs. {}'.format(
                err._get_reason()))


def _generate_job_id():
    """Returns a unique job id prefixed with 'tf_train'."""
    # CAIP job id can contains only numbers, letters and underscores.
    unique_tag = str(uuid.uuid4()).replace('-', '_')
    return 'tf_train_{}'.format(unique_tag)
