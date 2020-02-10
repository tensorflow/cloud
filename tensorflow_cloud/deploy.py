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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import string
import subprocess

from googleapiclient import discovery
from googleapiclient import errors

from . import gcp


def deploy_job(region, image_uri, chief_config, worker_count, worker_config,
               entry_point_args, enable_stream_logs):
    job_name = _get_name()
    project_id = gcp.get_project_name()
    ml_apis = discovery.build('ml', 'v1')

    request_dict = _create_request_dict(
        job_name, region, image_uri, chief_config, worker_count, worker_config,
        entry_point_args)
    try:
        response = ml_apis.projects().jobs().create(
            parent='projects/{}'.format(project_id),
            body=request_dict
        ).execute()
        print('Job submitted successfully.')
        _print_logs_info(job_name, project_id)
        # TODO(psv): Add support for streaming logs.
    except errors.HttpError as err:
        print('There was an error submitting the job.')
        print(err._get_reason())
    return job_name


def _create_request_dict(job_name, region, image_uri, chief_config,
                         worker_count, worker_config, entry_point_args):
    trainingInput = {}
    trainingInput['region'] = region
    trainingInput['scaleTier'] = 'custom'
    trainingInput['masterType'] = gcp.get_machine_type(
        chief_config.cpu_cores, chief_config.memory)

    # Set master config
    masterConfig = {}
    masterConfig['imageUri'] = image_uri
    masterConfig['acceleratorConfig'] = {}
    masterConfig['acceleratorConfig']['count'] = str(
        chief_config.accelerator_count)
    masterConfig['acceleratorConfig']['type'] = gcp.get_accelerator_type(
        chief_config.accelerator_type.value)

    trainingInput['masterConfig'] = masterConfig
    trainingInput['workerCount'] = str(worker_count)

    if worker_count > 0:
        trainingInput['workerType'] = gcp.get_machine_type(
            worker_config.cpu_cores, worker_config.memory)

        workerConfig = {}
        workerConfig['imageUri'] = image_uri
        workerConfig['acceleratorConfig'] = {}
        workerConfig['acceleratorConfig']['count'] = str(
            worker_config.accelerator_count)
        workerConfig['acceleratorConfig']['type'] = gcp.get_accelerator_type(
            worker_config.accelerator_type.value)
        trainingInput['workerConfig'] = workerConfig

    if entry_point_args is not None:
        trainingInput['args'] = entry_point_args
    trainingInput['use_chief_in_tf_config'] = True
    request_dict = {}
    request_dict['jobId'] = job_name
    request_dict['trainingInput'] = trainingInput
    return request_dict


def _print_logs_info(job_name, project_id):
    print('Your job ID is: ', job_name)
    print('Please access your job logs at the following URL:')
    print('https://console.cloud.google.com/mlengine/jobs/{}?project={}'
          .format(job_name, project_id))


def _get_name():
    unique_tag = ''.join(random.choice(
        string.ascii_lowercase + string.digits) for _ in range(8))
    return 'tf_train_{}'.format(unique_tag)
