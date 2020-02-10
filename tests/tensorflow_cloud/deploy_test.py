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
import io
import mock
import os
import shutil
import sys
import tarfile
import unittest

from tensorflow_cloud import deploy
from tensorflow_cloud import machine_config
from tensorflow_cloud import package

from mock import call, patch


class TestDeploy(unittest.TestCase):

    def setup(self, MockDiscovery):
        self.mock_job_name = 'tf-train-abcde'
        self.mock_project_name = 'my-gcp-project'
        self.entry_point = 'testdata/sample_compile_fit.py'
        self.chief_config = machine_config.COMMON_MACHINE_CONFIGS['K80_4X']
        self.worker_count = 2
        self.worker_config = machine_config.COMMON_MACHINE_CONFIGS['K80_1X']
        self.region = 'us-central-a'
        self.docker_img = 'custom-image-tag'
        self.entry_point_args = ['1000']
        self.stream_logs = False

        self.expected_request_dict = {
            'jobId': self.mock_job_name,
            'trainingInput': {
                'use_chief_in_tf_config': True,
                'scaleTier': 'custom',
                'region': self.region,
                'args': self.entry_point_args,
                'masterType': 'n1-standard-16',
                'workerType': 'n1-standard-8',
                'workerCount': str(self.worker_count),
                'workerConfig': {
                    'acceleratorConfig': {
                        'count': '1',
                        'type': 'NVIDIA_TESLA_K80'
                    },
                    'imageUri': self.docker_img,
                },
                'masterConfig': {
                    'acceleratorConfig': {
                        'count': '4',
                        'type': 'NVIDIA_TESLA_K80'
                    },
                    'imageUri': self.docker_img,
                }
            },
        }

        # Verify mocking is correct and setup method mocks.
        assert MockDiscovery is deploy.discovery

        def _mock_get_name():
            return self.mock_job_name
        deploy._get_name = _mock_get_name

        def _mock_get_project_name():
            return self.mock_project_name
        deploy.gcp.get_project_name = _mock_get_project_name

    @patch('sys.stdout', new_callable=io.BytesIO)
    @patch('tensorflow_cloud.deploy.discovery')
    def test_deploy_job(self, MockDiscovery, MockStdOut):
        self.setup(MockDiscovery)

        job_name = deploy.deploy_job(
            self.region, self.docker_img, self.chief_config, self.worker_count,
            self.worker_config, self.entry_point_args, self.stream_logs)

        self.assertEqual(job_name, self.mock_job_name)

        # Verify discovery API is invoked as expected.
        self.assertEqual(MockDiscovery.build.call_count, 1)
        args, _ = MockDiscovery.build.call_args
        self.assertListEqual(list(args), ['ml', 'v1'])

        # Verify job is created as expected
        build_ret_val = MockDiscovery.build.return_value
        self.assertEqual(build_ret_val.projects.call_count, 1)
        proj_ret_val = build_ret_val.projects.return_value
        self.assertEqual(proj_ret_val.jobs.call_count, 1)
        jobs_ret_val = proj_ret_val.jobs.return_value
        self.assertEqual(jobs_ret_val.create.call_count, 1)

        # Verify job creation args
        _, kwargs = jobs_ret_val.create.call_args
        self.assertDictEqual(kwargs, {
            'parent': 'projects/' + self.mock_project_name,
            'body': self.expected_request_dict})

        # Verify print statement
        self.assertEqual(
            MockStdOut.getvalue(),
            'Job submitted successfully.\nYour job ID is:  {}\nPlease access '
            'your job logs at the following URL:\nhttps://'
            'console.cloud.google.com/mlengine/jobs/{}?project={}\n'.format(
                self.mock_job_name,
                self.mock_job_name,
                self.mock_project_name))

    @patch('tensorflow_cloud.deploy.discovery')
    def test_request_dict_without_workers(self, MockDiscovery):
        self.setup(MockDiscovery)
        worker_count = 0

        job_name = deploy.deploy_job(
            self.region, self.docker_img, self.chief_config, worker_count,
            None, self.entry_point_args, self.stream_logs)
        build_ret_val = MockDiscovery.build.return_value
        proj_ret_val = build_ret_val.projects.return_value
        jobs_ret_val = proj_ret_val.jobs.return_value

        self.expected_request_dict['trainingInput']['workerCount'] = str(
            worker_count)
        del self.expected_request_dict['trainingInput']['workerType']
        del self.expected_request_dict['trainingInput']['workerConfig']

        # Verify job creation args
        _, kwargs = jobs_ret_val.create.call_args
        self.assertDictEqual(kwargs, {
            'parent': 'projects/' + self.mock_project_name,
            'body': self.expected_request_dict})

    @patch('tensorflow_cloud.deploy.discovery')
    def test_request_dict_without_user_args(self, MockDiscovery):
        self.setup(MockDiscovery)
        job_name = deploy.deploy_job(
            self.region, self.docker_img, self.chief_config, self.worker_count,
            self.worker_config, None, self.stream_logs)
        build_ret_val = MockDiscovery.build.return_value
        proj_ret_val = build_ret_val.projects.return_value
        jobs_ret_val = proj_ret_val.jobs.return_value

        del self.expected_request_dict['trainingInput']['args']

        # Verify job creation args
        _, kwargs = jobs_ret_val.create.call_args
        self.assertDictEqual(kwargs, {
            'parent': 'projects/' + self.mock_project_name,
            'body': self.expected_request_dict})
