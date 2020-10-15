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
"""Tests for the validation module."""

import os

from absl.testing import absltest
import mock

from tensorflow_cloud.core import machine_config
from tensorflow_cloud.core import validate
from tensorflow_cloud.core.tests.unit import constants


class TestValidate(absltest.TestCase):

    def setUp(self):
        super(TestValidate, self).setUp()
        self.script_entry_point = os.path.join(
            constants.TESTDATA_DIR, "mnist_example_using_fit.py"
        )
        self.notebook_entry_point = os.path.join(
            constants.TESTDATA_DIR, "mnist_example_using_fit.ipynb"
        )
        self.requirements_file = os.path.join(
            constants.TESTDATA_DIR, "requirements.txt"
        )

    def test_valid_args_chief_worker(self):
        validate.validate(
            entry_point=self.script_entry_point,
            distribution_strategy="auto",
            requirements_txt=self.requirements_file,
            chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
            worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
            worker_count=1,
            entry_point_args=None,
            stream_logs=True,
            docker_image_build_bucket=None,
            called_from_notebook=False,
        )

    def test_valid_args_chief_only(self):
        validate.validate(
            entry_point=self.script_entry_point,
            distribution_strategy=None,
            requirements_txt=self.requirements_file,
            chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
            worker_config=None,
            worker_count=0,
            entry_point_args=["1000"],
            stream_logs=False,
            docker_image_build_bucket=None,
            called_from_notebook=False,
        )

    def test_valid_args_notebook_entry_point(self):
        validate.validate(
            entry_point=self.notebook_entry_point,
            distribution_strategy=None,
            requirements_txt=self.requirements_file,
            chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
            worker_config=None,
            worker_count=0,
            entry_point_args=["1000"],
            stream_logs=False,
            docker_image_build_bucket=None,
            called_from_notebook=False,
        )

    def test_valid_args_no_entry_point(self):
        validate.validate(
            entry_point=None,
            distribution_strategy=None,
            requirements_txt=self.requirements_file,
            chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
            worker_config=None,
            worker_count=0,
            entry_point_args=["1000"],
            stream_logs=False,
            docker_image_build_bucket="abc",
            called_from_notebook=True,
        )

    def test_valid_args_no_entry_point_with_job_label(self):
        validate.validate(
            entry_point=None,
            distribution_strategy=None,
            requirements_txt=self.requirements_file,
            chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
            worker_config=None,
            worker_count=0,
            entry_point_args=["1000"],
            stream_logs=False,
            docker_image_build_bucket="abc",
            called_from_notebook=True,
            job_labels={"a": "b"},
        )

    def test_invalid_entry_point(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `entry_point`"):
            validate.validate(
                entry_point="/path/to/nonexistent.py",
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_requirements_txt(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `requirements_txt`"):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt="/path/to/nonexistent/requirements.txt",
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_distribution_strategy(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid `distribution_strategy`",
        ):
            validate.validate(
                entry_point=None,
                distribution_strategy="MirroredStrategy",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_chief_config(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `chief_config`"):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=None,
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_worker_config(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `worker_config`"):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_config=None,
                worker_count=1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_worker_count(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `worker_count`"):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=-1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_args(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `entry_point_args`"):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=1,
                entry_point_args="1000",
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_stream_logs(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `stream_logs`"):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=1,
                entry_point_args=None,
                stream_logs="True",
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_cloud_bucket_name(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid `docker_config.image_build_bucket`",
        ):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=1,
                entry_point_args=None,
                stream_logs=False,
                docker_image_build_bucket=None,
                called_from_notebook=True,
            )

    def test_invalid_tpu_chief_config(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid `chief_config`",
        ):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["TPU"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["K80_1X"],
                worker_count=1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_tpu_worker_count(self):
        with self.assertRaisesRegex(ValueError, r"Invalid `worker_count`"):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["CPU"],
                worker_config=machine_config.COMMON_MACHINE_CONFIGS["TPU"],
                worker_count=2,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    def test_invalid_tpu_accelerator_count(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid machine configuration",
        ):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["CPU"],
                worker_config=machine_config.MachineConfig(
                    accelerator_type=machine_config.AcceleratorType.TPU_V3
                ),
                worker_count=1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
            )

    @mock.patch("tensorflow_cloud.utils.tf_utils.get_version")  # pylint: disable=line-too-long
    def test_invalid_tpu_accelerator_tf_version(self, mock_get_version):
        mock_get_version.return_value = "2.2.0"
        with self.assertRaisesRegex(
            NotImplementedError,
            r"TPUs are only supported for TF version <= 2.1.0",
        ):
            validate.validate(
                entry_point=None,
                distribution_strategy="auto",
                requirements_txt=None,
                chief_config=machine_config.COMMON_MACHINE_CONFIGS["CPU"],
                worker_config=machine_config.MachineConfig(
                    accelerator_type=machine_config.AcceleratorType.TPU_V2,
                    accelerator_count=8,
                ),
                worker_count=1,
                entry_point_args=None,
                stream_logs=True,
                docker_image_build_bucket=None,
                called_from_notebook=False,
                docker_parent_image=None,
            )


if __name__ == "__main__":
    absltest.main()
