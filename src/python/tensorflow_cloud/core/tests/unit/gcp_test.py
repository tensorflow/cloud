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
"""Tests for gcp module."""

from absl.testing import absltest

from tensorflow_cloud.core import gcp
from tensorflow_cloud.core import machine_config


class TestGcp(absltest.TestCase):

    def test_get_region(self):
        assert gcp.get_region() == "us-central1"

    def test_get_accelerator_type(self):
        assert gcp.get_accelerator_type("CPU") == "ACCELERATOR_TYPE_UNSPECIFIED"
        assert gcp.get_accelerator_type("K80") == "NVIDIA_TESLA_K80"
        assert gcp.get_accelerator_type("P100") == "NVIDIA_TESLA_P100"
        assert gcp.get_accelerator_type("V100") == "NVIDIA_TESLA_V100"
        assert gcp.get_accelerator_type("P4") == "NVIDIA_TESLA_P4"
        assert gcp.get_accelerator_type("T4") == "NVIDIA_TESLA_T4"
        assert gcp.get_accelerator_type("TPU_V2") == "TPU_V2"
        assert gcp.get_accelerator_type("TPU_V3") == "TPU_V3"

    def test_get_machine_type(self):
        assert (
            gcp.get_machine_type(4, 15,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-standard-4"
        )
        assert (
            gcp.get_machine_type(8, 30,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-standard-8"
        )
        assert (
            gcp.get_machine_type(16, 60,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-standard-16"
        )
        assert (
            gcp.get_machine_type(32, 120,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-standard-32"
        )
        assert (
            gcp.get_machine_type(64, 240,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-standard-64"
        )
        assert (
            gcp.get_machine_type(96, 360,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-standard-96"
        )
        assert (
            gcp.get_machine_type(2, 13,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highmem-2"
        )
        assert (
            gcp.get_machine_type(4, 26,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highmem-4"
        )
        assert (
            gcp.get_machine_type(8, 52,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highmem-8"
        )
        assert (
            gcp.get_machine_type(16, 104,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highmem-16"
        )
        assert (
            gcp.get_machine_type(32, 208,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highmem-32"
        )
        assert (
            gcp.get_machine_type(64, 416,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highmem-64"
        )
        assert (
            gcp.get_machine_type(96, 624,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highmem-96"
        )
        assert (
            gcp.get_machine_type(16, 14.4,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highcpu-16"
        )
        assert (
            gcp.get_machine_type(32, 28.8,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highcpu-32"
        )
        assert (
            gcp.get_machine_type(
                64, 57.6, machine_config.AcceleratorType.NO_ACCELERATOR
            )
            == "n1-highcpu-64"
        )
        assert (
            gcp.get_machine_type(96, 86.4,
                                 machine_config.AcceleratorType.NO_ACCELERATOR)
            == "n1-highcpu-96"
        )
        assert (
            gcp.get_machine_type(96, 86.4,
                                 machine_config.AcceleratorType.TPU_V3)
            == "cloud_tpu"
        )
        assert (
            gcp.get_machine_type(96, 86.4,
                                 machine_config.AcceleratorType.TPU_V2)
            == "cloud_tpu"
        )

    def test_get_cloud_tpu_supported_tf_versions(self):
        self.assertListEqual(gcp.get_cloud_tpu_supported_tf_versions(), ["2.1"])

    def test_validate_machine_configuration(self):
        # valid GPU config
        gcp.validate_machine_configuration(
            4, 15, machine_config.AcceleratorType.NVIDIA_TESLA_K80, 4
        )

        # valid TPU config
        gcp.validate_machine_configuration(
            4, 15, machine_config.AcceleratorType.TPU_V3, 8
        )

        # test invalid config
        with self.assertRaisesRegex(ValueError,
                                    r"Invalid machine configuration"):
            gcp.validate_machine_configuration(
                1, 15, machine_config.AcceleratorType.NVIDIA_TESLA_K80, 4
            )

    def test_validate_invalid_job_label(self):
        with self.assertRaisesRegex(ValueError, r"Invalid job labels"):
            # must start with lower case
            gcp.validate_job_labels(job_labels={"": ""},)

        with self.assertRaisesRegex(ValueError, r"Invalid job labels"):
            # must start with lower case
            gcp.validate_job_labels(job_labels={"test": "-label"})

        with self.assertRaisesRegex(ValueError, r"Invalid job labels"):
            # must start with lower case
            gcp.validate_job_labels(job_labels={"Test": "label"})

        with self.assertRaisesRegex(ValueError, r"Invalid job labels"):
            # no upper case
            gcp.validate_job_labels(job_labels={"test": "labelA"})

        with self.assertRaisesRegex(ValueError, r"Invalid job labels"):
            # no symbol
            gcp.validate_job_labels(job_labels={"test": "label@"})

        with self.assertRaisesRegex(ValueError, r"Invalid job labels"):
            # label cannot be too long
            gcp.validate_job_labels(job_labels={"test": "a" * 64})

        with self.assertRaisesRegex(ValueError, r"Invalid job labels"):
            # label cannot be too many
            gcp.validate_job_labels(
                job_labels={"key{}".format(i):
                            "val{}".format(i) for i in range(65)}
            )

    def testValidateServiceAccount_NotEndWithCorrectDomain_RaisesValueError(
        self):
        with self.assertRaisesRegex(ValueError, r"Invalid service_account"):
            # must end with .iam.gserviceaccount.com
            gcp.validate_service_account(
                "test_sa_name@test-project.wrong_domain.com")

    def testValidateServiceAccount_ShortProjectId_RaisesValueError(self):
        with self.assertRaisesRegex(ValueError, r"Invalid service_account"):
            # Project id must be greater than 6 characters
            short_project_id = "a" * 5
            gcp.validate_service_account(
                f"test_sa_name@{short_project_id}.iam.gserviceaccount.com")

    def testValidateServiceAccount_LongProjectId_RaisesValueError(self):
        with self.assertRaisesRegex(ValueError, r"Invalid service_account"):
            # Project id must be less than 30 characters
            long_project_id = "a" * 31
            gcp.validate_service_account(
                f"test_sa_name@{long_project_id}.iam.gserviceaccount.com")

    def testValidateServiceAccount_ProjectIdWithDot_RaisesValueError(self):
        with self.assertRaisesRegex(ValueError, r"Invalid service_account"):
            # Project id can not contain .
            gcp.validate_service_account(
                "test_sa_name@test.projectid.iam.gserviceaccount.com")

    def testValidateServiceAccount_ProjectIdWithUnderScore_RaisesValueError(
        self):
        with self.assertRaisesRegex(ValueError, r"Invalid service_account"):
            # Project id can not contain _
            gcp.validate_service_account(
                "test_sa_name@test_projectid.iam.gserviceaccount.com")

if __name__ == "__main__":
    absltest.main()
