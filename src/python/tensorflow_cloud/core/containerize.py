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
"""Docker related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import tarfile
import tempfile
import time
import uuid
import requests
import warnings

from . import machine_config
import docker
from google.cloud import storage
from google.cloud.exceptions import NotFound
from googleapiclient import discovery
from googleapiclient import errors

try:
    from tensorflow import __version__ as VERSION
except ImportError:
    # Use the latest TF docker image if a local installation is not available.
    VERSION = "latest"


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ContainerBuilder(object):
    """Container builder for building and pushing a docker image.

    Args:
        entry_point: Optional string. File path to the python file or iPython
            notebook that contains the TensorFlow code.
            Note: This path must be in the current working directory tree.
            Example: 'train.py', 'training/mnist.py', 'mnist.ipynb'
            If `entry_point` is not provided, then
            - If you are in an iPython notebook environment, then the
                current notebook is taken as the `entry_point`.
            - Otherwise, the current python script is taken as the
                `entry_point`.
        preprocessed_entry_point: Optional `preprocessed_entry_point`
            file path.
        chief_config: `MachineConfig` that represents the configuration for
            the chief worker in a distribution cluster.
        worker_config: `MachineConfig` that represents the configuration
            for the workers in a distribution cluster.
        docker_registry: The docker registry name.
        project_id: Project id string.
        requirements_txt: Optional string. File path to requirements.txt file
            containing aditionally pip dependencies, if any.
        destination_dir: Optional working directory in the docker container
            filesystem.
        docker_base_image: Optional base docker image to use. Defaults to None.
        docker_image_bucket_name: Optional string that specifies the docker
            image cloud storage bucket name.
        called_from_notebook: Optional boolean which indicates whether run has
            been called in a notebook environment.
    """

    def __init__(
        self,
        entry_point,
        preprocessed_entry_point,
        chief_config,
        worker_config,
        docker_registry,
        project_id,
        requirements_txt=None,
        destination_dir="/app/",
        docker_base_image=None,
        docker_image_bucket_name=None,
        called_from_notebook=False,
    ):
        self.entry_point = entry_point
        self.preprocessed_entry_point = preprocessed_entry_point
        self.chief_config = chief_config
        self.worker_config = worker_config
        self.docker_registry = docker_registry
        self.requirements_txt = requirements_txt
        self.destination_dir = destination_dir
        self.docker_base_image = docker_base_image
        self.docker_image_bucket_name = docker_image_bucket_name
        self.called_from_notebook = called_from_notebook
        self.project_id = project_id

        # Those will be populated lazily.
        self.tar_file_path = None
        self.docker_client = None

    def get_docker_image(
        self, max_status_check_attempts=None, delay_between_status_checks=None
    ):
        """Builds, publishes and returns a docker image.

        Args:
            max_status_check_attempts: Maximum number of times allowed to check
                build status. Applicable only to cloud container builder.
            delay_between_status_checks: Time is seconds to wait between status
                checks.
        """
        raise NotImplementedError

    def get_generated_files(self):
        return [self.docker_file_path, self.tar_file_path]

    def _get_tar_file_path(self):
        """Packages files into a tarball."""
        self._create_docker_file()
        file_path_map = self._get_file_path_map()

        _, self.tar_file_path = tempfile.mkstemp()
        with tarfile.open(self.tar_file_path, "w:gz", dereference=True) as tar:
            for source, destination in file_path_map.items():
                tar.add(source, arcname=destination)

    def _create_docker_file(self):
        """Creates a Dockerfile."""
        if self.docker_base_image is None:
            # Updating the name for RC's to match with the TF generated RC docker image names.
            tf_version = VERSION.replace("-rc", "rc")
            # Get the TF docker base image to use based on the current
            # TF version.
            self.docker_base_image = "tensorflow/tensorflow:{}".format(tf_version)
            if (
                self.chief_config.accelerator_type
                != machine_config.AcceleratorType.NO_ACCELERATOR
            ):
                self.docker_base_image += "-gpu"

            # Add python 3 tag for TF version <= 2.1.0
            # https://hub.docker.com/r/tensorflow/tensorflow
            if VERSION != "latest":
                v = VERSION.split(".")
                if float(v[0] + "." + v[1]) <= 2.1:
                    self.docker_base_image += "-py3"

        if not self._base_image_exist():
            warnings.warn(
                "Docker base image {} does not exist.".format(self.docker_base_image)
            )
            if "dev" in self.docker_base_image:
                # Except for the latest TF nightly, other nightlies
                # do not have corresponding docker images.
                newtag = "nightly"
                if self.docker_base_image.endswith("-gpu"):
                    newtag += "-gpu"
                self.docker_base_image = (
                    self.docker_base_image.split(":")[0] + ":" + newtag
                )
                warnings.warn("Using the latest TF nightly build.")
            else:
                warnings.warn(
                    "Using the latest stable TF docker image available: "
                    "`tensorflow/tensorflow:latest`"
                    "Please see https://hub.docker.com/r/tensorflow/tensorflow/ "
                    "for details on available docker images."
                )
                newtag = "tensorflow/tensorflow:latest"
                if self.docker_base_image.endswith("-gpu"):
                    newtag += "-gpu"
                self.docker_base_image = newtag

        lines = [
            "FROM {}".format(self.docker_base_image),
            "WORKDIR {}".format(self.destination_dir),
        ]

        if self.requirements_txt is not None:
            _, requirements_txt_name = os.path.split(self.requirements_txt)
            dst_requirements_txt = os.path.join(requirements_txt_name)
            requirements_txt_path = os.path.join(
                self.destination_dir, requirements_txt_name
            )
            lines.append(
                "COPY {} {}".format(requirements_txt_path, requirements_txt_path)
            )
            # install pip requirements from requirements_txt if it exists.
            lines.append(
                "RUN if [ -e {} ]; "
                "then pip install --no-cache -r {}; "
                "fi".format(dst_requirements_txt, dst_requirements_txt)
            )
        if self.entry_point is None:
            lines.append("RUN pip install tensorflow-cloud")

        if self.worker_config is not None and machine_config.is_tpu_config(
            self.worker_config
        ):
            lines.append("RUN pip install cloud-tpu-client")

        # Copies the files from the `destination_dir` in docker daemon location
        # to the `destination_dir` in docker container filesystem.
        lines.append("COPY {} {}".format(self.destination_dir, self.destination_dir))

        docker_entry_point = self.preprocessed_entry_point or self.entry_point
        _, docker_entry_point_file_name = os.path.split(docker_entry_point)

        # Using `ENTRYPOINT` here instead of `CMD` specifically because
        # we want to support passing user code flags.
        lines.extend(
            ['ENTRYPOINT ["python", "{}"]'.format(docker_entry_point_file_name)]
        )

        content = "\n".join(lines)
        _, self.docker_file_path = tempfile.mkstemp()
        with open(self.docker_file_path, "w") as f:
            f.write(content)

    def _get_file_path_map(self):
        """Maps local file paths to the docker daemon process location.

        Dictionary mapping file paths in the local file system to the paths
        in the docker daemon process location. The `key` or source is the path
        of the file that will be used when creating the archive. The `value`
        or destination is set as the `arcname` for the file at this time.
        When extracting files from the archive, they are extracted to the
        destination path.

        Returns:
            A file path map.
        """
        location_map = {}
        if self.entry_point is None and sys.argv[0].endswith("py"):
            self.entry_point = sys.argv[0]

        # Map entry_point directory to the dst directory.
        if not self.called_from_notebook:
            entry_point_dir, _ = os.path.split(self.entry_point)
            if entry_point_dir == "":  # Current directory
                entry_point_dir = "."
            location_map[entry_point_dir] = self.destination_dir

        # Place preprocessed_entry_point in the dst directory.
        if self.preprocessed_entry_point is not None:
            _, preprocessed_entry_point_file_name = os.path.split(
                self.preprocessed_entry_point
            )
            location_map[self.preprocessed_entry_point] = os.path.join(
                self.destination_dir, preprocessed_entry_point_file_name
            )

        # Place requirements_txt in the dst directory.
        if self.requirements_txt is not None:
            _, requirements_txt_name = os.path.split(self.requirements_txt)
            location_map[self.requirements_txt] = os.path.join(
                self.destination_dir, requirements_txt_name
            )

        # Place docker file in the root directory.
        location_map[self.docker_file_path] = "Dockerfile"
        return location_map

    def _generate_name(self):
        """Returns unique name+tag for the docker image."""
        # Keeping this name format uniform with the job id.
        unique_tag = str(uuid.uuid4()).replace("-", "_")
        return "{}/{}:{}".format(self.docker_registry, "tf_cloud_train", unique_tag)

    def _base_image_exist(self):
        """Check whether the docker base image exists on dockerhub.

        Use docker api v2 to check if base image is available.
        """
        repo_name, tag_name = self.docker_base_image.split(":")
        r = requests.get(
            "http://hub.docker.com/v2/repositories/{}/tags/{}".format(
                repo_name, tag_name
            )
        )
        return r.ok


class LocalContainerBuilder(ContainerBuilder):
    """Container builder that uses local docker daemon process."""

    def get_docker_image(
        self, max_status_check_attempts=None, delay_between_status_checks=None
    ):
        """Builds, publishes and returns a docker image.

        Args:
            max_status_check_attempts: Maximum number of times allowed to check
                build status. Not applicable to this builder.
            delay_between_status_checks: Time is seconds to wait between status
                checks. Not applicable to this builder.
        """
        self.docker_client = docker.APIClient(version="auto")
        self._get_tar_file_path()

        # create docker image from tarball
        image_uri = self._build_docker_image()
        # push to the registry
        self._publish_docker_image(image_uri)
        return image_uri

    def _build_docker_image(self):
        """Builds docker image."""
        image_uri = self._generate_name()
        logger.info("Building docker image: {}".format(image_uri))

        # `fileobj` is generally set to the Dockerfile file path. If a tar file
        # is used for docker build context (ones that includes a Dockerfile)
        # then `custom_context` should be enabled.
        with open(self.tar_file_path, "rb") as fileobj:
            bld_logs_generator = self.docker_client.build(
                path=".",
                custom_context=True,
                fileobj=fileobj,
                tag=image_uri,
                encoding="utf-8",
                decode=True,
            )
        self._get_logs(bld_logs_generator, "build", image_uri)
        return image_uri

    def _publish_docker_image(self, image_uri):
        """Publishes docker image.

        Args:
            image_uri: String, the registry name and tag.
        """
        logger.info("Publishing docker image: {}".format(image_uri))
        pb_logs_generator = self.docker_client.push(image_uri, stream=True, decode=True)
        self._get_logs(pb_logs_generator, "publish", image_uri)

    def _get_logs(self, logs_generator, name, image_uri):
        """Decodes logs from docker and generates user friendly logs.

        Args:
            logs_generator: Generator returned from docker build/push APIs.
            name: String, 'build' or 'publish' used to identify where the
                generator came from.
            image_uri: String, the docker image URI.

        Raises:
            RuntimeError: if there are any errors when building or publishing a
            docker image.
        """
        for chunk in logs_generator:
            if "stream" in chunk:
                for line in chunk["stream"].splitlines():
                    logger.info(line)

            if "error" in chunk:
                raise RuntimeError(
                    "Docker image {} failed: {}\nImage URI: {}".format(
                        name, str(chunk["error"]), image_uri
                    )
                )


class CloudContainerBuilder(ContainerBuilder):
    """Container builder that uses Google cloud build."""

    def get_docker_image(
        self, max_status_check_attempts=20, delay_between_status_checks=30
    ):
        """Builds, publishes and returns a docker image.

        Args:
            max_status_check_attempts: Maximum number of times allowed to check
                build status. Applicable only to cloud container builder.
            delay_between_status_checks: Time is seconds to wait between status
                checks.
        """
        self._get_tar_file_path()
        storage_object_name = self._upload_tar_to_gcs()
        image_uri = self._generate_name()

        logger.info(
            "Building and publishing docker image using Google "
            "Cloud Build: {}".format(image_uri)
        )
        build_service = discovery.build("cloudbuild", "v1", cache_discovery=False)
        request_dict = self._create_cloud_build_request_dict(
            image_uri, storage_object_name
        )

        try:
            # Call to queue request to build and push docker image.
            create_response = (
                build_service.projects()
                .builds()
                .create(projectId=self.project_id, body=request_dict)
                .execute()
            )

            # `create` returns a long-running `Operation`.
            # https://cloud.google.com/cloud-build/docs/api/reference/rest/v1/operations#Operation
            # This contains the build id, which we can use to get the status.

            attempts = 1
            while attempts <= max_status_check_attempts:
                # Call to check on the status of the queued request.
                get_response = (
                    build_service.projects()
                    .builds()
                    .get(
                        projectId=self.project_id,
                        id=create_response["metadata"]["build"]["id"],
                    )
                    .execute()
                )

                # `get` response is a `Build` object which contains `Status`.
                # https://cloud.google.com/cloud-build/docs/api/reference/rest/v1/projects.builds#Build.Status
                status = get_response["status"]
                if status != "WORKING" and status != "QUEUED":
                    break

                attempts += 1
                # Wait for 30 seconds before we check on status again.
                time.sleep(delay_between_status_checks)
            if status != "SUCCESS":
                raise RuntimeError(
                    "There was an error executing the cloud build job. "
                    "Job status: " + status
                )

        except errors.HttpError as err:
            raise RuntimeError(
                "There was an error submitting the cloud build job. "
                + err._get_reason()
            )
        return image_uri

    def _upload_tar_to_gcs(self):
        """Uploads tarfile to GCS and returns the GCS object name."""
        logger.info("Uploading files to GCS.")
        storage_client = storage.Client()
        try:
            bucket = storage_client.get_bucket(self.docker_image_bucket_name)
        except NotFound:
            bucket = storage_client.create_bucket(self.docker_image_bucket_name)

        unique_tag = str(uuid.uuid4()).replace("-", "_")
        storage_object_name = "tf_cloud_train_tar_{}".format(unique_tag)

        blob = bucket.blob(storage_object_name)
        blob.upload_from_filename(self.tar_file_path)
        return storage_object_name

    def _create_cloud_build_request_dict(self, image_uri, storage_object_name):
        """Creates request body for cloud build JSON API call.

        `create` body should be a `Build` object
        https://cloud.google.com/cloud-build/docs/api/reference/rest/v1/projects.builds#Build

        Args:
            image_uri: GCR docker image uri.
            storage_object_name: Name of the tarfile object in GCS.

        Returns:
            Build request dictionary.
        """
        request_dict = {}
        request_dict["projectId"] = self.project_id
        request_dict["images"] = [[image_uri]]
        request_dict["steps"] = {
            "name": "gcr.io/cloud-builders/docker",
            "args": ["build", "-t", image_uri, "."],
        }
        request_dict["source"] = {
            "storageSource": {
                "bucket": self.docker_image_bucket_name,
                "object": storage_object_name,
            }
        }
        return request_dict
