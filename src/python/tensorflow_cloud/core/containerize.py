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

import logging
import os
import sys
import tarfile
import tempfile
import time
import uuid
import warnings

from . import gcp
from . import machine_config
import docker
from google.cloud import storage
from google.cloud.exceptions import NotFound
from googleapiclient import discovery
from googleapiclient import errors
import requests
from ..utils import google_api_client
from ..utils import tf_utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_image_uri():
    """Returns unique name+tag for a Docker image."""
    # Keeping this name format uniform with the job id.
    unique_tag = str(uuid.uuid4()).replace("-", "_")
    docker_registry = "gcr.io/{}".format(gcp.get_project_name())
    return "{}/{}:{}".format(docker_registry, "tf_cloud_train", unique_tag)


class ContainerBuilder(object):
    """Container builder for building and pushing a Docker image."""

    def __init__(
        self,
        entry_point,
        preprocessed_entry_point,
        chief_config,
        worker_config,
        requirements_txt=None,
        destination_dir="/app/",
        docker_config=None,
        called_from_notebook=False,
    ):
        """Constructor.

        Args:
            entry_point: Optional string. File path to the python file or
                iPython notebook that contains the TensorFlow code.
                Note) This path must be in the current working directory tree.
                Example) 'train.py', 'training/mnist.py', 'mnist.ipynb'
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
            requirements_txt: Optional string. File path to requirements.txt
                file containing additionally pip dependencies, if any.
            destination_dir: Optional working directory in the Docker container
                filesystem.
            docker_config: Optional Docker configuration.
            called_from_notebook: Optional boolean which indicates whether run
                has been called in a notebook environment.
        """
        self.entry_point = entry_point
        self.preprocessed_entry_point = preprocessed_entry_point
        self.chief_config = chief_config
        self.worker_config = worker_config
        self.requirements_txt = requirements_txt
        self.destination_dir = destination_dir
        self.docker_config = docker_config
        self.called_from_notebook = called_from_notebook
        self.project_id = gcp.get_project_name()

        # Those will be populated lazily.
        self.tar_file_path = None
        self.docker_client = None
        self.tar_file_descriptor = None
        self.docker_file_descriptor = None

    def get_docker_image(
        self, max_status_check_attempts=None, delay_between_status_checks=None
    ):
        """Builds, publishes and returns a Docker image.

        Args:
            max_status_check_attempts: Maximum number of times allowed to check
                build status. Applicable only to cloud container builder.
            delay_between_status_checks: Time is seconds to wait between status
                checks.
        """
        raise NotImplementedError

    def get_generated_files(self, return_descriptors=False):
        """Get generated file paths and/or descriptors for generated files.

        Args:
            return_descriptors: Whether to return descriptors as well.

        Returns:
          Docker and tar file paths. Depending on return_descriptors, possibly
          their file descriptors as well.
        """
        if return_descriptors:
            return [
                (self.docker_file_path, self.docker_file_descriptor),
                (self.tar_file_path, self.tar_file_descriptor)
            ]
        else:
            return [self.docker_file_path, self.tar_file_path]

    def _get_tar_file_path(self):
        """Packages files into a tarball."""
        self._create_docker_file()
        file_path_map = self._get_file_path_map()

        self.tar_file_descriptor, self.tar_file_path = tempfile.mkstemp()
        with tarfile.open(self.tar_file_path, "w:gz", dereference=True) as tar:
            for source, destination in file_path_map.items():
                tar.add(source, arcname=destination)

    def _get_docker_base_image(self):
        """Returns the docker image to be used as the default base image."""
        # If in a Kaggle environment, use the `KAGGLE_DOCKER_IMAGE` as the base
        # image.
        img = os.getenv("KAGGLE_DOCKER_IMAGE")
        if img:
            return img

        tf_version = tf_utils.get_version()
        if tf_version is not None:
            # Updating the name for RC's to match with the TF generated
            # RC Docker image names.
            tf_version = tf_version.replace("-rc", "rc")
            # Get the TF Docker parent image to use based on the current
            # TF version.
            img = "tensorflow/tensorflow:{}".format(tf_version)
            if (self.chief_config.accelerator_type !=
                machine_config.AcceleratorType.NO_ACCELERATOR):
                img += "-gpu"

            # Add python 3 tag for TF version <= 2.1.0
            # https://hub.docker.com/r/tensorflow/tensorflow
            v = tf_version.split(".")
            if float(v[0] + "." + v[1]) <= 2.1:
                img += "-py3"

        # Use the latest TF docker image if a local installation is not
        # available or if the docker image corresponding to the `tf_version`
        # does not exist.
        if not (img and self._image_exists(img)):
            warnings.warn(
                "TF Cloud `run` API uses docker, with a TF parent image "
                "matching your local TF version, for containerizing your "
                "code. A TF Docker image does not exist for the TF version "
                "you are using: {}"
                "We are replacing this with the latest stable TF Docker "
                "image available: `tensorflow/tensorflow:latest`"
                "Please see "
                "https://hub.docker.com/r/tensorflow/tensorflow/ "
                "for details on the available Docker images."
                "If you are seeing any code compatibility issues because of"
                " the TF version change, please try using a custom "
                "`docker_config.parent_image` with the required "
                "TF version.".format(tf_version))
            new_img = "tensorflow/tensorflow:latest"
            if img and img.endswith("-gpu"):
                new_img += "-gpu"
            img = new_img
        return img

    def _create_docker_file(self):
        """Creates a Dockerfile."""
        if self.docker_config:
          parent_image = self.docker_config.parent_image
        else:
          parent_image = None
        if parent_image is None:
            parent_image = self._get_docker_base_image()

        lines = [
            "FROM {}".format(parent_image),
            "WORKDIR {}".format(self.destination_dir),
        ]

        if self.requirements_txt is not None:
            _, requirements_txt_name = os.path.split(self.requirements_txt)
            dst_requirements_txt = os.path.join(requirements_txt_name)
            requirements_txt_path = os.path.join(
                self.destination_dir, requirements_txt_name
            )
            lines.append(
                "COPY {requirements_txt} {requirements_txt}".format(
                    requirements_txt=requirements_txt_path)
            )
            # install pip requirements from requirements_txt if it exists.
            lines.append(
                "RUN if [ -e {requirements_txt} ]; "
                "then pip install --no-cache -r {requirements_txt}; "
                "fi".format(requirements_txt=dst_requirements_txt)
            )
        if self.entry_point is None:
            lines.append("RUN pip install tensorflow-cloud")

        if self.worker_config is not None and machine_config.is_tpu_config(
            self.worker_config
        ):
            lines.append("RUN pip install cloud-tpu-client")

        # Copies the files from the `destination_dir` in Docker daemon location
        # to the `destination_dir` in Docker container filesystem.
        lines.append("COPY {} {}".format(self.destination_dir,
                                         self.destination_dir))

        docker_entry_point = self.preprocessed_entry_point or self.entry_point
        _, docker_entry_point_file_name = os.path.split(docker_entry_point)

        # Using `ENTRYPOINT` here instead of `CMD` specifically because
        # we want to support passing user code flags.
        lines.extend(
            ['ENTRYPOINT ["python", "{}"]'.format(docker_entry_point_file_name)]
        )

        content = "\n".join(lines)
        self.docker_file_descriptor, self.docker_file_path = tempfile.mkstemp()
        with open(self.docker_file_path, "w") as f:
            f.write(content)

    def _image_exists(self, image):
        """Checks whether the image exists on dockerhub using Docker v2 api.

        Args:
            image: image to check for.

        Returns:
            True if the image is found on dockerhub.
        """
        repo_name, tag_name = image.split(":")
        r = requests.get(
            "http://hub.docker.com/v2/repositories/{}/tags/{}".format(
                repo_name, tag_name
            )
        )
        return r.ok

    def _get_file_path_map(self):
        """Maps local file paths to the Docker daemon process location.

        Dictionary mapping file paths in the local file system to the paths
        in the Docker daemon process location. The `key` or source is the path
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
            if not entry_point_dir:  # Current directory
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

        # Place Docker file in the root directory.
        location_map[self.docker_file_path] = "Dockerfile"
        return location_map


class LocalContainerBuilder(ContainerBuilder):
    """Container builder that uses local Docker daemon process."""

    def get_docker_image(
        self, max_status_check_attempts=None, delay_between_status_checks=None
    ):
        """Builds, publishes and returns a Docker image.

        Args:
            max_status_check_attempts: Maximum number of times allowed to check
                build status. Not applicable to this builder.
            delay_between_status_checks: Time is seconds to wait between status
                checks. Not applicable to this builder.
        Returns:
            URI in a registory where the Docker image has been built and pushed.
        """
        self.docker_client = docker.APIClient(version="auto")
        self._get_tar_file_path()

        # create Docker image from tarball
        image_uri = self._build_docker_image()
        # push to the registry
        self._publish_docker_image(image_uri)
        return image_uri

    def _build_docker_image(self):
        """Builds Docker image.

        https://docker-py.readthedocs.io/en/stable/api.html#module-docker.api.build

        Returns:
            Image URI.
        """
        # Use the given Docker image given, if available.
        if self.docker_config:
            image_uri = self.docker_config.image
        else:
            image_uri = None
        image_uri = image_uri or generate_image_uri()
        logger.info("Building Docker image: %s", image_uri)

        # `fileobj` is generally set to the Dockerfile file path. If a tar file
        # is used for Docker build context (ones that includes a Dockerfile)
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
        """Publishes Docker image.

        Args:
            image_uri: String, the registry name and tag.
        """
        logger.info("Publishing Docker image: %s", image_uri)
        pb_logs_generator = self.docker_client.push(
            image_uri, stream=True, decode=True)
        self._get_logs(pb_logs_generator, "publish", image_uri)

    def _get_logs(self, logs_generator, name, image_uri):
        """Decodes logs from Docker and generates user friendly logs.

        Args:
            logs_generator: Generator returned from Docker build/push APIs.
            name: String, 'build' or 'publish' used to identify where the
                generator came from.
            image_uri: String, the Docker image URI.

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
        self, max_status_check_attempts=40, delay_between_status_checks=30
    ):
        """Builds, publishes and returns a Docker image.

        Args:
            max_status_check_attempts: Maximum number of times allowed to check
                build status. Applicable only to cloud container builder.
            delay_between_status_checks: Time is seconds to wait between status
                checks.
        Returns:
            URI in a registory where the Docker image has been built and pushed.
        """
        self._get_tar_file_path()
        storage_object_name = self._upload_tar_to_gcs()
        # Use the given Docker image name, if available.
        if self.docker_config:
            image_uri = self.docker_config.image
        else:
            image_uri = None
        image_uri = image_uri or generate_image_uri()

        logger.info(
            "Building and publishing Docker image using Google Cloud Build: %s",
            image_uri)
        build_service = discovery.build(
            "cloudbuild",
            "v1",
            cache_discovery=False,
            requestBuilder=google_api_client.TFCloudHttpRequest,
        )
        request_dict = self._create_cloud_build_request_dict(
            image_uri,
            storage_object_name,
            max_status_check_attempts*delay_between_status_checks
        )

        try:
            # Call to queue request to build and push Docker image.
            print("Submitting Docker build and push request to Cloud Build.")
            print("Please access your Cloud Build job information here:")
            print("https://console.cloud.google.com/cloud-build/builds")
            create_response = (
                build_service.projects()
                .builds()
                .create(projectId=self.project_id, body=request_dict)
                .execute()
            )

            # `create` returns a long-running `Operation`.
            # https://cloud.google.com/cloud-build/docs/api/reference/rest/v1/operations#Operation  # pylint: disable=line-too-long
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
                # https://cloud.google.com/cloud-build/docs/api/reference/rest/v1/projects.builds#Build.Status    # pylint: disable=line-too-long
                status = get_response["status"]
                if status != "WORKING" and status != "QUEUED":
                    break

                attempts += 1
                # Wait for 30 seconds before we check on status again.
                print("Waiting for Cloud Build, checking status in 30 seconds.")
                time.sleep(delay_between_status_checks)
            if status != "SUCCESS":
                raise RuntimeError(
                    "There was an error executing the cloud build job. "
                    "Job status: " + status
                )

        except errors.HttpError as err:
            raise RuntimeError(
                "There was an error submitting the cloud build job. ", err)
        return image_uri

    def _upload_tar_to_gcs(self):
        """Uploads tarfile to GCS and returns the GCS object name."""
        logger.info("Uploading files to GCS.")
        storage_client = storage.Client()
        try:
            bucket = storage_client.get_bucket(
                self.docker_config.image_build_bucket)
        except NotFound:
            bucket = storage_client.create_bucket(
                self.docker_config.image_build_bucket)

        unique_tag = str(uuid.uuid4()).replace("-", "_")
        storage_object_name = "tf_cloud_train_tar_{}".format(unique_tag)

        blob = bucket.blob(storage_object_name)
        blob.upload_from_filename(self.tar_file_path)
        return storage_object_name

    def _create_cloud_build_request_dict(
        self, image_uri, storage_object_name, timeout_sec
    ):
        """Creates request body for cloud build JSON API call.

        `create` body should be a `Build` object
        https://cloud.google.com/cloud-build/docs/api/reference/rest/v1/projects.builds#Build

        Args:
            image_uri: GCR Docker image URI.
            storage_object_name: Name of the tarfile object in GCS.
            timeout_sec: timeout for the CloudBuild in seconds.

        Returns:
            Build request dictionary.
        """
        request_dict = {}
        request_dict["projectId"] = self.project_id
        request_dict["images"] = [[image_uri]]
        request_dict["steps"] = []
        request_dict["timeout"] = "{}s".format(timeout_sec)
        build_args = ["build", "-t", image_uri, "."]

        if self.docker_config:
            cache_from = (self.docker_config.cache_from or
                          self.docker_config.image)

        if cache_from:
            # Use the given Docker image as cache.
            request_dict["steps"].append({
                "name": "gcr.io/cloud-builders/docker",
                "entrypoint": "bash",
                "args": [
                    "-c",
                    "docker pull {} || exit 0".format(cache_from),
                ],
            })
            build_args[3:3] = ["--cache-from", cache_from]

        request_dict["steps"].append({
            "name": "gcr.io/cloud-builders/docker",
            "args": build_args,
        })
        request_dict["source"] = {
            "storageSource": {
                "bucket": self.docker_config.image_build_bucket,
                "object": storage_object_name,
            }
        }
        return request_dict
