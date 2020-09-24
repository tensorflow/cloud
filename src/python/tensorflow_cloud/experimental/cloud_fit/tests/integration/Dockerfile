# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Base image to use for this docker
ARG BASE_IMAGE

# Dockerfile
FROM "${BASE_IMAGE}"

# Path to tensorflow_cloud whl file
ARG WHEEL_FILE

# `tensorflow==${version}, if different version than that of the base image.
ARG TENSORFLOW_PIP_PACKAGE

WORKDIR /root

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Update TensorFlow if explicitly specified.
RUN if [ -n "${TENSORFLOW_PIP_PACKAGE}" ] ; then \
  # Uninstall already-installed TensorFlow
  pip3 show tensorflow-cpu > /dev/null 2>&1 && \
  pip3 uninstall -y tensorflow-cpu tensorflow ; \
  # We override TENSORFLOW_PIP_PACAKGE for pre-released version.
  pip3 install --pre --upgrade "${TENSORFLOW_PIP_PACKAGE}"  && \
  echo "TensorFlow re-installed with ${TENSORFLOW_PIP_PACKAGE}" || exit 1; \
fi

# Copy and install tensorflow_enterprise_addons wheel file
ADD "${WHEEL_FILE}" /tmp/
RUN pip3 install --upgrade /tmp/tensorflow_cloud-*.whl

# Sets up the entry point to invoke cloud_fit.
ENTRYPOINT ["python3","-m","tensorflow_cloud.experimental.cloud_fit.remote"]
