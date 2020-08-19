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
"""Utilities for Google API client."""

from .. import version

from googleapiclient import http as googleapiclient_http


_USER_AGENT_FOR_TF_CLOUD_TRACKING = "tf-cloud/" + version.__version__


class TFCloudHttpRequest(googleapiclient_http.HttpRequest):
    """HttpRequest builder that sets a customized user-agent header for TF Cloud.

    This is used to track the usage of the TF Cloud.
    """

    def __init__(self, *args, **kwargs):
        """Construct a HttpRequest.

        Args:
            *args: Positional arguments to pass to the base class constructor.
            **kwargs: Keyword arguments to pass to the base class constructor.
        """
        headers = kwargs.setdefault("headers", {})
        headers["user-agent"] = _USER_AGENT_FOR_TF_CLOUD_TRACKING
        super(TFCloudHttpRequest, self).__init__(*args, **kwargs)
