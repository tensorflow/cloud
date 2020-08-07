# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for using googleapis targets as deps
"""

def gcp_monitoring_grpc_deps():
    """Cc grpc libraries for GCP monitoring APIs
    """
    return [
        "@com_google_googleapis//google/monitoring/v3:monitoring_cc_grpc",
    ]

def gcp_monitoring_proto_deps():
    """Cc proto libraries for GCP monitoring APIs
    """
    return [
        "@com_google_googleapis//google/api:distribution_cc_proto",
        "@com_google_googleapis//google/api:label_cc_proto",
        "@com_google_googleapis//google/api:metric_cc_proto",
        "@com_google_googleapis//google/monitoring/v3:monitoring_cc_proto",
    ]

def gcp_monitoring_proto_header_deps():
    """Cc proto headers for GCP monitoring APIs

      The headers are useful when the target depending on them cannot contain
    protobuf implementations, to avoid ODR violation when multiple shared
    objects are dynamically linked. When only static linking is used, the
    headers would be equivalent to the cc proto libraries.
    """
    return [dep + "_headers_only" for dep in gcp_monitoring_proto_deps()]
