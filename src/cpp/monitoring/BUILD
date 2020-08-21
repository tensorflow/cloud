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
load(
    "@org_tensorflow//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "build_config.bzl",
    "gcp_monitoring_grpc_deps",
    "gcp_monitoring_proto_deps",
    "gcp_monitoring_proto_header_deps",
)

cc_library(
    name = "stackdriver_client",
    srcs = [
        "stackdriver_client.cc",
    ],
    hdrs = [
        "stackdriver_client.h",
    ],
    deps = [
        "@com_google_protobuf//:protobuf_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
    ] + gcp_monitoring_grpc_deps() + gcp_monitoring_proto_header_deps(),
)

cc_test(
    name = "stackdriver_client_test",
    srcs = [
        "stackdriver_client_test.cc",
    ],
    deps = [
        ":stackdriver_client",
        "@com_google_googletest//:gtest_main",
    ] + gcp_monitoring_grpc_deps() + gcp_monitoring_proto_deps(),
)

cc_library(
    name = "stackdriver_config",
    srcs = [
        "stackdriver_config.cc",
    ],
    hdrs = [
        "stackdriver_config.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "stackdriver_exporter",
    srcs = [
        "stackdriver_exporter.cc",
    ],
    hdrs = [
        "stackdriver_exporter.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":stackdriver_client",
        ":stackdriver_config",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:periodic_function_dynamic",
    ],
)

cc_library(
    name = "stackdriver_exporter_protos",
    visibility = [
        "//visibility:public",
    ],
    deps = gcp_monitoring_proto_deps(),
)
