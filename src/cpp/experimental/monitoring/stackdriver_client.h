/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_STACKDRIVER_CLIENT_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_STACKDRIVER_CLIENT_H_

#include <memory>
#include <string>

#include "google/monitoring/v3/metric_service.grpc.pb.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace monitoring {

// A client which send metrics descriptors and values to Stackdriver APIs.
//
class StackdriverClient {
 public:
  struct Options {
    std::string project_id;
  };

  StackdriverClient(
      Options options,
      std::unique_ptr<
          google::monitoring::v3::grpc::MetricService::StubInterface>
          metric_service_stub);

  static StackdriverClient* Get();

  grpc::Status CreateTimeSeries(
      const std::map<string, std::unique_ptr<PointSet>>& point_set_map) const;

  grpc::Status CreateMetricDescriptor(
      const MetricDescriptor& metric_descriptor) const;

 private:
  explicit StackdriverClient(Options options);

  const Options options_;
  std::unique_ptr<google::monitoring::v3::grpc::MetricService::StubInterface>
      metric_service_stub_;
};

}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_STACKDRIVER_CLIENT_H_
