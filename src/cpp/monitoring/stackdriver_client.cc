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

#include "monitoring/stackdriver_client.h"

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "google/api/distribution.pb.h"
#include "google/api/label.pb.h"
#include "google/api/metric.pb.h"
#include "google/protobuf/timestamp.pb.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace monitoring {

namespace {

// Decide whether to enable StackdriverExporter
std::string GetStackdriverProjectIdFromEnv() {
  std::string project_id;
  TF_CHECK_OK(ReadStringFromEnvVar("TF_MONITORING_STACKDRIVER_PROJECT_ID",
                                   /* default */ "", &project_id));
  return project_id;
}

constexpr char kGoogleStackdriverStatsAddress[] = "monitoring.googleapis.com";
constexpr char kMetricTypePrefix[] = "custom.googleapis.com";
constexpr char kProjectNamePrefix[] = "projects/";
constexpr char kDefaultResourceType[] = "global";

std::unique_ptr<google::monitoring::v3::grpc::MetricService::StubInterface>
MakeMetricServiceStub() {
  grpc::ChannelArguments args;
  args.SetUserAgentPrefix("stackdriver_exporter");

  // The credential file path is configured by environment variable
  // GOOGLE_APPLICATION_CREDENTIALS
  auto credential = ::grpc::GoogleDefaultCredentials();
  auto channel = ::grpc::CreateCustomChannel(kGoogleStackdriverStatsAddress,
                                             credential, args);
  return google::monitoring::v3::grpc::MetricService::NewStub(channel);
}

void ConvertTimestamp(absl::Time ts, google::protobuf::Timestamp* proto) {
  const int64 sec = absl::ToUnixSeconds(ts);
  proto->set_seconds(sec);
  proto->set_nanos((ts - absl::FromUnixSeconds(sec)) / absl::Nanoseconds(1));
}

void ConvertDistribution(const HistogramProto& tf_histogram,
                         google::api::Distribution* distribution) {
  distribution->set_count(static_cast<int64>(tf_histogram.num()));

  // Set mean and stddev
  if (tf_histogram.num() != 0.0) {
    distribution->set_mean(tf_histogram.sum() / tf_histogram.num());
    distribution->set_sum_of_squared_deviation(
        (tf_histogram.sum_squares() * tf_histogram.num() -
         tf_histogram.sum() * tf_histogram.sum()) /
        tf_histogram.num());
  } else {
    distribution->set_mean(0.0);
    distribution->set_sum_of_squared_deviation(0.0);
  }

  // Set bucket limits. The last bucket limit from TF histogram shall be omitted
  const auto num_buckets = tf_histogram.bucket_limit_size();
  auto* output_bucket_limits =
      distribution->mutable_bucket_options()->mutable_explicit_buckets();
  for (int bucket_id = 0; bucket_id < num_buckets - 1; ++bucket_id) {
    auto bucket_limit = tf_histogram.bucket_limit(bucket_id);
    output_bucket_limits->add_bounds(bucket_limit);
  }

  // Set bucket counts.
  for (const auto& bucket_count : tf_histogram.bucket()) {
    distribution->add_bucket_counts(static_cast<int64>(bucket_count));
  }
}

void ConvertPoint(const Point& tf_point,
                  google::monitoring::v3::Point* stackdriver_point) {
  switch (tf_point.value_type) {
    case ValueType::kInt64:
      stackdriver_point->mutable_value()->set_int64_value(tf_point.int64_value);
      break;
    case ValueType::kString:
      stackdriver_point->mutable_value()->set_string_value(
          tf_point.string_value);
      break;
    case ValueType::kBool:
      stackdriver_point->mutable_value()->set_bool_value(tf_point.bool_value);
      break;
    case ValueType::kHistogram:
      ConvertDistribution(
          tf_point.histogram_value,
          stackdriver_point->mutable_value()->mutable_distribution_value());
      break;
    case ValueType::kPercentiles:
      break;
  }

  ConvertTimestamp(absl::FromUnixMillis(tf_point.end_timestamp_millis),
                   stackdriver_point->mutable_interval()->mutable_end_time());
}

void ConvertPointSet(const PointSet& point_set,
                     google::monitoring::v3::TimeSeries* time_series) {
  time_series->mutable_metric()->set_type(
      absl::StrCat(kMetricTypePrefix, point_set.metric_name));
  time_series->mutable_resource()->set_type(kDefaultResourceType);

  // Only keeps the first point for prototype
  if (!point_set.points.empty()) {
    ConvertPoint(*point_set.points.front(), time_series->add_points());
  }
}

void ConvertLabelDescriptor(const std::string& tf_label_name,
                            google::api::LabelDescriptor* stackdriver_label) {
  stackdriver_label->set_key(tf_label_name);
  stackdriver_label->set_value_type(google::api::LabelDescriptor::STRING);
  stackdriver_label->set_description("");
}

google::api::MetricDescriptor::MetricKind ConvertMetricKind(
    const MetricKind& tf_metric_kind) {
  switch (tf_metric_kind) {
    case MetricKind::kGauge:
      return google::api::MetricDescriptor::GAUGE;
    case MetricKind::kCumulative:
      return google::api::MetricDescriptor::CUMULATIVE;
  }
}

google::api::MetricDescriptor::ValueType ConvertValueType(
    const ValueType& tf_value_type) {
  switch (tf_value_type) {
    case ValueType::kInt64:
      return google::api::MetricDescriptor::INT64;
    case ValueType::kHistogram:
      TF_FALLTHROUGH_INTENDED;
    case ValueType::kPercentiles:
      return google::api::MetricDescriptor::DISTRIBUTION;
    case ValueType::kString:
      return google::api::MetricDescriptor::STRING;
    case ValueType::kBool:
      return google::api::MetricDescriptor::BOOL;
  }
}

void ConvertMetricDescriptor(
    const MetricDescriptor& tf_metric,
    google::api::MetricDescriptor* stackdriver_metric) {
  stackdriver_metric->set_name(tf_metric.name);
  stackdriver_metric->set_type(absl::StrCat(kMetricTypePrefix, tf_metric.name));
  stackdriver_metric->set_description(tf_metric.description);
  stackdriver_metric->clear_labels();
  for (const auto& tf_label_name : tf_metric.label_names) {
    ConvertLabelDescriptor(tf_label_name, stackdriver_metric->add_labels());
  }
  stackdriver_metric->set_metric_kind(ConvertMetricKind(tf_metric.metric_kind));
  stackdriver_metric->set_value_type(ConvertValueType(tf_metric.value_type));
}

}  // namespace

StackdriverClient::StackdriverClient(
    Options options,
    std::unique_ptr<google::monitoring::v3::grpc::MetricService::StubInterface>
        metric_service_stub)
    : options_(std::move(options)),
      metric_service_stub_(std::move(metric_service_stub)) {}

StackdriverClient::StackdriverClient(Options options)
    : options_(std::move(options)),
      metric_service_stub_(MakeMetricServiceStub()) {}

/* static */ StackdriverClient* StackdriverClient::Get() {
  static StackdriverClient* client = []() {
    StackdriverClient::Options options;
    options.project_id = GetStackdriverProjectIdFromEnv();
    return new StackdriverClient(std::move(options));
  }();
  return client;
}

grpc::Status StackdriverClient::CreateTimeSeries(
    const std::map<string, std::unique_ptr<PointSet>>& point_set_map) const {
  google::monitoring::v3::CreateTimeSeriesRequest request;
  request.set_name(absl::StrCat(kProjectNamePrefix, options_.project_id));

  for (const auto& kv : point_set_map) {
    ConvertPointSet(*kv.second, request.add_time_series());
  }

  if (request.time_series().empty()) {
    return grpc::Status(grpc::StatusCode::CANCELLED,
                        "CreateTimeSeriesRequest has no time series.");
  }

  grpc::ClientContext context;
  google::protobuf::Empty response;
  grpc::Status status =
      metric_service_stub_->CreateTimeSeries(&context, request, &response);
  return status;
}

grpc::Status StackdriverClient::CreateMetricDescriptor(
    const MetricDescriptor& metric_descriptor) const {
  google::monitoring::v3::CreateMetricDescriptorRequest request;
  request.set_name(absl::StrCat(kProjectNamePrefix, options_.project_id));

  ConvertMetricDescriptor(metric_descriptor,
                          request.mutable_metric_descriptor());

  grpc::ClientContext context;
  google::api::MetricDescriptor response;
  grpc::Status status = metric_service_stub_->CreateMetricDescriptor(
      &context, request, &response);
  return status;
}

}  // namespace monitoring
}  // namespace tensorflow
