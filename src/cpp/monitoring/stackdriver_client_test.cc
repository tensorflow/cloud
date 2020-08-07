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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <list>

#include "google/monitoring/v3/metric_service_mock.grpc.pb.h"
#include "google/protobuf/util/json_util.h"

namespace tensorflow {
namespace monitoring {

using ::testing::_;
using ::testing::Return;

namespace {
inline void SetPointValue(int64 value, Point* point) {
  point->value_type = ValueType::kInt64;
  point->int64_value = value;
}

inline void SetPointValue(std::list<double> value, Point* point) {
  point->value_type = ValueType::kHistogram;
  HistogramProto& histogram = point->histogram_value;

  // 2 buckets: (-oo, 0] and (0, +oo)
  histogram.add_bucket_limit(0.0);
  histogram.add_bucket(0.0);
  histogram.add_bucket_limit(DBL_MAX);
  histogram.add_bucket(0.0);

  if (value.empty()) {
    return;
  }

  histogram.set_min(value.front());
  histogram.set_max(value.front());
  histogram.set_num(value.size());
  for (double sample : value) {
    histogram.set_min(std::min(histogram.min(), sample));
    histogram.set_max(std::max(histogram.max(), sample));
    histogram.set_sum(histogram.sum() + sample);
    histogram.set_sum_squares(histogram.sum_squares() + sample * sample);
    size_t bucket_id = (sample < 0.0) ? 0 : 1;
    histogram.set_bucket(bucket_id, histogram.bucket(bucket_id) + 1);
  }
}

template <typename T>
std::unique_ptr<PointSet> MakePointSet(std::string name, int64 time_millis,
                                       std::initializer_list<T> values) {
  auto point_set = std::make_unique<PointSet>();
  point_set->metric_name = std::move(name);
  for (auto& value : values) {
    auto point = std::make_unique<Point>();
    SetPointValue(std::move(value), point.get());
    point->end_timestamp_millis = time_millis;
    point_set->points.emplace_back(std::move(point));
  }
  return point_set;
}

MATCHER_P(EqualsRequest, expected, "The expected request does not match") {
  *result_listener << "as the expected is " << expected.DebugString();
  return expected.SerializeAsString() == arg.SerializeAsString();
}

}  // namespace

TEST(StackdriverClientTest, CreateTimeSeries) {
  std::map<string, std::unique_ptr<PointSet>> point_set_map;
  point_set_map["/metric_1"] = MakePointSet(
      "/metric_1",  // metric type is "custom.googleapis.com/metric_1"
      12345,        // timestamp is 12 seconds and 345,000,000 nanoseconds
      {33, 44});    // the first number 33 is exported as int64_value
  point_set_map["/metric_2"] = MakePointSet(
      "/metric_2",  // metric type is "custom.googleapis.com/metric_2"
      12345,        // timestamp is 12 seconds and 345,000,000 nanoseconds
      {std::list<double>{-1.0, 1.0}});  // one histogram containing {-1.0, 1.0}

  const std::string expected_request_json = R"(
  {
    "name": "projects/test_project",
    "time_series": [
      {
        "metric": {
          "type": "custom.googleapis.com/metric_1",
        },
        "resource": {
          "type": "global"
        },
        "points": {
          "interval": {
            "end_time": {
              "seconds": 12,
              "nanos": 345000000,
            },
          },
          "value": {
            "int64_value": 33,
          },
        },
      },
      {
        "metric": {
          "type": "custom.googleapis.com/metric_2",
        },
        "resource": {
          "type": "global"
        },
        "points": {
          "interval": {
            "end_time": {
              "seconds": 12,
              "nanos": 345000000,
            },
          },
          "value": {
            "distribution_value": {
              "count": 2,
              "mean": 0.0,
              "sum_of_squared_deviation": 2.0,
              "bucket_options": {
                "explicit_buckets": {
                  "bounds": [
                    0.0,
                  ],
                },
              },
              "bucket_counts": [
                1,
                1,
              ],
            }
          },
        },
      },
    ],
  }
  )";
  google::monitoring::v3::CreateTimeSeriesRequest expected_request;
  auto status = google::protobuf::util::JsonStringToMessage(
      expected_request_json, &expected_request);
  ASSERT_TRUE(status.ok()) << status.error_message();

  auto mock_metric_service =
      std::make_unique<google::monitoring::v3::grpc::MockMetricServiceStub>();
  EXPECT_CALL(*mock_metric_service,
              CreateTimeSeries(_, EqualsRequest(expected_request), _))
      .WillOnce(Return(::grpc::Status::OK));
  StackdriverClient client({"test_project"}, std::move(mock_metric_service));
  EXPECT_OK(client.CreateTimeSeries(point_set_map));
}

TEST(StackdriverClientTest, CreateMetricDescriptor) {
  MetricDescriptor metric_descriptor{
      .name = "metric_1",
      .description = "desc",
      .label_names = {"a", "b"},
      .metric_kind = MetricKind::kCumulative,
      .value_type = ValueType::kInt64,
  };

  const std::string expected_request_json = R"(
  {
    name: "projects/test_project",
    metric_descriptor: {
      name: "metric_1",
      type: "custom.googleapis.commetric_1",
      description: "desc",
      labels: [
        {
          key: "a"
        },
        {
          key: "b"
        }
      ],
      metric_kind: 3,
      value_type: 2,
    },
  }
  )";
  google::monitoring::v3::CreateMetricDescriptorRequest expected_request;
  auto status = google::protobuf::util::JsonStringToMessage(
      expected_request_json, &expected_request);
  ASSERT_TRUE(status.ok()) << status.error_message();

  auto mock_metric_service =
      std::make_unique<google::monitoring::v3::grpc::MockMetricServiceStub>();
  EXPECT_CALL(*mock_metric_service,
              CreateMetricDescriptor(_, EqualsRequest(expected_request), _))
      .WillOnce(Return(::grpc::Status::OK));
  StackdriverClient client({"test_project"}, std::move(mock_metric_service));
  EXPECT_OK(client.CreateMetricDescriptor(metric_descriptor));
}

}  // namespace monitoring
}  // namespace tensorflow
