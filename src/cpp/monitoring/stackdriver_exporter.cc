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

#include "monitoring/stackdriver_exporter.h"

#include "monitoring/stackdriver_client.h"
#include "monitoring/stackdriver_config.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace monitoring {

namespace {
constexpr int64 kIntervalMicros = 10 * 1000 * 1000;  // 10 seconds

// Decide whether to enable StackdriverExporter
bool StackdriverExporterEnabled() {
  bool is_enabled;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_MONITORING_STACKDRIVER_EXPORTER_ENABLED",
                                 /* default */ false, &is_enabled));
  return is_enabled;
}

bool ShouldExport(const PointSet& point_set) {
  if (!StackdriverConfig::Get()->IsWhitelisted(point_set.metric_name)) {
    return false;
  }
  size_t non_empty_points = 0;
  for (const auto& point : point_set.points) {
    switch (point->value_type) {
      case ValueType::kInt64:
        non_empty_points += point->int64_value != 0;
        break;
      case ValueType::kHistogram:
        non_empty_points += point->histogram_value.num();
        break;
      default:
        // Other value types are not supported yet
        break;
    }
  }
  return non_empty_points > 0;
}

std::map<string, std::unique_ptr<PointSet>> FilterPointSetMap(
    std::map<string, std::unique_ptr<PointSet>> point_set_map) {
  std::map<string, std::unique_ptr<PointSet>> solid_map;
  for (auto& kv : point_set_map) {
    if (ShouldExport(*kv.second)) {
      solid_map.emplace(kv.first, std::move(kv.second));
    }
  }
  return solid_map;
}

}  // namespace

void StackdriverExporter::PeriodicallyExportMetrics() {
  if (!StackdriverExporterEnabled()) {
    return;
  }
  mutex_lock lock(mutex_);
  if (periodic_function_) {
    return;
  }
  LOG(INFO) << "Start exporting metrics periodically every " << kIntervalMicros
            << " us. " << StackdriverConfig::Get()->DebugString();
  periodic_function_ = std::make_unique<serving::PeriodicFunction>(
      [this]() { ExportMetrics(); }, kIntervalMicros);
}

void StackdriverExporter::ExportMetrics() {
  CollectionRegistry::CollectMetricsOptions collect_options;
  auto collected_metrics =
      CollectionRegistry::Default()->CollectMetrics(collect_options);
  collected_metrics->point_set_map =
      FilterPointSetMap(std::move(collected_metrics->point_set_map));
  if (collected_metrics->point_set_map.empty()) {
    return;
  }
  ExportMetricDescriptors(*collected_metrics);
  const auto status = StackdriverClient::Get()->CreateTimeSeries(
      collected_metrics->point_set_map);
  if (!status.ok()) {
    LOG(ERROR) << "CreateTimeSeries error: ["
               << static_cast<int>(status.error_code()) << "]"
               << status.error_message();
  }
}

void StackdriverExporter::ExportMetricDescriptors(
    const CollectedMetrics& collected_metrics) {
  mutex_lock ml(mutex_);
  for (const auto& kv : collected_metrics.metric_descriptor_map) {
    const auto& name = kv.first;
    const auto& descriptor = kv.second;
    if (collected_metrics.point_set_map.find(name) !=
            collected_metrics.point_set_map.end() &&
        exported_metric_names_.find(name) == exported_metric_names_.end()) {
      const auto status =
          StackdriverClient::Get()->CreateMetricDescriptor(*descriptor);
      if (status.ok() ||
          status.error_code() == grpc::StatusCode::ALREADY_EXISTS) {
        exported_metric_names_.insert(name);
      } else {
        LOG(ERROR) << "CreateMetricDescriptor error: ["
                   << static_cast<int>(status.error_code()) << "]"
                   << status.error_message();
      }
    }
  }
}

REGISTER_TF_METRICS_EXPORTER(StackdriverExporter);

}  // namespace monitoring
}  // namespace tensorflow
