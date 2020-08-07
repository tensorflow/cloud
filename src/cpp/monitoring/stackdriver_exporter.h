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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_STACKDRIVER_EXPORTER_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_STACKDRIVER_EXPORTER_H_

#include <memory>
#include <unordered_set>

#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace monitoring {

// An exporter which exports metrics to Stackdriver through the Stackdriver
// Client. Internally calls to the default CollectionRegistry to collect
// the metric descriptors and values.
//
class StackdriverExporter : public Exporter {
 public:
  // Idempotent. Thread safe.
  void PeriodicallyExportMetrics() override;

 private:
  void ExportMetrics() override;

  void ExportMetricDescriptors(const CollectedMetrics& collected_metrics);

  mutex mutex_;
  std::unique_ptr<serving::PeriodicFunction> periodic_function_
      TF_GUARDED_BY(mutex_);
  std::unordered_set<string> exported_metric_names_ TF_GUARDED_BY(mutex_);
};

}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_STACKDRIVER_EXPORTER_H_
