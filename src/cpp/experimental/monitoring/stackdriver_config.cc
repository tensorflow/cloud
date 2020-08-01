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

#include "monitoring/stackdriver_config.h"

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace monitoring {
namespace {

std::string StackdriverMetricsWhitelist() {
  std::string metrics_whitelist;
  TF_CHECK_OK(
      ReadStringFromEnvVar("TF_MONITORING_STACKDRIVER_METRICS_WHITELIST",
                           /* default */ "", &metrics_whitelist));
  return metrics_whitelist;
}

std::unordered_set<std::string> InitStackdriverMetricsWhitelist() {
  const std::string user_whitelist = StackdriverMetricsWhitelist();
  if (user_whitelist.empty()) {
    return {
        "/tensorflow/core/graph_optimization_usecs",
        "/tensorflow/core/graph_run_time_usecs_histogram",
        "/tensorflow/data/bytes_fetched",
        "/tensorflow/data/getnext_duration",
        "/tensorflow/data/getnext_period",
        "/tensorflow/data/optimization",
    };
  }
  const std::vector<std::string> metrics_whitelist =
      absl::StrSplit(user_whitelist, ',');
  return std::unordered_set<std::string>(metrics_whitelist.begin(),
                                         metrics_whitelist.end());
}

}  // namespace

/* static */ const StackdriverConfig* StackdriverConfig::Get() {
  static const StackdriverConfig* kConfig = new StackdriverConfig();
  return kConfig;
}

bool StackdriverConfig::IsWhitelisted(const std::string& metric_name) const {
  return metrics_whitelist_.find(metric_name) != metrics_whitelist_.end();
}

std::string StackdriverConfig::DebugString() const {
  return absl::StrCat("metrics_whitelist: ",
                      absl::StrJoin(metrics_whitelist_, ","));
}

StackdriverConfig::StackdriverConfig()
    : metrics_whitelist_(InitStackdriverMetricsWhitelist()) {}

}  // namespace monitoring
}  // namespace tensorflow
