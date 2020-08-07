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

#ifndef MONITORING_STACKDRIVER_CONFIG_H_
#define MONITORING_STACKDRIVER_CONFIG_H_

#include <unordered_set>

namespace tensorflow {
namespace monitoring {

class StackdriverConfig {
 public:
  static const StackdriverConfig* Get();

  bool IsWhitelisted(const std::string& metric_name) const;

  std::string DebugString() const;

 private:
  StackdriverConfig();

  const std::unordered_set<std::string> metrics_whitelist_;
};
}  // namespace monitoring
}  // namespace tensorflow

#endif  // MONITORING_STACKDRIVER_CONFIG_H_
