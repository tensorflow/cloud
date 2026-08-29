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
"""Constants definitions for tuner sub module."""

# By default, the Tuner worker(s) always requests one trial at a time because
# we would parallelize the tuning loop themselves as opposed to getting multiple
# trial suggestions in one tuning loop.
SUGGESTION_COUNT_PER_REQUEST = 1

# Number of tries to retry getting study if it was already created
MAX_NUM_TRIES_FOR_STUDIES = 3
