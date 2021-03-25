# Copyright 2020 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# LSQP examples

add_executable(dmr_lsqp_core_example
  ${CMAKE_CURRENT_SOURCE_DIR}/core_example.cc
)
target_link_libraries(dmr_lsqp_core_example
  absl::span
  absl::hash
  absl::hashtablez_sampler
  absl::strings
  absl::status
  absl::statusor
  dmr_support
  dmr_lsqp_core
  Eigen3::Eigen
)

add_executable(dmr_lsqp_common_example
  ${CMAKE_CURRENT_SOURCE_DIR}/common_example.cc
)
target_link_libraries(dmr_lsqp_common_example
  absl::span
  absl::hash
  absl::hashtablez_sampler
  absl::strings
  absl::status
  absl::statusor
  dmr_support
  dmr_lsqp_core
  dmr_lsqp_common
)
