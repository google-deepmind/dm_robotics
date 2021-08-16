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

cmake_minimum_required(VERSION 3.5)
project(dmr_support
  DESCRIPTION "DM Robotics: Infrastructure support library"
  VERSION 0.1
)

# This is a header-only library.
set(HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/support/logging.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/support/status_macros.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/support/status-matchers.h
)
add_library(dmr_support INTERFACE)
target_sources(dmr_support INTERFACE ${HDRS})
target_include_directories(dmr_support
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(dmr_support
  INTERFACE
    absl::status
    absl::statusor
)
