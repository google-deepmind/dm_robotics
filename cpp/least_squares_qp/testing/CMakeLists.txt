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

# LSQP Testing library

# Library
set(HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/testing/matchers.h
)
add_library(dmr_lsqp_testing)
set_target_properties(dmr_lsqp_testing PROPERTIES LINKER_LANGUAGE CXX)
target_sources(dmr_lsqp_testing PUBLIC ${HDRS})
target_include_directories(dmr_lsqp_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(dmr_lsqp_testing
  PUBLIC
    gtest
    gmock
    gtest_main
    absl::strings
    dmr_lsqp_core
)

