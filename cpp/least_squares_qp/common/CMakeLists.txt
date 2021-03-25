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

# LSQP Common library

# Library
set(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/box_constraint.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/identity_constraint_union.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/identity_task.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/math_utils.cc
)
set(HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/common/box_constraint.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/common/identity_constraint.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/common/identity_constraint_union.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/common/identity_task.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/common/math_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/common/minimize_norm_task.h
)
add_library(dmr_lsqp_common)
target_sources(dmr_lsqp_common PUBLIC ${HDRS} PRIVATE ${SRCS})
target_include_directories(dmr_lsqp_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(dmr_lsqp_common
  PUBLIC
    absl::span
    absl::hash
    absl::hashtablez_sampler
    absl::strings
    absl::status
    absl::statusor
    dmr_support
    dmr_lsqp_core
  PRIVATE
    Eigen3::Eigen
)

# Tests
if(DM_ROBOTICS_BUILD_TESTS)
  add_executable(dmr_lsqp_common_box_constraint_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/box_constraint_test.cc
  )
  target_link_libraries(dmr_lsqp_common_box_constraint_test
    gtest
    gmock
    gtest_main
    dmr_support
    dmr_lsqp_core
    dmr_lsqp_common
    dmr_lsqp_testing
  )
  gtest_discover_tests(dmr_lsqp_common_box_constraint_test)

  add_executable(dmr_lsqp_common_identity_constraint_union_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/identity_constraint_union_test.cc
  )
  target_link_libraries(dmr_lsqp_common_identity_constraint_union_test
    gtest
    gmock
    gtest_main
    dmr_support
    dmr_lsqp_core
    dmr_lsqp_common
    dmr_lsqp_testing
  )
  gtest_discover_tests(dmr_lsqp_common_identity_constraint_union_test)

  add_executable(dmr_lsqp_common_identity_task_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/identity_task_test.cc
  )
  target_link_libraries(dmr_lsqp_common_identity_task_test
    gtest
    gmock
    gtest_main
    dmr_support
    dmr_lsqp_core
    dmr_lsqp_common
    dmr_lsqp_testing
  )
  gtest_discover_tests(dmr_lsqp_common_identity_task_test)

  add_executable(dmr_lsqp_common_minimize_norm_task_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimize_norm_task_test.cc
  )
  target_link_libraries(dmr_lsqp_common_minimize_norm_task_test
    gtest
    gmock
    gtest_main
    dmr_support
    dmr_lsqp_core
    dmr_lsqp_common
    dmr_lsqp_testing
  )
  gtest_discover_tests(dmr_lsqp_common_minimize_norm_task_test)

  add_executable(dmr_lsqp_common_integration_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_test.cc
  )
  target_compile_definitions(dmr_lsqp_common_integration_test
    PUBLIC EIGEN_RUNTIME_NO_MALLOC
  )
  target_link_libraries(dmr_lsqp_common_integration_test
    gtest
    gmock
    gtest_main
    dmr_support
    dmr_lsqp_core
    dmr_lsqp_common
    dmr_lsqp_testing
    Eigen3::Eigen
  )
  gtest_discover_tests(dmr_lsqp_common_integration_test)
endif()
