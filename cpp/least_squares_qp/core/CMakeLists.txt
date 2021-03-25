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

# LSQP Core library

# Library
set(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lsqp_stack_of_tasks_solver.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lsqp_stack_of_tasks_solver_impl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lsqp_stack_of_tasks_solver_impl.cc
)
set(HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/core/lsqp_task.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/core/lsqp_constraint.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h
)
add_library(dmr_lsqp_core)
target_sources(dmr_lsqp_core PUBLIC ${HDRS} PRIVATE ${SRCS})
target_include_directories(dmr_lsqp_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(dmr_lsqp_core
  PUBLIC
    absl::span
    absl::hash
    absl::flat_hash_map
    absl::hashtablez_sampler
    absl::strings
    absl::status
    absl::statusor
    dmr_support
  PRIVATE
    Eigen3::Eigen
    osqp-cpp
)

# Tests
if(DM_ROBOTICS_BUILD_TESTS)
  add_executable(dmr_lsqp_core_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/lsqp_stack_of_tasks_solver_test.cc
  )
  target_compile_definitions(dmr_lsqp_core_test PUBLIC EIGEN_RUNTIME_NO_MALLOC)
  target_link_libraries(dmr_lsqp_core_test
    gtest
    gmock
    gtest_main
    dmr_lsqp_core
    Eigen3::Eigen
  )
  gtest_discover_tests(dmr_lsqp_core_test)
endif()
