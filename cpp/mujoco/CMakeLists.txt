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
project(dmr_mujoco
  DESCRIPTION "DM Robotics: MuJoCo utilities library"
  VERSION 0.2
)

# Pick up the version of mujoco to use from the environment,
# with an appropriate default if unset.
set(MUJOCO_VERSION $ENV{MUJOCO_VERSION} CACHE STRING "Mujoco library version")
if(NOT MUJOCO_VERSION)
    set(MUJOCO_VERSION "3.3.3")
endif()
message ("MUJOCO_VERSION: ${MUJOCO_VERSION}")

# Generate defs.h from configuration with the MuJoCo constants.
set(DMR_DMCONTROL_SUITE_PATH "$ENV{HOME}/.local/lib/python3.10/site-packages/dm_control/suite" CACHE STRING "Path to `dm_control.suite`")
set(DMR_DMCONTROL_SUITE_HUMANOID_XML_PATH "${DMR_DMCONTROL_SUITE_PATH}/humanoid.xml")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/defs.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/dm_robotics/mujoco/defs.h
  NEWLINE_STYLE LF
)

# Library
set(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cc
)
set(HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/mujoco/types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/mujoco/utils.h
  ${CMAKE_CURRENT_BINARY_DIR}/include/dm_robotics/mujoco/defs.h
)
add_library(dmr_mujoco)
target_sources(dmr_mujoco PUBLIC ${HDRS} PRIVATE ${SRCS})
target_include_directories(dmr_mujoco
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)
target_link_libraries(dmr_mujoco
  PUBLIC
    absl::span
    absl::hash
    absl::btree
    absl::hashtablez_sampler
    absl::strings
    absl::status
    absl::statusor
    mujoco
  PRIVATE
    dmr_support
    Eigen3::Eigen
)


# Tests
if(DM_ROBOTICS_BUILD_TESTS)
  # Testing lib
  add_library(dmr_mujoco_testing)
  target_sources(dmr_mujoco_testing
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/mujoco/test_with_mujoco_model.h
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/test_with_mujoco_model.cc
  )
  target_include_directories(dmr_mujoco_testing
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(dmr_mujoco_testing
    PUBLIC
      gtest
      gmock
      dmr_support
      dmr_mujoco
  )

  # test_with_mujoco_model_test
  add_executable(dmr_mujoco_test_with_mujoco_model_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_with_mujoco_model_test.cc
  )
  target_link_libraries(dmr_mujoco_test_with_mujoco_model_test
    gtest
    gmock
    gtest_main
    dmr_support
    dmr_mujoco
    dmr_mujoco_testing
  )
  gtest_discover_tests(dmr_mujoco_test_with_mujoco_model_test)

  # utils_test
  add_executable(dmr_mujoco_utils_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/utils_test.cc
  )
  target_link_libraries(dmr_mujoco_utils_test
    gtest
    gmock
    gtest_main
    absl::random_random
    absl::node_hash_set
    absl::btree
    dmr_support
    dmr_mujoco
    dmr_mujoco_testing
    Eigen3::Eigen
  )
  gtest_discover_tests(dmr_mujoco_utils_test)

endif()
