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

# DM Robotics LSQP Controllers

# Library
set(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cartesian_6d_to_joint_velocity_mapper.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cartesian_6d_velocity_task.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/collision_avoidance_constraint.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/joint_acceleration_constraint.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/joint_position_limit_constraint.cc
)
set(HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/controllers/lsqp/cartesian_6d_to_joint_velocity_mapper.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/controllers/lsqp/cartesian_6d_velocity_task.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/controllers/lsqp/collision_avoidance_constraint.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/controllers/lsqp/joint_acceleration_constraint.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/controllers/lsqp/joint_position_limit_constraint.h
)
add_library(dmr_controllers_lsqp)
target_sources(dmr_controllers_lsqp PUBLIC ${HDRS} PRIVATE ${SRCS})
target_include_directories(dmr_controllers_lsqp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(dmr_controllers_lsqp
  PUBLIC
    absl::span
    absl::hash
    absl::btree
    absl::hashtablez_sampler
    absl::strings
    absl::status
    absl::statusor
    dmr_support
    dmr_mujoco
    dmr_lsqp_core
    dmr_lsqp_common
  PRIVATE
    Eigen3::Eigen
)

# Tests
if(DM_ROBOTICS_BUILD_TESTS)
  # Testing lib
  add_library(dmr_controllers_lsqp_testing)
  target_sources(dmr_controllers_lsqp_testing
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include/dm_robotics/controllers/lsqp/test_utils.h
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/test_utils.cc
  )
  target_include_directories(dmr_controllers_lsqp_testing
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(dmr_controllers_lsqp_testing
    PUBLIC
      gtest
      gmock
      dmr_support
      dmr_mujoco
      dmr_controllers_lsqp
    PRIVATE
      Eigen3::Eigen
  )

  # Macro for creating test executables in dmr_controllers_lsqp.
  macro(AddDmrControllersLsqpTest)
    set(options DEF_EIGEN_RUNTIME_NO_MALLOC)
    set(one_value_args NAME SOURCE)
    set(multi_value_args)
    cmake_parse_arguments(_ARGS
      "${options}"
      "${one_value_args}"
      "${multi_value_args}"
      ${ARGN}
    )

    add_executable(${_ARGS_NAME} ${_ARGS_SOURCE})
    target_link_libraries(${_ARGS_NAME}
      gtest
      gmock
      gtest_main
      dmr_support
      dmr_mujoco_testing
      dmr_controllers_lsqp
      dmr_controllers_lsqp_testing
      Eigen3::Eigen
    )

    if(${_ARGS_DEF_EIGEN_RUNTIME_NO_MALLOC})
      target_compile_definitions(${_ARGS_NAME} PUBLIC EIGEN_RUNTIME_NO_MALLOC)
    endif()

    gtest_discover_tests(${_ARGS_NAME})
  endmacro()

  # Cartesian 6D to joint velocity mapper test.
  AddDmrControllersLsqpTest(
    NAME dmr_controllers_lsqp_cartesian_6d_to_joint_velocity_mapper_test
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cartesian_6d_to_joint_velocity_mapper_test.cc
  )

  # Cartesian 6D velocity task test.
  AddDmrControllersLsqpTest(
    DEF_EIGEN_RUNTIME_NO_MALLOC
    NAME dmr_controllers_lsqp_cartesian_6d_velocity_task_test
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cartesian_6d_velocity_task_test.cc
  )

  # Collision avoidance constraint test.
  AddDmrControllersLsqpTest(
    NAME dmr_controllers_lsqp_collision_avoidance_constraint_test
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/tests/collision_avoidance_constraint_test.cc
  )

  # Integration test.
  AddDmrControllersLsqpTest(
    DEF_EIGEN_RUNTIME_NO_MALLOC
    NAME dmr_controllers_lsqp_integration_test
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_test.cc
  )

  # Joint acceleration constraint test.
  AddDmrControllersLsqpTest(
    NAME dmr_controllers_lsqp_joint_acceleration_constraint_test
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/tests/joint_acceleration_constraint_test.cc
  )

  # Joint position limit constraint test.
  AddDmrControllersLsqpTest(
    NAME dmr_controllers_lsqp_joint_position_limit_constraint_test
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/tests/joint_position_limit_constraint_test.cc
  )
endif()
