# Copyright 2020 DeepMind Technologies Limited.
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
project(dmr_controllers_py
  DESCRIPTION "DM Robotics: Controllers (Python)"
  VERSION 0.1
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Variables.
set(DMR_PYTHON_VERSION "3.8" CACHE STRING "Python version.")
set(PYTHONVERSION ${DMR_PYTHON_VERSION})
set(PYBIND11_PYTHON_VERSION ${PYTHONVERSION} CACHE STRING "")
set(PYBIND11_PYTHON_VERSION ${PYTHONVERSION})

add_definitions(-DPYBIND11_PYTHON_VERSION="${PYTHONVERSION}")
add_definitions(-DDM_ROBOTICS_USE_NEW_MUJOCO_BINDINGS=1)

if(NOT CMAKE_CONFIGURATION_TYPES)
  if(NOT CMAKE_BUILD_TYPE)
      message(STATUS "Setting build type to 'Release' as none was specified.")
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
  endif()
  set(BUILD_TYPES "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${BUILD_TYPES})
endif()

if((${CMAKE_CXX_COMPILER_ID} MATCHES "GNU") OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
  set(PROJECT_WARNING_FLAGS_DEBUG "")
  set(PROJECT_WARNING_FLAGS_RELEASE "")
  list(APPEND PROJECT_WARNING_FLAGS_DEBUG -Wall)
  list(APPEND PROJECT_WARNING_FLAGS_DEBUG -Wextra)
  list(APPEND PROJECT_WARNING_FLAGS_DEBUG -Woverloaded-virtual)
  list(APPEND PROJECT_WARNING_FLAGS_DEBUG -pedantic)
  # For now use the same properties.
  set(PROJECT_WARNING_FLAGS_RELEASE ${PROJECT_WARNING_FLAGS_DEBUG})
endif()

# Options.
option(DMR_CONTROLLERS_PY_USE_SYSTEM_absl
  "If ON, it will attempt to find `absl` as a system dependency. If OFF, it will fetch `absl` from the git repo if not found."
  OFF
)
option(DMR_CONTROLLERS_PY_USE_SYSTEM_pybind11
  "If ON, it will attempt to find `pybind11` as a system dependency. If OFF, it will fetch `pybind11` from the git repo if not found."
  OFF
)

# Absl
FindOrFetch(
  USE_SYSTEM_PACKAGE
    DMR_CONTROLLERS_USE_SYSTEM_absl
  PACKAGE_NAME
    absl
  LIBRARY_NAME
    abseil-cpp
  GIT_REPO
    https://github.com/abseil/abseil-cpp.git
  GIT_TAG
    origin/master
  TARGETS
    absl::span
    absl::hash
    absl::btree
    absl::flat_hash_map
    absl::hashtablez_sampler
    absl::strings
    absl::status
)

# Pybind11
FindOrFetch(
  USE_SYSTEM_PACKAGE
    DMR_CONTROLLERS_USE_SYSTEM_pybind11
  PACKAGE_NAME
    pybind11
  LIBRARY_NAME
    pybind11
  GIT_REPO
    https://github.com/pybind/pybind11.git
  GIT_TAG
    origin/master
  TARGETS
    pybind11::headers
)

# Python Bindings
set(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/cartesian_6d_to_joint_velocity_mapper.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind_utils.cc
)
pybind11_add_module(cartesian_6d_to_joint_velocity_mapper_pybind
  ${SRCS}
)
set_target_properties(cartesian_6d_to_joint_velocity_mapper_pybind
  PROPERTIES
    OUTPUT_NAME "cartesian_6d_to_joint_velocity_mapper"
)
target_include_directories(cartesian_6d_to_joint_velocity_mapper_pybind
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(cartesian_6d_to_joint_velocity_mapper_pybind
  PRIVATE
    absl::span
    absl::hash
    absl::btree
    absl::flat_hash_map
    absl::hashtablez_sampler
    absl::strings
    absl::status
    dmr_mujoco
    dmr_lsqp_core
    dmr_controllers_lsqp
)
