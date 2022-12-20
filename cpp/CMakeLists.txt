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
project(dm_robotics
  DESCRIPTION "DM Robotics"
  VERSION 0.1
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# To build a manylinux compatible wheel:
# set DM_ROBOTICS_BUILD_WHEEL True
# set DM_ROBOTICS_VERSION_SCRIPT to a version script for gcc.
if( DM_ROBOTICS_BUILD_WHEEL )
  # Flags and setup to build manylinux wheels.
  add_link_options("-static-libgcc")
  if( DM_ROBOTICS_VERSION_SCRIPT )
    set(CMAKE_SHARED_LINKER_FLAGS
        ${CMAKE_SHARED_LINKER_FLAGS}
        "-Wl,--version-script=${DM_ROBOTICS_VERSION_SCRIPT},--no-whole-archive")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS
        ${CMAKE_SHARED_LINKER_FLAGS}
        "-Wl,--no-whole-archive")
  endif()

  # We also use a glibc_shim, but if your glibc is manylinux compatible you
  # won't need to make such a shim.
  if(EXISTS "/usr/lib/x86_64-linux-gnu/libglibc_shim.a")
    link_libraries("-l:libstdc++.a" "-lglibc_shim")
  else()
    link_libraries("-l:libstdc++.a")
  endif()

endif()

set(DMR_PYTHON_VERSION "3.10" CACHE STRING "Python version.")
set(PYTHONVERSION ${DMR_PYTHON_VERSION})
add_definitions(-DPYBIND11_PYTHON_VERSION="${PYTHONVERSION}")
set(PYBIND11_PYTHON_VERSION ${PYTHONVERSION} CACHE STRING "")
set(PYBIND11_PYTHON_VERSION ${PYTHONVERSION})

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

# Get CMake scripts.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(FindOrFetch)

# Build options.
option(DM_ROBOTICS_BUILD_dmr_support
  "Whether to build the `dmr_support` library." ON
)
option(DM_ROBOTICS_BUILD_dmr_lsqp
  "Whether to build the `dmr_lsqp` library." ON
)
option(DM_ROBOTICS_BUILD_dmr_mujoco
  "Whether to build the `dmr_mujoco` library." ON
)
option(DM_ROBOTICS_BUILD_dmr_controllers
  "Whether to build the `dmr_controllers` library." ON
)
option(DM_ROBOTICS_BUILD_dmr_controllers_python
  "Whether to build Python bindings for the `dmr_controllers` library." ON
)
option(DM_ROBOTICS_BUILD_TESTS
  "Whether to build `dm_robotics` tests." ON
)
option(DM_ROBOTICS_BUILD_EXAMPLES
  "Whether to build `dm_robotics` examples." ON
)

# Dependency options.
option(DM_ROBOTICS_USE_SYSTEM_absl
  "If ON, it will attempt to find `absl` as a system dependency. If OFF, it will fetch `absl` from the git repo if not found."
  OFF
)
option(DM_ROBOTICS_USE_SYSTEM_Eigen3
  "If ON, it will attempt to find `Eigen3` as a system dependency. If OFF, it will fetch `Eigen3` from the git repo if not found."
  OFF
)
option(DM_ROBOTICS_USE_SYSTEM_GTest
  "If ON, it will attempt to find `GTest` as a system dependency. If OFF, it will fetch `GTest` from the git repo if not found."
  OFF
)
option(DM_ROBOTICS_USE_SYSTEM_osqp-cpp
  "If ON, it will attempt to find `osqp-cpp` as a system dependency. If OFF, it will fetch `osqp-cpp` from the git repo if not found."
  OFF
)

# Absl
if (DM_ROBOTICS_BUILD_dmr_support
  OR DM_ROBOTICS_BUILD_dmr_lsqp
  OR DM_ROBOTICS_BUILD_dmr_mujoco
  OR DM_ROBOTICS_BUILD_dmr_controllers
  OR DM_ROBOTICS_BUILD_dmr_controllers_python
)
  FindOrFetch(
    USE_SYSTEM_PACKAGE
      DM_ROBOTICS_USE_SYSTEM_absl
    PACKAGE_NAME
      absl
    LIBRARY_NAME
      abseil-cpp
    GIT_REPO
      https://github.com/abseil/abseil-cpp.git
    GIT_TAG
      lts_2021_11_02
    TARGETS
      absl::span
      absl::hash
      absl::btree
      absl::flat_hash_map
      absl::hashtablez_sampler
      absl::strings
      absl::status
      absl::statusor
  )
endif()


# Eigen3
if (DM_ROBOTICS_BUILD_dmr_lsqp
  OR DM_ROBOTICS_BUILD_dmr_mujoco
  OR DM_ROBOTICS_BUILD_dmr_controllers
  OR DM_ROBOTICS_BUILD_dmr_controllers_python
)
  # Deactivate Eigen3 tests.
  option(BUILD_TESTING OFF)
  FindOrFetch(
    USE_SYSTEM_PACKAGE
      DM_ROBOTICS_USE_SYSTEM_Eigen3
    PACKAGE_NAME
      Eigen3
    LIBRARY_NAME
      eigen
    GIT_REPO
      https://gitlab.com/libeigen/eigen
    GIT_TAG
      3.4.0
    TARGETS
      Eigen3::Eigen
  )
endif()

# osqp-cpp
if (DM_ROBOTICS_BUILD_dmr_lsqp)
  # Deactivate OSQP tests.
  option(UNITTESTS OFF)
  option(OSQP-CPP_BUILD_TESTS OFF)
  FindOrFetch(
    USE_SYSTEM_PACKAGE
      DM_ROBOTICS_USE_SYSTEM_osqp-cpp
    PACKAGE_NAME
      osqp-cpp
    LIBRARY_NAME
      osqp-cpp
    GIT_REPO
      https://github.com/google/osqp-cpp.git
    GIT_TAG
      8cd904e2b49c24dd41d11f8c6e0adb113dd5e26c
    TARGETS
      osqp-cpp
  )
endif()


if (DM_ROBOTICS_BUILD_TESTS)
  enable_testing()
  FindOrFetch(
    USE_SYSTEM_PACKAGE
      DM_ROBOTICS_USE_SYSTEM_GTest
    PACKAGE_NAME
      GTest
    LIBRARY_NAME
      googletest
    GIT_REPO
      https://github.com/google/googletest.git
    GIT_TAG
      release-1.11.0
    TARGETS
      gtest
      gmock
      gtest_main
  )
  include(GoogleTest)
endif()

if (DM_ROBOTICS_BUILD_dmr_support)
add_subdirectory(support)
endif()

if (DM_ROBOTICS_BUILD_dmr_lsqp)
add_subdirectory(least_squares_qp)
endif()

if (DM_ROBOTICS_BUILD_dmr_mujoco)
add_subdirectory(mujoco)
endif()

if (DM_ROBOTICS_BUILD_dmr_controllers)
add_subdirectory(controllers)
endif()

if (DM_ROBOTICS_BUILD_dmr_controllers_python)
add_subdirectory(controllers_py)
endif()
