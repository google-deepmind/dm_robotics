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
project(dmr_lsqp
  DESCRIPTION "DM Robotics: Least Squares Quadratic Optimization Library"
  VERSION 0.1
)

add_subdirectory(core)

if (DM_ROBOTICS_BUILD_TESTS)
  add_subdirectory(testing)
endif()

add_subdirectory(common)

if (DM_ROBOTICS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# TODO(josechenf): Add install/uninstall phase.
