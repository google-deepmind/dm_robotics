// Copyright 2020 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DM_ROBOTICS_MUJOCO_TEST_WITH_MJLIB_H_
#define DM_ROBOTICS_MUJOCO_TEST_WITH_MJLIB_H_

#include "gtest/gtest.h"
#include "dm_robotics/mujoco/mjlib.h"

namespace dm_robotics::testing {

// A base class for test fixtures that require MjLib.
// This class loads libmujoco.so when the test suite is set up,
// and destroys (unloads) it when the test suite is torn down.
class TestWithMjLib : public ::testing::Test {
 protected:
  static void SetUpTestSuite();
  static void TearDownTestSuite();
  static MjLib* mjlib_;
};

}  // namespace dm_robotics::testing

#endif  // DM_ROBOTICS_MUJOCO_TEST_WITH_MJLIB_H_
