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

#include "dm_robotics/mujoco/test_with_mujoco_model.h"

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "dm_robotics/mujoco/defs.h"

namespace dm_robotics::testing {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;

TEST_F(TestWithMujocoModel, MjLibModelAndDataNotNull) {
  EXPECT_THAT(mjlib_, NotNull());
  EXPECT_THAT(model_, IsNull());
  EXPECT_THAT(data_, IsNull());

  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);
  EXPECT_THAT(model_, NotNull());
  EXPECT_THAT(data_, NotNull());
}

}  // namespace
}  // namespace dm_robotics::testing
