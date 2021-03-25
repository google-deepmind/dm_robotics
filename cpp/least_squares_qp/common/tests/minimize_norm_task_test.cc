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

#include "dm_robotics/least_squares_qp/common/minimize_norm_task.h"

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "dm_robotics/least_squares_qp/common/math_utils.h"
#include "dm_robotics/least_squares_qp/testing/matchers.h"

namespace dm_robotics {
namespace {

using ::dm_robotics::testing::LsqpTaskDimensionsAreValid;
using ::testing::DoubleEq;
using ::testing::Each;
using ::testing::Pointwise;

TEST(MinimizeNormTaskTest,
     CoefficientMatrixAndBiasHaveValidDimensionsAndValues) {
  const int kNumDof = 777;
  const MinimizeNormTask task(kNumDof);

  EXPECT_THAT(task, LsqpTaskDimensionsAreValid());
  EXPECT_EQ(task.GetNumberOfDof(), kNumDof);
  EXPECT_THAT(task.GetBias(), Each(DoubleEq(0.0)));
  EXPECT_THAT(task.GetCoefficientMatrix(),
              Pointwise(DoubleEq(), MakeIdentityMatrix(kNumDof)));
}

}  // namespace
}  // namespace dm_robotics
