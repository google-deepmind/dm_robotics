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

#include "dm_robotics/least_squares_qp/common/box_constraint.h"

#include <limits>
#include <vector>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "dm_robotics/least_squares_qp/common/math_utils.h"
#include "dm_robotics/least_squares_qp/testing/matchers.h"

namespace dm_robotics {
namespace {

using ::dm_robotics::testing::LsqpConstraintDimensionsAreValid;
using ::testing::DoubleEq;
using ::testing::Pointwise;

TEST(BoxConstraintTest,
     CoefficientMatrixAndBoundsHaveValidDimensionsAndValues) {
  const std::vector<double> kUpperBound{
      0, -2, 1, 2, 3, std::numeric_limits<double>::infinity(), -1};
  const std::vector<double> kLowerBound{
      -std::numeric_limits<double>::infinity(), -2, -1, 0, 1, 2, -3};
  const BoxConstraint constraint(kLowerBound, kUpperBound);

  EXPECT_THAT(constraint, LsqpConstraintDimensionsAreValid());
  EXPECT_THAT(constraint.GetUpperBound(), Pointwise(DoubleEq(), kUpperBound));
  EXPECT_THAT(constraint.GetLowerBound(), Pointwise(DoubleEq(), kLowerBound));
  EXPECT_THAT(constraint.GetCoefficientMatrix(),
              Pointwise(DoubleEq(), MakeIdentityMatrix(kLowerBound.size())));
}

}  // namespace
}  // namespace dm_robotics
