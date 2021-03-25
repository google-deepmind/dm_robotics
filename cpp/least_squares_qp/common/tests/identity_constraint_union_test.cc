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

#include "dm_robotics/least_squares_qp/common/identity_constraint_union.h"

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "dm_robotics/least_squares_qp/common/box_constraint.h"
#include "dm_robotics/least_squares_qp/common/math_utils.h"
#include "dm_robotics/least_squares_qp/testing/matchers.h"

namespace dm_robotics {
namespace {

using ::dm_robotics::testing::LsqpConstraintDimensionsAreValid;
using ::testing::DoubleEq;
using ::testing::Pointwise;

TEST(IdentityConstraintUnionTest,
     FailsToUpdateFeasibleSpaceWithConflictingConstraints) {
  BoxConstraint box_constraint_one({-4, -3, -2, -1, 0, 1, 2},
                                   {4, 5, 6, 7, 8, 9, 10});
  BoxConstraint box_constraint_two({-8, -1, -4, 2, 9, -5, 1},
                                   {-2, -1, 6, 10, 10, 10, 11});

  IdentityConstraintUnion constraint_union(box_constraint_one.GetNumberOfDof());
  EXPECT_THAT(constraint_union, LsqpConstraintDimensionsAreValid());
  EXPECT_EQ(constraint_union
                .UpdateFeasibleSpace({&box_constraint_one, &box_constraint_two})
                .code(),
            absl::StatusCode::kNotFound);
  EXPECT_THAT(constraint_union, LsqpConstraintDimensionsAreValid());
}

TEST(IdentityConstraintUnionTest, MinimumFeasibleBoundsAreCorrect) {
  BoxConstraint box_constraint_one({-4, -3, -2, -1, 0, 1, 2},
                                   {4, 5, 6, 7, 8, 9, 10});
  BoxConstraint box_constraint_two({-8, -1, -4, 2, -1, -5, 1},
                                   {-2, -1, 6, 10, 7, 10, 11});
  IdentityConstraintUnion constraint_union(box_constraint_one.GetNumberOfDof());

  EXPECT_THAT(constraint_union, LsqpConstraintDimensionsAreValid());
  EXPECT_OK(constraint_union.UpdateFeasibleSpace(
      {&box_constraint_one, &box_constraint_two}));
  EXPECT_THAT(constraint_union, LsqpConstraintDimensionsAreValid());

  EXPECT_THAT(constraint_union.GetNumberOfDof(),
              box_constraint_one.GetNumberOfDof());
  EXPECT_THAT(constraint_union.GetBoundsLength(),
              box_constraint_one.GetBoundsLength());
  EXPECT_THAT(constraint_union.GetLowerBound(),
              Pointwise(DoubleEq(), {-4, -1, -2, 2, 0, 1, 2}));
  EXPECT_THAT(constraint_union.GetUpperBound(),
              Pointwise(DoubleEq(), {-2, -1, 6, 7, 7, 9, 10}));
  EXPECT_THAT(constraint_union.GetCoefficientMatrix(),
              Pointwise(DoubleEq(), MakeIdentityMatrix(
                                        box_constraint_one.GetNumberOfDof())));
}

}  // namespace
}  // namespace dm_robotics
