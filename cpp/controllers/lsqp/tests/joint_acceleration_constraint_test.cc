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

#include "dm_robotics/controllers/lsqp/joint_acceleration_constraint.h"

#include <ostream>
#include <vector>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "dm_robotics/least_squares_qp/common/math_utils.h"
#include "dm_robotics/least_squares_qp/testing/matchers.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/test_with_mujoco_model.h"

namespace dm_robotics {
namespace {

using JointAccelerationConstraintTest =
    ::dm_robotics::testing::TestWithMujocoModel;
using ::dm_robotics::testing::LsqpConstraintDimensionsAreValid;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Pointwise;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

struct TestParams {
  absl::Duration integration_timestep;
  absl::btree_set<int> joint_ids;
  std::vector<double> acceleration_magnitude_limits;
  std::vector<double> qvel_curr;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const TestParams& param) {
    return stream << "(integration_timestep["
                  << absl::ToDoubleSeconds(param.integration_timestep)
                  << " s], joint_ids["
                  << absl::StrJoin(param.joint_ids.begin(),
                                   param.joint_ids.end(), ", ")
                  << "], acceleration_magnitude_limits["
                  << absl::StrJoin(param.acceleration_magnitude_limits.begin(),
                                   param.acceleration_magnitude_limits.end(),
                                   ", ")
                  << "], qvel_curr["
                  << absl::StrJoin(param.qvel_curr.begin(),
                                   param.qvel_curr.end(), ", ")
                  << "])";
  }
};

// Helper function and struct for computing the expected lower and upper
// bounds of a JointAccelerationConstraint based on TestParams.
struct LowerAndUpperBounds {
  std::vector<double> lower_bound;
  std::vector<double> upper_bound;
};
LowerAndUpperBounds ComputeExpectedLowerAndUpperBounds(
    const TestParams& params) {
  LowerAndUpperBounds bounds;
  bounds.lower_bound.resize(params.joint_ids.size());
  bounds.upper_bound.resize(params.joint_ids.size());

  // The upper (lower) bounds are the joint velocities obtained accelerating
  // (decelerating) by the acceleration limit for `params.integration_timestep`.
  for (int i = 0; i < params.joint_ids.size(); ++i) {
    bounds.lower_bound[i] =
        params.qvel_curr[i] -
        params.acceleration_magnitude_limits[i] *
            absl::ToDoubleSeconds(params.integration_timestep);
    bounds.upper_bound[i] =
        params.qvel_curr[i] +
        params.acceleration_magnitude_limits[i] *
            absl::ToDoubleSeconds(params.integration_timestep);
  }
  return bounds;
}

// Test fixture for parameterized tests on a variety of MuJoCo joints.
// Note: Only 1 DoF joints are used.
class ParameterizedJointAccelerationConstraintTest
    : public WithParamInterface<TestParams>,
      public JointAccelerationConstraintTest {};
INSTANTIATE_TEST_SUITE_P(
    ParameterizedJointAccelerationConstraintTests,
    ParameterizedJointAccelerationConstraintTest,
    ValuesIn({TestParams{.integration_timestep = absl::Seconds(1.0),
                         .joint_ids = {16, 17, 18},
                         .acceleration_magnitude_limits = {1.0, 2.0, 3.0},
                         .qvel_curr = {-1.3, 0.0, 2.6}},
              TestParams{.integration_timestep = absl::Seconds(2.0),
                         .joint_ids = {19, 20, 21},
                         .acceleration_magnitude_limits = {0.1, 0.2, 0.3},
                         .qvel_curr = {3.0, -0.1, 0.0}},
              TestParams{
                  .integration_timestep = absl::Seconds(1.0e-3),
                  .joint_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9},
                  .acceleration_magnitude_limits = {0.1, 0.2, 0.3, 0.4, 0.5,
                                                    0.6, 0.7, 0.8, 0.9},
                  .qvel_curr = {0.1, 20, -0.3, 4, -0.5, 0.0, 0.0, -1, -10}}}));

// Tests that the bounds and coefficients are accurate immediately after
// construction.
TEST_P(ParameterizedJointAccelerationConstraintTest,
       CoefficientMatrixAndBoundsHaveValidDimensionsAndValues) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  auto test_params = GetParam();
  JointAccelerationConstraint::Parameters params;
  params.model = model_.get();
  params.joint_ids = test_params.joint_ids;
  params.integration_timestep = test_params.integration_timestep;
  params.acceleration_magnitude_limits =
      test_params.acceleration_magnitude_limits;

  // Set joint velocities in mjData, and construct constraint.
  int joint_id_idx = 0;
  for (int joint_id : params.joint_ids) {
    int dof_id = model_->jnt_dofadr[joint_id];
    data_->qvel[dof_id] = test_params.qvel_curr[joint_id_idx];
    ++joint_id_idx;
  }
  JointAccelerationConstraint constraint(params, *data_);

  // Ensure interface output is as expected.
  LowerAndUpperBounds expected_bounds =
      ComputeExpectedLowerAndUpperBounds(test_params);
  EXPECT_THAT(constraint, LsqpConstraintDimensionsAreValid());
  EXPECT_THAT(
      constraint.GetCoefficientMatrix(),
      Pointwise(DoubleEq(), MakeIdentityMatrix(params.joint_ids.size())));
  EXPECT_THAT(constraint.GetLowerBound(),
              Pointwise(DoubleNear(1.0e-10), expected_bounds.lower_bound));
  EXPECT_THAT(constraint.GetUpperBound(),
              Pointwise(DoubleNear(1.0e-10), expected_bounds.upper_bound));
}

// Tests that the bounds and coefficients are correctly updated with a call to
// UpdateBounds.
TEST_P(ParameterizedJointAccelerationConstraintTest,
       BoundsAndCoefficientsAreCorrectlyUpdated) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  auto test_params = GetParam();
  JointAccelerationConstraint::Parameters params;
  params.model = model_.get();
  params.joint_ids = test_params.joint_ids;
  params.integration_timestep = test_params.integration_timestep;
  params.acceleration_magnitude_limits =
      test_params.acceleration_magnitude_limits;

  // Initialize the JointAccelerationConstraint without updating mjData.
  JointAccelerationConstraint constraint(params, *data_);

  // Call update bounds on a mjData with new velocities to ensure the constraint
  // interface outputs get updated.
  int joint_id_idx = 0;
  for (int joint_id : params.joint_ids) {
    int dof_id = model_->jnt_dofadr[joint_id];
    data_->qvel[dof_id] = test_params.qvel_curr[joint_id_idx];
    ++joint_id_idx;
  }
  constraint.UpdateBounds(*data_);

  // Ensure interface output is as expected.
  LowerAndUpperBounds expected_bounds =
      ComputeExpectedLowerAndUpperBounds(test_params);
  EXPECT_THAT(constraint, LsqpConstraintDimensionsAreValid());
  EXPECT_THAT(
      constraint.GetCoefficientMatrix(),
      Pointwise(DoubleEq(), MakeIdentityMatrix(params.joint_ids.size())));
  EXPECT_THAT(constraint.GetLowerBound(),
              Pointwise(DoubleNear(1.0e-10), expected_bounds.lower_bound));
  EXPECT_THAT(constraint.GetUpperBound(),
              Pointwise(DoubleNear(1.0e-10), expected_bounds.upper_bound));
}

}  // namespace
}  // namespace dm_robotics
