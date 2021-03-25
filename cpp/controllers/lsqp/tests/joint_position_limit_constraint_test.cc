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

#include "dm_robotics/controllers/lsqp/joint_position_limit_constraint.h"

#include <algorithm>
#include <ostream>
#include <vector>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_join.h"
#include "dm_robotics/least_squares_qp/common/math_utils.h"
#include "dm_robotics/least_squares_qp/testing/matchers.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/test_with_mujoco_model.h"

namespace dm_robotics {
namespace {

using JointPositionLimitConstraintTest =
    ::dm_robotics::testing::TestWithMujocoModel;
using ::dm_robotics::testing::LsqpConstraintDimensionsAreValid;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Pointwise;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

struct TestParams {
  double min_distance_from_limits;
  double gain;
  absl::btree_set<int> joint_ids;
  std::vector<double> qpos_curr;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const TestParams& param) {
    return stream << "(min_distance_from_limits["
                  << param.min_distance_from_limits << "], gain[" << param.gain
                  << "], joint_ids["
                  << absl::StrJoin(param.joint_ids.begin(),
                                   param.joint_ids.end(), ", ")
                  << "], qpos_curr["
                  << absl::StrJoin(param.qpos_curr.begin(),
                                   param.qpos_curr.end(), ", ")
                  << "])";
  }
};

// Helper function and struct for computing the expected lower and upper bounds
// of a JointPositionLimitConstraint based on TestParams.
struct LowerAndUpperBounds {
  std::vector<double> lower_bound;
  std::vector<double> upper_bound;
};
LowerAndUpperBounds ComputeExpectedLowerAndUpperBounds(
    const mjModel& model, const TestParams& params) {
  LowerAndUpperBounds bounds;
  bounds.lower_bound.resize(params.joint_ids.size());
  bounds.upper_bound.resize(params.joint_ids.size());

  int joint_id_idx = 0;
  for (const int joint_id : params.joint_ids) {
    bounds.lower_bound[joint_id_idx] =
        params.gain *
        (model.jnt_range[2 * joint_id] + params.min_distance_from_limits -
         params.qpos_curr[joint_id_idx]);

    bounds.upper_bound[joint_id_idx] =
        params.gain *
        (model.jnt_range[2 * joint_id + 1] - params.min_distance_from_limits -
         params.qpos_curr[joint_id_idx]);

    // Clip to zero.
    bounds.lower_bound[joint_id_idx] =
        std::min(0.0, bounds.lower_bound[joint_id_idx]);
    bounds.upper_bound[joint_id_idx] =
        std::max(0.0, bounds.upper_bound[joint_id_idx]);

    ++joint_id_idx;
  }
  return bounds;
}

// Test fixture for parameterized tests on a variety of MuJoCo joints.
// Note: Only 1 DoF joints are used.
class ParameterizedJointPositionLimitConstraintTest
    : public WithParamInterface<TestParams>,
      public JointPositionLimitConstraintTest {};
INSTANTIATE_TEST_SUITE_P(
    ParameterizedJointPositionLimitConstraintTests,
    ParameterizedJointPositionLimitConstraintTest,
    ValuesIn({TestParams{.min_distance_from_limits = 0.0,
                         .gain = 1.0,
                         .joint_ids = {16, 17, 18},
                         .qpos_curr = {-1.39, 1.05, 1.6}},
              TestParams{.min_distance_from_limits = 1.0e-3,
                         .gain = 100,
                         .joint_ids = {19, 20, 21},
                         .qpos_curr = {1.39, 0.0, 0.5}},
              TestParams{.min_distance_from_limits = 0.1,
                         .gain = 0.5,
                         .joint_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9},
                         .qpos_curr = std::vector<double>(8, 0.0)}}));

TEST_P(ParameterizedJointPositionLimitConstraintTest,
       CoefficientMatrixAndBoundsHaveValidDimensionsAndValues) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  auto test_params = GetParam();
  JointPositionLimitConstraint::Parameters params;
  params.model = model_.get();
  params.min_distance_from_limits = test_params.min_distance_from_limits;
  params.gain = test_params.gain;
  params.joint_ids = test_params.joint_ids;

  // Set joint positions in mjData, and construct constraint.
  // Note that we do not clamp the qpos field, as the constraint in this case
  // should output a boundary velocity of zero.
  int joint_id_idx = 0;
  for (int joint_id : params.joint_ids) {
    int qpos_adr = model_->jnt_qposadr[joint_id];
    data_->qpos[qpos_adr] = test_params.qpos_curr[joint_id_idx];
    ++joint_id_idx;
  }
  JointPositionLimitConstraint constraint(params, *data_);

  // Validate all fields.
  LowerAndUpperBounds expected_bounds =
      ComputeExpectedLowerAndUpperBounds(*model_, test_params);
  EXPECT_THAT(constraint, LsqpConstraintDimensionsAreValid());
  EXPECT_THAT(
      constraint.GetCoefficientMatrix(),
      Pointwise(DoubleEq(), MakeIdentityMatrix(params.joint_ids.size())));
  EXPECT_THAT(constraint.GetLowerBound(),
              Pointwise(DoubleNear(1.0e-10), expected_bounds.lower_bound));
  EXPECT_THAT(constraint.GetUpperBound(),
              Pointwise(DoubleNear(1.0e-10), expected_bounds.upper_bound));
}

// Tests that by constantly integrating the boundary velocities, the
// resultant joint positions never go out of limits.
// Assumes a gain G = 0.99999/T, i.e. K = 0.99999. This is to avoid numerical
// errors causing it to go over the limit.
TEST_P(ParameterizedJointPositionLimitConstraintTest,
       IntegratingBoundaryVelocitiesResultsInPositionsWithinLimits) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  auto test_params = GetParam();
  JointPositionLimitConstraint::Parameters params;
  params.model = model_.get();
  params.min_distance_from_limits = test_params.min_distance_from_limits;
  params.gain = test_params.gain;
  params.joint_ids = test_params.joint_ids;
  const double integration_timestep = 0.99999 / params.gain;

  // Note that for this test, we need to ensure that the initial qpos field in
  // mjData is always within the limits by clamping. This is to avoid the
  // constraint from computing bounds for soft-contacts, in which case the
  // boundary velocity (which is zero) would still result in the joint being
  // outside the limits.
  std::vector<double> qpos_curr_clamped(params.joint_ids.size());
  int joint_id_idx = 0;
  for (int joint_id : params.joint_ids) {
    qpos_curr_clamped[joint_id_idx] = std::clamp(
        test_params.qpos_curr[joint_id_idx],
        model_->jnt_range[joint_id * 2] + params.min_distance_from_limits,
        model_->jnt_range[joint_id * 2 + 1] - params.min_distance_from_limits);
    ++joint_id_idx;
  }

  // Construct constraint.
  JointPositionLimitConstraint constraint(params, *data_);

  // Test for upper bounds starting from (clamped) qpos_curr.
  joint_id_idx = 0;
  for (int joint_id : params.joint_ids) {
    int qpos_adr = model_->jnt_qposadr[joint_id];
    data_->qpos[qpos_adr] = qpos_curr_clamped[joint_id_idx];
    ++joint_id_idx;
  }
  for (int iteration = 0; iteration < 1000; ++iteration) {
    constraint.UpdateBounds(*data_);
    auto upper_bound = constraint.GetUpperBound();
    joint_id_idx = 0;
    for (int joint_id : params.joint_ids) {
      // Note that we do not need to clamp here, as the boundary velocity
      // should never allow it to go outside the limits.
      int qpos_adr = model_->jnt_qposadr[joint_id];
      data_->qpos[qpos_adr] += integration_timestep * upper_bound[joint_id_idx];
      EXPECT_LE(data_->qpos[qpos_adr], model_->jnt_range[joint_id * 2 + 1] -
                                           params.min_distance_from_limits);
      ++joint_id_idx;
    }
  }

  // Test for lower bounds starting from (clamped) qpos_curr.
  joint_id_idx = 0;
  for (int joint_id : params.joint_ids) {
    int qpos_adr = model_->jnt_qposadr[joint_id];
    data_->qpos[qpos_adr] = qpos_curr_clamped[joint_id_idx];
    ++joint_id_idx;
  }
  for (int iteration = 0; iteration < 1000; ++iteration) {
    constraint.UpdateBounds(*data_);
    auto lower_bound = constraint.GetLowerBound();
    joint_id_idx = 0;
    for (int joint_id : params.joint_ids) {
      // Note that we do not need to clamp here, as the boundary velocity
      // should never allow it to go outside the limits.
      int qpos_adr = model_->jnt_qposadr[joint_id];
      data_->qpos[qpos_adr] += integration_timestep * lower_bound[joint_id_idx];
      EXPECT_GE(data_->qpos[qpos_adr], model_->jnt_range[joint_id * 2] +
                                           params.min_distance_from_limits);
      ++joint_id_idx;
    }
  }
}

}  // namespace
}  // namespace dm_robotics
