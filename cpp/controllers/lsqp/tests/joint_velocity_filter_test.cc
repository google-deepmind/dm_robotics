// Copyright 2022 DeepMind Technologies Limited
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

#include "dm_robotics/controllers/lsqp/joint_velocity_filter.h"
#include <algorithm>
#include <array>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dm_robotics/controllers/lsqp/test_utils.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/test_with_mujoco_model.h"
#include "dm_robotics/mujoco/types.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {
namespace {

using ::dm_robotics::testing::SetSubsetOfJointVelocities;
using ::dm_robotics::testing::TestWithMujocoModel;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class JointVelocityFilterTest :
    public TestWithMujocoModel,
    public WithParamInterface<bool> {};

constexpr bool kJointVelocityFilterParameterSet[] = {
    true, false
};

INSTANTIATE_TEST_SUITE_P(
    JointVelocityFilterTests,
    JointVelocityFilterTest,
    ValuesIn(kJointVelocityFilterParameterSet));

TEST_P(JointVelocityFilterTest,
       SolutionIsOkAndRealizesTarget) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);
  const std::array<double, 3> kTargetVelocity = {
      0.0450566, 0.0199436, 0.0199436
  };
  const absl::btree_set<int> kJointIds{16, 17, 18};
  constexpr double kSolutionTolerance = 1.0e-15;

  // Instantiate filter and solve once.
  JointVelocityFilter::Parameters params;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.integration_timestep = absl::Seconds(1);
  params.solution_tolerance = kSolutionTolerance;
  params.use_adaptive_step_size = GetParam();

  ASSERT_OK(JointVelocityFilter::ValidateParameters(params));
  JointVelocityFilter filter(params);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution,
                       filter.FilterJointVelocities(*data_, kTargetVelocity));
  // Ensure the computed joint velocities are within the tolerance of the target
  // joint velocities.
  auto solution_vec = Eigen::Map<const Eigen::VectorXd>(
          solution.data(), solution.size());
  auto target_vec = Eigen::Map<const Eigen::VectorXd>(
          kTargetVelocity.data(), kTargetVelocity.size());
  double diff_norm = (solution_vec - target_vec).lpNorm<Eigen::Infinity>();
  EXPECT_LE(diff_norm, kSolutionTolerance);
}

TEST_P(JointVelocityFilterTest,
       SolutionWithAllConstraintsAndNotInCollision) {
  bool use_adaptive_step_size = GetParam();
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Place the humanoid in a position where the left hand can collide with the
  // floor if it moves down.
  data_->qpos[2] = 0.3;
  mj_fwdPosition(model_.get(), data_.get());

  // We make the left hand move down towards the plane.
  // These values were chosen to produce an initial downward motion that later
  // isn't purely downwards but would still hit the floor.
  const std::array<double, 3> kTargetVelocity = {-0.02, 0.02, -0.02};
  const absl::btree_set<int> kJointIds{19, 20, 21};
  const double kSolutionTolerance = use_adaptive_step_size ? 1.0e-3 : 1.0e-6;

  JointVelocityFilter::Parameters params;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.integration_timestep = absl::Milliseconds(5);

  params.enable_joint_position_limits = true;
  params.joint_position_limit_velocity_scale = 0.95;
  params.minimum_distance_from_joint_position_limit = 0.01;  // ~0.5deg.

  params.enable_joint_velocity_limits = true;
  params.joint_velocity_magnitude_limits = {0.5, 0.5, 0.5};

  params.enable_collision_avoidance = true;
  params.collision_avoidance_normal_velocity_scale = 0.01;
  params.minimum_distance_from_collisions = 0.005;
  params.collision_detection_distance = 10.0;
  params.collision_pairs = {
      CollisionPair(GeomGroup{"left_upper_arm", "left_lower_arm", "left_hand"},
                    GeomGroup{"floor"})};

  params.check_solution_validity = true;
  params.max_qp_solver_iterations = 300;
  params.solution_tolerance = kSolutionTolerance;
  params.use_adaptive_step_size = use_adaptive_step_size;

  ASSERT_OK(JointVelocityFilter::ValidateParameters(params));
  JointVelocityFilter filter(params);

  // Convert to geom pairs and get geom IDs to query MuJoCo for collision
  // information.
  auto geom_pairs = CollisionPairsToGeomIdPairs(
      *model_, params.collision_pairs, false, false);
  int left_hand_id =
      mj_name2id(model_.get(), mjtObj::mjOBJ_GEOM, "left_hand");
  int floor_id = mj_name2id(model_.get(), mjtObj::mjOBJ_GEOM, "floor");
  auto maybe_dist = ComputeMinimumContactDistance(*model_, *data_,
                                                  left_hand_id, floor_id, 10.0);
  ASSERT_TRUE(maybe_dist.has_value());
  double left_hand_to_floor_dist = *maybe_dist;

  // Compute velocities and integrate, for 5000 steps.
  // We expect that the distance from the hand to the floor will decrease from
  // iteration to iteration, and eventually settle at <0.006. None of the geoms
  // in the collision pair should ever penetrate.
  for (int i = 0; i < 5000; ++i) {
    ASSERT_OK_AND_ASSIGN(
        absl::Span<const double> solution,
        filter.FilterJointVelocities(*data_, kTargetVelocity));
    SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());
    mj_integratePos(model_.get(), data_->qpos, data_->qvel,
                            absl::ToDoubleSeconds(params.integration_timestep));
    mj_fwdPosition(model_.get(), data_.get());

    // Ensure the new distance always decreases from iteration to iteration,
    // until it settles.
    auto maybe_new_dist = ComputeMinimumContactDistance(
        *model_, *data_, left_hand_id, floor_id, 10.0);
    ASSERT_TRUE(maybe_new_dist.has_value());
    EXPECT_LE(*maybe_new_dist, std::max(0.006, left_hand_to_floor_dist));
    left_hand_to_floor_dist = *maybe_new_dist;

    // Ensure no contacts are detected on the collision pairs.
    ASSERT_OK_AND_ASSIGN(
        int num_contacts,
        ComputeContactsForGeomPairs(*model_, *data_, geom_pairs,
                                    params.minimum_distance_from_collisions,
                                    absl::Span<mjContact>()));
    EXPECT_EQ(num_contacts, 0);
  }
}

TEST_P(
    JointVelocityFilterTest,
    SolutionWithAllConstraintsAndNullspaceIsOkAndNotInCollisionSingleContacts) {
  bool use_adaptive_step_size = GetParam();
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Place the humanoid in a position where the left hand can collide with the
  // floor if it moves down.
  data_->qpos[2] = 0.3;
  mj_fwdPosition(model_.get(), data_.get());

  // We make the left hand move down towards the plane.
  // These values were chosen to produce an initial downward motion that later
  // isn't purely downwards but would still hit the floor.
  const std::array<double, 3> kTargetVelocity = {-0.02, 0.02, -0.02};
  const absl::btree_set<int> kJointIds{19, 20, 21};
  const double kSolutionTolerance = use_adaptive_step_size ? 1.0e-3 : 1.0e-6;

  JointVelocityFilter::Parameters params;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.integration_timestep = absl::Milliseconds(5);

  params.enable_joint_position_limits = true;
  params.joint_position_limit_velocity_scale = 0.95;
  params.minimum_distance_from_joint_position_limit = 0.01;  // ~0.5deg.

  params.enable_joint_velocity_limits = true;
  params.joint_velocity_magnitude_limits = {0.5, 0.5, 0.5};

  params.enable_collision_avoidance = true;
  params.use_minimum_distance_contacts_only = true;
  params.collision_avoidance_normal_velocity_scale = 0.01;
  params.minimum_distance_from_collisions = 0.005;
  params.collision_detection_distance = 10.0;
  params.collision_pairs = {
      CollisionPair(GeomGroup{"left_upper_arm", "left_lower_arm", "left_hand"},
                    GeomGroup{"floor"})};

  params.check_solution_validity = true;
  params.solution_tolerance = kSolutionTolerance;
  params.use_adaptive_step_size = use_adaptive_step_size;

  ASSERT_OK(JointVelocityFilter::ValidateParameters(params));
  JointVelocityFilter filter(params);

  // Convert to geom pairs and get geom IDs to query MuJoCo for collision
  // information.
  auto geom_pairs = CollisionPairsToGeomIdPairs(
      *model_, params.collision_pairs, false, false);
  int left_hand_id =
      mj_name2id(model_.get(), mjtObj::mjOBJ_GEOM, "left_hand");
  int floor_id = mj_name2id(model_.get(), mjtObj::mjOBJ_GEOM, "floor");
  auto maybe_dist = ComputeMinimumContactDistance(*model_, *data_,
                                                  left_hand_id, floor_id, 10.0);
  ASSERT_TRUE(maybe_dist.has_value());
  double left_hand_to_floor_dist = *maybe_dist;

  // Compute velocities and integrate, for 5000 steps.
  // We expect that the distance from the hand to the floor will decrease from
  // iteration to iteration, and eventually settle at <0.006. None of the geoms
  // in the collision pair should ever penetrate.
  for (int i = 0; i < 5000; ++i) {
    ASSERT_OK_AND_ASSIGN(
        absl::Span<const double> solution,
        filter.FilterJointVelocities(*data_, kTargetVelocity));
    SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());
    mj_integratePos(model_.get(), data_->qpos, data_->qvel,
                            absl::ToDoubleSeconds(params.integration_timestep));
    mj_fwdPosition(model_.get(), data_.get());

    // Ensure the new distance always decreases from iteration to iteration,
    // until it settles.
    auto maybe_new_dist = ComputeMinimumContactDistance(
        *model_, *data_, left_hand_id, floor_id, 10.0);
    ASSERT_TRUE(maybe_new_dist.has_value());
    EXPECT_LE(*maybe_new_dist, std::max(0.006, left_hand_to_floor_dist));
    left_hand_to_floor_dist = *maybe_new_dist;

    // Ensure no contacts are detected on the collision pairs.
    ASSERT_OK_AND_ASSIGN(
        int num_contacts,
        ComputeContactsForGeomPairs(*model_, *data_, geom_pairs,
                                    params.minimum_distance_from_collisions,
                                    absl::Span<mjContact>()));
    EXPECT_EQ(num_contacts, 0);
  }
}

}  // namespace
}  // namespace dm_robotics
