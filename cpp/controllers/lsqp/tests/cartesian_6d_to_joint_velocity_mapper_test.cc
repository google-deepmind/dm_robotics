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

#include "dm_robotics/controllers/lsqp/cartesian_6d_to_joint_velocity_mapper.h"

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "dm_robotics/controllers/lsqp/test_utils.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/test_with_mujoco_model.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"

namespace dm_robotics {
namespace {

using ::dm_robotics::testing::ComputeObject6dJacobianForJoints;
using ::dm_robotics::testing::ComputeObjectCartesian6dVelocityWithJacobian;
using ::dm_robotics::testing::SetSubsetOfJointVelocities;
using ::dm_robotics::testing::TestWithMujocoModel;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class Cartesian6dToJointVelocityMapperTest : public TestWithMujocoModel,
                                             public WithParamInterface<bool> {};

constexpr bool kCartesian6dToJointVelocityMapperParameterSet[] = {true, false};

INSTANTIATE_TEST_SUITE_P(
    Cartesian6dToJointVelocityMapperTests, Cartesian6dToJointVelocityMapperTest,
    ValuesIn(kCartesian6dToJointVelocityMapperParameterSet));

TEST_P(Cartesian6dToJointVelocityMapperTest,
       SolutionWithoutNullspaceIsOkAndRealizesTarget) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);
  const std::string kObjectName = "right_hand";
  const mjtObj kObjectType = mjtObj::mjOBJ_GEOM;
  const std::array<double, 6> kTargetVelocity = {
      0.0450566, 0.0199436, 0.0199436, 0, 0.0071797, -0.0071797};
  const absl::btree_set<int> kJointIds{16, 17, 18};
  constexpr double kRegularizationWeight = 0.0;
  constexpr double kSolutionTolerance = 1.0e-15;

  // Instantiate mapper and solve once.
  Cartesian6dToJointVelocityMapper::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  params.integration_timestep = absl::Seconds(1);
  params.solution_tolerance = kSolutionTolerance;
  params.regularization_weight = kRegularizationWeight;
  params.use_adaptive_step_size = GetParam();

  ASSERT_OK(Cartesian6dToJointVelocityMapper::ValidateParameters(params));
  Cartesian6dToJointVelocityMapper mapper(params);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution,
                       mapper.ComputeJointVelocities(*data_, kTargetVelocity));

  // Compute the realized Cartesian velocity and compare it with the target
  // Cartesian velocity.
  SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());
  Eigen::Vector<double, 6> realized_cartesian_6d_vel(
      ComputeObjectCartesian6dVelocityWithJacobian(*mjlib_, *model_, *data_,
                                                   kObjectName, kObjectType)
          .data());
  Eigen::Map<const Eigen::Vector<double, 6>> target_cartesian_6d_vel(
      kTargetVelocity.data());

  // Ensure the realized Cartesian velocity is within tolerance of the target
  // velocity.
  // Note that for an unconstrained stack-of-tasks problem with one task that is
  // realizable, the `absolute_tolerance` represents how far from optimality the
  // solution is, measured by:
  //   e_dual = W ||J^T J qvel - (xdot_target^T J)^T||_inf
  //   e_dual = W ||J^T xdot_target - J^T xdot_realized||_inf
  Eigen::MatrixXd jacobian = Eigen::Map<Eigen::MatrixXd>(
      ComputeObject6dJacobianForJoints(*mjlib_, *model_, *data_, kObjectType,
                                       kObjectName, kJointIds)
          .data(),
      6, kJointIds.size());
  double e_dual = (jacobian.transpose() *
                   (realized_cartesian_6d_vel - target_cartesian_6d_vel))
                      .lpNorm<Eigen::Infinity>();
  EXPECT_LE(e_dual, kSolutionTolerance);
}

TEST_P(Cartesian6dToJointVelocityMapperTest,
       SolutionWithNullspaceIsOkAndRealizesTarget) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);
  const std::string kObjectName = "right_foot";
  const mjtObj kObjectType = mjtObj::mjOBJ_BODY;
  const std::array<double, 6> kTargetVelocity = {1.0, 0, 0, 0, 0, 0};
  const absl::btree_set<int> kJointIds{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::array<double, 9> nullspace_bias = {-1.0, 0.0,  1.0, -1.0, 0.0,
                                                1.0,  -1.0, 0.0, 1.0};
  constexpr double kSolutionTolerance = 1.0e-6;
  constexpr double kRegularizationWeight = 1.0e-3;
  constexpr double kNullspaceProjectionSlack = 1.0e-7;

  // Shared parameters for optimization problem with and without nullspace
  // hierarchy.
  Cartesian6dToJointVelocityMapper::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  params.integration_timestep = absl::Seconds(1);
  params.solution_tolerance = kSolutionTolerance;
  params.regularization_weight = kRegularizationWeight;
  params.use_adaptive_step_size = GetParam();

  ASSERT_OK(Cartesian6dToJointVelocityMapper::ValidateParameters(params));

  // First compute the solution without nullspace projection and the realized
  // Cartesian velocity. We need this in order to compute the accuracy achieved
  // in Cartesian space and compare that with the solution *with* nullspace
  // projection.
  std::array<double, 6> no_nullspace_cartesian_vel;
  std::vector<double> no_nullspace_solution;
  {
    Cartesian6dToJointVelocityMapper mapper(params);
    ASSERT_OK_AND_ASSIGN(
        absl::Span<const double> solution,
        mapper.ComputeJointVelocities(*data_, kTargetVelocity));
    SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());

    no_nullspace_solution =
        std::vector<double>(solution.begin(), solution.end());
    no_nullspace_cartesian_vel = ComputeObjectCartesian6dVelocityWithJacobian(
        *mjlib_, *model_, *data_, kObjectName, kObjectType);
  }

  // Reuse the same parameters but add nullspace projection, and compute the
  // solution to the optimization problem with nullspace.
  params.enable_nullspace_control = true;
  params.return_error_on_nullspace_failure = true;
  params.nullspace_projection_slack = kNullspaceProjectionSlack;
  ASSERT_OK(Cartesian6dToJointVelocityMapper::ValidateParameters(params));

  // Compute the solution with nullspace projection and the realized Cartesian
  // velocity. The solution with nullspace projection should keep the same
  // accuracy in Cartesian space, while increasing the accuracy of the nullspace
  // task.
  std::array<double, 6> nullspace_cartesian_vel;
  std::vector<double> nullspace_solution;
  Cartesian6dToJointVelocityMapper mapper(params);
  ASSERT_OK_AND_ASSIGN(
      absl::Span<const double> solution,
      mapper.ComputeJointVelocities(*data_, kTargetVelocity, nullspace_bias));
  SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());

  nullspace_solution = std::vector<double>(solution.begin(), solution.end());
  nullspace_cartesian_vel = ComputeObjectCartesian6dVelocityWithJacobian(
      *mjlib_, *model_, *data_, kObjectName, kObjectType);

  // The nullspace solution should be different than the no-nullspace solution.
  // For this problem, we computed the Euclidean distance of both solutions to
  // be around ~0.85; this is expected since there's 10 DoF and only 6 DoF are
  // being used for Cartesian control. Test that the solutions differ by at
  // least 0.8, and that the nullspace solution is closer to the nullspace
  // target.
  Eigen::Map<const Eigen::VectorXd> no_nullspace_solution_map(
      no_nullspace_solution.data(), no_nullspace_solution.size());
  Eigen::Map<const Eigen::VectorXd> nullspace_solution_map(
      nullspace_solution.data(), nullspace_solution.size());
  Eigen::Map<const Eigen::VectorXd> nullspace_target_map(nullspace_bias.data(),
                                                         nullspace_bias.size());
  double solution_diff =
      (nullspace_solution_map - no_nullspace_solution_map).norm();
  double no_nullspace_sol_bias_error =
      (nullspace_target_map - no_nullspace_solution_map).norm();
  double nullspace_sol_bias_error =
      (nullspace_target_map - nullspace_solution_map).norm();
  EXPECT_GE(solution_diff, 0.8);
  EXPECT_LE(nullspace_sol_bias_error, no_nullspace_sol_bias_error);

  // The nullspace solution should respect the nullspace inequality constraint:
  //  xdot_first - slack  <= xdot <= xdot_first + slack,
  // with a tolerance of kSolution tolerance (and thus increasing the slack).
  // This means that the norm of the difference must be within
  // kSolutionTolerance + kNullspaceProjectionSlack from the Cartesian velocity
  // achieved without nullspace.
  // Note that other constraints would result in the norm being limited to a
  // lower value, but kSolutionTolerance + kNullspaceProjectionSlack is the
  // upper bound when no other constraints are active.
  Eigen::Map<const Eigen::VectorXd> nullspace_cartesian_vel_map(
      nullspace_cartesian_vel.data(), nullspace_cartesian_vel.size());
  Eigen::Map<const Eigen::VectorXd> no_nullspace_cartesian_vel_map(
      no_nullspace_cartesian_vel.data(), no_nullspace_cartesian_vel.size());
  EXPECT_LE((nullspace_cartesian_vel_map - no_nullspace_cartesian_vel_map)
                .lpNorm<Eigen::Infinity>(),
            kSolutionTolerance + kNullspaceProjectionSlack);
}

TEST_P(Cartesian6dToJointVelocityMapperTest,
       SolutionWithNonIdentityWeightingMatrixIsOkAndRealizesTarget) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);
  const std::string kObjectName = "right_hand";
  const mjtObj kObjectType = mjtObj::mjOBJ_GEOM;
  const std::array<double, 6> kTargetVelocity = {
      0.0450566, 1.0e9, 0.0199436, 1.0e9, 0.0071797, -0.0071797};
  const absl::btree_set<int> kJointIds{16, 17, 18};
  constexpr double kRegularizationWeight = 0.0;
  constexpr double kSolutionTolerance = 1.0e-15;

  // Instantiate mapper.
  Cartesian6dToJointVelocityMapper::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  params.integration_timestep = absl::Seconds(1);
  params.cartesian_velocity_task_weighting_matrix = {
      1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // Enable Vx
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // Disable Vy
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  // Enable Vz
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // Disable Wx
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  // Enable Wy
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0   // Enable Wz
  };
  params.solution_tolerance = kSolutionTolerance;
  params.regularization_weight = kRegularizationWeight;
  params.use_adaptive_step_size = GetParam();

  ASSERT_OK(Cartesian6dToJointVelocityMapper::ValidateParameters(params));
  Cartesian6dToJointVelocityMapper mapper(params);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution,
                       mapper.ComputeJointVelocities(*data_, kTargetVelocity));

  // Compute the realized Cartesian 4D velocity and Jacobian for the non-zero
  // axes. Note that the axes with zero-weight are not controlled by the mapper,
  // and thus they can be anything.
  SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());
  Eigen::Vector<double, 6> realized_cartesian_6d_vel(
      ComputeObjectCartesian6dVelocityWithJacobian(*mjlib_, *model_, *data_,
                                                   kObjectName, kObjectType)
          .data());
  Eigen::MatrixXd jacobian_6d = Eigen::Map<Eigen::MatrixXd>(
      ComputeObject6dJacobianForJoints(*mjlib_, *model_, *data_, kObjectType,
                                       kObjectName, kJointIds)
          .data(),
      6, kJointIds.size());
  Eigen::MatrixXd jacobian_4d = jacobian_6d({0, 2, 4, 5}, Eigen::indexing::all);
  Eigen::Vector<double, 4> realized_cartesian_4d_vel;
  Eigen::Vector<double, 4> target_cartesian_4d_vel;
  int index_4d = 0;
  for (const int index_6d : {0, 2, 4, 5}) {
    realized_cartesian_4d_vel[index_4d] = realized_cartesian_6d_vel[index_6d];
    target_cartesian_4d_vel[index_4d] = kTargetVelocity[index_6d];
    ++index_4d;
  }

  // Ensure the realized Cartesian velocity is within tolerance of the target
  // velocity.
  // Note that for an unconstrained stack-of-tasks problem with one task that is
  // realizable, the `absolute_tolerance` represents how far from optimality the
  // solution is, measured by:
  //   e_dual = W ||J^T J qvel - (xdot_target^T J)^T||_inf
  //   e_dual = W ||J^T xdot_target - J^T xdot_realized||_inf
  double e_dual = (jacobian_4d.transpose() *
                   (realized_cartesian_4d_vel - target_cartesian_4d_vel))
                      .lpNorm<Eigen::Infinity>();
  EXPECT_LE(e_dual, kSolutionTolerance);
}

TEST_P(Cartesian6dToJointVelocityMapperTest,
       SolutionWithAllConstraintsAndNullspaceIsOkAndNotInCollision) {
  bool use_adaptive_step_size = GetParam();
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Place the humanoid in a position where the left hand can collide with the
  // floor if it moves down.
  data_->qpos[2] = 0.3;
  mjlib_->mj_fwdPosition(model_.get(), data_.get());

  // We make the left hand move down towards the plane.
  const std::string kObjectName = "left_hand";
  const mjtObj kObjectType = mjtObj::mjOBJ_SITE;
  const std::array<double, 6> kTargetVelocity = {0.0, 0.0, -1.0, 0, 0.0, 0.0};
  const absl::btree_set<int> kJointIds{19, 20, 21};
  const std::array<double, 3> nullspace_bias = {-1.0, 0.0, 1.0};
  constexpr double kRegularizationWeight = 1.0e-3;
  const double kSolutionTolerance = use_adaptive_step_size ? 1.0e-3 : 1.0e-6;
  const double kNullspaceProjectionSlack = 0.1 * kSolutionTolerance;

  Cartesian6dToJointVelocityMapper::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  params.integration_timestep = absl::Milliseconds(5);

  params.enable_joint_position_limits = true;
  params.joint_position_limit_velocity_scale = 0.95;
  params.minimum_distance_from_joint_position_limit = 0.01;  // ~0.5deg.

  params.enable_joint_velocity_limits = true;
  params.joint_velocity_magnitude_limits = {0.5, 0.5, 0.5};

  params.enable_joint_acceleration_limits = true;
  params.remove_joint_acceleration_limits_if_in_conflict = true;
  params.joint_acceleration_magnitude_limits = {1.0, 1.0, 1.0};

  params.enable_collision_avoidance = true;
  params.collision_avoidance_normal_velocity_scale = 0.01;
  params.minimum_distance_from_collisions = 0.005;
  params.collision_detection_distance = 10.0;
  params.collision_pairs = {
      CollisionPair(GeomGroup{"left_upper_arm", "left_lower_arm", "left_hand"},
                    GeomGroup{"floor"})};

  params.check_solution_validity = true;
  params.solution_tolerance = kSolutionTolerance;
  params.regularization_weight = kRegularizationWeight;
  params.enable_nullspace_control = true;
  params.return_error_on_nullspace_failure = false;
  params.nullspace_projection_slack = kNullspaceProjectionSlack;
  params.use_adaptive_step_size = use_adaptive_step_size;
  params.log_nullspace_failure_warnings = false;

  ASSERT_OK(Cartesian6dToJointVelocityMapper::ValidateParameters(params));
  Cartesian6dToJointVelocityMapper mapper(params);

  // Convert to geom pairs and get geom IDs to query MuJoCo for collision
  // information.
  auto geom_pairs = CollisionPairsToGeomIdPairs(
      *mjlib_, *model_, params.collision_pairs, false, false);
  int left_hand_id =
      mjlib_->mj_name2id(model_.get(), mjtObj::mjOBJ_GEOM, "left_hand");
  int floor_id = mjlib_->mj_name2id(model_.get(), mjtObj::mjOBJ_GEOM, "floor");
  auto maybe_dist = ComputeMinimumContactDistance(*mjlib_, *model_, *data_,
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
        mapper.ComputeJointVelocities(*data_, kTargetVelocity, nullspace_bias));
    SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());
    mjlib_->mj_integratePos(model_.get(), data_->qpos, data_->qvel,
                            absl::ToDoubleSeconds(params.integration_timestep));
    mjlib_->mj_fwdPosition(model_.get(), data_.get());

    // Ensure the new distance always decreases from iteration to iteration,
    // until it settles.
    auto maybe_new_dist = ComputeMinimumContactDistance(
        *mjlib_, *model_, *data_, left_hand_id, floor_id, 10.0);
    ASSERT_TRUE(maybe_new_dist.has_value());
    EXPECT_LE(*maybe_new_dist, std::max(0.006, left_hand_to_floor_dist));
    left_hand_to_floor_dist = *maybe_new_dist;

    // Ensure no contacts are detected on the collision pairs.
    ASSERT_OK_AND_ASSIGN(
        int num_contacts,
        ComputeContactsForGeomPairs(*mjlib_, *model_, *data_, geom_pairs,
                                    params.minimum_distance_from_collisions,
                                    absl::Span<mjContact>()));
    EXPECT_EQ(num_contacts, 0);
  }
}

TEST_P(
    Cartesian6dToJointVelocityMapperTest,
    SolutionWithAllConstraintsAndNullspaceIsOkAndNotInCollisionSingleContacts) {
  bool use_adaptive_step_size = GetParam();
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Place the humanoid in a position where the left hand can collide with the
  // floor if it moves down.
  data_->qpos[2] = 0.3;
  mjlib_->mj_fwdPosition(model_.get(), data_.get());

  // We make the left hand move down towards the plane.
  const std::string kObjectName = "left_hand";
  const mjtObj kObjectType = mjtObj::mjOBJ_SITE;
  const std::array<double, 6> kTargetVelocity = {0.0, 0.0, -1.0, 0, 0.0, 0.0};
  const absl::btree_set<int> kJointIds{19, 20, 21};
  const std::array<double, 3> nullspace_bias = {-1.0, 0.0, 1.0};
  constexpr double kRegularizationWeight = 1.0e-3;
  const double kSolutionTolerance = use_adaptive_step_size ? 1.0e-3 : 1.0e-6;
  const double kNullspaceProjectionSlack = 0.1 * kSolutionTolerance;

  Cartesian6dToJointVelocityMapper::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  params.integration_timestep = absl::Milliseconds(5);

  params.enable_joint_position_limits = true;
  params.joint_position_limit_velocity_scale = 0.95;
  params.minimum_distance_from_joint_position_limit = 0.01;  // ~0.5deg.

  params.enable_joint_velocity_limits = true;
  params.joint_velocity_magnitude_limits = {0.5, 0.5, 0.5};

  params.enable_joint_acceleration_limits = true;
  params.remove_joint_acceleration_limits_if_in_conflict = true;
  params.joint_acceleration_magnitude_limits = {1.0, 1.0, 1.0};

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
  params.regularization_weight = kRegularizationWeight;
  params.enable_nullspace_control = true;
  params.return_error_on_nullspace_failure = false;
  params.nullspace_projection_slack = kNullspaceProjectionSlack;
  params.use_adaptive_step_size = use_adaptive_step_size;
  params.log_nullspace_failure_warnings = false;

  ASSERT_OK(Cartesian6dToJointVelocityMapper::ValidateParameters(params));
  Cartesian6dToJointVelocityMapper mapper(params);

  // Convert to geom pairs and get geom IDs to query MuJoCo for collision
  // information.
  auto geom_pairs = CollisionPairsToGeomIdPairs(
      *mjlib_, *model_, params.collision_pairs, false, false);
  int left_hand_id =
      mjlib_->mj_name2id(model_.get(), mjtObj::mjOBJ_GEOM, "left_hand");
  int floor_id = mjlib_->mj_name2id(model_.get(), mjtObj::mjOBJ_GEOM, "floor");
  auto maybe_dist = ComputeMinimumContactDistance(*mjlib_, *model_, *data_,
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
        mapper.ComputeJointVelocities(*data_, kTargetVelocity, nullspace_bias));
    SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());
    mjlib_->mj_integratePos(model_.get(), data_->qpos, data_->qvel,
                            absl::ToDoubleSeconds(params.integration_timestep));
    mjlib_->mj_fwdPosition(model_.get(), data_.get());

    // Ensure the new distance always decreases from iteration to iteration,
    // until it settles.
    auto maybe_new_dist = ComputeMinimumContactDistance(
        *mjlib_, *model_, *data_, left_hand_id, floor_id, 10.0);
    ASSERT_TRUE(maybe_new_dist.has_value());
    EXPECT_LE(*maybe_new_dist, std::max(0.006, left_hand_to_floor_dist));
    left_hand_to_floor_dist = *maybe_new_dist;

    // Ensure no contacts are detected on the collision pairs.
    ASSERT_OK_AND_ASSIGN(
        int num_contacts,
        ComputeContactsForGeomPairs(*mjlib_, *model_, *data_, geom_pairs,
                                    params.minimum_distance_from_collisions,
                                    absl::Span<mjContact>()));
    EXPECT_EQ(num_contacts, 0);
  }
}

}  // namespace
}  // namespace dm_robotics
