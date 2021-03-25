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

#include <array>
#include <ostream>
#include <vector>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_task.h"
#include "dm_robotics/controllers/lsqp/collision_avoidance_constraint.h"
#include "dm_robotics/controllers/lsqp/test_utils.h"
#include "dm_robotics/least_squares_qp/common/identity_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/mjlib.h"
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

using IdentityTaskAndCollisionAvoidanceIntegrationTest = TestWithMujocoModel;

std::string ToString(mjtObj obj) {
  switch (obj) {
    case mjtObj::mjOBJ_GEOM:
      return "Geom";
    case mjtObj::mjOBJ_SITE:
      return "Site";
    case mjtObj::mjOBJ_BODY:
      return "Body";
    default:
      return "Unknown";
  }
}

// Parameters and fixture for tests on a variety of MuJoCo objects.
// Note: Only 1 DoF joints are used.
struct Cartesian6dVelocityTaskTestParams {
  std::string object_name;
  mjtObj object_type;
  std::array<double, 6> target_velocity;
  absl::btree_set<int> joint_ids;

  friend std::ostream& operator<<(
      std::ostream& stream, const Cartesian6dVelocityTaskTestParams& param) {
    return stream << "(object_name[" << param.object_name << "], object_type["
                  << ToString(param.object_type) << "], target_velocity["
                  << absl::StrJoin(param.target_velocity.begin(),
                                   param.target_velocity.end(), ", ")
                  << "], joint_ids["
                  << absl::StrJoin(param.joint_ids.begin(),
                                   param.joint_ids.end(), ", ")
                  << "])";
  }
};

const Cartesian6dVelocityTaskTestParams& GetRightHandGeomTestParams() {
  static const Cartesian6dVelocityTaskTestParams* const
      kRightHandGeomTestParams = new Cartesian6dVelocityTaskTestParams{
          "right_hand",
          mjtObj::mjOBJ_GEOM,
          {0.0450566, 0.0199436, 0.0199436, 0, 0.0071797, -0.0071797},
          {16, 17, 18}};
  return *kRightHandGeomTestParams;
}

const Cartesian6dVelocityTaskTestParams& GetLeftHandSiteTestParams() {
  static const Cartesian6dVelocityTaskTestParams* const
      kLeftHandSiteTestParams = new Cartesian6dVelocityTaskTestParams{
          "left_hand",
          mjtObj::mjOBJ_SITE,
          {0.0312323, -0.0138245, 0.0138245, 0, 0.00497681, 0.00497681},
          {19, 20, 21}};
  return *kLeftHandSiteTestParams;
}

const Cartesian6dVelocityTaskTestParams& GetRightFootBodyTestParams() {
  static const Cartesian6dVelocityTaskTestParams* const
      kRightFootBodyTestParams =
          new Cartesian6dVelocityTaskTestParams{"right_foot",
                                                mjtObj::mjOBJ_BODY,
                                                {1.0, 0, 0, 0, 0, 0},
                                                {1, 2, 3, 4, 5, 6, 7, 8, 9}};
  return *kRightFootBodyTestParams;
}

class ParameterizedCartesian6dVelocityTaskTest
    : public WithParamInterface<Cartesian6dVelocityTaskTestParams>,
      public TestWithMujocoModel {};
INSTANTIATE_TEST_SUITE_P(ParameterizedCartesian6dVelocityTaskTests,
                         ParameterizedCartesian6dVelocityTaskTest,
                         ValuesIn({GetRightHandGeomTestParams(),
                                   GetLeftHandSiteTestParams(),
                                   GetRightFootBodyTestParams()}));

// Tests that the computed joint velocities never result in collisions when
// integrated, even when an identity task is biasing the resultant velocities
// towards a Qpos in collision.
TEST_F(IdentityTaskAndCollisionAvoidanceIntegrationTest,
       ComputedJointVelocitiesDoNotResultInCollisionsWhenIntegrated) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);
  constexpr double kCollisionDistanceTolerance = 1.0e-3;
  constexpr double kIntegrationTimestepSeconds = 1.0;

  CollisionAvoidanceConstraint::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.collision_detection_distance = 0.3;
  params.minimum_normal_distance = 0.05;
  params.gain = 0.85 / kIntegrationTimestepSeconds;
  for (int i = 1; i < model_->njnt; ++i) {
    params.joint_ids.insert(i);
  }
  for (int i = 0; i < model_->ngeom; ++i) {
    for (int j = i + 1; j < model_->ngeom; ++j) {
      params.geom_pairs.insert(std::make_pair(i, j));
    }
  }
  Eigen::VectorXd qvel_to_collision(
      Eigen::VectorXd::Zero(params.joint_ids.size()));

  // Instantiate a solver and add a task to move towards a pre-determined qpos
  // in collision, and a constraint to avoid collisions. Note that this is a
  // difficult problem to solve, with 326 constraint rows and 21 DoF, where the
  // target velocity is infeasible.
  LsqpStackOfTasksSolver qp_solver(LsqpStackOfTasksSolver::Parameters{
      /*return_error_on_nullspace_failure=*/false,
      /*verbosity=*/LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
      /*absolute_tolerance=*/kCollisionDistanceTolerance,
      /*relative_tolerance=*/0.0,
      /*hierarchical_projection_slack=*/1.0e-5,
      /*primal_infeasibility_tolerance=*/1.0e-6,
      /*dual_infeasibility_tolerance=*/1.0e-6});
  auto task =
      qp_solver.AddNewTaskHierarchy(10000)
          ->InsertOrAssignTask(
              "MoveToCollision",
              absl::make_unique<IdentityTask>(qvel_to_collision), 1.0, false)
          .first;
  auto constraint =
      qp_solver
          .InsertOrAssignConstraint(
              "CollisionAvoidance",
              absl::make_unique<CollisionAvoidanceConstraint>(params, *data_))
          .first;
  ASSERT_OK(qp_solver.SetupProblem());

  // Go through 1k integrations, to ensure that the computed velocities do not
  // eventually cause collisions.
  for (int i = 0; i < 1000; ++i) {
    // Set the collision velocities to move towards a collision position, and
    // update the collision avoidance constraint with the latest mjData.
    int joint_id_idx = 0;
    for (int joint_id : params.joint_ids) {
      const int qpos_adr = model_->jnt_qposadr[joint_id];
      const double hi_lim = model_->jnt_range[2 * joint_id + 1];
      const double low_lim = model_->jnt_range[2 * joint_id];
      const double collision_qpos = 0.1 * hi_lim + 0.9 * low_lim;
      qvel_to_collision[joint_id_idx] =
          (collision_qpos - data_->qpos[qpos_adr]) /
          kIntegrationTimestepSeconds;
      ++joint_id_idx;
    }
    task->SetTarget(qvel_to_collision);
    constraint->UpdateCoefficientsAndBounds(*data_);

    // Solve the problem. Note that sparsity of the coefficient matrix may
    // change, and thus the solver may need to re-allocate memory.
    ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver.Solve());

    // Integrate the computed joint velocities in MuJoCo, and ensure that no
    // contacts are detected.
    joint_id_idx = 0;
    for (int joint_id : params.joint_ids) {
      const int dof_adr = model_->jnt_dofadr[joint_id];
      data_->qvel[dof_adr] = solution[joint_id_idx];
      ++joint_id_idx;
    }
    mjlib_->mj_integratePos(model_.get(), data_->qpos, data_->qvel,
                            kIntegrationTimestepSeconds);
    mjlib_->mj_fwdPosition(model_.get(), data_.get());
    EXPECT_EQ(data_->ncon, 0);
  }
}

// Tests that the computed joint velocities realize the target Cartesian
// velocity.
TEST_P(ParameterizedCartesian6dVelocityTaskTest,
       ComputedJointVelocitiesResultInTargetCartesian6dVelocities) {
  constexpr double kSolutionTolerance = 1.0e-10;
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  auto [kObjectName, kObjectType, kTargetVelocity, kJointIds] = GetParam();
  Cartesian6dVelocityTask::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;

  // Instantiate the solver and find a solution to the problem.
  LsqpStackOfTasksSolver qp_solver(LsqpStackOfTasksSolver::Parameters{
      /*return_error_on_nullspace_failure=*/false,
      /*verbosity=*/LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
      /*absolute_tolerance=*/kSolutionTolerance,
      /*relative_tolerance=*/0.0,
      /*primal_infeasibility_tolerance=*/0.1 * kSolutionTolerance,
      /*dual_infeasibility_tolerance=*/0.1 * kSolutionTolerance});
  qp_solver.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Cartesian",
      absl::make_unique<Cartesian6dVelocityTask>(params, *data_,
                                                 kTargetVelocity),
      1.0, false);
  ASSERT_OK(qp_solver.SetupProblem());
  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver.Solve());
  Eigen::internal::set_is_malloc_allowed(true);

  // Compute the realized Cartesian velocity and compare it with the target
  // Cartesian velocity.
  SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());
  Eigen::Vector<double, 6> realized_cartesian_6d_vel(
      ComputeObjectCartesian6dVelocityWithJacobian(*mjlib_, *model_, *data_,
                                                   kObjectName, kObjectType)
          .data());
  Eigen::Map<Eigen::Vector<double, 6>> target_cartesian_6d_vel(
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

// Tests that UpdateCoefficientsAndBias updates the coefficients matrix and bias
// to the correct values.
TEST_P(ParameterizedCartesian6dVelocityTaskTest,
       UpdateCoefficientsAndBiasUpdatesCoefficientMatrixAndBias) {
  constexpr double kSolutionTolerance = 1.0e-10;
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  auto [kObjectName, kObjectType, kTargetVelocity, kJointIds] = GetParam();
  Cartesian6dVelocityTask::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;

  // Instantiate the solver and set up the problem, but do not solve yet.
  LsqpStackOfTasksSolver qp_solver(LsqpStackOfTasksSolver::Parameters{
      /*return_error_on_nullspace_failure=*/false,
      /*verbosity=*/LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
      /*absolute_tolerance=*/kSolutionTolerance,
      /*relative_tolerance=*/0.0,
      /*primal_infeasibility_tolerance=*/1.0e-6,
      /*dual_infeasibility_tolerance=*/1.0e-6});
  auto hierarchy = qp_solver.AddNewTaskHierarchy(10000);
  auto [task, is_inserted] =
      hierarchy->InsertOrAssignTask("Cartesian",
                                    absl::make_unique<Cartesian6dVelocityTask>(
                                        params, *data_, kTargetVelocity),
                                    1.0, false);
  ASSERT_OK(qp_solver.SetupProblem());

  // Update coefficients and bias to set the object target velocity to zero, and
  // solve the problem.
  task->UpdateCoefficientsAndBias(*data_, {0, 0, 0, 0, 0, 0});
  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver.Solve());
  Eigen::internal::set_is_malloc_allowed(true);

  // Compute the realized Cartesian velocity and compare it with the target
  // Cartesian velocity.
  SetSubsetOfJointVelocities(*model_, kJointIds, solution, data_.get());
  Eigen::Vector<double, 6> realized_cartesian_6d_vel(
      ComputeObjectCartesian6dVelocityWithJacobian(*mjlib_, *model_, *data_,
                                                   kObjectName, kObjectType)
          .data());

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
  double e_dual = (jacobian.transpose() * realized_cartesian_6d_vel)
                      .lpNorm<Eigen::Infinity>();
  EXPECT_LE(e_dual, kSolutionTolerance);
}

}  // namespace
}  // namespace dm_robotics
