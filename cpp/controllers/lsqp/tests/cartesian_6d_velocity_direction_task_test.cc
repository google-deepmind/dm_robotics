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

#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_direction_task.h"

#include <array>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "absl/types/span.h"
#include "dm_robotics/controllers/lsqp/test_utils.h"
#include "dm_robotics/least_squares_qp/testing/matchers.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/test_with_mujoco_model.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {
namespace {

using Cartesian6dVelocityDirectionTaskTest =
    ::dm_robotics::testing::TestWithMujocoModel;
using ::dm_robotics::testing::ComputeObject6dJacobianForJoints;
using ::dm_robotics::testing::LsqpTaskDimensionsAreValid;
using ::testing::DoubleEq;
using ::testing::Pointwise;

constexpr char kObjectName[] = "right_foot";
constexpr mjtObj kObjectType = mjtObj::mjOBJ_BODY;
constexpr std::array<double, 6> kTargetVelocity = {1.0, 0, 0, 0, 0, 0};

TEST_F(Cartesian6dVelocityDirectionTaskTest,
       CoefficientMatrixAndBiasHaveValidDimensionsAndValues) {
  const absl::btree_set<int> kJointIds = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  Cartesian6dVelocityDirectionTask::Parameters params;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  Cartesian6dVelocityDirectionTask task(params, *data_, kTargetVelocity);

  EXPECT_THAT(task, LsqpTaskDimensionsAreValid());
  EXPECT_THAT(task.GetNumberOfDof(), kJointIds.size());
  EXPECT_THAT(task.GetBiasLength(), 6);

  // The bias for this task is always zero.
  std::vector<double> expected_bias(6, 0.0);
  EXPECT_THAT(task.GetBias(), Pointwise(DoubleEq(), expected_bias));

  // The coefficients for this task should be:
  //   C = W (v_d v_d^T - I) J
  // where v_d is the target velocity direction.
  // In this test we test for W = I.
  std::vector<double> expected_coefficients(6 * kJointIds.size());
  std::vector<double> jacobian = ComputeObject6dJacobianForJoints(
      *model_, *data_, kObjectType, kObjectName, kJointIds);
  Eigen::Vector<double, 6> target_vel(kTargetVelocity.data());
  target_vel.normalize();
  Eigen::Map<Eigen::MatrixXd>(expected_coefficients.data(), 6,
                              kJointIds.size()) =
      (target_vel * target_vel.transpose() -
       Eigen::Matrix<double, 6, 6>::Identity()) *
      Eigen::Map<const Eigen::MatrixXd>(jacobian.data(), 6, kJointIds.size());
  EXPECT_THAT(task.GetCoefficientMatrix(),
              Pointwise(DoubleEq(), expected_coefficients));
}

TEST_F(Cartesian6dVelocityDirectionTaskTest,
       DoesNotAllocateMemoryAfterConstruction) {
  const absl::btree_set<int> kJointIds = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  Cartesian6dVelocityDirectionTask::Parameters params;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  Cartesian6dVelocityDirectionTask task(params, *data_, kTargetVelocity);

  // Die if Eigen allocates memory.
  Eigen::internal::set_is_malloc_allowed(false);

  // Ensure that getting the coefficient matrix and bias do not allocate memory.
  task.GetCoefficientMatrix();
  task.GetBias();

  // Ensure that updating the coefficient matrix and bias do not allocate
  // memory.
  task.UpdateCoefficientsAndBias(*data_, std::vector<double>(6, 0.0));

  Eigen::internal::set_is_malloc_allowed(true);
}

TEST_F(Cartesian6dVelocityDirectionTaskTest,
       WeightingMatrixIsAppliedToCoefficientMatrixAndBias) {
  const absl::btree_set<int> kJointIds = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  Cartesian6dVelocityDirectionTask::Parameters params;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  Cartesian6dVelocityDirectionTask unweighted_task(params, *data_,
                                                   kTargetVelocity);

  Eigen::Map<Eigen::Matrix<double, 6, 6>> weighting_matrix(
      params.weighting_matrix.data());
  weighting_matrix.setZero();
  weighting_matrix.diagonal() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  Cartesian6dVelocityDirectionTask weighted_task(params, *data_,
                                                 kTargetVelocity);

  const auto unweighted_bias = unweighted_task.GetBias();
  const auto unweighted_coeff_matrix = unweighted_task.GetCoefficientMatrix();

  // The bias and coefficient matrix of the weighted task must be equal to the
  // unweighted task's bias and coefficient pre-multiplied by the weighting
  // matrix.
  std::array<double, 6> weighted_bias;
  std::array<double, 54> weighted_coeff_matrix;
  Eigen::Map<Eigen::Vector<double, 6>>(weighted_bias.data()) =
      weighting_matrix *
      Eigen::Map<const Eigen::Vector<double, 6>>(unweighted_bias.data());
  Eigen::Map<Eigen::Matrix<double, 6, 9>>(weighted_coeff_matrix.data()) =
      weighting_matrix * (Eigen::Map<const Eigen::Matrix<double, 6, 9>>(
                             unweighted_coeff_matrix.data()));
  EXPECT_THAT(weighted_task.GetBias(), Pointwise(DoubleEq(), weighted_bias));
  EXPECT_THAT(weighted_task.GetCoefficientMatrix(),
              Pointwise(DoubleEq(), weighted_coeff_matrix));
}

}  // namespace
}  // namespace dm_robotics
