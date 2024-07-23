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

#include "dm_robotics/controllers/lsqp/collision_avoidance_constraint.h"
#include <limits>
#include <utility>
#include <vector>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/testing/matchers.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/test_with_mujoco_model.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {
namespace {

using CollisionAvoidanceConstraintTest =
    ::dm_robotics::testing::TestWithMujocoModel;
using ::dm_robotics::testing::LsqpConstraintDimensionsAreValid;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::Ge;
using ::testing::Pointwise;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

absl::btree_set<std::pair<int, int>> GetAllGeomPairs(const mjModel& model) {
  absl::btree_set<std::pair<int, int>> geom_pairs;
  for (int i = 0; i < model.ngeom; ++i) {
    for (int j = i + 1; j < model.ngeom; ++j) {
      geom_pairs.insert(std::make_pair(i, j));
    }
  }
  return geom_pairs;
}

// Returns the Jacobian mapping the joint velocities to the normal linear
// velocity between both objects in a specified contact.
Eigen::MatrixXd ComputeContactNormalJacobianForJoints(
    const mjModel& model, const mjData& data,
    const mjContact& contact, const absl::btree_set<int>& joint_ids) {
  // We need a vector of DoF IDs for indexing Eigen::VectorXd.
  absl::btree_set<int> dof_ids = JointIdsToDofIds(model, joint_ids);
  std::vector<int> dof_ids_vector(dof_ids.begin(), dof_ids.end());

  // Get the contact normal Jacobian for all the joints.
  // This Jacobian maps joint velocities to the linear Cartesian velocity at
  // which the geoms move away from each other.
  std::vector<double> jacobian_buffer(3 * model.nv);
  Eigen::MatrixXd jacobian(1, model.nv);
  ComputeContactNormalJacobian(
      model, data, contact, absl::MakeSpan(jacobian_buffer),
      absl::MakeSpan(jacobian.data(), jacobian.size()));

  // Index the complete Jacobian to get a subset of joints.
  // We multiply by -1.0 to ensure that the Jacobain maps joint velocities to
  // the velocity of the geoms towards each other, not away from each other.
  return -1.0 * jacobian(Eigen::indexing::all, dof_ids_vector);
}

class CollisionAvoidanceConstraintWithParamsTest
    : public CollisionAvoidanceConstraintTest,
      public WithParamInterface<bool> {};

constexpr bool CollisionAvoidanceConstraintParameterSet[] = {true, false};

INSTANTIATE_TEST_SUITE_P(CollisionAvoidanceConstraintWithParamsTests,
                         CollisionAvoidanceConstraintWithParamsTest,
                         ValuesIn(CollisionAvoidanceConstraintParameterSet));

TEST_P(CollisionAvoidanceConstraintWithParamsTest,
       CoefficientMatrixAndBoundsDimensionsAndValuesAreValid) {
  bool use_minimum_distance_contacts_only = GetParam();
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Initialize constraint.
  CollisionAvoidanceConstraint::Parameters params;
  params.model = model_.get();
  params.use_minimum_distance_contacts_only =
      use_minimum_distance_contacts_only;
  params.collision_detection_distance = 1.0;
  params.minimum_normal_distance = 0.05;
  params.gain = 0.85;
  params.bound_relaxation = -1.0e-3;
  for (int i = 1; i < model_->njnt; ++i) {
    params.joint_ids.insert(i);
  }
  params.geom_pairs = GetAllGeomPairs(*model_);
  CollisionAvoidanceConstraint constraint(params, *data_);

  // Ensure dimensions are valid.
  int max_num_contacts;
  if (use_minimum_distance_contacts_only) {
    max_num_contacts = params.geom_pairs.size();
  } else {
    max_num_contacts =
        ComputeMaximumNumberOfContacts(*model_, params.geom_pairs);
  }
  EXPECT_THAT(constraint, LsqpConstraintDimensionsAreValid());
  EXPECT_EQ(constraint.GetNumberOfDof(), params.joint_ids.size());
  EXPECT_EQ(constraint.GetBoundsLength(), max_num_contacts);

  // The lower-bound should be -infinity everywhere.
  EXPECT_THAT(
      constraint.GetLowerBound(),
      Pointwise(DoubleEq(),
                std::vector<double>(max_num_contacts,
                                    -std::numeric_limits<double>::infinity())));

  // The upper bound should always be greater or equal to the bound relaxation
  // parameter.
  EXPECT_THAT(constraint.GetUpperBound(), Each(Ge(params.bound_relaxation)));
}

// Tests that the coefficient matrix and bounds have the expected values in
// the rows that do not have contacts.
TEST_P(CollisionAvoidanceConstraintWithParamsTest,
       CoefficientMatrixAndBoundsValuesAsExpectedInRowsWithNoContacts) {
  bool use_minimum_distance_contacts_only = GetParam();
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Initialize constraint.
  CollisionAvoidanceConstraint::Parameters params;
  params.model = model_.get();
  params.use_minimum_distance_contacts_only =
      use_minimum_distance_contacts_only;
  params.collision_detection_distance = 0.3;
  params.minimum_normal_distance = 0.05;
  params.gain = 0.85;
  for (int i = 1; i < model_->njnt; ++i) {
    params.joint_ids.insert(i);
  }
  params.geom_pairs = GetAllGeomPairs(*model_);
  CollisionAvoidanceConstraint constraint(params, *data_);

  // Ensure that the number of detected contacts in the current configuration
  // is not zero, for this test to be valid.
  int max_num_contacts;
  int num_contacts;
  if (use_minimum_distance_contacts_only) {
    max_num_contacts = params.geom_pairs.size();
    num_contacts = 0;
    for (const auto& pair : params.geom_pairs) {
      absl::optional<mjContact> maybe_contact =
          ComputeContactWithMinimumDistance(
              *model_, *data_, pair.first, pair.second,
              params.collision_detection_distance);
      if (maybe_contact.has_value()) {
        ++num_contacts;
      }
    }
  } else {
    max_num_contacts =
        ComputeMaximumNumberOfContacts(*model_, params.geom_pairs);
    std::vector<mjContact> contacts(max_num_contacts);
    ASSERT_OK_AND_ASSIGN(
        num_contacts,
        ComputeContactsForGeomPairs(*model_, *data_, params.geom_pairs,
                                    params.collision_detection_distance,
                                    absl::MakeSpan(contacts)));
  }
  ASSERT_NE(num_contacts, 0);

  // The bottom rows corresponding to the number of empty contacts of the
  // coefficient matrix must be zero.
  ASSERT_EQ(constraint.GetCoefficientMatrix().size(),
            max_num_contacts * params.joint_ids.size());
  Eigen::Map<const Eigen::MatrixXd> coefficient_matrix(
      constraint.GetCoefficientMatrix().data(), max_num_contacts,
      params.joint_ids.size());
  EXPECT_EQ(coefficient_matrix.bottomRows(max_num_contacts - num_contacts),
            Eigen::MatrixXd::Zero(max_num_contacts - num_contacts,
                                  params.joint_ids.size()));

  // The bottom rows corresponding to the number of empty contacts of the
  // upper bound must be infinity.
  absl::Span<const double> upper_bound_bottom_rows(
      constraint.GetUpperBound().data() + num_contacts,
      max_num_contacts - num_contacts);
  EXPECT_THAT(
      upper_bound_bottom_rows,
      Pointwise(DoubleEq(),
                std::vector<double>(max_num_contacts - num_contacts,
                                    std::numeric_limits<double>::infinity())));
}

// Tests that the coefficient matrix and bounds have the expected values in
// the rows that have contacts.
TEST_F(CollisionAvoidanceConstraintTest,
       CoefficientMatrixValuesAsExpectedInRowsWithContactsMultiContact) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Initialize constraint.
  CollisionAvoidanceConstraint::Parameters params;
  params.model = model_.get();
  params.use_minimum_distance_contacts_only = false;
  params.collision_detection_distance = 0.3;
  params.minimum_normal_distance = 0.05;
  params.gain = 0.85;
  for (int i = 1; i < model_->njnt; ++i) {
    params.joint_ids.insert(i);
  }
  params.geom_pairs = GetAllGeomPairs(*model_);
  CollisionAvoidanceConstraint constraint(params, *data_);

  // Ensure that the number of detected contacts in the current configuration
  // is not zero, for this test to be valid.
  int max_num_contacts =
      ComputeMaximumNumberOfContacts(*model_, params.geom_pairs);
  std::vector<mjContact> contacts(max_num_contacts);
  ASSERT_OK_AND_ASSIGN(
      int num_contacts,
      ComputeContactsForGeomPairs(*model_, *data_, params.geom_pairs,
                                  params.collision_detection_distance,
                                  absl::MakeSpan(contacts)));
  ASSERT_NE(num_contacts, 0);

  // Every row of the coefficient matrix should be the jacobian, in order,
  // corresponding to each contact.
  ASSERT_EQ(constraint.GetCoefficientMatrix().size(),
            max_num_contacts * params.joint_ids.size());
  Eigen::Map<const Eigen::MatrixXd> coefficient_matrix(
      constraint.GetCoefficientMatrix().data(), max_num_contacts,
      params.joint_ids.size());
  for (int i = 0; i < contacts.size(); ++i) {
    // We don't expect but Jacobians to be exactly the same, but they should be
    // within a very tight tolerance of each other.
    Eigen::MatrixXd jacobian = ComputeContactNormalJacobianForJoints(
        *model_, *data_, contacts[i], params.joint_ids);
    EXPECT_THAT(coefficient_matrix.row(i),
                Pointwise(DoubleNear(1.0e-10), jacobian.row(0)));
  }
}

TEST_F(CollisionAvoidanceConstraintTest,
       CoefficientMatrixValuesAsExpectedInRowsWithContactsSingleContact) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Initialize constraint.
  CollisionAvoidanceConstraint::Parameters params;
  params.model = model_.get();
  params.use_minimum_distance_contacts_only = true;
  params.collision_detection_distance = 0.3;
  params.minimum_normal_distance = 0.05;
  params.gain = 0.85;
  for (int i = 1; i < model_->njnt; ++i) {
    params.joint_ids.insert(i);
  }
  params.geom_pairs = GetAllGeomPairs(*model_);
  CollisionAvoidanceConstraint constraint(params, *data_);

  // Every row of the coefficient matrix should be the jacobian, in order,
  // corresponding to each contact.
  int max_num_contacts = params.geom_pairs.size();
  ASSERT_EQ(constraint.GetCoefficientMatrix().size(),
            max_num_contacts * params.joint_ids.size());
  Eigen::Map<const Eigen::MatrixXd> coefficient_matrix(
      constraint.GetCoefficientMatrix().data(), max_num_contacts,
      params.joint_ids.size());
  int contact_counter = 0;
  for (const auto& pair : params.geom_pairs) {
    absl::optional<mjContact> maybe_contact = ComputeContactWithMinimumDistance(
        *model_, *data_, pair.first, pair.second,
        params.collision_detection_distance);
    if (maybe_contact.has_value()) {
      Eigen::MatrixXd jacobian = ComputeContactNormalJacobianForJoints(
          *model_, *data_, *maybe_contact, params.joint_ids);
      EXPECT_THAT(coefficient_matrix.row(contact_counter),
                  Pointwise(DoubleNear(1.0e-10), jacobian.row(0)));
      ++contact_counter;
    }
  }

  // Ensure that the number of detected contacts in the current configuration
  // is not zero, for this test to be valid.
  ASSERT_NE(contact_counter, 0);
}

}  // namespace
}  // namespace dm_robotics
