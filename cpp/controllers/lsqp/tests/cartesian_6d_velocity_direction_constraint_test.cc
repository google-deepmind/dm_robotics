#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_direction_constraint.h"

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "dm_robotics/controllers/lsqp/test_utils.h"
#include "dm_robotics/least_squares_qp/testing/matchers.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/test_with_mujoco_model.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"

namespace dm_robotics {
namespace {

using Cartesian6dVelocityDirectionConstraintTest =
    ::dm_robotics::testing::TestWithMujocoModel;
using ::dm_robotics::testing::ComputeObject6dJacobianForJoints;
using ::dm_robotics::testing::LsqpConstraintDimensionsAreValid;
using ::testing::DoubleEq;
using ::testing::Each;
using ::testing::Pointwise;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

constexpr char kObjectName[] = "right_foot";
constexpr mjtObj kObjectType = mjtObj::mjOBJ_BODY;
constexpr std::array<double, 6> kTargetVelocity = {1.0,  0.5, -0.2,
                                                   -0.7, 0.9, -1.0};

TEST_F(Cartesian6dVelocityDirectionConstraintTest,
       CoefficientMatrixAndBoundsDimensionsAreValid) {
  const absl::btree_set<int> kJointIds = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  Cartesian6dVelocityDirectionConstraint::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  Cartesian6dVelocityDirectionConstraint constraint(params, *data_,
                                                    kTargetVelocity);

  // Ensure dimensions are valid.
  EXPECT_THAT(constraint, LsqpConstraintDimensionsAreValid());
  EXPECT_EQ(constraint.GetNumberOfDof(), kJointIds.size());
  EXPECT_EQ(constraint.GetBoundsLength(), 1);
}

class Cartesian6dVelocityDirectionConstraintParametricTest
    : public Cartesian6dVelocityDirectionConstraintTest,
      public WithParamInterface<std::array<bool, 6>> {};

constexpr std::array<bool, 6>
    kCartesian6dVelocityDirectionConstraintParameterSet[] = {
        {true, false, false, false, false, false},
        {false, false, false, false, false, true},
        {true, true, true, false, false, false},
        {true, true, true, true, true, true}};

INSTANTIATE_TEST_SUITE_P(
    Cartesian6dVelocityDirectionConstraintParametricTests,
    Cartesian6dVelocityDirectionConstraintParametricTest,
    ValuesIn(kCartesian6dVelocityDirectionConstraintParameterSet));

TEST_P(Cartesian6dVelocityDirectionConstraintParametricTest,
       CoefficientMatrixAndBoundsValuesAsExpected) {
  const absl::btree_set<int> kJointIds = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  Cartesian6dVelocityDirectionConstraint::Parameters params;
  params.lib = mjlib_;
  params.model = model_.get();
  params.joint_ids = kJointIds;
  params.object_type = kObjectType;
  params.object_name = kObjectName;
  params.enable_axis_constraint = GetParam();
  Cartesian6dVelocityDirectionConstraint constraint(params, *data_,
                                                    kTargetVelocity);

  // The upper-bound should be infinity.
  EXPECT_THAT(constraint.GetUpperBound(),
              Each(DoubleEq(std::numeric_limits<double>::infinity())));

  // The lower-bound should be zero.
  EXPECT_THAT(constraint.GetLowerBound(), Each(DoubleEq(0.0)));

  // Make an Eigen indexing vector for the enabled velocities.
  std::vector<int> indexer;
  for (int i = 0; i < 6; ++i) {
    if (params.enable_axis_constraint[i]) {
      indexer.push_back(i);
    }
  }

  // The coefficients for this task should be:
  //   C = v_d^T J
  // where v_d is the target velocity direction.
  std::vector<double> jacobian_vec = ComputeObject6dJacobianForJoints(
      *mjlib_, *model_, *data_, kObjectType, kObjectName, kJointIds);
  const Eigen::MatrixXd jacobian = Eigen::Map<const Eigen::MatrixXd>(
      jacobian_vec.data(), 6, kJointIds.size());
  const Eigen::Vector<double, 6> target_vel(kTargetVelocity.data());
  const Eigen::MatrixXd indexed_jacobian =
      jacobian(indexer, Eigen::indexing::all);
  Eigen::VectorXd indexed_target_vel = target_vel(indexer);
  indexed_target_vel.normalize();
  std::vector<double> expected_coefficients(1 * kJointIds.size());
  Eigen::Map<Eigen::MatrixXd>(expected_coefficients.data(), 1,
                              kJointIds.size()) =
      indexed_target_vel.transpose() * indexed_jacobian;
  EXPECT_THAT(constraint.GetCoefficientMatrix(),
              Pointwise(DoubleEq(), expected_coefficients));
}

}  // namespace
}  // namespace dm_robotics
