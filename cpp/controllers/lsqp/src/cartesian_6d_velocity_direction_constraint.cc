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

#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_direction_constraint.h"

#include <limits>

#include "dm_robotics/support/logging.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {
namespace {

// Returns the ID of the MuJoCo object with object_type and object_name.
//
// Check-fails if such object does not exist.
int GetObjectId(const mjModel& model, mjtObj object_type,
                const std::string& object_name) {
  // Note: object_name must be a null-terminated string for the MuJoCo
  // interface, and we enforce this by having object_name be a reference to an
  // std::string.
  int id = mj_name2id(&model, object_type, object_name.c_str());
  CHECK(id >= 0) << absl::Substitute(
      "GetObjectId: Could not find MuJoCo object with name [$0] and type [$1] "
      "in the provided model.",
      object_name, mju_type2Str(object_type));
  return id;
}

std::vector<int> GetVelocityIndexer(const std::array<bool, 6>& enable_flags) {
  std::vector<int> indexer;
  for (int i = 0; i < 6; ++i) {
    if (enable_flags[i]) {
      indexer.push_back(i);
    }
  }
  return indexer;
}

// Copies the row-major jacobian elements corresponding to the joint_dof_ids
// into the column-major jacobian of the joint DoF IDs.
void RowMajorJacobianToJointDofJacobian(
    const mjModel& model, const absl::Span<const double> jacobian,
    const absl::btree_set<int>& joint_dof_ids,
    absl::Span<const int> velocity_indexer,
    absl::Span<double> joint_dof_jacobian) {
  // Note that each joint DoF in joint_dof_ids accounts for exactly one column
  // in the coefficient matrix.
  Eigen::Map<const Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor>>
      jacobian_map(jacobian.data(), 6, model.nv);
  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
      joint_dof_jacobian_map(joint_dof_jacobian.data(), velocity_indexer.size(),
                             joint_dof_ids.size());

  int coefficient_column_counter = 0;
  for (int joint_dof_id : joint_dof_ids) {
    joint_dof_jacobian_map.col(coefficient_column_counter) =
        jacobian_map.col(joint_dof_id)(velocity_indexer);
    ++coefficient_column_counter;
  }
}

}  // namespace

Cartesian6dVelocityDirectionConstraint::Cartesian6dVelocityDirectionConstraint(
    const Parameters& params, const mjData& data,
    absl::Span<const double> target_cartesian_velocity)
    : model_(*DieIfNull(params.model)),
      object_type_(params.object_type),
      object_id_(
          GetObjectId(model_, params.object_type, params.object_name)),
      velocity_indexer_(GetVelocityIndexer(params.enable_axis_constraint)),
      joint_dof_ids_(JointIdsToDofIds(model_, params.joint_ids)),
      jacobian_buffer_(6 * model_.nv),
      joint_dof_jacobian_buffer_(
          velocity_indexer_.size() * joint_dof_ids_.size(), 0.0),
      velocity_direction_buffer_(velocity_indexer_.size()),
      lower_bound_({0.0}),
      upper_bound_({std::numeric_limits<double>::infinity()}),
      coefficient_matrix_(1 * joint_dof_ids_.size(), 0.0) {
  CHECK(params.joint_ids.size() == joint_dof_ids_.size()) << absl::Substitute(
      "Cartesian6dVelocityDirectionConstraint: One or more joints "
      "have more than 1 DoF. Only 1 DoF joints are supported at the moment. "
      "Number of joint IDs provided was [$0], but total number of DoF resulted "
      "in [$1].",
      params.joint_ids.size(), joint_dof_ids_.size());
  CHECK(object_type_ == mjtObj::mjOBJ_BODY ||
        object_type_ == mjtObj::mjOBJ_GEOM ||
        object_type_ == mjtObj::mjOBJ_SITE)
      << absl::Substitute(
             "Cartesian6dVelocityDirectionConstraint: Objects of type [$0] are "
             "not supported. Only bodies, geoms, and sites are supported.",
             mju_type2Str(object_type_));
  CHECK(!velocity_indexer_.empty())
      << "Cartesian6dVelocityDirectionConstraint: all elements of the "
         "`enable_axis_constraint` parameter cannot be false.";

  UpdateCoefficients(data, target_cartesian_velocity);
}

// 0.0 <= v_target_dir^T J qdot <= infinity
void Cartesian6dVelocityDirectionConstraint::UpdateCoefficients(
    const mjData& data, absl::Span<const double> target_cartesian_velocity) {
  CHECK(target_cartesian_velocity.size() == 6) << absl::Substitute(
      "UpdateCoefficients: Invalid target_cartesian_velocity size [$0]. Must "
      "be of size 6.",
      target_cartesian_velocity.size());

  // Compute Jacobian for the enabled velocities.
  ComputeObject6dJacobian(model_, data, object_type_, object_id_,
                          absl::MakeSpan(jacobian_buffer_));
  RowMajorJacobianToJointDofJacobian(
      model_, jacobian_buffer_, joint_dof_ids_, velocity_indexer_,
      absl::MakeSpan(joint_dof_jacobian_buffer_));
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      joint_dof_jacobian_map(joint_dof_jacobian_buffer_.data(),
                             velocity_indexer_.size(), joint_dof_ids_.size());

  // Compute velocity direction.
  Eigen::Map<const Eigen::Vector<double, 6>> target_vel_map(
      target_cartesian_velocity.data());
  Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> target_vel_dir_map(
      velocity_direction_buffer_.data(), velocity_direction_buffer_.size());
  target_vel_dir_map = target_vel_map(velocity_indexer_).normalized();

  // Compute coefficient matrix.
  Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic>> coefficient_matrix_map(
      coefficient_matrix_.data(), 1, joint_dof_ids_.size());
  coefficient_matrix_map =
      target_vel_dir_map.transpose() * joint_dof_jacobian_map;
}

absl::Span<const double>
Cartesian6dVelocityDirectionConstraint::GetCoefficientMatrix() const {
  return coefficient_matrix_;
}

absl::Span<const double> Cartesian6dVelocityDirectionConstraint::GetLowerBound()
    const {
  return lower_bound_;
}

absl::Span<const double> Cartesian6dVelocityDirectionConstraint::GetUpperBound()
    const {
  return upper_bound_;
}

int Cartesian6dVelocityDirectionConstraint::GetNumberOfDof() const {
  return joint_dof_ids_.size();
}

int Cartesian6dVelocityDirectionConstraint::GetBoundsLength() const {
  return 1;
}

}  // namespace dm_robotics
