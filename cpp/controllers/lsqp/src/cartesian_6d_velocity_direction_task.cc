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

#include <algorithm>
#include <iterator>

#include "dm_robotics/support/logging.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"

namespace dm_robotics {
namespace {

// Returns the ID of the MuJoCo object with object_type and object_name.
//
// Check-fails if such object does not exist.
int GetObjectId(const MjLib& lib, const mjModel& model, mjtObj object_type,
                const std::string& object_name) {
  // Note: object_name must be a null-terminated string for the MuJoCo
  // interface, and we enforce this by having object_name be a reference to an
  // std::string.
  int id = lib.mj_name2id(&model, object_type, object_name.c_str());
  CHECK(id >= 0) << absl::Substitute(
      "GetObjectId: Could not find MuJoCo object with name [$0] and type [$1] "
      "in the provided model.",
      object_name, lib.mju_type2Str(object_type));
  return id;
}

// Copies the row-major jacobian elements corresponding to the joint_dof_ids
// into the column-major jacobian of the joint DoF IDs.
void RowMajorJacobianToJointDofJacobian(
    const mjModel& model, const absl::Span<const double> jacobian,
    const absl::btree_set<int>& joint_dof_ids,
    absl::Span<double> coefficient_matrix) {
  // Note that each joint DoF in joint_dof_ids accounts for exactly one column
  // in the coefficient matrix.
  Eigen::Map<const Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor>>
      jacobian_map(jacobian.data(), 6, model.nv);
  Eigen::Map<Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::ColMajor>>
      coefficients_map(coefficient_matrix.data(), 6, joint_dof_ids.size());

  int coefficient_column_counter = 0;
  for (int joint_dof_id : joint_dof_ids) {
    coefficients_map.col(coefficient_column_counter) =
        jacobian_map.col(joint_dof_id);
    ++coefficient_column_counter;
  }
}

}  // namespace

Cartesian6dVelocityDirectionTask::Cartesian6dVelocityDirectionTask(
    const Parameters& params, const mjData& data,
    absl::Span<const double> target_cartesian_velocity)
    : lib_(*DieIfNull(params.lib)),
      model_(*DieIfNull(params.model)),
      object_type_(params.object_type),
      object_id_(
          GetObjectId(lib_, model_, params.object_type, params.object_name)),
      weighting_matrix_(params.weighting_matrix),
      joint_dof_ids_(JointIdsToDofIds(model_, params.joint_ids)),
      jacobian_buffer_(6 * model_.nv),
      joint_dof_jacobian_buffer_(6 * joint_dof_ids_.size(), 0.0),
      weighted_coefficient_matrix_(6 * joint_dof_ids_.size(), 0.0) {
  CHECK(params.joint_ids.size() == joint_dof_ids_.size()) << absl::Substitute(
      "Cartesian6dVelocityDirectionTask: One or more joints have more than 1 "
      "DoF. Only 1 DoF joints are supported at the moment. Number of joint IDs "
      "provided was [$0], but total number of DoF resulted in [$1].",
      params.joint_ids.size(), joint_dof_ids_.size());
  CHECK(object_type_ == mjtObj::mjOBJ_BODY ||
        object_type_ == mjtObj::mjOBJ_GEOM ||
        object_type_ == mjtObj::mjOBJ_SITE)
      << absl::Substitute(
             "Cartesian6dVelocityDirectionTask: Objects of type [$0] are not "
             "supported. "
             "Only bodies, geoms, and sites are supported.",
             lib_.mju_type2Str(object_type_));

  std::fill(bias_.begin(), bias_.end(), 0.0);
  UpdateCoefficientsAndBias(data, target_cartesian_velocity);
}

// Parallel component vector:
// v_realized_component_in_target_direction * v_target_dir
//   = [v_target_dir^T v_realized] v_target_dir
//   = [v_target_dir v_target_dir^T] v_realized
//
// Perpendicular component vector:
// [v_target_dir v_target_dir^T] v_realized - v_realized
//  = [v_target_dir v_target_dir^T - I_6] v_realized
//  = [v_target_dir v_target_dir^T - I_6] J qdot
//
// This tasks minimizes the perpendicular component, i.e.
// it attempts to achieve a zero perpendicular component.
//
// C = (v_d v_d^T - I) J
// b = 0
void Cartesian6dVelocityDirectionTask::UpdateCoefficientsAndBias(
    const mjData& data, absl::Span<const double> target_cartesian_velocity) {
  CHECK(target_cartesian_velocity.size() == 6) << absl::Substitute(
      "UpdateCoefficientsAndBias: Invalid target_cartesian_velocity size [$0]. "
      "Must be of size 6.",
      target_cartesian_velocity.size());

  // Compute coefficient and bias.
  ComputeObject6dJacobian(lib_, model_, data, object_type_, object_id_,
                          absl::MakeSpan(jacobian_buffer_));
  RowMajorJacobianToJointDofJacobian(
      model_, jacobian_buffer_, joint_dof_ids_,
      absl::MakeSpan(joint_dof_jacobian_buffer_));
  Eigen::Map<const Eigen::Matrix<double, 6, Eigen::Dynamic>>
      joint_dof_jacobian_map(joint_dof_jacobian_buffer_.data(), 6,
                             joint_dof_ids_.size());

  // Compute [v_target_dir v_target_dir^T - I_6].
  Eigen::Map<const Eigen::Vector<double, 6>> target_vel_map(
      target_cartesian_velocity.data());
  Eigen::Map<Eigen::Vector<double, 6>> target_vel_dir_map(
      velocity_direction_buffer_.data());
  target_vel_dir_map = target_vel_map.normalized();
  Eigen::Matrix<double, 6, 6> jacobian_pre_multiplier =
      target_vel_dir_map * target_vel_dir_map.transpose() -
      Eigen::Matrix<double, 6, 6>::Identity();

  // Compute weighted coefficient matrix.
  Eigen::Map<Eigen::Matrix<double, 6, 6>> weighting_matrix_map(
      weighting_matrix_.data());
  Eigen::Map<Eigen::Matrix<double, 6, Eigen::Dynamic>>
      weighted_coefficient_matrix_map(weighted_coefficient_matrix_.data(), 6,
                                      joint_dof_ids_.size());
  weighted_coefficient_matrix_map.noalias() =
      weighting_matrix_map * jacobian_pre_multiplier * joint_dof_jacobian_map;
}

absl::Span<const double>
Cartesian6dVelocityDirectionTask::GetCoefficientMatrix() const {
  return weighted_coefficient_matrix_;
}

absl::Span<const double> Cartesian6dVelocityDirectionTask::GetBias() const {
  return bias_;
}

int Cartesian6dVelocityDirectionTask::GetNumberOfDof() const {
  return joint_dof_ids_.size();
}

int Cartesian6dVelocityDirectionTask::GetBiasLength() const {
  return bias_.size();
}

}  // namespace dm_robotics
