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

#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_task.h"

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
// into the column-major coefficient_matrix.
void RowMajorJacobianToCoefficientMatrix(
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

Cartesian6dVelocityTask::Cartesian6dVelocityTask(
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
      coefficient_matrix_(6 * joint_dof_ids_.size(), 0.0),
      weighted_coefficient_matrix_(6 * joint_dof_ids_.size(), 0.0) {
  CHECK(params.joint_ids.size() == joint_dof_ids_.size()) << absl::Substitute(
      "Cartesian6dVelocityTask: One or more joints have more than 1 DoF. Only "
      "1 DoF joints are supported at the moment. Number of joint IDs provided "
      "was [$0], but total number of DoF resulted in [$1].",
      params.joint_ids.size(), joint_dof_ids_.size());
  CHECK(object_type_ == mjtObj::mjOBJ_BODY ||
        object_type_ == mjtObj::mjOBJ_GEOM ||
        object_type_ == mjtObj::mjOBJ_SITE)
      << absl::Substitute(
             "Cartesian6dVelocityTask: Objects of type [$0] are not supported. "
             "Only bodies, geoms, and sites are supported.",
             lib_.mju_type2Str(object_type_));

  UpdateCoefficientsAndBias(data, target_cartesian_velocity);
}

void Cartesian6dVelocityTask::UpdateCoefficientsAndBias(
    const mjData& data, absl::Span<const double> target_cartesian_velocity) {
  CHECK(target_cartesian_velocity.size() == 6) << absl::Substitute(
      "UpdateCoefficientsAndBias: Invalid target_cartesian_velocity size [$0]. "
      "Must be of size 6.",
      target_cartesian_velocity.size());

  // Compute un-weighted coefficient and bias.
  ComputeObject6dJacobian(lib_, model_, data, object_type_, object_id_,
                          absl::MakeSpan(jacobian_buffer_));
  RowMajorJacobianToCoefficientMatrix(model_, jacobian_buffer_, joint_dof_ids_,
                                      absl::MakeSpan(coefficient_matrix_));
  std::copy(target_cartesian_velocity.begin(), target_cartesian_velocity.end(),
            bias_.begin());

  // Add weights for coefficient and bias.
  Eigen::Map<Eigen::Matrix<double, 6, 6>> weighting_matrix_map(
      weighting_matrix_.data());

  Eigen::Map<Eigen::Matrix<double, 6, Eigen::Dynamic>> coefficient_matrix_map(
      coefficient_matrix_.data(), 6, joint_dof_ids_.size());
  Eigen::Map<Eigen::Matrix<double, 6, Eigen::Dynamic>>
      weighted_coefficient_matrix_map(weighted_coefficient_matrix_.data(), 6,
                                      joint_dof_ids_.size());
  weighted_coefficient_matrix_map.noalias() =
      weighting_matrix_map * coefficient_matrix_map;

  Eigen::Map<Eigen::Vector<double, 6>> bias_map(bias_.data());
  bias_map = weighting_matrix_map * bias_map;
}

absl::Span<const double> Cartesian6dVelocityTask::GetCoefficientMatrix() const {
  return weighted_coefficient_matrix_;
}

absl::Span<const double> Cartesian6dVelocityTask::GetBias() const {
  return bias_;
}

int Cartesian6dVelocityTask::GetNumberOfDof() const {
  return joint_dof_ids_.size();
}

int Cartesian6dVelocityTask::GetBiasLength() const { return bias_.size(); }

}  // namespace dm_robotics
