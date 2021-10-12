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

#include "dm_robotics/controllers/lsqp/test_utils.h"

#include <algorithm>
#include <string>

#include "dm_robotics/support/logging.h"
#include "absl/container/btree_set.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"

namespace dm_robotics::testing {

std::vector<double> ComputeObject6dJacobianForJoints(
    const MjLib& lib, const mjModel& model, const mjData& data,
    mjtObj object_type, const std::string& object_name,
    const absl::btree_set<int>& joint_ids) {
  int object_id = lib.mj_name2id(&model, object_type, object_name.c_str());

  // We need a vector of DoF IDs for indexing Eigen::MatrixXd.
  absl::btree_set<int> dof_ids = JointIdsToDofIds(model, joint_ids);
  std::vector<int> dof_ids_vector(dof_ids.begin(), dof_ids.end());

  // Compute Jacobian and convert to column-major.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      row_jacobian_buffer(6, model.nv);
  ComputeObject6dJacobian(
      lib, model, data, object_type, object_id,
      absl::MakeSpan(row_jacobian_buffer.data(), row_jacobian_buffer.size()));
  Eigen::MatrixXd col_jacobian_buffer = row_jacobian_buffer;

  // Get the columns corresponding to the DoF IDs.
  std::vector<double> jacobian(6 * dof_ids.size());
  Eigen::Map<Eigen::MatrixXd>(jacobian.data(), 6, dof_ids.size()) =
      col_jacobian_buffer(Eigen::indexing::all, dof_ids_vector);
  return jacobian;
}

std::array<double, 6> ComputeObjectCartesian6dVelocityWithJacobian(
    const MjLib& lib, const mjModel& model, const mjData& data,
    const std::string& object_name, mjtObj object_type) {
  Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor> jacobian(6,
                                                                     model.nv);
  int object_id = lib.mj_name2id(&model, object_type, object_name.c_str());
  ComputeObject6dJacobian(lib, model, data, object_type, object_id,
                          absl::MakeSpan(jacobian.data(), jacobian.size()));

  std::array<double, 6> cartesian_velocity;
  Eigen::Map<Eigen::Vector<double, 6>> cartesian_map(cartesian_velocity.data());
  Eigen::Map<const Eigen::VectorXd> qvel_map(data.qvel, model.nv);
  cartesian_map = jacobian * qvel_map;
  return cartesian_velocity;
}

void SetSubsetOfJointVelocities(const mjModel& model,
                                const absl::btree_set<int>& joint_ids,
                                absl::Span<const double> joint_velocities,
                                mjData* data) {
  CHECK(joint_ids.size() == joint_velocities.size()) << absl::Substitute(
      "SetSubsetOfJointVelocities: Number of joints[$0] does not match the "
      "number of joint velocities[$1]. Only 1 DoF joints are supported.",
      joint_ids.size(), joint_velocities.size());

  // Update MuJoCo data.
  std::fill_n(data->qvel, model.nv, 0.0);
  int decision_var_counter = 0;
  for (int joint_id : joint_ids) {
    int dof_adr = model.jnt_dofadr[joint_id];
    data->qvel[dof_adr] = joint_velocities.at(decision_var_counter);
    ++decision_var_counter;
  }
}

}  // namespace dm_robotics::testing
