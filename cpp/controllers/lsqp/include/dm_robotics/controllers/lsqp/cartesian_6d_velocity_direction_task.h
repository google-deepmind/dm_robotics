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

#ifndef DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_VELOCITY_DIRECTION_TASK_H_
#define DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_VELOCITY_DIRECTION_TASK_H_

#include <array>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"
#include "dm_robotics/mujoco/mjlib.h"

namespace dm_robotics {

// MuJoCo joint velocity control task that attempts to realize a target
// Cartesian 6D velocity direction by minimizing the velocity components
// orthogonal to the target velocity direction.
//
// The QP's decision variables are assumed to be the joint velocities of a
// pre-defined subset of MuJoCo joints, in ascending order according to their
// DoF IDs.
//
// Instances of this class cannot be moved or copied.
class Cartesian6dVelocityDirectionTask : public LsqpTask {
 public:
  // Initialization parameters for Cartesian6dVelocityDirectionTask that define
  // a MuJoCo reference frame and a subset of MuJoCo joints.
  //
  // The reference frame is defined by its MuJoCo mjtObj type and a string
  // representing its name. The MuJoCo object can be either a body, geom, or
  // site. Only 1 DoF joints are supported at the moment.
  //
  // The `weighting_matrix` parameter represents a 6x6 matrix (column-major)
  // containing the weights for each component of the Cartesian velocity being
  // controlled by this task, such that the quadratic cost term of this task is
  // defined as:
  //   || W (C q_dot - b) ||^2
  // where `W` is the weighting matrix; `C` is the coefficient matrix; `q_dot`
  // are the joint velocities; and `b` is the bias.
  //
  // The caller retains ownership of lib and model.
  // It is the caller's responsibility to ensure that the *lib and *model
  // objects outlive any Cartesian6dVelocityDirectionTask instances created with
  // this object.
  struct Parameters {
    const MjLib* lib;
    const mjModel* model;
    absl::btree_set<int> joint_ids;
    mjtObj object_type;
    std::string object_name;
    std::array<double, 36> weighting_matrix = {
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,  //
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  //
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  //
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  //
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0   //
    };
  };

  // Constructs a Cartesian6dVelocityDirectionTask.
  //
  // The task's coefficients and bias are initialized according to the provided
  // `data` and `target_cartesian_velocity` parameters, which encode the current
  // MuJoCo environment state and the target Cartesian 6D velocity,
  // respectively. At every iteration, the coefficients and bias can be updated
  // through a call to `UpdateCoefficientsAndBias`.
  // Note: all the necessary MuJoCo computations should have been performed on
  // the `data` parameter for the Jacobian computations to to be accurate.
  // Namely, mj_kinematics and mj_comPos at the very least.
  Cartesian6dVelocityDirectionTask(
      const Parameters& params, const mjData& data,
      absl::Span<const double> target_cartesian_velocity);

  Cartesian6dVelocityDirectionTask(const Cartesian6dVelocityDirectionTask&) =
      delete;
  Cartesian6dVelocityDirectionTask& operator=(
      const Cartesian6dVelocityDirectionTask&) = delete;

  // Updates the coefficient matrix and bias based on an mjData and the target
  // Cartesian velocity.
  //
  // Note: all the necessary MuJoCo computations should have been performed on
  // the `data` parameter for the Jacobian computations to to be accurate.
  // Namely, mj_kinematics and mj_comPos at the very least.
  //
  // Check-fails if target_cartesian_velocity is not a view over a 6-dimensional
  // array.
  void UpdateCoefficientsAndBias(
      const mjData& data, absl::Span<const double> target_cartesian_velocity);

  // LsqpTask virtual members.
  absl::Span<const double> GetCoefficientMatrix() const override;
  absl::Span<const double> GetBias() const override;
  int GetNumberOfDof() const override;
  int GetBiasLength() const override;

 private:
  const MjLib& lib_;
  const mjModel& model_;
  mjtObj object_type_;
  int object_id_;
  std::array<double, 6> velocity_direction_buffer_;
  std::array<double, 6> bias_;
  std::array<double, 36> weighting_matrix_;

  absl::btree_set<int> joint_dof_ids_;
  std::vector<double> jacobian_buffer_;
  std::vector<double> joint_dof_jacobian_buffer_;
  std::vector<double> weighted_coefficient_matrix_;
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_VELOCITY_DIRECTION_TASK_H_
