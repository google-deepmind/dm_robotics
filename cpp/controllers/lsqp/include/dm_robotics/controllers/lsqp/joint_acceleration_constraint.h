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

#ifndef DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_ACCELERATION_CONSTRAINT_H_
#define DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_ACCELERATION_CONSTRAINT_H_

#include <vector>

#include "absl/container/btree_set.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {

// Inequality constraint on the velocity of MuJoCo joints to impose a
// restriction on their acceleration. The QP's decision variables are assumed to
// be the joint velocities of a pre-defined subset of MuJoCo joints, in
// ascending order according to their DoF IDs. At the moment, only 1 DoF joints
// are supported.
//
// This inequality constraint is formulated as:
//   V_curr - A_lim * T < V < V_curr + A_lim * T
// where
//   * V is the vector of joint velocities to be computed;
//   * V_curr is the vector of current joint velocities;
//   * A_lim is the acceleration limit for each joint;
//   * T is a positive real number that determines how long the velocity will be
//     executed for (a.k.a. integration timestep).
class JointAccelerationConstraint : public IdentityConstraint {
 public:
  // Initialization parameters for JointAccelerationLimits.
  //
  // Only 1 DoF joints are supported at the moment. The elements of the
  // `acceleration_magnitude_limits` array must be ordered in ascending order of
  // DoF IDs. If any element of the `acceleration_magnitude_limits` array
  // is negative, its absolute value is used instead.
  //
  // The caller retains ownership of model.
  // It is the caller's responsibility to ensure that the *model object outlives
  // any JointAccelerationConstraint instances created with this object.
  struct Parameters {
    const mjModel* model;
    absl::btree_set<int> joint_ids;
    absl::Duration integration_timestep;
    std::vector<double> acceleration_magnitude_limits;
  };

  // Constructs a JointAccelerationConstraint, and sets the bounds based on an
  // mjData object.
  JointAccelerationConstraint(const Parameters& params, const mjData& data);

  JointAccelerationConstraint(const JointAccelerationConstraint&) = delete;
  JointAccelerationConstraint& operator=(const JointAccelerationConstraint&) =
      delete;

  // Updates the bounds of the constraint based on an mjData object. The mjData
  // must contain updated joint velocity information.
  //
  // This function does not perform dynamic memory allocation.
  void UpdateBounds(const mjData& data);

  // LsqpConstraint virtual members.
  absl::Span<const double> GetLowerBound() const override;
  absl::Span<const double> GetUpperBound() const override;

 private:
  const mjModel& model_;
  double integration_timestep_seconds_;
  absl::btree_set<int> joint_dof_ids_;
  std::vector<double> acceleration_magnitude_limits_;

  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_ACCELERATION_CONSTRAINT_H_
