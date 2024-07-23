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

#ifndef DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_POSITION_LIMIT_CONSTRAINT_H_
#define DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_POSITION_LIMIT_CONSTRAINT_H_

#include <vector>

#include "absl/container/btree_set.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {

// Inequality constraint on the velocity of MuJoCo joints to prevent joints from
// going off-limits. At the moment, only 1 DoF joints are supported.
//
// This inequality constraint is formulated as:
//   G (low_lim - curr_pos + min_dist) < V < G (hi_lim - curr_pos - min_dist)
// where
//   * V is the vector of joint velocities to be computed;
//   * G is the gain;
//   * curr_pos is the current joint position;
//   * min_dist is the minimum allowed distance from limits;
//   * low_lim and hi_lim are the low and high limits for the joint,
//     respectively.
//
// The gain parameter G is usually set to:
//  G = K/T
// where K is a positive real number (0,1] that determines the
// percentage of the maximum velocity allowed in each timestep (the smaller the
// value the more conservative the constraint will be); and T is a positive real
// number that determines how long the velocity will be executed for, (a.k.a.
// integration timestep).
//
// If any joint position is already violating its lower (upper) limit, the
// lower (upper) bound on the velocity for that joint will be set to zero.
// This is to prevent spring-like velocities in the case that the current
// position violates the limits, by ensuring that the zero-velocity
// vector is always a valid solution to the decision variables.
class JointPositionLimitConstraint : public IdentityConstraint {
 public:
  // Initialization parameters for JointPositionLimitConstraint.
  // Only 1 DoF joints are supported at the moment.
  //
  // The caller retains ownership of model.
  // It is the caller's responsibility to ensure that the *model object outlives
  // any JointPositionLimitConstraint instances created with this object.
  struct Parameters {
    const mjModel* model;
    double min_distance_from_limits;
    double gain;
    absl::btree_set<int> joint_ids;
  };

  // Constructs a JointPositionLimitConstraint.
  JointPositionLimitConstraint(const Parameters& params, const mjData& data);

  JointPositionLimitConstraint(const JointPositionLimitConstraint&) = delete;
  JointPositionLimitConstraint& operator=(const JointPositionLimitConstraint&) =
      delete;

  // Does not perform dynamic memory allocation.
  void UpdateBounds(const mjData& data);

  // IdentityConstraint virtual members.
  absl::Span<const double> GetLowerBound() const override;
  absl::Span<const double> GetUpperBound() const override;

 private:
  const mjModel& model_;
  double min_distance_from_limits_;
  double gain_;
  absl::btree_set<int> joint_ids_;

  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
};

}  // namespace dm_robotics
#endif  // DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_POSITION_LIMIT_CONSTRAINT_H_
