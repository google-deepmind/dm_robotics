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

#include "dm_robotics/controllers/lsqp/joint_position_limit_constraint.h"

#include "dm_robotics/support/logging.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {

JointPositionLimitConstraint::JointPositionLimitConstraint(
    const Parameters& params, const mjData& data)
    : IdentityConstraint(params.joint_ids.size()),
      model_(*DieIfNull(params.model)),
      min_distance_from_limits_(params.min_distance_from_limits),
      gain_(params.gain),
      joint_ids_(params.joint_ids),
      lower_bound_(params.joint_ids.size()),
      upper_bound_(params.joint_ids.size()) {
  for (const int joint_id : params.joint_ids) {
    mjtJoint type = static_cast<mjtJoint>(model_.jnt_type[joint_id]);
    CHECK(type == mjtJoint::mjJNT_SLIDE || type == mjtJoint::mjJNT_HINGE)
        << absl::Substitute(
               "JointPositionLimitConstraint: Joint with ID [$0] is not a 1 "
               "DoF joint. Only 1 DoF joints are allowed at the moment.",
               joint_id);
    CHECK(model_.jnt_limited[joint_id]) << absl::Substitute(
        "JointPositionLimitConstraint: Joint with ID [$0] does not have "
        "limits. This constraint can only be defined for 1 DoF joints with "
        "limits.",
        joint_id);
  }
  UpdateBounds(data);
}

void JointPositionLimitConstraint::UpdateBounds(const mjData& data) {
  // joint_id_idx is the index of the upper/lower bound element for that joint.
  int joint_id_idx = 0;
  for (const int joint_id : joint_ids_) {
    // hi_lim/lo_lim are the joint limits once the minimum allowed distance from
    // limits is included.
    const int qpos_adr = model_.jnt_qposadr[joint_id];
    const double hi_lim =
        model_.jnt_range[2 * joint_id + 1] - min_distance_from_limits_;
    const double lo_lim =
        model_.jnt_range[2 * joint_id] + min_distance_from_limits_;

    // Compute the distances from the hi_lim/lo_lim (negative means
    // penetration).
    const double hi_lim_dist = hi_lim - data.qpos[qpos_adr];
    const double lo_lim_dist = data.qpos[qpos_adr] - lo_lim;

    // If hi_lim <= lo_lim, both bounds are set to zero; either it is already
    // violating the limits, or any non-zero joint velocity would cause it to
    // violate the limits. If any limit distance is less or equal to zero, the
    // corresponding uni-lateral bound for that joint is set to zero.
    if (hi_lim_dist > 0.0 && hi_lim > lo_lim) {
      upper_bound_[joint_id_idx] = gain_ * hi_lim_dist;
    } else {
      upper_bound_[joint_id_idx] = 0.0;
    }
    if (lo_lim_dist > 0.0 && hi_lim > lo_lim) {
      lower_bound_[joint_id_idx] = -gain_ * lo_lim_dist;
    } else {
      lower_bound_[joint_id_idx] = 0.0;
    }

    ++joint_id_idx;
  }
}

absl::Span<const double> JointPositionLimitConstraint::GetLowerBound() const {
  return lower_bound_;
}

absl::Span<const double> JointPositionLimitConstraint::GetUpperBound() const {
  return upper_bound_;
}

}  // namespace dm_robotics
