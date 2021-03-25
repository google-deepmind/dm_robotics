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

#include "dm_robotics/controllers/lsqp/joint_acceleration_constraint.h"

#include "dm_robotics/support/logging.h"
#include "absl/container/btree_set.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/utils.h"

namespace dm_robotics {

// Computes the lower and upper bounds for the JointAccelerationConstraint.
//
// Assumes that `lower_bound` and `upper_bound` is the same length as the number
// of DoF.
void ComputeLowerAndUpperBounds(
    double integration_timestep_seconds, const absl::btree_set<int>& dof_ids,
    const mjData& data, absl::Span<const double> acceleration_magnitude_limits,
    absl::Span<double> lower_bound, absl::Span<double> upper_bound) {
  // Note that the joint velocities in mjData that we are interested in may not
  // necessarily be contiguous, but `acceleration_magnitude_limits`,
  // `lower_bound`, and `upper_bound` are.
  int bounds_idx = 0;
  for (int dof_id : dof_ids) {
    double acceleration_limit =
        std::abs(acceleration_magnitude_limits[bounds_idx]);
    lower_bound[bounds_idx] =
        data.qvel[dof_id] - integration_timestep_seconds * acceleration_limit;
    upper_bound[bounds_idx] =
        data.qvel[dof_id] + integration_timestep_seconds * acceleration_limit;
    ++bounds_idx;
  }
}

JointAccelerationConstraint::JointAccelerationConstraint(
    const Parameters& params, const mjData& data)
    : IdentityConstraint(params.acceleration_magnitude_limits.size()),
      model_(*DieIfNull(params.model)),
      integration_timestep_seconds_(
          absl::ToDoubleSeconds(params.integration_timestep)),
      joint_dof_ids_(JointIdsToDofIds(model_, params.joint_ids)),
      acceleration_magnitude_limits_(params.acceleration_magnitude_limits),
      lower_bound_(acceleration_magnitude_limits_.size()),
      upper_bound_(acceleration_magnitude_limits_.size()) {
  CHECK(params.joint_ids.size() == joint_dof_ids_.size()) << absl::Substitute(
      "JointAccelerationConstraint: One or more joints have more than 1 DoF. "
      "Only 1 DoF joints are supported at the moment. Number of joint IDs "
      "provided was [$0], but total number of DoF resulted in [$1].",
      params.joint_ids.size(), joint_dof_ids_.size());
  CHECK(acceleration_magnitude_limits_.size() == params.joint_ids.size())
      << absl::Substitute(
             "JointAccelerationConstraint: `acceleration_magnitude_limits` "
             "array [$0] is not the same size as the number of DoF [$1].",
             acceleration_magnitude_limits_.size(), params.joint_ids.size());

  ComputeLowerAndUpperBounds(integration_timestep_seconds_, joint_dof_ids_,
                             data, acceleration_magnitude_limits_,
                             absl::MakeSpan(lower_bound_),
                             absl::MakeSpan(upper_bound_));
}

void JointAccelerationConstraint::UpdateBounds(const mjData& data) {
  ComputeLowerAndUpperBounds(integration_timestep_seconds_, joint_dof_ids_,
                             data, acceleration_magnitude_limits_,
                             absl::MakeSpan(lower_bound_),
                             absl::MakeSpan(upper_bound_));
}

absl::Span<const double> JointAccelerationConstraint::GetLowerBound() const {
  return lower_bound_;
}

absl::Span<const double> JointAccelerationConstraint::GetUpperBound() const {
  return upper_bound_;
}

}  // namespace dm_robotics
