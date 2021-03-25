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

#include "dm_robotics/least_squares_qp/common/identity_constraint_union.h"

#include <algorithm>
#include <limits>

#include "dm_robotics/support/logging.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"

namespace dm_robotics {
namespace {

// Returns true if the elements in lower_bound are equal or lower than the
// respective elements in upper_bound. Returns false otherwise.
bool AreBoundsFeasible(absl::Span<const double> lower_bound,
                       absl::Span<const double> upper_bound) {
  int num_dof = lower_bound.size();
  for (int i = 0; i < num_dof; ++i) {
    if (lower_bound[i] > upper_bound[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

IdentityConstraintUnion::IdentityConstraintUnion(int num_dof)
    : IdentityConstraint(num_dof),
      lower_bound_(num_dof, -std::numeric_limits<double>::infinity()),
      upper_bound_(num_dof, std::numeric_limits<double>::infinity()) {}

absl::Status IdentityConstraintUnion::UpdateFeasibleSpace(
    absl::Span<const IdentityConstraint* const> constraints) {
  std::fill(lower_bound_.begin(), lower_bound_.end(),
            -std::numeric_limits<double>::infinity());
  std::fill(upper_bound_.begin(), upper_bound_.end(),
            std::numeric_limits<double>::infinity());

  int num_dof = GetNumberOfDof();
  for (int i = 0; i < constraints.size(); ++i) {
    CHECK(constraints[i]->GetNumberOfDof() == num_dof) << absl::Substitute(
        "IdentityConstraintUnion::UpdateFeasibleSpace: "
        "Constraint with index [$0] with number of Dof [$1] has a "
        "different number of DoF than this constraint [$2]. Number of "
        "DoF cannot be updated after construction.",
        i, constraints[i]->GetNumberOfDof(), num_dof);
    for (int j = 0; j < num_dof; ++j) {
      lower_bound_[j] =
          std::max(lower_bound_[j], constraints[i]->GetLowerBound()[j]);
      upper_bound_[j] =
          std::min(upper_bound_[j], constraints[i]->GetUpperBound()[j]);
    }
  }

  // If bounds are not feasible, set bounds to infinity and return an error.
  if (!AreBoundsFeasible(lower_bound_, upper_bound_)) {
    std::fill(lower_bound_.begin(), lower_bound_.end(),
              -std::numeric_limits<double>::infinity());
    std::fill(upper_bound_.begin(), upper_bound_.end(),
              std::numeric_limits<double>::infinity());
    return absl::NotFoundError(
        "IdentityConstraintUnion::UpdateFeasibleSpace: Resulting bounds are "
        "infeasible. This can happen if two or more constraints have disjoint "
        "feasible spaces.");
  }

  return absl::OkStatus();
}

absl::Span<const double> IdentityConstraintUnion::GetLowerBound() const {
  return lower_bound_;
}
absl::Span<const double> IdentityConstraintUnion::GetUpperBound() const {
  return upper_bound_;
}

}  // namespace dm_robotics
