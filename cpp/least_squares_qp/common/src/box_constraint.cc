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

#include "dm_robotics/least_squares_qp/common/box_constraint.h"

#include "dm_robotics/support/logging.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"

namespace dm_robotics {

BoxConstraint::BoxConstraint(absl::Span<const double> lower_bound,
                             absl::Span<const double> upper_bound)
    : IdentityConstraint(lower_bound.size()),
      lower_bound_(lower_bound.begin(), lower_bound.end()),
      upper_bound_(upper_bound.begin(), upper_bound.end()) {
  // Ensure same size.
  CHECK(lower_bound.size() == upper_bound.size()) << absl::Substitute(
      "BoxConstraint: Number of DoF mismatch between `lower_bounds` with size "
      "[$0] and `upper_bounds` with size [$1].",
      lower_bound.size(), upper_bound.size());

  // Ensure constraint is feasible.
  for (int i = 0; i < lower_bound.size(); ++i) {
    CHECK(lower_bound[i] <= upper_bound[i]) << absl::Substitute(
        "BoxConstraint: Constraint is infeasible. Element [$0] of "
        "`lower_bound` with value [$1] is larger than the respective element "
        "in `upper_bound` with value [$2]",
        i, lower_bound[i], upper_bound[i]);
  }
}

absl::Span<const double> BoxConstraint::GetLowerBound() const {
  return lower_bound_;
}

absl::Span<const double> BoxConstraint::GetUpperBound() const {
  return upper_bound_;
}

}  // namespace dm_robotics
