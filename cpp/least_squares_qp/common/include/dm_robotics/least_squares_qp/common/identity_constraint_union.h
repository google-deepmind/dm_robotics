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

#ifndef DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_CONSTRAINT_UNION_H_
#define DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_CONSTRAINT_UNION_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"

namespace dm_robotics {

// LsqpConstraint resulting from the union of several identity constraints.
// The resultant constraint will have the same number of DoF as the provided
// constraints, and be such that if this constraint is satisfied, all the
// identity constraints will also be satisfied.
class IdentityConstraintUnion : public IdentityConstraint {
 public:
  // Constructs an empty union for IdentityConstraints with number of Dof equal
  // to num_dof. All the elements of the lower and upper bounds are initialized
  // to -infinity and infinity, respectively.
  explicit IdentityConstraintUnion(int num_dof);

  // Computes the minimal feasible bounds that satisfy all of the
  // IdentityConstraints in `constraints` and updates the lower and upper bounds
  // to match the computed bounds. If the constraints array is empty, all the
  // elements of the lower and upper bounds will be set to -infinity and
  // infinity, respectively.
  //
  // Returns a not-found error if it was not possible to find a set of feasible
  // bounds, i.e. if some constraints are conflicting with each other. In this
  // case, the lower and upper bounds will be set to -infinity and infinity,
  // respectively.
  //
  // Check-fails if the number of DoF of any provided constraint does not match
  // the number of DoF for the union.
  absl::Status UpdateFeasibleSpace(
      absl::Span<const IdentityConstraint* const> constraints);

  // IdentityConstraint virtual members.
  absl::Span<const double> GetLowerBound() const override;
  absl::Span<const double> GetUpperBound() const override;

 private:
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_CONSTRAINT_UNION_H_
