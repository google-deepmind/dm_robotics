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

#ifndef DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_BOX_CONSTRAINT_H_
#define DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_BOX_CONSTRAINT_H_

#include <vector>

#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"

namespace dm_robotics {

// Constraint that bounds the solution vector to be within an n-dimensional box.
// One-sided bounds can be implemented by setting the opposite bound to
// std::numeric_limits<double>::infinity.
class BoxConstraint : public IdentityConstraint {
 public:
  // Check-fails if both arrays are not the same size, or if any element in
  // `lower_bound` is greater than the respective element in `upper_bound`.
  BoxConstraint(absl::Span<const double> lower_bound,
                absl::Span<const double> upper_bound);

  // IdentityConstraint virtual members.
  absl::Span<const double> GetLowerBound() const override;
  absl::Span<const double> GetUpperBound() const override;

 private:
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_BOX_CONSTRAINT_H_
