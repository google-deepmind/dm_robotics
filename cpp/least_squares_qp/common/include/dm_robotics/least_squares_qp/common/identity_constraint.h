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

#ifndef DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_CONSTRAINT_H_
#define DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_CONSTRAINT_H_

#include <vector>

#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/math_utils.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"

namespace dm_robotics {

// Abstraction for an LsqpConstraint with an identity coefficient matrix.
class IdentityConstraint : public LsqpConstraint {
 public:
  explicit IdentityConstraint(int num_dof)
      : num_dof_(num_dof), coefficient_matrix_(MakeIdentityMatrix(num_dof)) {}

  IdentityConstraint(const IdentityConstraint&) = delete;
  IdentityConstraint& operator=(const IdentityConstraint&) = delete;
  ~IdentityConstraint() override = default;

  // LsqpConstraint virtual members.
  absl::Span<const double> GetLowerBound() const override = 0;
  absl::Span<const double> GetUpperBound() const override = 0;

  absl::Span<const double> GetCoefficientMatrix() const final {
    return coefficient_matrix_;
  }
  int GetNumberOfDof() const final { return num_dof_; }
  int GetBoundsLength() const final { return num_dof_; }

 private:
  int num_dof_;
  std::vector<double> coefficient_matrix_;
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_CONSTRAINT_H_
