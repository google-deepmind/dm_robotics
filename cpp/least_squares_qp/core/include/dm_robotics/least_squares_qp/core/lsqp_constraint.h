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

#ifndef LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_CONSTRAINT_H_
#define LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_CONSTRAINT_H_

#include "absl/types/span.h"

namespace dm_robotics {

// Abstraction for a least-squares QP constraint.
//
// A constraint is defined by the following term:
//  l <= Ax <= u
// where A is the coefficient matrix; l are the lower bounds; and u are the
// upper bounds.
//
// A set of constraints on the same vector of decision variables defines the
// feasible space for the decision variables.
class LsqpConstraint {
 public:
  // Returns a view of the column-major ordered array representing the
  // coefficient matrix. The array will have its number of rows equal to the
  // bounds length, and its number of columns equal to the number of DoF of the
  // constraint.
  virtual absl::Span<const double> GetCoefficientMatrix() const = 0;

  // Returns a view of the array representing the lower bound for the
  // constraint.
  virtual absl::Span<const double> GetLowerBound() const = 0;

  // Returns a view of the array representing the upper bound for the
  // constraint.
  virtual absl::Span<const double> GetUpperBound() const = 0;

  // Returns the number of DoF for the constraint, i.e. length of `x`.
  virtual int GetNumberOfDof() const = 0;

  // Returns the bounds length for the constraint, i.e. length of `u` and `l`.
  virtual int GetBoundsLength() const = 0;

  virtual ~LsqpConstraint() = default;
};

}  // namespace dm_robotics

#endif  // LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_CONSTRAINT_H_
