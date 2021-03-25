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

#ifndef LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_TASK_H_
#define LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_TASK_H_

#include "absl/types/span.h"

namespace dm_robotics {

// Abstraction for a least-squares QP task.
//
// A task is defined by the following term:
//  ||Mx - b||^2
// where M is the coefficient matrix; x is the vector of decision variables, and
// b is the bias vector.
//
// A set of tasks on the same vector of decision variables can be used to build
// a QP optimization problem, which attempts to find the values of decision
// variables that minimize the summation of all the tasks.
class LsqpTask {
 public:
  // Returns a view of the column-major ordered array representing the
  // coefficient matrix. The array will have its number of rows equal to the
  // bias length, and its number of columns equal to the number of DoF of the
  // task.
  virtual absl::Span<const double> GetCoefficientMatrix() const = 0;

  // Returns a view of the array representing the bias for the task.
  virtual absl::Span<const double> GetBias() const = 0;

  // Returns the number of DoF for the task, i.e. length of `x`.
  virtual int GetNumberOfDof() const = 0;

  // Returns the bias length for the task, i.e. length of `b`.
  virtual int GetBiasLength() const = 0;

  virtual ~LsqpTask() = default;
};

}  // namespace dm_robotics

#endif  // LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_TASK_H_
