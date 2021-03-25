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

#ifndef DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_TASK_H_
#define DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_TASK_H_

#include <vector>

#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"

namespace dm_robotics {

// Task that biases the solution variables to a target vector through an
// identity transformation.
//
// This task's coefficient matrix is always an identity matrix with the number
// of rows and columns equal to the number of DoF, and the bias is the
// user-defined target vector.
class IdentityTask : public LsqpTask {
 public:
  // Constructs an identity task from a provided target vector and weight. The
  // number of DoF will be initialized to the length of the target vector.
  explicit IdentityTask(absl::Span<const double> target);

  IdentityTask(const IdentityTask&) = delete;
  IdentityTask& operator=(const IdentityTask&) = delete;

  // Sets a new target. Does not allocate memory.
  //
  // Check-fails if the number of DoF of the provided target vector is different
  // than the task's number of DoF.
  void SetTarget(absl::Span<const double> target);

  // LsqpTask virtual members.
  absl::Span<const double> GetCoefficientMatrix() const override;
  absl::Span<const double> GetBias() const override;
  int GetNumberOfDof() const override;
  int GetBiasLength() const override;

 private:
  std::vector<double> bias_;
  std::vector<double> coefficient_matrix_;
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_IDENTITY_TASK_H_
