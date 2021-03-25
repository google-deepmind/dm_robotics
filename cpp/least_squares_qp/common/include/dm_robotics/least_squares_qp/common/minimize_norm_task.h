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

#ifndef DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_MINIMIZE_NORM_TASK_H_
#define DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_MINIMIZE_NORM_TASK_H_

#include <vector>

#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/identity_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"

namespace dm_robotics {

// Task that attempts to minimize the magnitude of the solution variables.
//
// This task's coefficient matrix is always an identity matrix with the number
// of rows and columns equal to the number of DoF, and the bias is always the
// zero vector with length equal to the number of DoF.
class MinimizeNormTask : public LsqpTask {
 public:
  explicit MinimizeNormTask(const int num_dof)
      : identity_task_(std::vector<double>(num_dof, 0.0)) {}

  MinimizeNormTask(const MinimizeNormTask&) = delete;
  MinimizeNormTask& operator=(const MinimizeNormTask&) = delete;

  absl::Span<const double> GetCoefficientMatrix() const override {
    return identity_task_.GetCoefficientMatrix();
  }

  absl::Span<const double> GetBias() const override {
    return identity_task_.GetBias();
  }

  int GetNumberOfDof() const override {
    return identity_task_.GetNumberOfDof();
  }

  int GetBiasLength() const override { return identity_task_.GetBiasLength(); }

 private:
  IdentityTask identity_task_;
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_LEAST_SQUARES_QP_COMMON_MINIMIZE_NORM_TASK_H_
