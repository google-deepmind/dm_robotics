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

#include "dm_robotics/least_squares_qp/common/identity_task.h"

#include <algorithm>
#include <vector>

#include "dm_robotics/support/logging.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/math_utils.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"

namespace dm_robotics {

IdentityTask::IdentityTask(absl::Span<const double> target)
    : bias_(target.begin(), target.end()),
      coefficient_matrix_(MakeIdentityMatrix(bias_.size())) {}

void IdentityTask::SetTarget(absl::Span<const double> target) {
  CHECK(target.size() == bias_.size()) << absl::Substitute(
      "IdentityTask::SetTarget: Number of DoF mismatch. Provided target "
      "size is [$0] but task was constructed with [$1] DoF.",
      target.size(), bias_.size());
  std::copy(target.begin(), target.end(), bias_.begin());
}

absl::Span<const double> IdentityTask::GetCoefficientMatrix() const {
  return coefficient_matrix_;
}

absl::Span<const double> IdentityTask::GetBias() const { return bias_; }

int IdentityTask::GetNumberOfDof() const { return bias_.size(); }

int IdentityTask::GetBiasLength() const { return bias_.size(); }

}  // namespace dm_robotics
