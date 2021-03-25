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

#ifndef LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_TESTING_MATCHERS_H_
#define LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_TESTING_MATCHERS_H_

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "absl/strings/substitute.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"

namespace dm_robotics::testing {

MATCHER(LsqpTaskDimensionsAreValid,
        "coefficient matrix and bias dimensions are consistent with return "
        "values of LsqpTask::GetNumberOfDof and LsqpTask::GetBiasLength") {
  const LsqpTask& task = arg;
  auto m = task.GetCoefficientMatrix();
  auto b = task.GetBias();
  int dof = task.GetNumberOfDof();
  int b_length = task.GetBiasLength();

  if (m.size() != b_length * dof) {
    *result_listener << absl::Substitute(
        "coefficient matrix size is [$0] but expected [$1]", m.size(),
        b_length * dof);
    return false;
  }

  if (b.size() != b_length) {
    *result_listener << absl::Substitute(
        "bias length is [$0] but expected [$1]", b.size(), b_length);
    return false;
  }

  return true;
}

MATCHER(
    LsqpConstraintDimensionsAreValid,
    "coefficient matrix and bounds dimensions are consistent with return "
    "values of LsqpConstraint::GetNumberOfDof and LsqpTask::GetBoundsLength") {
  const LsqpConstraint& constraint = arg;
  auto m = constraint.GetCoefficientMatrix();
  auto l = constraint.GetUpperBound();
  auto u = constraint.GetLowerBound();
  int dof = constraint.GetNumberOfDof();
  int bounds_length = constraint.GetBoundsLength();

  if (m.size() != bounds_length * dof) {
    *result_listener << absl::Substitute(
        "coefficient matrix size is [$0] but expected [$1]", m.size(),
        bounds_length * dof);
    return false;
  }

  if (l.size() != bounds_length) {
    *result_listener << absl::Substitute(
        "lower bound length is [$0] but expected [$1]", l.size(),
        bounds_length);
    return false;
  }

  if (u.size() != bounds_length) {
    *result_listener << absl::Substitute(
        "upper bound length is [$0] but expected [$1]", u.size(),
        bounds_length);
    return false;
  }

  return true;
}

}  // namespace dm_robotics::testing

#endif  // LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_TESTING_MATCHERS_H_
