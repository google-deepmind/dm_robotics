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

#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include "dm_robotics/support/logging.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h"
#include "dm_robotics/least_squares_qp/core/utils.h"
#include "Eigen/Core"

namespace {

constexpr int kNumDof = 3;
constexpr double kDesiredValue = 1.23;
constexpr double kUpperBound = 1.0;
constexpr double kLowerBound = -std::numeric_limits<double>::infinity();

// Task to set all DoFs to kDesiredValue.
class MyTask final : public dm_robotics::LsqpTask {
 public:
  MyTask()
      : bias_(kNumDof, kDesiredValue), coefficient_matrix_(kNumDof * kNumDof) {
    Eigen::Map<Eigen::MatrixXd>(coefficient_matrix_.data(), kNumDof, kNumDof)
        .setIdentity();
  }

  MyTask(const MyTask&) = delete;
  MyTask& operator=(const MyTask&) = delete;

  absl::Span<const double> GetCoefficientMatrix() const override {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetBias() const override { return bias_; }

  int GetNumberOfDof() const override { return kNumDof; }

  int GetBiasLength() const override { return kNumDof; }

 private:
  const std::vector<double> bias_;
  std::vector<double> coefficient_matrix_;
};

// Constraint for all variables between kLowerBound and kUpperBound.
class MyConstraint final : public dm_robotics::LsqpConstraint {
 public:
  MyConstraint()
      : lower_bound_(kNumDof, kLowerBound),
        upper_bound_(kNumDof, kUpperBound),
        coefficient_matrix_(kNumDof * kNumDof) {
    Eigen::Map<Eigen::MatrixXd>(coefficient_matrix_.data(), kNumDof, kNumDof)
        .setIdentity();
  }

  MyConstraint(const MyConstraint&) = delete;
  MyConstraint& operator=(const MyConstraint&) = delete;

  absl::Span<const double> GetCoefficientMatrix() const override {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetUpperBound() const override {
    return upper_bound_;
  }

  absl::Span<const double> GetLowerBound() const override {
    return lower_bound_;
  }

  int GetNumberOfDof() const override { return kNumDof; }

  int GetBoundsLength() const override { return kNumDof; }

 private:
  const std::vector<double> lower_bound_;
  const std::vector<double> upper_bound_;
  std::vector<double> coefficient_matrix_;
};

}  // namespace

int main() {
  // Instantiate solver
  dm_robotics::LsqpStackOfTasksSolver qp_solver(
      dm_robotics::LsqpStackOfTasksSolver::Parameters{
          /*use_adaptive_rho=*/false,
          /*return_error_on_nullspace_failure=*/true,
          /*verbosity=*/
          dm_robotics::LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
          /*absolute_tolerance=*/1.0e-6,
          /*relative_tolerance=*/0.0,
          /*hierarchical_projection_slack=*/1.0e-6,
          /*primal_infeasibility_tolerance=*/1.0e-8,
          /*dual_infeasibility_tolerance=*/1.0e-8});

  // Create a task hierarchy to hold the task.
  dm_robotics::LsqpTaskHierarchy* task_hierarchy =
      qp_solver.AddNewTaskHierarchy(/*max_iterations*/ 10000);

  // Add task to solver.
  // [MyTask*, bool]
  auto [task_ptr, is_task_inserted] = task_hierarchy->InsertOrAssignTask(
      /*name*/ "MyTaskName",
      /*task*/ absl::make_unique<MyTask>(),
      /*weight*/ 1.0,
      /*should_ignore_nullspace*/ false);
  CHECK(task_ptr != nullptr);
  CHECK(is_task_inserted);

  // Add constraint to solver.
  // [MyConstraint*, bool]
  auto [constraint_ptr, is_constraint_inserted] =
      qp_solver.InsertOrAssignConstraint(
          /*name*/ "MyConstraintName",
          /*constraint*/ absl::make_unique<MyConstraint>());
  CHECK(constraint_ptr != nullptr);
  CHECK(is_constraint_inserted);

  // Setup and solve.
  absl::StatusOr<absl::Span<const double>> solution_or =
      qp_solver.SetupAndSolve();
  CHECK_EQ(solution_or.status(), absl::OkStatus())
      << "Failed to solve problem.";
  std::vector<double> solution = dm_robotics::AsCopy(*solution_or);
  std::cout << "Found solution: "
            << absl::StrJoin(solution.begin(), solution.end(), ", ")
            << std::endl;
}
