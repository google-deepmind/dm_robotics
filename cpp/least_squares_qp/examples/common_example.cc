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
#include "dm_robotics/least_squares_qp/common/box_constraint.h"
#include "dm_robotics/least_squares_qp/common/identity_task.h"
#include "dm_robotics/least_squares_qp/common/minimize_norm_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h"
#include "dm_robotics/least_squares_qp/core/utils.h"

namespace {

constexpr int kNumDof = 3;
constexpr double kFirstDofDesiredValue = 1.23;
constexpr double kSecondHierarchyDofValues = 0.42;
constexpr double kUpperBound = 1.0;
constexpr double kLowerBound = -std::numeric_limits<double>::infinity();

// Task to set the first DoF to kFirstDofDesiredValue, other DoFs are
// unassigned.
class FirstDofTask final : public dm_robotics::LsqpTask {
 public:
  FirstDofTask() : coefficient_matrix_(kNumDof * kNumDof, 0.0) {
    bias_[0] = kFirstDofDesiredValue;
    coefficient_matrix_[0] = 1.0;
  }

  FirstDofTask(const FirstDofTask&) = delete;
  FirstDofTask& operator=(const FirstDofTask&) = delete;

  absl::Span<const double> GetCoefficientMatrix() const override {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetBias() const override { return bias_; }

  int GetNumberOfDof() const override { return kNumDof; }

  int GetBiasLength() const override { return kNumDof; }

 private:
  std::array<double, kNumDof> bias_;
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

  // Create two task hierarchies.
  dm_robotics::LsqpTaskHierarchy* first_hierarchy =
      qp_solver.AddNewTaskHierarchy(/*max_iterations*/ 10000);
  dm_robotics::LsqpTaskHierarchy* second_hierarchy =
      qp_solver.AddNewTaskHierarchy(/*max_iterations*/ 10000);

  // Add tasks to first hierarchy.
  // [FirstDofTask*, bool]
  auto [first_dof_task_ptr, is_first_dof_task_inserted] =
      first_hierarchy->InsertOrAssignTask(
          /*name*/ "MyFirstDofTask",
          /*task*/ absl::make_unique<FirstDofTask>(),
          /*weight*/ 1.0,
          /*should_ignore_nullspace*/ false);
  CHECK(first_dof_task_ptr != nullptr);
  CHECK(is_first_dof_task_inserted);

  // [MinimizeNormTask*, bool]
  auto [minimize_norm_task_ptr, is_minimize_norm_task_inserted] =
      first_hierarchy->InsertOrAssignTask(
          /*name*/ "MyMinimizeNormTask",
          /*task*/ absl::make_unique<dm_robotics::MinimizeNormTask>(kNumDof),
          /*weight*/ 1.0e-6,
          /*should_ignore_nullspace*/ true);
  CHECK(minimize_norm_task_ptr != nullptr);
  CHECK(is_minimize_norm_task_inserted);

  // Add task to second hierarchy.
  // [IdentityTask*, bool]
  auto [identity_task_ptr, is_identity_task_inserted] =
      second_hierarchy->InsertOrAssignTask(
          /*name*/ "MyIdentityTask",
          /*task*/
          absl::make_unique<dm_robotics::IdentityTask>(
              std::vector<double>(kNumDof, kSecondHierarchyDofValues)),
          /*weight*/ 1.0,
          /*should_ignore_nullspace*/ false);
  CHECK(identity_task_ptr != nullptr);
  CHECK(is_identity_task_inserted);

  // Add constraint to solver.
  // [BoxConstraint*, bool]
  auto [constraint_ptr, is_constraint_inserted] =
      qp_solver.InsertOrAssignConstraint(
          /*name*/ "MyBoxConstraint",
          /*constraint*/
          absl::make_unique<dm_robotics::BoxConstraint>(
              std::vector<double>(kNumDof, kLowerBound),
              std::vector<double>(kNumDof, kUpperBound)));
  CHECK(constraint_ptr != nullptr);
  CHECK(is_constraint_inserted);

  // Setup.
  CHECK_EQ(qp_solver.SetupProblem(), absl::OkStatus());

  // Solve.
  absl::StatusOr<absl::Span<const double>> solution_or;
  solution_or = qp_solver.Solve();

  CHECK_EQ(solution_or.status(), absl::OkStatus())
      << "Failed to solve problem(1).";
  std::vector<double> solution = dm_robotics::AsCopy(*solution_or);
  std::cout << "Found solution(1): "
            << absl::StrJoin(solution.begin(), solution.end(), ", ")
            << std::endl;

  // Update second hierarchy task and solve again.
  identity_task_ptr->SetTarget(
      std::vector<double>(kNumDof, -kSecondHierarchyDofValues));
  solution_or = qp_solver.Solve();

  CHECK_EQ(solution_or.status(), absl::OkStatus())
      << "Failed to solve problem(2).";
  solution = dm_robotics::AsCopy(*solution_or);
  std::cout << "Found solution(2): "
            << absl::StrJoin(solution.begin(), solution.end(), ", ")
            << std::endl;
}
