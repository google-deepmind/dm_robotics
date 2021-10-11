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

#include "dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dm_robotics/support/logging.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "lsqp_stack_of_tasks_solver_impl.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h"
#include "Eigen/Core"
#include "osqp++.h"
#include "dm_robotics/support/status_macros.h"

namespace dm_robotics {

constexpr char kErrorMessageUnableToInitializeNoHierarchies[] =
    "Unable to initialize QP solver as no task hierarchies have been created.";

constexpr bool kDefaultIsPolish = false;    // Prevents memory allocation.
constexpr bool kDefaultIsWarmStart = true;  // Ensures determinism.

// Implementation structure for LsqpStackOfTasks.
class LsqpStackOfTasksSolver::Impl {
 public:
  explicit Impl(const Parameters& params);

  // Solves the optimization problem. The problem is solved iteratively,
  // warm-started by the current values in `lsqp_problem.primal_solution`.
  absl::StatusOr<absl::Span<const double>> Solve();

  bool return_error_on_nullspace_failure;
  bool enable_nullspace_warnings;

  double hierarchical_projection_slack;
  internal::LsqpStackOfTasksProblem lsqp_problem;
  internal::LsqpStackOfTasksProblemBuffer lsqp_buffer;

  osqp::OsqpSettings osqp_settings;
  std::vector<internal::OsqpProblem> osqp_problems;
};

LsqpStackOfTasksSolver::Impl::Impl(const Parameters& params)
    : return_error_on_nullspace_failure(
          params.return_error_on_nullspace_failure),
      enable_nullspace_warnings(
          (params.verbosity & Parameters::VerboseFlags::kNullspaceWarnings) ==
          Parameters::VerboseFlags::kNullspaceWarnings),
      hierarchical_projection_slack(params.hierarchical_projection_slack) {
  // Ensure at least one of absolute/relative tolerance is valid.
  CHECK(params.absolute_tolerance > 0.0 || params.relative_tolerance > 0.0)
      << absl::Substitute(
             "LsqpStackOfTasksSolver: The absolute tolerance [$0] and the "
             "relative tolerance [$1] are both invalid. At least one must be "
             "positive.",
             params.absolute_tolerance, params.relative_tolerance);

  // Ensure both infeasibility tolerances are valid.
  CHECK(params.primal_infeasibility_tolerance > 0.0 &&
        params.dual_infeasibility_tolerance > 0.0)
      << absl::Substitute(
             "LsqpStackOfTasksSolver: The primal infeasibility tolerance "
             "[$0] or the dual infeasibility tolerance [$1] is invalid. They "
             "must both be positive.",
             params.primal_infeasibility_tolerance,
             params.dual_infeasibility_tolerance);

  // Ensure the hierarchical projection slack is valid.
  CHECK_GE(params.hierarchical_projection_slack, 0.0) << absl::Substitute(
      "LsqpStackOfTasksSolver: The hierarchical projection slack [$0] must "
      "be zero or positive.",
      params.hierarchical_projection_slack);

  osqp_settings.warm_start = kDefaultIsWarmStart;
  osqp_settings.eps_abs = std::max(params.absolute_tolerance, 0.0);
  osqp_settings.eps_rel = std::max(params.relative_tolerance, 0.0);
  osqp_settings.eps_prim_inf = params.primal_infeasibility_tolerance;
  osqp_settings.eps_dual_inf = params.dual_infeasibility_tolerance;
  osqp_settings.adaptive_rho = params.use_adaptive_rho;
  osqp_settings.polish = kDefaultIsPolish;
  osqp_settings.verbose =
      (params.verbosity & Parameters::VerboseFlags::kInternalSolver) ==
      Parameters::VerboseFlags::kInternalSolver;
}

absl::StatusOr<absl::Span<const double>> LsqpStackOfTasksSolver::Impl::Solve() {
  const int num_hierarchies = lsqp_buffer.num_hierarchies;
  const int num_dof = lsqp_buffer.num_dof;
  CHECK_GE(num_hierarchies, 0) << absl::Substitute(
      "LsqpStackOfTasksSolver::Solve: Unexpected number of hierarchies [$0]. "
      "Has `SetupProblem` been called?",
      num_hierarchies);
  CHECK_GE(num_dof, 0) << absl::Substitute(
      "LsqpStackOfTasksSolver::Solve: Unexpected number of DoF [$0]. "
      "Has `SetupProblem` been called?",
      num_dof);

  // Update buffer and OSQP problems with latest constraint/task data.
  CHECK_EQ(
      internal::UpdateStackOfTasksProblemBuffer(lsqp_problem, &lsqp_buffer),
      absl::OkStatus())
      << "LsqpStackOfTasksSolver::Solve: Unable to update the tasks and "
         "constraints. Has `SetupProblem` been called?";
  RETURN_IF_ERROR(internal::UpdateOsqpProblems(lsqp_buffer, &osqp_settings,
                                               &osqp_problems));

  // Immutable references.
  const std::vector<int>& constraint_rows = lsqp_buffer.constraint_rows;
  const std::vector<internal::LsqpStackOfTasks>& stacks_of_tasks =
      lsqp_buffer.stacks_of_tasks;

  // Mutable references. Note that the stack of constraints needs a mutable
  // reference since we need to update the bounds based on the solution of the
  // previous hierarchy.
  internal::LsqpStackOfConstraints& stack_of_constraints =
      lsqp_buffer.stack_of_constraints;
  Eigen::VectorXd& primal_solution = lsqp_buffer.primal_solution;
  Eigen::VectorXd& dual_solution = lsqp_buffer.dual_solution;

  // Solve Hierarchies. Note that the initial guess is whichever value is held
  // inside `primal_solution`.
  dual_solution.setZero();
  for (int i = 0; i < num_hierarchies; i++) {
    // constraint_rows holds the size of the constraint for each hierarchy.
    const int constraint_size = constraint_rows[i];

    Eigen::VectorXd& l = stack_of_constraints.lower_bound;
    Eigen::VectorXd& u = stack_of_constraints.upper_bound;

    // Update bounds.
    internal::OsqpProblem& osqp_problem = osqp_problems[i];
    osqp_problem.instance.lower_bounds = l.topRows(constraint_size);
    osqp_problem.instance.upper_bounds = u.topRows(constraint_size);
    RETURN_IF_ERROR(
        osqp_problem.solver.SetBounds(osqp_problem.instance.lower_bounds,
                                      osqp_problem.instance.upper_bounds));

    // We warm start with zero for the first hierarchy, and with the previous
    // solution for the second hierarchy and onwards. For the dual solution, we
    // warm start with zero for the first hierarchy. For the second hierarchy
    // and onwards, only the indexes corresponding to the previous hierarchy
    // constraints are warm started with the previous solution, and the
    // remaining indices are warm started to zero.
    RETURN_IF_ERROR(osqp_problem.solver.SetWarmStart(
        primal_solution, dual_solution.topRows(constraint_size)));

    // Solve QP.
    osqp::OsqpExitCode exit_code = osqp_problem.solver.Solve();

    // OSQP backend saves the old SIGINT handler and installs its own custom
    // handler when osqp_solve is called, if the CTRLC MACRO is defined. In this
    // case, when the call to osqp_solve returns, the custom handler is
    // uninstalled and the old handler is re-installed. Solve() wraps osqp_solve
    // and returns kInterrupted if SIGINT was intercepted. In this case, we also
    // return this error and let the user decide what to do.
    if (exit_code == osqp::OsqpExitCode::kInterrupted) {
      return absl::CancelledError(
          "LsqpStackOfTasksSolver::Solve: OSQP was interrupted while solving "
          "the optimization problem.");
    }

    // We consider a success only if a feasible optimal solution is found within
    // the specified tolerances.
    const bool success = exit_code == osqp::OsqpExitCode::kOptimal;

    // If unable to find a feasible solution on the first hierarchy, return an
    // error. If on subsequent hierarchies and
    // `return_error_on_nullspace_failure` is false, return the previous
    // hierarchy's feasible solution. This is acceptable as subsequent
    // hierarchies are always nullspace projections of the previous ones.
    if (!success) {
      if (i == 0 || return_error_on_nullspace_failure) {
        return absl::InternalError(
            absl::Substitute("LsqpStackOfTasksSolver::Solve: Unable to solve "
                             "the optimization problem for hierarchy [$0].",
                             i));
      } else {
        if (enable_nullspace_warnings) {
          LOG(WARNING) << absl::Substitute(
              "LsqpStackOfTasksSolver::Solve: Unable to find feasible "
              "solution for hierarchy [$0]. Returning last feasible solution. "
              "In other words, the solution of hierarchy [$1] without "
              "nullspace projection.",
              i, i - 1);
        }
        return primal_solution;
      }
    }

    // If successful, update the primal solution and dual solutions.
    primal_solution = osqp_problem.solver.primal_solution();
    dual_solution.topRows(constraint_size) =
        osqp_problem.solver.dual_solution();

    // If it's not the last QP optimization problem, and there are hierarchical
    // projection constraints, add nullspace projection constraint bounds for
    // the next hierarchy.
    const int current_projection_rows =
        stacks_of_tasks[i].task_nullspace_matrix.rows();
    if (i < (num_hierarchies - 1) && current_projection_rows != 0) {
      Eigen::Ref<Eigen::VectorXd> projection_l_block =
          l.segment(constraint_size, current_projection_rows);
      Eigen::Ref<Eigen::VectorXd> projection_u_block =
          u.segment(constraint_size, current_projection_rows);

      // Using auto makes use of Eigen's expression templates, delaying
      // evaluation and preventing the creation of a temporary.
      // We use .noalias() to ensure that task_space_solution is evaluated
      // directly into the blocks.
      // We use .array() to perform coefficient-wise operations on the blocks
      // with the hierarchical_projection_slack, which is a scalar.
      auto task_space_solution =
          stacks_of_tasks[i].task_nullspace_matrix * primal_solution;
      projection_l_block.noalias() = task_space_solution;
      projection_u_block.noalias() = task_space_solution;
      projection_l_block.array() -= hierarchical_projection_slack;
      projection_u_block.array() += hierarchical_projection_slack;
    }
  }

  return absl::Span<const double>(primal_solution.data(),
                                  primal_solution.size());
}

// Solver
LsqpStackOfTasksSolver::LsqpStackOfTasksSolver(const Parameters& params)
    : pimpl_(absl::make_unique<Impl>(params)) {}

LsqpStackOfTasksSolver::~LsqpStackOfTasksSolver() = default;

LsqpTaskHierarchy* LsqpStackOfTasksSolver::AddNewTaskHierarchy(
    int max_iterations) {
  CHECK_GT(max_iterations, 0) << "LsqpStackOfTasksSolver::AddNewTaskHierarchy: "
                                 "`max_iterations` cannot be negative or zero.";

  pimpl_->lsqp_problem.task_hierarchies.push_back(
      absl::make_unique<LsqpTaskHierarchy>());
  pimpl_->lsqp_problem.max_iterations.push_back(max_iterations);
  return pimpl_->lsqp_problem.task_hierarchies.back().get();
}

LsqpTaskHierarchy* LsqpStackOfTasksSolver::GetTaskHierarchy(int index) {
  CHECK(index >= 0) << absl::Substitute(
      "LsqpStackOfTasksSolver::GetTaskHierarchy: Index [$0] must be zero or "
      "positive.",
      index);
  CHECK(index < pimpl_->lsqp_problem.task_hierarchies.size())
      << absl::Substitute(
             "LsqpStackOfTasksSolver::GetTaskHierarchy: Index [$0] is out of "
             "bounds. Current number of tasks: [$1]",
             index, pimpl_->lsqp_problem.task_hierarchies.size());
  return pimpl_->lsqp_problem.task_hierarchies[index].get();
}

int LsqpStackOfTasksSolver::GetNumberOfTaskHierarchies() const {
  return pimpl_->lsqp_problem.task_hierarchies.size();
}

bool LsqpStackOfTasksSolver::RemoveConstraint(absl::string_view name) {
  return pimpl_->lsqp_problem.constraints.erase(name) != 0;
}

void LsqpStackOfTasksSolver::ClearAllConstraints() {
  pimpl_->lsqp_problem.constraints.clear();
}

bool LsqpStackOfTasksSolver::HasConstraint(absl::string_view name) const {
  return pimpl_->lsqp_problem.constraints.find(name) !=
         pimpl_->lsqp_problem.constraints.end();
}

absl::Status LsqpStackOfTasksSolver::SetupProblem() {
  CHECK(!pimpl_->lsqp_problem.task_hierarchies.empty())
      << "LsqpStackOfTasksSolver::SetupProblem: "
      << kErrorMessageUnableToInitializeNoHierarchies;

  // Allocate memory. This can fail if no tasks have been added to any
  // hierarchy, or if tasks don't all have the same number of DoF > 0.
  {
    auto lsqp_buffer_or =
        internal::CreateStackOfTasksProblemBuffer(pimpl_->lsqp_problem);
    CHECK_EQ(lsqp_buffer_or.status(), absl::OkStatus())
        << "LsqpStackOfTasksSolver::SetupProblem: Unable to set up problem.";
    pimpl_->lsqp_buffer = *std::move(lsqp_buffer_or);
  }
  pimpl_->osqp_problems = internal::CreateOsqpProblems(pimpl_->lsqp_buffer);

  // Update to ensure objective/constraint matrices with same sparsity do
  // not cause re-allocation.
  //
  // Updating the problem buffer should never fail, as the problem has just been
  // set-up.
  CHECK_EQ(internal::UpdateStackOfTasksProblemBuffer(pimpl_->lsqp_problem,
                                                     &pimpl_->lsqp_buffer),
           absl::OkStatus())
      << "LsqpStackOfTasksSolver::SetupProblem: Unable to set up problem.";
  //
  // Internal OSQP solver errors are returned to the user.
  RETURN_IF_ERROR(internal::UpdateOsqpProblems(
      pimpl_->lsqp_buffer, &pimpl_->osqp_settings, &pimpl_->osqp_problems));
  return absl::OkStatus();
}

absl::StatusOr<absl::Span<const double>> LsqpStackOfTasksSolver::Solve() {
  pimpl_->lsqp_buffer.primal_solution.setZero();
  return pimpl_->Solve();
}

absl::StatusOr<absl::Span<const double>> LsqpStackOfTasksSolver::Solve(
    absl::Span<const double> initial_guess) {
  const int num_dof = pimpl_->lsqp_buffer.num_dof;
  CHECK_EQ(initial_guess.size(), num_dof) << absl::Substitute(
      "LsqpStackOfTasksSolver::Solve: the size of `initial_guess` [$0] must be "
      "the same as the number of DoF [$1].",
      initial_guess.size(), num_dof);

  pimpl_->lsqp_buffer.primal_solution =
      Eigen::Map<const Eigen::VectorXd>(initial_guess.data(), num_dof);
  return pimpl_->Solve();
}

absl::StatusOr<absl::Span<const double>>
LsqpStackOfTasksSolver::SetupAndSolve() {
  // We do not make a call to SetupProblem as this would require us to call
  // UpdateStackOfTasksProblemBuffer and UpdateOsqpProblems twice. We instead
  // call only the required functions.
  CHECK(!pimpl_->lsqp_problem.task_hierarchies.empty())
      << "LsqpStackOfTasksSolver::SetupAndSolve: "
      << kErrorMessageUnableToInitializeNoHierarchies;
  {
    auto lsqp_buffer_or =
        internal::CreateStackOfTasksProblemBuffer(pimpl_->lsqp_problem);
    CHECK_EQ(lsqp_buffer_or.status(), absl::OkStatus())
        << "LsqpStackOfTasksSolver::SetupAndSolve: Unable to set up problem.";
    pimpl_->lsqp_buffer = *std::move(lsqp_buffer_or);
  }
  pimpl_->osqp_problems = internal::CreateOsqpProblems(pimpl_->lsqp_buffer);

  // Solve.
  return Solve();
}

// Private members.
bool LsqpStackOfTasksSolver::InsertOrAssignConstraintImpl(
    absl::string_view name, std::unique_ptr<LsqpConstraint> constraint) {
  CHECK(constraint != nullptr)
      << "LsqpStackOfTasksSolver::InsertOrAssignConstraintImpl: "
         "'constraint' cannot be null.";

  return pimpl_->lsqp_problem.constraints
      .insert_or_assign(name, std::move(constraint))
      .second;
}

}  // namespace dm_robotics
