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

#include "lsqp_stack_of_tasks_solver_impl.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dm_robotics/support/logging.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h"
#include "Eigen/Core"
#include "osqp++.h"
#include "dm_robotics/support/status_macros.h"

namespace dm_robotics::internal {
namespace {

// Returns true if the number of non-zeros and sparsity pattern is the same
// between a and b. False otherwise.
bool IsSparsityEqual(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, osqp::c_int>& a,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, osqp::c_int>& b) {
  CHECK(a.isCompressed() && b.isCompressed());
  CHECK_EQ(a.rows(), b.rows());
  CHECK_EQ(a.cols(), b.cols());

  // If number of non-zeros are different, sparsity is different.
  if (a.nonZeros() != b.nonZeros()) {
    return false;
  }

  // Check sparsity through inner indices. Note that the number of inner indices
  // is equal to the number of non-zeros. Eigen::SparseMatrix<...>::innerSize
  // returns the number of inner dimensions (i.e. number of [zero + non-zero]
  // rows for col-major matrices), NOT the size of the innerIndexPtr array.
  for (size_t i = 0; i < a.nonZeros(); ++i) {
    if (a.innerIndexPtr()[i] != b.innerIndexPtr()[i]) {
      return false;
    }
  }

  // Check sparsity through outer indices. Note that the number of outer indices
  // is equal to the number of outer dimensions (i.e. number of [zero +
  // non-zero] columns for col-major matrices).
  for (size_t i = 0; i < a.outerSize(); ++i) {
    if (a.outerIndexPtr()[i] != b.outerIndexPtr()[i]) {
      return false;
    }
  }

  return true;
}

// Returns the number of hierarchies to solve the problem.
// Returns an invalid-argument error if there are no hierarchies to solve.
absl::StatusOr<int> GetNumberOfNonEmptyHierarchies(
    const LsqpStackOfTasksProblem& problem) {
  int hierarchies = 0;
  for (const auto& hierarchy : problem.task_hierarchies) {
    if (hierarchy != nullptr && !hierarchy->IsEmpty()) {
      hierarchies++;
    }
  }

  if (hierarchies == 0) {
    return absl::InvalidArgumentError(
        "LsqpStackOfTasksSolverImpl::GetNumberOfNonEmptyHierarchies: Number of "
        "non-empty task hierarchies cannot be zero.");
  }
  return hierarchies;
}

// Returns the number of DoF for the problem.
// Returns an invalid-argument error if the problem does not have a consistent
// number of DoF.
absl::StatusOr<int> GetNumberOfDof(const LsqpStackOfTasksProblem& problem) {
  absl::optional<int> num_dof;

  // Note that the user may have left some hierarchies empty. We ignore those as
  // per the class's documentation.
  // All tasks must have the same number of DoFs.
  for (const auto& hierarchy : problem.task_hierarchies) {
    if (hierarchy != nullptr && !hierarchy->IsEmpty()) {
      for (const auto& [task_name, task_info] : *hierarchy) {
        const int task_dof = task_info.task->GetNumberOfDof();
        if (task_dof <= 0) {
          return absl::InvalidArgumentError(absl::Substitute(
              "LsqpStackOfTasksSolverImpl::GetNumberOfDof: The number of DoFs "
              "for every task must be larger than zero. Task with name [$0] "
              "has [$1] DoFs.",
              task_name, task_dof));
        }

        if (!num_dof.has_value()) {
          num_dof = task_dof;
        }

        if (task_dof != num_dof.value()) {
          return absl::InvalidArgumentError(absl::Substitute(
              "LsqpStackOfTasksSolverImpl::GetNumberOfDof: All tasks must have "
              "the same number of DoFs. Task with name [$0] has [$1] DoFs, "
              "which does not match with the last value found [$2].",
              task_name, task_dof, num_dof.value()));
        }
      }
    }
  }

  // All constraints must have the same number of DoFs as the tasks.
  for (const auto& [constraint_name, constraint] : problem.constraints) {
    const int constraint_dof = constraint->GetNumberOfDof();
    if (constraint_dof != num_dof.value()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "LsqpStackOfTasksSolverImpl::GetNumberOfDof: All tasks and "
          "constraints must have the same number of DoFs. Constraint with name "
          "[$0] has [$1] DoFs, which does not match with the number of DoFs "
          "for the tasks [$2].",
          constraint_name, constraint_dof, num_dof.value()));
    }
  }

  return num_dof.value();
}

// Allocates and returns a vector of LsqpStackOfTasks buffers for the problem's
// stacks of tasks.
std::vector<LsqpStackOfTasks> CreateStacksOfTasks(
    const LsqpStackOfTasksProblem& problem, int num_dof) {
  std::vector<LsqpStackOfTasks> stacks_of_tasks;

  // Create a stack for every valid hierarchy and append it to the array.
  // c is of length num_dof;
  // Q has size [num_dof, num_dof];
  // M_null has size [r_null, num_dof];
  // where r_null is the number of total rows of the coefficient matrices
  // for tasks with nullspace projection enabled.
  for (const auto& hierarchy : problem.task_hierarchies) {
    if (hierarchy != nullptr && !hierarchy->IsEmpty()) {
      int nullspace_projection_rows = 0;
      for (const auto& [task_name, task_info] : *hierarchy) {
        const int bias_length = task_info.task->GetBiasLength();
        if (!task_info.should_ignore_nullspace) {
          nullspace_projection_rows += bias_length;
        }
      }
      stacks_of_tasks.push_back(LsqpStackOfTasks{
          Eigen::VectorXd::Zero(num_dof),
          Eigen::MatrixXd::Zero(num_dof, num_dof),
          Eigen::MatrixXd::Zero(nullspace_projection_rows, num_dof)});
    }
  }
  return stacks_of_tasks;
}

// Returns a vector containing the constraint rows for each hierarchy.
std::vector<int> GetConstraintRowsPerHierarchy(
    const LsqpStackOfTasksProblem& problem,
    const std::vector<LsqpStackOfTasks>& stacks_of_tasks) {
  std::vector<int> constraint_rows;

  // All hierarchies share the constraints without nullspace projection.
  int constraint_rows_counter = 0;
  for (const auto& [constraint_name, constraint] : problem.constraints) {
    constraint_rows_counter += constraint->GetBoundsLength();
  }

  // The nullspace projection constraint rows need to be added to the following
  // hierarchy, if it exists.
  for (const auto& stack : stacks_of_tasks) {
    constraint_rows.push_back(constraint_rows_counter);
    constraint_rows_counter += stack.task_nullspace_matrix.rows();
  }
  return constraint_rows;
}

// Updates the stacks of tasks with the problem's coefficients and biases for
// each task, as well as the max iterations per hierarchy with the problems'
// task hierarchies' settings. We update the stacks_of_tasks and the
// max_iterations together to avoid having to loop through all the hierarchies
// twice.
//
// CHECK-fails if stacks_of_tasks or max_iterations is null.
absl::Status UpdateStacksOfTasksAndMaxIterationsPerHierarchy(
    const LsqpStackOfTasksProblem& problem,
    std::vector<LsqpStackOfTasks>* stacks_of_tasks,
    std::vector<int>* max_iterations) {
  CHECK(stacks_of_tasks != nullptr);
  CHECK(max_iterations != nullptr);

  int stacks_hierarchy_index = 0;
  for (int problem_hierarchy_index = 0;
       problem_hierarchy_index < problem.task_hierarchies.size();
       ++problem_hierarchy_index) {
    const auto& hierarchy = problem.task_hierarchies[problem_hierarchy_index];

    // Ignore all null and empty hierarchies.
    if (hierarchy != nullptr && !hierarchy->IsEmpty()) {
      if (stacks_hierarchy_index >= stacks_of_tasks->size() ||
          stacks_hierarchy_index >= max_iterations->size()) {
        return absl::FailedPreconditionError(absl::Substitute(
            "UpdateStacksOfTasksAndMaxIterationsPerHierarchy: The "
            "stacks_of_tasks vector or max_iterations vector does not have "
            "enough memory for hierarchy with index [$0]. This may happen if a "
            "hierarchy is added but memory is not reallocated.",
            stacks_hierarchy_index));
      }

      // Update the max iterations.
      (*max_iterations)[stacks_hierarchy_index] =
          problem.max_iterations[problem_hierarchy_index];

      // Get the corresponding stack, and initialize it to zeros.
      // Dimensions:
      // Q: num_dof x num_dof
      // c: num_dof
      // M_null: sum(bias_lengths) x num_dof
      LsqpStackOfTasks& stack = (*stacks_of_tasks)[stacks_hierarchy_index];
      Eigen::MatrixXd& Q = stack.cost_matrix;
      Eigen::VectorXd& c = stack.cost_vector;
      Eigen::MatrixXd& M_null = stack.task_nullspace_matrix;
      Q.setZero();
      c.setZero();
      M_null.setZero();

      // Update Q, c, and M_null with each of the tasks' coefficients and
      // biases.
      int nullspace_projection_rows_counter = 0;
      for (const auto& [task_name, task_info] : *hierarchy) {
        const int task_dof = task_info.task->GetNumberOfDof();
        const int bias_length = task_info.task->GetBiasLength();
        const double weight = task_info.weight;

        Eigen::Map<const Eigen::MatrixXd> m(
            task_info.task->GetCoefficientMatrix().data(), bias_length,
            task_dof);
        Eigen::Map<const Eigen::VectorXd> b(task_info.task->GetBias().data(),
                                            bias_length);

        // We use noalias to prevent a temporary, as per Eigen's
        // documentation.
        Q.noalias() += weight * m.transpose() * m;
        c.noalias() -= weight * b.transpose() * m;

        // Copy matrices with nullspace projection into M_hierc_vec.
        if (!task_info.should_ignore_nullspace) {
          M_null.block(nullspace_projection_rows_counter, 0, m.rows(),
                       m.cols()) = m;
          nullspace_projection_rows_counter += bias_length;
        }
      }

      // As per this function's documentation, we zero out elements that are not
      // upper-triangular.
      Q.triangularView<Eigen::StrictlyLower>().setZero();

      // Increment the hierarchy index for every valid hierarchy.
      ++stacks_hierarchy_index;
    }
  }
  return absl::OkStatus();
}

// Updates the stack of constraints with the problem's coefficients and bounds
// for each constraint. The stacks_of_tasks nullspace coefficient matrices are
// added below the user-specified constraints, but note that the bounds cannot
// be updated until the problem for each hierarchy is solved.
//
// CHECK-fails if stack_of_constraints is null.
absl::Status UpdateStackOfConstraints(
    const LsqpStackOfTasksProblem& problem,
    const std::vector<LsqpStackOfTasks>& stacks_of_tasks,
    LsqpStackOfConstraints* stack_of_constraints) {
  CHECK(stack_of_constraints != nullptr);

  Eigen::MatrixXd& A = stack_of_constraints->coefficient_matrix;
  Eigen::VectorXd& l = stack_of_constraints->lower_bound;
  Eigen::VectorXd& u = stack_of_constraints->upper_bound;
  A.setZero();
  l.setZero();
  u.setZero();

  int constraint_rows_counter = 0;
  for (const auto& [constraint_name, constraint] : problem.constraints) {
    if (constraint_rows_counter >= A.rows()) {
      return absl::FailedPreconditionError(absl::Substitute(
          "UpdateStackOfConstraints: The stacks_of_constraints does not have "
          "enough memory to add constraint with name [$0]. This may happen "
          "if a constraint is added but memory is not reallocated.",
          constraint_name));
    }

    const int num_dof = constraint->GetNumberOfDof();
    const int bounds_length = constraint->GetBoundsLength();
    absl::Span<const double> coefficient_matrix =
        constraint->GetCoefficientMatrix();
    absl::Span<const double> lower_bound = constraint->GetLowerBound();
    absl::Span<const double> upper_bound = constraint->GetUpperBound();

    A.block(constraint_rows_counter, 0, bounds_length, A.cols()) =
        Eigen::Map<const Eigen::MatrixXd>(coefficient_matrix.data(),
                                          bounds_length, num_dof);
    l.segment(constraint_rows_counter, bounds_length) =
        Eigen::Map<const Eigen::VectorXd>(lower_bound.data(),
                                          lower_bound.size());
    u.segment(constraint_rows_counter, bounds_length) =
        Eigen::Map<const Eigen::VectorXd>(upper_bound.data(),
                                          upper_bound.size());

    constraint_rows_counter += bounds_length;
  }

  // Add nullspace constraints in order at the bottom, if they exist.
  // Note that the last stack's nullspace task coefficients are not added as a
  // constraint, as there is no need to project these.
  for (int i = 0; i < (stacks_of_tasks.size() - 1); ++i) {
    const Eigen::MatrixXd& M_null = stacks_of_tasks[i].task_nullspace_matrix;
    if (M_null.rows() > 0) {
      A.block(constraint_rows_counter, 0, M_null.rows(), A.cols()) = M_null;
      constraint_rows_counter += M_null.rows();
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<LsqpStackOfTasksProblemBuffer> CreateStackOfTasksProblemBuffer(
    const LsqpStackOfTasksProblem& problem) {
  LsqpStackOfTasksProblemBuffer buffer;

  ASSIGN_OR_RETURN(buffer.num_hierarchies,
                   GetNumberOfNonEmptyHierarchies(problem));
  ASSIGN_OR_RETURN(buffer.num_dof, GetNumberOfDof(problem));

  buffer.max_iterations.resize(buffer.num_hierarchies);
  buffer.stacks_of_tasks = CreateStacksOfTasks(problem, buffer.num_dof);
  buffer.constraint_rows =
      GetConstraintRowsPerHierarchy(problem, buffer.stacks_of_tasks);

  int max_num_constraint_rows = buffer.constraint_rows.back();
  buffer.stack_of_constraints = LsqpStackOfConstraints{
      Eigen::VectorXd::Zero(max_num_constraint_rows),
      Eigen::VectorXd::Zero(max_num_constraint_rows),
      Eigen::MatrixXd::Zero(max_num_constraint_rows, buffer.num_dof)};

  buffer.primal_solution = Eigen::VectorXd::Zero(buffer.num_dof);
  buffer.dual_solution = Eigen::VectorXd::Zero(max_num_constraint_rows);
  return buffer;
}

std::vector<OsqpProblem> CreateOsqpProblems(
    const LsqpStackOfTasksProblemBuffer& buffer) {
  std::vector<OsqpProblem> osqp_problems(buffer.num_hierarchies);

  for (int i = 0; i < buffer.num_hierarchies; ++i) {
    // Allocate instance matrices.
    osqp_problems[i].instance.objective_matrix =
        Eigen::SparseMatrix<double, Eigen::ColMajor, osqp::c_int>(
            buffer.num_dof, buffer.num_dof);
    osqp_problems[i].instance.objective_vector =
        Eigen::VectorXd::Zero(buffer.num_dof);
    osqp_problems[i].instance.constraint_matrix =
        Eigen::SparseMatrix<double, Eigen::ColMajor, osqp::c_int>(
            buffer.constraint_rows[i], buffer.num_dof);
    osqp_problems[i].instance.lower_bounds = Eigen::VectorXd::Constant(
        buffer.constraint_rows[i], -std::numeric_limits<double>::infinity());
    osqp_problems[i].instance.upper_bounds = Eigen::VectorXd::Constant(
        buffer.constraint_rows[i], std::numeric_limits<double>::infinity());

    osqp_problems[i].objective_matrix_sparse_buffer =
        osqp_problems[i].instance.objective_matrix;
    osqp_problems[i].constraint_matrix_sparse_buffer =
        osqp_problems[i].instance.constraint_matrix;
    osqp_problems[i].constraint_matrix_dense_buffer =
        Eigen::MatrixXd(buffer.constraint_rows[i], buffer.num_dof);
  }
  return osqp_problems;
}

absl::Status UpdateStackOfTasksProblemBuffer(
    const LsqpStackOfTasksProblem& problem,
    LsqpStackOfTasksProblemBuffer* buffer) {
  CHECK(buffer != nullptr);
  RETURN_IF_ERROR(UpdateStacksOfTasksAndMaxIterationsPerHierarchy(
      problem, &buffer->stacks_of_tasks, &buffer->max_iterations));
  RETURN_IF_ERROR(UpdateStackOfConstraints(problem, buffer->stacks_of_tasks,
                                           &buffer->stack_of_constraints));
  return absl::OkStatus();
}

absl::Status UpdateOsqpProblems(const LsqpStackOfTasksProblemBuffer& buffer,
                                osqp::OsqpSettings* osqp_settings,
                                std::vector<OsqpProblem>* osqp_problems) {
  CHECK(osqp_problems != nullptr);
  const Eigen::MatrixXd& A = buffer.stack_of_constraints.coefficient_matrix;
  for (int i = 0; i < buffer.num_hierarchies; ++i) {
    const LsqpStackOfTasks& stack = buffer.stacks_of_tasks[i];
    const Eigen::MatrixXd& Q = stack.cost_matrix;
    const Eigen::VectorXd& c = stack.cost_vector;
    int constraint_rows = buffer.constraint_rows[i];
    OsqpProblem& osqp_problem = (*osqp_problems)[i];

    // As per this function's documentation, we can avoid dynamic allocation if
    // `Q` and `A` have the same sparsity pattern.
    osqp_problem.objective_matrix_sparse_buffer = Q.sparseView();
    osqp_problem.constraint_matrix_dense_buffer = A.topRows(constraint_rows);
    osqp_problem.constraint_matrix_sparse_buffer =
        osqp_problem.constraint_matrix_dense_buffer.sparseView();
    bool is_objective_sparsity_equal =
        IsSparsityEqual(osqp_problem.objective_matrix_sparse_buffer,
                        osqp_problem.instance.objective_matrix);
    bool is_constraint_sparsity_equal =
        IsSparsityEqual(osqp_problem.constraint_matrix_sparse_buffer,
                        osqp_problem.instance.constraint_matrix);

    // Update the OSQP problem instance/setting.
    osqp_problem.instance.objective_matrix =
        osqp_problem.objective_matrix_sparse_buffer;
    osqp_problem.instance.objective_vector = c;
    osqp_problem.instance.constraint_matrix =
        osqp_problem.constraint_matrix_sparse_buffer;
    osqp_settings->max_iter = buffer.max_iterations[i];

    // Avoid calling Init, if possible.
    if (!osqp_problem.solver.IsInitialized() || !is_objective_sparsity_equal ||
        !is_constraint_sparsity_equal) {
      RETURN_IF_ERROR(
          osqp_problem.solver.Init(osqp_problem.instance, *osqp_settings));
    } else {
      // UpdateObjectiveAndConstraintMatrices performs memory allocation if the
      // objective matrix is not upper triangular.
      //
      // Note that this update may cause the solution to be different when
      // compared to when Init is used due to small numerical errors. This is
      // deterministic based on the last time Init was called, and the
      // difference is usually less than 1e-10.
      RETURN_IF_ERROR(osqp_problem.solver.UpdateObjectiveAndConstraintMatrices(
          osqp_problem.instance.objective_matrix,
          osqp_problem.instance.constraint_matrix));
      RETURN_IF_ERROR(osqp_problem.solver.SetObjectiveVector(
          osqp_problem.instance.objective_vector));
      RETURN_IF_ERROR(
          osqp_problem.solver.UpdateMaxIter(osqp_settings->max_iter));
      RETURN_IF_ERROR(osqp_problem.solver.UpdateEpsAbs(osqp_settings->eps_abs));
    }
  }
  return absl::OkStatus();
}

}  // namespace dm_robotics::internal
