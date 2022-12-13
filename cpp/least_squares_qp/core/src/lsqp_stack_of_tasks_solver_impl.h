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

#ifndef LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_STACK_OF_TASKS_SOLVER_IMPL_H_
#define LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_STACK_OF_TASKS_SOLVER_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h"
#include "Eigen/Core"
#include "osqp++.h"

namespace dm_robotics::internal {

// Defines a complete LsqpStackOfTasks problem to be solved by the
// LsqpStackOfTasksSolver.
struct LsqpStackOfTasksProblem {
  std::vector<int> max_iterations;
  std::vector<std::unique_ptr<LsqpTaskHierarchy>> task_hierarchies;
  absl::flat_hash_map<std::string, std::unique_ptr<const LsqpConstraint>>
      constraints;
};

// Every stack of tasks defines the following cost function:
//  minimize: x^T Q x + c^T x
// where Q is the cost matrix, c is the cost vector, and x are the decision
// variables. The matrix Q in LSQP problems is, by definition, real-valued
// positive (semi-)definite and thus symmetric.
//
// Each stack of tasks also defines an equality constraint for nullspace
// projection:
//  M_null x = M_null x_sol
// where M_null is formed by vertically stacking all the coefficient matrices
// for the tasks with enabled nullspace projection, and x_sol are the values of
// the decision variable that minimize the stack's cost function. The number of
// rows in M_null is equal to the sum of the bias lengths for the tasks with
// enabled nullspace projection.
struct LsqpStackOfTasks {
  Eigen::VectorXd cost_vector;            // dim: num_dof
  Eigen::MatrixXd cost_matrix;            // dim: num_dof x num_dof
  Eigen::MatrixXd task_nullspace_matrix;  // dim: sum(bias_lengths) x num_dof
};

// A stack of constraints defines the following inequality:
//  l <= A x <= u
// where A, l, and u are formed by vertically stacking the coefficient matrices,
// lower bounds, and upper bounds of all constraints in the stack, respectively.
struct LsqpStackOfConstraints {
  Eigen::VectorXd lower_bound;         // dim: sum(bound_lengths)
  Eigen::VectorXd upper_bound;         // dim: sum(bound_lengths)
  Eigen::MatrixXd coefficient_matrix;  // dim: sum(bound_lengths) x num_dof
};

// Buffer for holding all data necessary to instantiate and solve
// an LSQP Stack-of-Tasks problem without allocating more memory.
//
// In the current implementation, the `stacks_of_tasks` field only stores the
// upper-triangular elements of the cost matrices (Q), and the remaining
// elements are set to zero.
struct LsqpStackOfTasksProblemBuffer {
  int num_hierarchies;
  int num_dof;
  std::vector<int> max_iterations;
  std::vector<int> constraint_rows;
  std::vector<LsqpStackOfTasks> stacks_of_tasks;
  LsqpStackOfConstraints stack_of_constraints;
  Eigen::VectorXd primal_solution;  // dim: # decision variables
  Eigen::VectorXd dual_solution;    // dim: # constraint rows
};

// Problem instances to be solved by OSQP.
struct OsqpProblem {
  osqp::OsqpInstance instance;
  osqp::OsqpSolver solver;

  // Buffers for preventing internal re-allocations.
  Eigen::SparseMatrix<double, Eigen::ColMajor, osqp::c_int>
      objective_matrix_sparse_buffer;
  Eigen::SparseMatrix<double, Eigen::ColMajor, osqp::c_int>
      constraint_matrix_sparse_buffer;
  Eigen::MatrixXd constraint_matrix_dense_buffer;
};

// Helper functions ----------------------------------

// Allocates and returns a buffer that can hold all data necessary to
// instantiate and solve an LSQP Stack-of-Tasks problem without allocating more
// memory.
absl::StatusOr<LsqpStackOfTasksProblemBuffer> CreateStackOfTasksProblemBuffer(
    const LsqpStackOfTasksProblem& problem);

// Allocates and returns an array of OsqpProblems, one per hierarchy, that can
// be solved through osqp::OsqpSolver.
std::vector<OsqpProblem> CreateOsqpProblems(
    const LsqpStackOfTasksProblemBuffer& buffer);

// Updates the buffer to match the latest task hierarchies and constraints
// information. The buffer's `primal_solution` and `dual_solution` fields are
// unaffected.
//
// In the current implementation, this function only sets the
// upper-triangular elements of the cost matrices (Q) in the `stacks_of_tasks`
// field in `buffer`, and the remaining elements are set to zero.
//
// CHECK-fails if buffer is null.
absl::Status UpdateStackOfTasksProblemBuffer(
    const LsqpStackOfTasksProblem& problem,
    LsqpStackOfTasksProblemBuffer* buffer);

// Updates the OSQP problems with the `buffer` data. OSQP setting fields other
// than the `max_iter` and `eps_abs` must be set before the first call to this
// function. The `eps_abs` field can be updated in every call. The `max_iter`
// field is read from the `max_iterations` field in `buffer` during every call.
//
// Note: The upper/lower bounds of osqp_problems is not updated, as these depend
// on the previous hierarchy's solution. Warm starting of the primal/dual
// solution, if enabled, must also be performed after calling this function.
//
// Note: If the cost matrices (Q) in the `stacks_of_tasks` field in `buffer` and
// the constraint matrix (A) in the `stack of constraints` field have the same
// sparsity pattern as the previous call to this function, and the cost matrices
// are all upper-triangular, this function does not perform dynamic memory
// allocation. However, in this case, the solution may have a small numerical
// difference (<1e-10 in our tests) when compared to a full update (i.e.
// CreateOsqpProblems followed by UpdateOsqpProblems) due to numerical scaling
// errors. If deterministic solutions are important to the point where a
// difference 1e-10 is too large, the user should always call CreateOsqpProblems
// followed by UpdateOsqpProblems.
//
// CHECK-fails if osqp_problems is null.
absl::Status UpdateOsqpProblems(const LsqpStackOfTasksProblemBuffer& buffer,
                                osqp::OsqpSettings* osqp_settings,
                                std::vector<OsqpProblem>* osqp_problems);

}  // namespace dm_robotics::internal

#endif  // LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_STACK_OF_TASKS_SOLVER_IMPL_H_
