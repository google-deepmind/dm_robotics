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

#ifndef LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_STACK_OF_TASKS_SOLVER_H_
#define LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_STACK_OF_TASKS_SOLVER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h"

namespace dm_robotics {

// A hierarchical least-squares Quadratic Optimization solver for stacks of
// affine tasks and constraints. Abstract interfaces are provided for the tasks
// and constraints that the solver takes as inputs.
//
// In every hierarchy, solves the following least squares optimization problem:
//   argmin_x Sum_i w_i || M_i x - b_i ||^2;
//   subject to l_j <= C_j x <= u_j  for all j
// where w_i, M_i, and b_i are the weight, coefficient matrix, and bias of the
// i-th task, respectively; and C_j, l_j, and u_j are the coefficient matrix,
// lower bound, and upper bound for the j-th constraint, respectively.
//
// The tasks of each hierarchy are projected to the remaining hierarchies as a
// nullspace projection constraint of the form:
//   M_null x_sol - e_slack <= M_null x <= M_null x_sol + e_slack
// where M_null are the stacked coefficient matrices of the tasks in the
// previous hierarchy with enabled nullspace projection, x_sol is the solution
// from the previous hierarchy, and e_slack is a slack added to make the
// hierarchical projection numerically stable.
//
// The internal QP solver (OSQP) is used to iterate on the resulting
// optimization problem from each hierarchy until it is solved or the maximum
// number of iterations is reached. The optimization problem will be considered
// solved if a set of decision variables is found such that the infinity norm of
// the primal and dual residuals of the resultant QP problem are within the
// specified tolerances. Note that task weights higher (lower) than `1.0`
// effectively tighten (loosen) the tolerance of the optimization problem.
// Please refer to OSQP's algorithm publication for more information:
// https://arxiv.org/pdf/1711.08013.pdf
//
// Solutions are returned in the form of an absl::Span<const double> object,
// which represents a view over the solution array. It is the user's
// responsibility to ensure that the view remains valid while it is being used.
// If a copy is desired, we recommend the use of the function `AsCopy` defined
// in the utils.h header in the same directory as this file.
class LsqpStackOfTasksSolver {
 public:
  // Initialization parameters for LsqpStackOfTasksSolver.
  //
  // For more information on OSQP-specific parameters, refer to:
  // https://buildmedia.readthedocs.org/media/pdf/osqp/stable/osqp.pdf
  struct Parameters {
    // If `true`, the solver will return an error if the nullspace projection
    // fails i.e. if the solver fails to find a solution to any hierarchy after
    // the first. If `false`, the solution to the last solved hierarchy will be
    // returned instead (which by definition is a valid but sub-optimal solution
    // to the entire optimization problem).
    bool return_error_on_nullspace_failure = true;

    // Defines the level of verbosity of the solver.
    enum class VerboseFlags : std::size_t {
      kNone = 0,
      kInternalSolver = 1 << 0,     // Print internal solver (OSQP) information.
      kNullspaceWarnings = 1 << 1,  // Print if nullspace fails to solve.
    } verbosity = VerboseFlags::kNone;

    // Absolute and relative tolerances for the internal OSQP solver. A smaller
    // tolerance level may be more accurate but may require a large number of
    // iterations. The problem will be considered solved if either the absolute
    // or relative tolerance is satisfied, and thus the tolerances should be set
    // to the smallest permissible value. One of the tolerances may set to zero
    // or negative to prevent it from being taken into consideration when
    // determining if the problem is solved.
    //
    // These tolerances affect both, the optimality of the solution as well as
    // the validity of the solution with respect to the constraints.
    //
    // Must not be zero or negative. (Precondition violation may cause
    // CHECK-failure.)
    double absolute_tolerance = 1.0e-5;
    double relative_tolerance = 0.0;

    // Sets the hierarchical projection slack "e_slack" to the provided value.
    // A smaller value will result in stricter hierarchical projection
    // constraints, but may be numerically unstable.
    //
    // Must not be negative. (Precondition violation may cause CHECK-failure.)
    double hierarchical_projection_slack = 1.0e-5;

    // Primal and dual infeasibility tolerances for the internal OSQP solver. A
    // smaller tolerance level results in stricter feasibility checks (i.e.
    // whether the problem is considered infeasible), but may require a large
    // number of iterations.
    // Must not be zero or negative. (Precondition violation may cause
    // CHECK-failure.)
    double primal_infeasibility_tolerance = 1.0e-6;
    double dual_infeasibility_tolerance = 1.0e-6;
  };

  explicit LsqpStackOfTasksSolver(const Parameters& params);

  LsqpStackOfTasksSolver(const LsqpStackOfTasksSolver&) = delete;
  LsqpStackOfTasksSolver& operator=(const LsqpStackOfTasksSolver&) = delete;

  ~LsqpStackOfTasksSolver();

  // Creates a new task hierarchy, appends it to the end of the list, and
  // returns a pointer to the newly created hierarchy. `max_iterations` defines
  // the maximum number of iterations that the solver is allowed to spend on
  // solving the optimization problem for this hierarchy.
  //
  // SetupProblem must be called before Solve if a new hierarchy is created, if
  // tasks are added/removed from an existing hierarchy, or if the number of
  // DoF/bias length of any added task changes.
  //
  // Note: The solver retains ownership of the created hierarchy, and of all
  // the tasks added to it.
  //
  // `max_iterations` must be positive. (Precondition violation may cause
  // CHECK-failure.)
  LsqpTaskHierarchy* AddNewTaskHierarchy(int max_iterations);

  // Returns a pointer to the hierarchy at the provided index.
  //
  // Check-fails if index is negative or out of bounds.
  LsqpTaskHierarchy* GetTaskHierarchy(int index);

  // Returns the number of task hierarchies added to the solver. Note that this
  // number includes empty hierarchies, i.e. hierarchies without any tasks.
  int GetNumberOfTaskHierarchies() const;

  // Inserts or assigns a constraint and returns a pointer to the constraint
  // that was inserted or assigned. The object being pointed to will remain
  // valid until it is removed or re-assigned.
  //
  // If an object already exists under the key `name`, assigns the constraint to
  // the existing object under the same key, and returns false in the bool
  // component. If no object exists under `name`, inserts the new constraint and
  // returns true in the bool component.
  //
  // SetupProblem must be called before Solve if a new constraint is
  // added/removed, or if the number of DoF/bounds length of any added
  // constraint changes.
  //
  // The type `T` must accessibly derive from `LsqpConstraint`. The `constraint`
  // pointer must not be null. (Precondition violation may cause CHECK-failure.)
  template <class T>
  std::pair<T*, bool> InsertOrAssignConstraint(absl::string_view name,
                                               std::unique_ptr<T> constraint) {
    static_assert(std::is_base_of_v<LsqpConstraint, T>,
                  "The constraint must derive from LsqpConstraint.");

    T* constraint_ptr = constraint.get();
    return {constraint_ptr,
            InsertOrAssignConstraintImpl(name, std::move(constraint))};
  }

  // Removes a constraint from the solver, if it exists. Returns true if the
  // constraint was removed, false otherwise.
  //
  // Calling this function requires calling SetupProblem() before the next call
  // to Solve.
  bool RemoveConstraint(absl::string_view name);

  // Clears all constraints from the solver.
  //
  // Calling this function requires calling SetupProblem() before the next call
  // to Solve.
  void ClearAllConstraints();

  // Returns true if constraint exists within the solver.
  bool HasConstraint(absl::string_view name) const;

  // Performs all necessary dynamic memory allocation for the added tasks and
  // constraints. If the size of the matrices and vectors returned by the added
  // tasks and constraints do not change, this only needs to be called once.
  //
  // The added tasks and constraints must remain constant while this function is
  // being executed.
  //
  // This function must be called before Solve() if any new tasks or constraints
  // are added/removed, or if the size of the matrices/vectors returned by the
  // tasks and constraints change.
  //
  // Before calling this function, all tasks and constraints must be fully
  // initialized and have the same number of DoF, which must always be larger
  // than zero. (Precondition violation may cause CHECK-failure.)
  absl::Status SetupProblem();

  // Solves the QP stack of tasks problem and returns a status or a view over
  // the solution array. The view is valid until the next call to SetupProblem,
  // Solve, or SetupAndSolve - whichever comes first.
  //
  // The added tasks and constraints must remain constant while this function is
  // being executed.
  //
  // If the coefficient matrices for the tasks and constraints keep the same
  // sparsity, this call does not allocate memory. Note that when this is the
  // case, the solution may have a small numerical difference (<1e-10 in our
  // tests) when compared to the result of SetupAndSolve due to numerical
  // scaling errors. If deterministic solutions are important to the point where
  // a difference of 1e-10 is too large, we recommend the use of SetupAndSolve
  // instead.
  //
  // Internally, it calls OSQP to solve the optimization problems. The OSQP
  // solver's primal and dual variables are initialized (warm-started) with the
  // zero-vector, which is used as a starting point to iterate towards the
  // optimal solution. If a SIGINT signal is raised and detected by OSQP while
  // one of the optimization problems is being solved, this function returns a
  // cancelled-error. Note that whether or not OSQP detects the signal depends
  // on OSQP's compilation options. Please refer to the OSQP's documentation for
  // more information.
  //
  // `SetupProblem` must have been called at least once before this function is
  // called. New tasks or constraints must not have been added or removed, and
  // the size of the matrices/vectors must not have changed before the last call
  // to `SetupProblem`. (Precondition violation may cause CHECK-failure.)
  absl::StatusOr<absl::Span<const double>> Solve();

  // Similar to `Solve`, but the OSQP solver's primal variables are initialized
  // (warm-started) with `initial_guess`. Please refer to the OSQP's
  // documentation for more information on the algorithm and the effects of
  // warm-starting.
  //
  // `initial_guess` must be a view over an array with a size equal to the
  // number of DoF of the optimization problem. (Precondition violation may
  // cause CHECK-failure.)
  absl::StatusOr<absl::Span<const double>> Solve(
      absl::Span<const double> initial_guess);

  // Convenience function that is equivalent to a call to SetupProblem followed
  // by a call to Solve, and returns an error status or a view over the solution
  // array. Similar to Solve, the view is valid until the next call to
  // SetupProblem, Solve, or SetupAndSolve - whichever comes first.
  absl::StatusOr<absl::Span<const double>> SetupAndSolve();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

  // Inserts or assigns a constraint and returns a boolean identifying whether
  // the constraint was inserted (true) or assigned (false).
  //
  // SetupProblem must be called before Solve if a new constraint is
  // added/removed, or if the number of DoF/bounds length of any added
  // constraint changes.
  //
  // Check-fails if constraint is null.
  bool InsertOrAssignConstraintImpl(absl::string_view name,
                                    std::unique_ptr<LsqpConstraint> constraint);
};

inline LsqpStackOfTasksSolver::Parameters::VerboseFlags operator|(
    LsqpStackOfTasksSolver::Parameters::VerboseFlags lhs,
    LsqpStackOfTasksSolver::Parameters::VerboseFlags rhs) {
  return static_cast<LsqpStackOfTasksSolver::Parameters::VerboseFlags>(
      static_cast<std::size_t>(lhs) | static_cast<std::size_t>(rhs));
}
inline LsqpStackOfTasksSolver::Parameters::VerboseFlags operator&(
    LsqpStackOfTasksSolver::Parameters::VerboseFlags lhs,
    LsqpStackOfTasksSolver::Parameters::VerboseFlags rhs) {
  return static_cast<LsqpStackOfTasksSolver::Parameters::VerboseFlags>(
      static_cast<std::size_t>(lhs) & static_cast<std::size_t>(rhs));
}

}  // namespace dm_robotics

#endif  // LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_STACK_OF_TASKS_SOLVER_H_
