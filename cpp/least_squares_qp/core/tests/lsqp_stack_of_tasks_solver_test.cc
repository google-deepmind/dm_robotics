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

#include <limits>
#include <ostream>
#include <vector>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h"
#include "Eigen/Core"
#include "dm_robotics/support/status_macros.h"

namespace dm_robotics {
namespace {

using ::testing::DoubleNear;
using ::testing::Test;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

constexpr int kDeterministicTestIterations = 1000;
constexpr double kQpSolutionTolerance = 1.0e-6;
constexpr double kHierarchicalProjectionSlack = 1.0e-6;

// Task for var_i = value for all i.
class FixedValueTask final : public LsqpTask {
 public:
  FixedValueTask(int num_dof, double value)
      : num_dof_(num_dof),
        bias_(num_dof_, value),
        coefficient_matrix_(num_dof_ * num_dof_) {
    Eigen::Map<Eigen::MatrixXd>(coefficient_matrix_.data(), num_dof_, num_dof_)
        .setIdentity();
  }

  FixedValueTask(const FixedValueTask&) = delete;
  FixedValueTask& operator=(const FixedValueTask&) = delete;

  absl::Span<const double> GetCoefficientMatrix() const final {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetBias() const final { return bias_; }

  int GetNumberOfDof() const final { return num_dof_; }

  int GetBiasLength() const final { return num_dof_; }

 private:
  const int num_dof_;
  const std::vector<double> bias_;
  std::vector<double> coefficient_matrix_;
};

// Task for var_{dof_id} = value, and zero for the rest.
class OneDofFixedValueTask final : public LsqpTask {
 public:
  OneDofFixedValueTask(int num_dof, int dof_id, double value)
      : num_dof_(num_dof),
        dof_id_(dof_id),
        bias_(1, value),
        coefficient_matrix_(num_dof_, 0.0) {
    coefficient_matrix_[dof_id_] = 1;
  }

  OneDofFixedValueTask(const OneDofFixedValueTask&) = delete;
  OneDofFixedValueTask& operator=(const OneDofFixedValueTask&) = delete;

  absl::Span<const double> GetCoefficientMatrix() const final {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetBias() const final { return bias_; }

  int GetNumberOfDof() const final { return num_dof_; }

  int GetBiasLength() const final { return 1; }

 private:
  const int num_dof_;
  const int dof_id_;
  const std::vector<double> bias_;
  std::vector<double> coefficient_matrix_;
};

// Task for var_i = value for all i != dof_id, and zero for i == dof_id.
class ExceptOneDofFixedValueTask final : public LsqpTask {
 public:
  ExceptOneDofFixedValueTask(int num_dof, int dof_id, double value)
      : num_dof_(num_dof),
        dof_id_(dof_id),
        bias_(num_dof_, value),
        coefficient_matrix_(num_dof_ * num_dof_) {
    Eigen::Map<Eigen::MatrixXd> m_map(coefficient_matrix_.data(), num_dof_,
                                      num_dof_);
    m_map.setIdentity();
    m_map(dof_id_, dof_id_) = 0;
    bias_[dof_id_] = 0;
  }

  ExceptOneDofFixedValueTask(const ExceptOneDofFixedValueTask&) = delete;
  ExceptOneDofFixedValueTask& operator=(const ExceptOneDofFixedValueTask&) =
      delete;

  absl::Span<const double> GetCoefficientMatrix() const final {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetBias() const final { return bias_; }

  int GetNumberOfDof() const final { return num_dof_; }

  int GetBiasLength() const final { return num_dof_; }

 private:
  const int num_dof_;
  const int dof_id_;
  std::vector<double> bias_;
  std::vector<double> coefficient_matrix_;
};

// Constraint for lower_bound < var_i < upper_bound for all i.
class FixedLimitConstraint final : public LsqpConstraint {
 public:
  FixedLimitConstraint(int num_dof, double lower_bound, double upper_bound)
      : num_dof_(num_dof),
        lower_bound_(num_dof_, lower_bound),
        upper_bound_(num_dof_, upper_bound),
        coefficient_matrix_(num_dof_ * num_dof_) {
    Eigen::Map<Eigen::MatrixXd>(coefficient_matrix_.data(), num_dof, num_dof)
        .setIdentity();
  }

  FixedLimitConstraint(const FixedLimitConstraint&) = delete;
  FixedLimitConstraint& operator=(const FixedLimitConstraint&) = delete;

  absl::Span<const double> GetCoefficientMatrix() const final {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetUpperBound() const final { return upper_bound_; }

  absl::Span<const double> GetLowerBound() const final { return lower_bound_; }

  int GetNumberOfDof() const final { return num_dof_; }

  int GetBoundsLength() const final { return num_dof_; }

 private:
  const int num_dof_;
  const std::vector<double> lower_bound_;
  const std::vector<double> upper_bound_;
  std::vector<double> coefficient_matrix_;
};

// Test that nullspace failure results in error when
// `return_error_on_nullspace_failure` is true.
TEST(SolverTestWithoutFixture,
     NullspaceFailureResultsInErrorWhenReturnErrorOnNullspaceFailureIsTrue) {
  LsqpStackOfTasksSolver qp_solver(LsqpStackOfTasksSolver::Parameters{
      /*return_error_on_nullspace_failure=*/true,
      /*verbosity=*/LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
      /*absolute_tolerance=*/kQpSolutionTolerance,
      /*relative_tolerance=*/0.0,
      /*hierarchical_projection_slack=*/kHierarchicalProjectionSlack,
      /*primal_infeasibility_tolerance=*/kQpSolutionTolerance / 100,
      /*dual_infeasibility_tolerance=*/kQpSolutionTolerance / 100});

  // Add two hierarchies, where the second hierarchy conflicts on every DoF with
  // the first. We expect the solver to take more than 1 iteration to solve the
  // nullspace task, and thus fail.
  qp_solver.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Task", absl::make_unique<FixedValueTask>(20, -42.42), 1, false);
  qp_solver.AddNewTaskHierarchy(1)->InsertOrAssignTask(
      "NullspaceTask", absl::make_unique<FixedValueTask>(20, 42.42), 1.0e3,
      false);

  EXPECT_OK(qp_solver.SetupProblem());
  EXPECT_EQ(qp_solver.Solve().status().code(), absl::StatusCode::kInternal);
}

// Test fixture for the LsqpStackOfTasksSolver.
class SolverTest : public Test {
 public:
  SolverTest()
      : qp_solver_(LsqpStackOfTasksSolver::Parameters{
            /*return_error_on_nullspace_failure=*/false,
            /*verbosity=*/
            LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
            /*absolute_tolerance=*/kQpSolutionTolerance,
            /*relative_tolerance=*/0.0,
            /*hierarchical_projection_slack=*/kHierarchicalProjectionSlack,
            /*primal_infeasibility_tolerance=*/kQpSolutionTolerance / 100,
            /*dual_infeasibility_tolerance=*/kQpSolutionTolerance / 100}) {}

 protected:
  LsqpStackOfTasksSolver qp_solver_;
};

TEST_F(SolverTest, SolverGetTaskHierarchyReturnsCorrectHierarchy) {
  // Add two empty hierarchyies and one with a fixed value task.
  qp_solver_.AddNewTaskHierarchy(10000);
  qp_solver_.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Task", absl::make_unique<FixedValueTask>(2, 0), 1.0e3, false);
  qp_solver_.AddNewTaskHierarchy(10000);

  EXPECT_EQ(3, qp_solver_.GetNumberOfTaskHierarchies());
  EXPECT_TRUE(qp_solver_.GetTaskHierarchy(0)->IsEmpty());
  EXPECT_TRUE(qp_solver_.GetTaskHierarchy(2)->IsEmpty());

  EXPECT_FALSE(qp_solver_.GetTaskHierarchy(1)->IsEmpty());
  EXPECT_TRUE(qp_solver_.GetTaskHierarchy(1)->HasTask("Task"));
}

TEST_F(SolverTest,
       InvertedConstraintBoundsResultsInInvalidArgumentErrorDuringSolve) {
  // Add task
  qp_solver_.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Task", absl::make_unique<FixedValueTask>(2, 0), 1.0e3, false);

  // Add Constraint
  qp_solver_.InsertOrAssignConstraint(
      "Constraint", absl::make_unique<FixedLimitConstraint>(2, 1, -1));

  EXPECT_OK(qp_solver_.SetupProblem());
  EXPECT_THAT(qp_solver_.SetupAndSolve().status().code(),
              absl::StatusCode::kInvalidArgument);
}

// UnfeasibleConstraints-> kInternal.
TEST_F(SolverTest, UnfeasibleConstraintsResultsInUnfeasibleErrorDuringSolve) {
  // Add task
  qp_solver_.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Task", absl::make_unique<FixedValueTask>(2, 0), 1.0e3, false);

  // Add Constraint
  qp_solver_.InsertOrAssignConstraint(
      "Constraint1", absl::make_unique<FixedLimitConstraint>(2, 0, 1));
  qp_solver_.InsertOrAssignConstraint(
      "Constraint2", absl::make_unique<FixedLimitConstraint>(2, -2, -1));

  // Both `Solve` and `SetupAndSolve` should fail.
  EXPECT_OK(qp_solver_.SetupProblem());
  EXPECT_THAT(qp_solver_.Solve().status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(qp_solver_.SetupAndSolve().status().code(),
              absl::StatusCode::kInternal);
}

// Parameterized test for unconstrained LSQP problems.
// Solves an LSQP with a single task s.t.
//   var_i = target for all i in N dof;
// The solution should be an array with every element equal to `target`.
struct UnconstrainedTestParameters {
  int num_dof;
  double target;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const UnconstrainedTestParameters& param) {
    return stream << "(num_dof[" << param.num_dof << "], target["
                  << param.target << "])";
  }
};

class UnconstrainedTest
    : public SolverTest,
      public WithParamInterface<UnconstrainedTestParameters> {};

constexpr UnconstrainedTestParameters kUnconstrainedTestParametersSet[] = {
    {3, -10}, {4, -0.5}, {5, 0}, {6, 7.1}, {7, 10}};

INSTANTIATE_TEST_SUITE_P(UnconstrainedTests, UnconstrainedTest,
                         ValuesIn(kUnconstrainedTestParametersSet));

TEST_P(UnconstrainedTest, UnconstrainedProblemIsSolvedWithSolve) {
  const auto [num_dof, target] = GetParam();

  // Create Task, add to solver, and setup problem.
  qp_solver_.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Task", absl::make_unique<FixedValueTask>(num_dof, target), 1.0, false);
  ASSERT_OK(qp_solver_.SetupProblem());

  // Solve, this should not allocate memory.
  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver_.Solve());
  Eigen::internal::set_is_malloc_allowed(true);
  EXPECT_EQ(solution.size(), num_dof);

  // Ensure the solution is within the expected tolerance, i.e. the norm of the
  // difference must be less than kQPSolutionTolerance.
  // e_dual = ||W C^T C x - W (b^T C)^T||_inf
  //        = W ||C^T C x - C^T b||_inf
  // where C is the identity matrix for this problem, and b is the target.
  // Note that e_primal and A^T y are irrelevant because this problem is
  // unconstrained.
  const Eigen::VectorXd solution_eigen =
      Eigen::Map<const Eigen::VectorXd>(solution.data(), solution.size());
  const Eigen::VectorXd expected_solution =
      Eigen::VectorXd::Constant(num_dof, target);
  EXPECT_LE((solution_eigen - expected_solution).lpNorm<Eigen::Infinity>(),
            kQpSolutionTolerance);
}

TEST_P(UnconstrainedTest, UnconstrainedProblemIsSolvedWithSetupAndSolve) {
  const auto [num_dof, target] = GetParam();

  // Create Task, add to solver, and setup problem.
  qp_solver_.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Task", absl::make_unique<FixedValueTask>(num_dof, target), 1.0, false);
  ASSERT_OK(qp_solver_.SetupProblem());

  // Solve through `SetupAndSolve`.
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution,
                       qp_solver_.SetupAndSolve());
  EXPECT_EQ(solution.size(), num_dof);

  // Ensure solution from `SetupAndSolve` is always exactly the same, and
  // satisfies the solution tolerance.
  const Eigen::VectorXd setup_and_solve_solution =
      Eigen::Map<const Eigen::VectorXd>(solution.data(), num_dof);
  for (unsigned int i = 0; i < kDeterministicTestIterations; i++) {
    EXPECT_EQ(setup_and_solve_solution,
              Eigen::Map<const Eigen::VectorXd>(
                  qp_solver_.SetupAndSolve().value().data(), num_dof));
  }

  // Ensure the solution is within the expected tolerance, i.e. the norm of the
  // difference must be less than kQPSolutionTolerance.
  // e_dual = ||W C^T C x - W (b^T C)^T||_inf
  //        = W ||C^T C x - C^T b||_inf
  // where C is the identity matrix for this problem, and b is the target.
  // Note that e_primal and A^T y are irrelevant because this problem is
  // unconstrained.
  const Eigen::VectorXd expected_solution =
      Eigen::VectorXd::Constant(num_dof, target);
  EXPECT_LE(
      (setup_and_solve_solution - expected_solution).lpNorm<Eigen::Infinity>(),
      kQpSolutionTolerance);
}

TEST_P(UnconstrainedTest, UnconstrainedProblemIsSolvedWithWarmStartedSolve) {
  const auto [num_dof, target] = GetParam();
  const std::vector<double> initial_guess(num_dof, 0.1235);

  // Create Task, add to solver, and setup problem.
  qp_solver_.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Task", absl::make_unique<FixedValueTask>(num_dof, target), 1.0, false);
  ASSERT_OK(qp_solver_.SetupProblem());

  // Solve, this should not allocate memory.
  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution,
                       qp_solver_.Solve(initial_guess));
  Eigen::internal::set_is_malloc_allowed(true);
  EXPECT_EQ(solution.size(), num_dof);

  // Ensure the solution is within the expected tolerance, i.e. the norm of the
  // difference must be less than kQPSolutionTolerance.
  // e_dual = ||W C^T C x - W (b^T C)^T||_inf
  //        = W ||C^T C x - C^T b||_inf
  // where C is the identity matrix for this problem, and b is the target.
  // Note that e_primal and A^T y are irrelevant because this problem is
  // unconstrained.
  const Eigen::VectorXd warm_started_solution_eigen =
      Eigen::Map<const Eigen::VectorXd>(solution.data(), solution.size());
  const Eigen::VectorXd expected_solution =
      Eigen::VectorXd::Constant(num_dof, target);
  EXPECT_LE((warm_started_solution_eigen - expected_solution)
                .lpNorm<Eigen::Infinity>(),
            kQpSolutionTolerance);
}

// Parameterized test for constrained LSQP problems.
// Solves an LSQP with a single task s.t.
//   var_i = target for all i in N dof;
// With a constraint s.t.
//   lower_bound < var_i < upper_bound for all i in N dof;
// The solution must be as close as possible to `target` without violating the
// bounds.
struct ConstrainedTestParameters {
  int num_dof;
  double target;
  double lower_bound;
  double upper_bound;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const ConstrainedTestParameters& param) {
    return stream << "(num_dof[" << param.num_dof << "], target["
                  << param.target << "], lower_bound[" << param.lower_bound
                  << "], upper_bound[" << param.upper_bound << "])";
  }
};

class ConstrainedTest : public SolverTest,
                        public WithParamInterface<ConstrainedTestParameters> {};

constexpr ConstrainedTestParameters kConstraintTestParametersSet[] = {
    {3, -0.5, -7.6, -7.5},
    {4, 0, 1, 3},
    {5, 7.1, 7.1, 7.1},
    {6, 10, 9.9, std::numeric_limits<double>::infinity()},
    {7, -10, -std::numeric_limits<double>::infinity(), 0}};

INSTANTIATE_TEST_SUITE_P(ConstrainedTests, ConstrainedTest,
                         ValuesIn(kConstraintTestParametersSet));

TEST_P(ConstrainedTest, ConstrainedProblemIsSolved) {
  const auto [num_dof, target, lower_bound, upper_bound] = GetParam();

  // Add task & constraint to solver and setup.
  qp_solver_.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Task", absl::make_unique<FixedValueTask>(num_dof, target), 1.0, false);
  qp_solver_.InsertOrAssignConstraint("FixedLimitConstraint",
                                      absl::make_unique<FixedLimitConstraint>(
                                          num_dof, lower_bound, upper_bound));
  ASSERT_OK(qp_solver_.SetupProblem());

  // Solve, this should not allocate memory.
  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver_.Solve());
  Eigen::internal::set_is_malloc_allowed(true);

  // Ensure the solution is within the expected tolerance.
  // e_primal = ||Ax - z||_inf
  // e_dual = || W [C^T C x - C^T b] + A^T y ||_inf
  EXPECT_EQ(solution.size(), num_dof);
  const Eigen::VectorXd solution_eigen =
      Eigen::Map<const Eigen::VectorXd>(solution.data(), solution.size());

  // Compute how much the solution violates the constraints, if any, and ensure
  // the constraint violation is within the tolerance.
  if (lower_bound > solution_eigen.minCoeff()) {
    EXPECT_LE(lower_bound - solution_eigen.minCoeff(), kQpSolutionTolerance);
  } else if (upper_bound < solution_eigen.maxCoeff()) {
    EXPECT_LE(solution_eigen.maxCoeff() - upper_bound, kQpSolutionTolerance);
  }

  // We approximate `e_dual` by computing the difference between the solution
  // and the optimal solution (computed manually). Note that this is not
  // exactly equal to `e_dual`, but we found (A^T y) to be close to zero for
  // this problem.
  Eigen::VectorXd expected_solution;
  if (lower_bound > target) {
    expected_solution = Eigen::VectorXd::Constant(num_dof, lower_bound);
  } else if (upper_bound < target) {
    expected_solution = Eigen::VectorXd::Constant(num_dof, upper_bound);
  } else {
    expected_solution = Eigen::VectorXd::Constant(num_dof, target);
  }
  EXPECT_LE((solution_eigen - expected_solution).lpNorm<Eigen::Infinity>(),
            kQpSolutionTolerance);
}

// Parameterized test for weighted LSQP problems.
// Solves a two-task weighted problem with conflicting goals in same hierarchy:
//   First task:  var_i = first_task_target for all i in N dof;
//   Second task: var_i = second_task_target for all i in N dof;
// The solution has to satisfy:
//   W1 * abs(sol - first_task_target) = W2 * abs(sol - second_task_target)
struct WeightedTestParameters {
  int num_dof;
  double first_task_target;
  double first_task_weight;
  double second_task_target;
  double second_task_weight;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const WeightedTestParameters& param) {
    return stream << "(num_dof[" << param.num_dof << "], first_task_target["
                  << param.first_task_target << "], first_task_weight["
                  << param.first_task_weight << "], second_task_target["
                  << param.second_task_target << "], second_task_weight["
                  << param.second_task_weight << "])";
  }
};

class WeightedTest : public SolverTest,
                     public WithParamInterface<WeightedTestParameters> {};

constexpr WeightedTestParameters kWeightedTestParametersSet[] = {
    {2, 3.890, 3.5, -3.890, 3.5},
    {3, 0, 3, 10, 0.001},
    {4, -0.5, 1.0e3, -7.6, 1},
    {5, 2.53, 1.0e3, 2.53, 1.0e-3},
    {6, -894, 1.0e-3, 243, 1.0e3}};

INSTANTIATE_TEST_SUITE_P(WeightedTests, WeightedTest,
                         ValuesIn(kWeightedTestParametersSet));

TEST_P(WeightedTest, WeightedProblemIsSolved) {
  const auto [num_dof, first_task_target, first_task_weight, second_task_target,
              second_task_weight] = GetParam();

  // Add tasks to solver and setup.
  LsqpTaskHierarchy* hierarchy = qp_solver_.AddNewTaskHierarchy(10000);
  hierarchy->InsertOrAssignTask(
      "FixedValueTask1",
      absl::make_unique<FixedValueTask>(num_dof, first_task_target),
      first_task_weight, false);
  hierarchy->InsertOrAssignTask(
      "FixedValueTask2",
      absl::make_unique<FixedValueTask>(num_dof, second_task_target),
      second_task_weight, false);
  ASSERT_OK(qp_solver_.SetupProblem());

  // Solve, this should not allocate memory.
  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver_.Solve());
  Eigen::internal::set_is_malloc_allowed(true);

  // Ensure that the solution results in equal costs when multiplied with the
  // weight. Solution has to satisfy:
  //   W1 * abs(sol - first_task_target) = W2 * abs(sol - second_task_target)
  // The solution to this problem is the weighted sum of each task divided by
  // the total sum of the weights.
  EXPECT_EQ(solution.size(), num_dof);
  const Eigen::VectorXd solution_eigen =
      Eigen::Map<const Eigen::VectorXd>(solution.data(), solution.size());
  const Eigen::VectorXd expected_solution = Eigen::VectorXd::Constant(
      num_dof, (first_task_weight * first_task_target +
                second_task_weight * second_task_target) /
                   (second_task_weight + first_task_weight));

  // Ensure that the residual is within the specified tolerance.
  // e_dual =  || (W1 C^T C + W2 C^T C) x - (W1 b1^T C + W2 b2^T C)^T||_inf
  const Eigen::VectorXd e_dual_first_term =
      (first_task_weight + second_task_weight) * solution_eigen;
  const Eigen::VectorXd e_dual_second_term =
      first_task_weight *
          Eigen::VectorXd::Constant(num_dof, first_task_target) +
      second_task_weight *
          Eigen::VectorXd::Constant(num_dof, second_task_target);
  EXPECT_LE((e_dual_first_term - e_dual_second_term).lpNorm<Eigen::Infinity>(),
            kQpSolutionTolerance);
}

// Parameterized test for hierarchical LSQP problems.
// Solves a three-task two-hierarchy LSQP.
// First hierarchy:
//  Task1: var_0 = value1, with nullspace projection;
//  Task2: var_i = 0 for i != 0 in N dof, without nullspace projection;
// Second hierarchy:
//  Task: var_i = value2 for all i in N dof;
// Solution must be var_0 = value1 and var_i = value2 for all i != 0.
struct HierarchicalTestParameters {
  int num_dof;
  double first_dof_target;          // For first DoF.
  double all_dof_nullspace_target;  // For all DoF.
  double task_weight;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const HierarchicalTestParameters& param) {
    return stream << "(num_dof[" << param.num_dof << "], first_dof_target["
                  << param.first_dof_target << "], all_dof_nullspace_target["
                  << param.all_dof_nullspace_target << "], task_weight["
                  << param.task_weight << "])";
  }
};

class HierarchicalTest : public SolverTest,
                         public WithParamInterface<HierarchicalTestParameters> {
};

constexpr HierarchicalTestParameters kHierarchicalTestParametersSet[] = {
    {2, 5, 10, 1},
    {3, -5, -10, 10},
    {4, 0.3, -8, 1.0e3},
    {5, -2.53, 100, 1.0e-2},
    {6, -0.001, -0.001, 243}};

INSTANTIATE_TEST_SUITE_P(HierarchicalTests, HierarchicalTest,
                         ValuesIn(kHierarchicalTestParametersSet));

TEST_P(HierarchicalTest, HierarchicalProblemIsSolved) {
  const auto [num_dof, first_dof_target, all_dof_nullspace_target, weight] =
      GetParam();

  // Add tasks to solver and setup.
  // Regularization task is added to ensure problem is well defined.
  // Dummy hierarchies are added to ensure that they are ignored as per
  // LsqpStackOfTasksSolver documentation.
  LsqpTaskHierarchy* first_hierarchy = qp_solver_.AddNewTaskHierarchy(10000);
  first_hierarchy->InsertOrAssignTask(
      "FirstDofTask",
      absl::make_unique<OneDofFixedValueTask>(num_dof, 0, first_dof_target),
      weight, false);
  first_hierarchy->InsertOrAssignTask(
      "Regularization",
      absl::make_unique<ExceptOneDofFixedValueTask>(num_dof, 0, 0), weight,
      true);  // Ignore nullspace

  // Dummy hierarchies.
  qp_solver_.AddNewTaskHierarchy(10000);
  qp_solver_.AddNewTaskHierarchy(10000);
  qp_solver_.AddNewTaskHierarchy(10000);

  // Second valid hierarchy.
  qp_solver_.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "AllDofTask",
      absl::make_unique<FixedValueTask>(num_dof, all_dof_nullspace_target),
      weight, false);

  // Dummy hierarchies.
  qp_solver_.AddNewTaskHierarchy(10000);
  qp_solver_.AddNewTaskHierarchy(10000);
  qp_solver_.AddNewTaskHierarchy(10000);
  ASSERT_OK(qp_solver_.SetupProblem());

  // Solve, this should not allocate memory.
  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver_.Solve());
  Eigen::internal::set_is_malloc_allowed(true);

  // The first DoF must converge to first_dof_target.
  // For the first hierarchy:
  //   e_dual < kQpSolutionTolerance
  //   W || x - b||_inf < kQpSolutionTolerance
  //   W abs(x_first_dof - b_first_dof) < kQpSolutionTolerance
  // Note that the maximum error for any DoF is kQpSolutionTolerance/weight.
  //
  // When projected into the second hierarchy, a slack of
  // kHierarchicalProjectionSlack will be added, and the solution may violate
  // the nullspace projection constraint by up to kQpSolutionTolerance.
  //
  // Thus, the first DoF value maximum error will be:
  //   kQpSolutionTolerance / weight
  //   + kHierarchicalProjectionSlack
  //   + kQpSolutionTolerance
  ASSERT_EQ(solution.size(), num_dof);
  EXPECT_THAT(solution[0],
              DoubleNear(first_dof_target, kQpSolutionTolerance / weight +
                                               kHierarchicalProjectionSlack +
                                               kQpSolutionTolerance));

  // The rest of the DoFs must converge to `all_dof_nullspace_target`.
  // This hierarchy has a constraint resulting from the first hierarchy, and
  // thus the `e_dual` residual takes the form:
  //   e_dual = || W [C^T C x - C^T b] + A^T y ||_inf
  // This is cannot be evaluated exactly since the LSQP solver does not expose
  // the value of the lagrange multiplier y, but we note that A is zero for all
  // the other DoFs, and thus the norm of the e_dual elements for the other DoFs
  // must be less than kQpSolutionTolerance.
  Eigen::VectorXd solution_eigen =
      Eigen::Map<const Eigen::VectorXd>(solution.data(), solution.size());
  Eigen::VectorXd e_dual_first_term =
      weight * (solution_eigen -
                Eigen::VectorXd::Constant(num_dof, all_dof_nullspace_target));
  EXPECT_LE(e_dual_first_term.bottomRows(num_dof - 1).lpNorm<Eigen::Infinity>(),
            kQpSolutionTolerance);
}

}  // namespace
}  // namespace dm_robotics
