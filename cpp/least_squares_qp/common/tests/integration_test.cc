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

#include <vector>

#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/common/box_constraint.h"
#include "dm_robotics/least_squares_qp/common/identity_task.h"
#include "dm_robotics/least_squares_qp/common/minimize_norm_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h"
#include "Eigen/Core"

namespace dm_robotics {
namespace {

using ::testing::TestWithParam;
using ::testing::ValuesIn;

// Fixture for parameterizing tests on a MinimizeNormTask with a variety of
// number of DoF.
using ParameterizedMinimizeNormTaskDofTest = TestWithParam<int>;
INSTANTIATE_TEST_SUITE_P(SolutionsAreZeroVectors,
                         ParameterizedMinimizeNormTaskDofTest,
                         ValuesIn({1, 2, 3, 4, 5}));

// Fixture for parameterizing tests on an IdentityTask with a variety of
// targets.
using ParameterizedIdentityTaskTargetTest = TestWithParam<std::vector<double>>;
INSTANTIATE_TEST_SUITE_P(
    SolutionsAreEqualToTargets, ParameterizedIdentityTaskTargetTest,
    ValuesIn({std::vector<double>(5, 0.0),  //
              std::vector<double>{1, 2, 3, 4, 5},
              std::vector<double>{-5, -4, -3, -2, -1},
              std::vector<double>{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5},
              std::vector<double>{-1.0e3, -1.0e2, -1.0e1, 0.0, 1.0e1, 1.0e2,
                                  1.0e3}}));

// The solution of the QP problem formed with only the MinimizeNorm task should
// always be the zero vector.
TEST_P(ParameterizedMinimizeNormTaskDofTest, SolutionIsZeroVector) {
  const double kTolerance = 1e-6;
  const int kNumDof = GetParam();
  LsqpStackOfTasksSolver qp_solver(LsqpStackOfTasksSolver::Parameters{
      /*use_adaptive_rho=*/false,
      /*return_error_on_nullspace_failure=*/false,
      /*verbosity=*/LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
      /*absolute_tolerance=*/kTolerance,
      /*relative_tolerance=*/0.0,
      /*hierarchical_projection_slack=*/1.0e-5,
      /*primal_infeasibility_tolerance=*/1.0e-6,
      /*dual_infeasibility_tolerance=*/1.0e-6});

  qp_solver.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "MinNorm", absl::make_unique<MinimizeNormTask>(kNumDof), 1.0, true);
  ASSERT_OK(qp_solver.SetupProblem());

  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver.Solve());
  Eigen::internal::set_is_malloc_allowed(true);

  // The solution must be within kTolerance of zero.
  Eigen::Map<const Eigen::VectorXd> solution_map(solution.data(),
                                                 solution.size());
  EXPECT_LE(solution_map.lpNorm<Eigen::Infinity>(), kTolerance);
}

// The QP problem formed with only the IdentityTask should always find a
// solution equal to the bias.
TEST_P(ParameterizedIdentityTaskTargetTest, SolutionIsEqualToTarget) {
  const double kTolerance = 1e-6;
  const std::vector<double> kTarget = GetParam();
  LsqpStackOfTasksSolver qp_solver(LsqpStackOfTasksSolver::Parameters{
      /*use_adaptive_rho=*/false,
      /*return_error_on_nullspace_failure=*/false,
      /*verbosity=*/LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
      /*absolute_tolerance=*/kTolerance,
      /*relative_tolerance=*/0.0,
      /*hierarchical_projection_slack=*/1.0e-5,
      /*primal_infeasibility_tolerance=*/1.0e-6,
      /*dual_infeasibility_tolerance=*/1.0e-6});

  qp_solver.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "Identity", absl::make_unique<IdentityTask>(kTarget), 1.0, true);
  ASSERT_OK(qp_solver.SetupProblem());

  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver.Solve());
  Eigen::internal::set_is_malloc_allowed(true);

  // The solver's tolerance defines the norm of the difference between the
  // expected and computed solution.
  Eigen::Map<const Eigen::VectorXd> solution_map(solution.data(),
                                                 solution.size());
  Eigen::Map<const Eigen::VectorXd> expected_map(kTarget.data(),
                                                 kTarget.size());
  EXPECT_LE((solution_map - expected_map).lpNorm<Eigen::Infinity>(),
            kTolerance);
}

// Solution with the MinimumNormTask and a BoxConstraint should be as close as
// possible to zero without violating the upper/lower bounds.
TEST(MinNormBoxConstraintIntegrationTest,
     SolutionIsAsCloseToZeroWithoutViolatingBounds) {
  const double kTolerance = 1.0e-6;
  const std::vector<double> kUpperBound{-2, -1, 0, 1, 2, 3, 4};
  const std::vector<double> kLowerBound{-4, -3, -2, -1, 0, 1, 2};
  const std::vector<double> kExpected{-2, -1, 0, 0, 0, 1, 2};

  // Initialize solver.
  LsqpStackOfTasksSolver qp_solver(LsqpStackOfTasksSolver::Parameters{
      /*use_adaptive_rho=*/false,
      /*return_error_on_nullspace_failure=*/false,
      /*verbosity=*/LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
      /*absolute_tolerance=*/kTolerance,
      /*relative_tolerance=*/0.0,
      /*hierarchical_projection_slack=*/1.0e-5,
      /*primal_infeasibility_tolerance=*/1.0e-6,
      /*dual_infeasibility_tolerance=*/1.0e-6});

  // Create problem with a min-norm task and a box constraint.
  qp_solver.AddNewTaskHierarchy(10000)->InsertOrAssignTask(
      "MinNorm", absl::make_unique<MinimizeNormTask>(kLowerBound.size()), 1.0,
      true);
  qp_solver.InsertOrAssignConstraint(
      "Box", absl::make_unique<BoxConstraint>(kLowerBound, kUpperBound));
  ASSERT_OK(qp_solver.SetupProblem());

  // These tasks should not cause memory allocation after Setup.
  Eigen::internal::set_is_malloc_allowed(false);
  ASSERT_OK_AND_ASSIGN(absl::Span<const double> solution, qp_solver.Solve());
  Eigen::internal::set_is_malloc_allowed(true);

  // The solver's tolerance defines the maximum norm of the difference between
  // the expected and computed solution. Note that the solution is allowed to
  // violate the box constraint limits by up to kTolerance.
  Eigen::Map<const Eigen::VectorXd> solution_map(solution.data(),
                                                 solution.size());
  Eigen::Map<const Eigen::VectorXd> expected_map(kExpected.data(),
                                                 kExpected.size());
  EXPECT_LE((solution_map - expected_map).lpNorm<Eigen::Infinity>(),
            kTolerance);
}

}  // namespace
}  // namespace dm_robotics
