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

#ifndef DM_ROBOTICS_SUPPORT_STATUS_MATCHERS_H_
#define DM_ROBOTICS_SUPPORT_STATUS_MATCHERS_H_

#define EXPECT_OK(exp)                              \
  do {                                              \
    auto&& _dm_robotics_status_matcher_s = (exp);   \
    EXPECT_TRUE(_dm_robotics_status_matcher_s.ok()) \
        << _dm_robotics_status_matcher_s;           \
  } while (false)

#define ASSERT_OK(exp)                              \
  do {                                              \
    auto&& _dm_robotics_status_matcher_s = (exp);   \
    ASSERT_TRUE(_dm_robotics_status_matcher_s.ok()) \
        << _dm_robotics_status_matcher_s;           \
  } while (false)

// Evaluates `status_or_expr` and asserts that it is Ok. If the assert passes,
// it moves its value to the variable resulting from evaluating `var_exp`.
//
// `status_or_expr` must be an expression of (or convertible to)
// absl::StatusOr<T>, and T must be moveable (or copyable) to the variable
// resulting from `var_exp`.
#define ASSERT_OK_AND_ASSIGN(var_exp, status_or_exp) \
  ASSERT_OK_AND_ASSIGN_(var_exp, status_or_exp,      \
                        STATUS_MATCHERS_CONCAT_(s_or, __COUNTER__))

#define STATUS_MATCHERS_CONCAT_IMPL_(a, b) a##b
#define STATUS_MATCHERS_CONCAT_(a, b) STATUS_MATCHERS_CONCAT_IMPL_(a, b)
#define ASSERT_OK_AND_ASSIGN_(var_exp, status_or_exp, status_or_unique_name) \
  auto&& status_or_unique_name = (status_or_exp);                            \
  ASSERT_OK(status_or_unique_name.status());                                 \
  var_exp = *std::move(status_or_unique_name)

#endif  // DM_ROBOTICS_SUPPORT_STATUS_MATCHERS_H_
