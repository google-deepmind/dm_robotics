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

#ifndef DM_ROBOTICS_SUPPORT_STATUS_MACROS_H_
#define DM_ROBOTICS_SUPPORT_STATUS_MACROS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// Evaluates `status_expr` and returns it from the current function if it is not
// Ok.
//
// `status_expr` must be an expression of (or convertible to) absl::Status.
#define RETURN_IF_ERROR(status_expr)                       \
  do {                                                     \
    if (absl::Status s = (status_expr); !s.ok()) return s; \
  } while (false)

// Evaluates `status_or_expr` and returns the status from the current function
// if it is not Ok. Otherwise, it moves its value to the variable resulting from
// evaluating `var_exp`.
//
// `status_or_expr` must be an expression of (or convertible to)
// absl::StatusOr<T>, and T must be moveable (or copyable) to the variable
// resulting from `var_exp`.
#define ASSIGN_OR_RETURN(var_exp, status_or_exp) \
  ASSIGN_OR_RETURN_(var_exp, status_or_exp,      \
                    STATUS_MACROS_CONCAT_(s_or, __COUNTER__))

#define STATUS_MACROS_CONCAT_IMPL_(a, b) a##b
#define STATUS_MACROS_CONCAT_(a, b) STATUS_MACROS_CONCAT_IMPL_(a, b)
#define ASSIGN_OR_RETURN_(var_exp, status_or_exp, status_or_unique_name) \
  auto&& status_or_unique_name = (status_or_exp);                        \
  if (!status_or_unique_name.ok()) {                                     \
    return std::move(status_or_unique_name).status();                    \
  }                                                                      \
  var_exp = *std::move(status_or_unique_name)

#endif  // DM_ROBOTICS_SUPPORT_STATUS_MACROS_H_
