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

#ifndef LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_UTILS_H_
#define LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_UTILS_H_

#include <type_traits>
#include <vector>

#include "absl/types/span.h"

namespace dm_robotics {

// Copies the array referred to by an absl::Span<T> object into a newly
// allocated std::vector<U> object, where U is the same type as T with its
// topmost cv-qualifiers removed.
//
// Example usage:
// absl::Span<const double> span = ...;
// std::vector<double> copy = AsCopy(span);
template <typename T, typename Alloc = std::allocator<std::remove_cv_t<T>>>
std::vector<std::remove_cv_t<T>, Alloc> AsCopy(absl::Span<T> s,
                                               const Alloc& alloc = Alloc()) {
  return std::vector<std::remove_cv_t<T>, Alloc>(s.begin(), s.end(), alloc);
}

}  // namespace dm_robotics

#endif  // LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_UTILS_H_
