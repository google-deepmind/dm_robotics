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

#ifndef LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_TASK_HIERARCHY_H_
#define LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_TASK_HIERARCHY_H_

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "dm_robotics/support/logging.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "dm_robotics/least_squares_qp/core/lsqp_task.h"

namespace dm_robotics {

struct LsqpTaskInfo {
  std::unique_ptr<const LsqpTask> task;
  bool should_ignore_nullspace;
  double weight;
};

using LsqpTaskMap = absl::flat_hash_map<std::string, LsqpTaskInfo>;

// Container class that holds a set of tasks with relative weights between them.
// Every hierarchy of tasks defines the following weighted least-squares
// optimization problem:
//   argmin_x Sum_i w_i || M_i x - b_i ||^2;
// where w_i, M_i, and b_i are the weight, coefficient matrix, and bias of the
// i-th task, respectively.
class LsqpTaskHierarchy {
 public:
  LsqpTaskHierarchy() = default;
  LsqpTaskHierarchy(const LsqpTaskHierarchy&) = delete;
  LsqpTaskHierarchy& operator=(const LsqpTaskHierarchy&) = delete;

  // Inserts or assigns a task to the hierarchy. Returns a pair, where the first
  // field contains a pointer to the task that was inserted or assigned, and the
  // second field contains a boolean identifying whether the task was
  // inserted(true) or assigned(false).
  //
  // The returned task will remain valid until it is removed or re-assigned. If
  // the should_ignore_nullspace flag is true, the task will not be included in
  // the hierarchical nullspace projection constraint - this is useful for
  // regularization tasks.
  //
  // The type `T` must accessibly derive from `LspqTask`. The `task` pointer
  // must not be null. (Precondition violation may cause CHECK-failure.)
  template <class T>
  std::pair<T*, bool> InsertOrAssignTask(absl::string_view name,
                                         std::unique_ptr<T> task, double weight,
                                         bool should_ignore_nullspace) {
    static_assert(std::is_base_of_v<LsqpTask, T>,
                  "The task must derive from LsqpTask.");
    CHECK(task != nullptr)
        << "LsqpTaskHierarchy::InsertOrAssignTask: 'task' must not be null.";

    // Get the task pointer before inserting to ensure we return a pointer of
    // type T*.
    T* task_ptr = task.get();
    return {task_ptr,
            tasks_
                .insert_or_assign(
                    name, {std::move(task), should_ignore_nullspace, weight})
                .second};
  }

  // Removes a task from the hierarchy.
  // Returns true if the task was removed, false otherwise.
  bool RemoveTask(absl::string_view name) { return tasks_.erase(name) != 0; }

  // Clears all tasks from the hierarchy.
  void ClearAllTasks() { tasks_.clear(); }

  // Returns true if a task with the provided name exists within the hierarchy.
  bool HasTask(absl::string_view name) const {
    return tasks_.find(name) != tasks_.end();
  }

  // Returns true if the hierarchy has no tasks.
  // Note: Empty hierarchies are ignored by LsqpStackOfTasksSolver.
  bool IsEmpty() const { return tasks_.empty(); }

  // Returns a const_iterator to the first element of the dictionary containing
  // all the tasks for this hierarchy.
  LsqpTaskMap::const_iterator begin() const { return tasks_.begin(); }

  // Returns a const_iterator to the end of the dictionary containing all the
  // tasks for this hierarchy.
  LsqpTaskMap::const_iterator end() const { return tasks_.end(); }

 private:
  LsqpTaskMap tasks_;
};

}  // namespace dm_robotics

#endif  // LEARNING_DEEPMIND_ROBOTICS_LEAST_SQUARES_QP_CORE_LSQP_TASK_HIERARCHY_H_
