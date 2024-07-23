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

#ifndef DM_ROBOTICS_CONTROLLERS_LSQP_TEST_UTILS_H_
#define DM_ROBOTICS_CONTROLLERS_LSQP_TEST_UTILS_H_

#include "absl/container/btree_set.h"
#include "absl/types/span.h"
#include "dm_robotics/mujoco/utils.h"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics::testing {

// Returns the Jacobian mapping the joint velocities to the object's 6D
// Cartesian velocity, in column-major ordering.
std::vector<double> ComputeObject6dJacobianForJoints(
    const mjModel& model, const mjData& data,
    mjtObj object_type, const std::string& object_name,
    const absl::btree_set<int>& joint_ids);

// Computes an object's 6D Cartesian velocity by multiplying the object's
// Jacobian by the joint velocities.
// This is necessary to make sure that the QP-based solution, which is
// Jacobian-based, can be compared with the expected tolerance, as MuJoCo's
// native computation results in a slightly different value.
std::array<double, 6> ComputeObjectCartesian6dVelocityWithJacobian(
    const mjModel& model, const mjData& data,
    const std::string& object_name, mjtObj object_type);

// Sets a subset of the joint velocities of the MuJoCo humanoid. All other
// joint velocities are set to zero.
//
// The `joint_velocities` array must contain the joint velocities of each
// joint, in ascending order according to `joint_ids`. Only 1 DoF joints are
// supported.
void SetSubsetOfJointVelocities(const mjModel& model,
                                const absl::btree_set<int>& joint_ids,
                                absl::Span<const double> joint_velocities,
                                mjData* data);

}  // namespace dm_robotics::testing

#endif  // DM_ROBOTICS_CONTROLLERS_LSQP_TEST_UTILS_H_
