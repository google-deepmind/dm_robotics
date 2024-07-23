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

#ifndef DM_ROBOTICS_MUJOCO_UTILS_H_
#define DM_ROBOTICS_MUJOCO_UTILS_H_

#include <utility>

#include "absl/container/btree_set.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dm_robotics/mujoco/types.h"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {

// Returns the number of degrees of freedom (DoF) of the joint with ID
// `joint_id`.
//
// Free joints have 6 DoF; ball joints have 3 DoF; and prismatic/revolute
// joints have 1 DoF.
//
// CHECK-fails if `joint_id` is not valid for `model.
int GetJointDofSize(const mjModel& model, int joint_id);

// Returns the number of MuJoCo `qpos` values of the joint with ID `joint_id`.
//
// In MuJoCo, free joints have 7 `qpos` values that represent the position and
// orientation of the joint; ball joints have 4 `qpos` values that represent the
// orientation of the joint; and prismatic/revolute joints have 1 `qpos` value.
//
// CHECK-fails if `joint_id` is not valid for `model`.
int GetJointQposSize(const mjModel& model, int joint_id);

// Converts a set of MuJoCo joint IDs to a set of DoF IDs.
// Note that several DoF IDs may be associated to a single MuJoCo joint.
//
// CHECK-fails if any joint ID is not valid for `model`.
absl::btree_set<int> JointIdsToDofIds(const mjModel& model,
                                      const absl::btree_set<int>& joint_ids);

// Returns the maximum number of contacts that MuJoCo can detect between the
// geoms specified by `geom_pairs`. It is the user's responsibility to ensure
// that geom pairs are not provided for unnecessary contacts, e.g. between two
// geoms corresponding to the same body, or providing the same pair twice in
// different order.
int ComputeMaximumNumberOfContacts(
    const mjModel& model,
    const absl::btree_set<std::pair<int, int>>& geom_pairs);

// Returns a set of geom ID pairs for all the possible geom-to-geom collisions.
// For each pair in the set, the ID in the first element is always guaranteed to
// be different and less than the ID in the second element.
//
// The contacts are added based on the following heuristics:
// 1) Contacts between geoms that are part of the same body or weld are never
//    included;
// 2) If `allow_parent_child_collisions` is false, the returned set will not
//    include pairs where the body of one geom is a parent of the body of the
//    other geom;
// 3) If `allow_worldbody_collisions` is true, the returned set will include
//    pairs where one of the geoms is part of the worldbody and the other is
//    not, even if it violates condition 2. Note that condition 1 takes
//    prescedence.
//
// Note:
//  - If two bodies are kinematically welded together (no joints between them,
//    body_weldid[a] == body_weldid[b]), they are considered to be the same
//    body within this function;
//  - This function ignores any MuJoCo contype/conaffinity fields (i.e. acts as
//    if contype/conaffinity is 1 for every geom);
//  - This function ignores any <pair> or <exclude> XML elements (i.e. acts as
//    if these two fields are empty).
//
// MuJoCo's default collision-filtering can be achieved by setting
// allow_parent_child_collisions to false and allow_worldbody_collisions to
// true.
//
// CHECK-fails if any of the geom names in `collision_pairs` is invalid.
absl::btree_set<std::pair<int, int>> CollisionPairsToGeomIdPairs(
    const mjModel& model,
    const absl::btree_set<CollisionPair>& collision_pairs,
    bool allow_parent_child_collisions, bool allow_worldbody_collisions);

// Computes the `dist`, `pos`, `frame` ([0]-[2] only), `geom1`, and `geom2`
// mjContact fields for every detected collision between the geoms specified by
// the `geom_pairs` parameter, and writes them into the provided `contacts`
// array starting from index 0. The `contacts` array must be large enough to
// hold all the detected contacts.
//
// This function ignores any collision-filtering mechanism by MuJoCo, and
// computes the contacts fields between all the geom pairs provided. It is the
// user's responsibility to ensure that geom pairs are not provided for
// unnecessary contacts, e.g. between two geoms corresponding to the same body,
// or providing the same pair twice in different order.
//
// The provided `data` object must have updated kinematic information, i.e. it
// must have called the following MuJoCo routine (either directly or through
// other MuJoCo computations):
// - mj_kinematics
//
// If successful, returns the number of collisions detected, i.e. the number of
// mjContact objects updated in the `contacts` parameter.
// Returns an error if:
// - the `contacts` array is not large enough to hold all the detected contacts.
//
// CHECK-fails if:
// - the ID of any geom is invalid for the provided model;
// - both elements in any pair contain the same geom ID;
// - MuJoCo failed to compute contacts for any geom pair.
absl::StatusOr<int> ComputeContactsForGeomPairs(
    const mjModel& model, const mjData& data,
    const absl::btree_set<std::pair<int, int>>& geom_pairs,
    double collision_detection_distance, absl::Span<mjContact> contacts);

// Computes the Jacobian mapping the model's joint velocities to the normal
// component of relative Cartesian linear velocity between the two geoms in a
// MuJoCo contact.
//
// The Jacobian-velocity relationship is given as:
//   J q_dot = n^T (v_2 - v_1)
// where
// * J is the computed Jacobian;
// * q_dot is the joint velocity vector;
// * n^T is the transpose of the normal pointing from contact.geom1 to
//   contact.geom2;
// * v_1, v_2 are the linear components of the Cartesian velocity of the two
//   closest points in contact.geom1 and contact.geom2.
// Note: n^T (v_2 - v_1) is a scalar that is positive if the geoms are moving
// away from each other, and negative if they are moving towards each other.
//
// This function does not allocate memory; the resultant Jacobian is written
// into the `jacobian` output parameter, which must be a view over an array of
// length model.nv. The `jacobian_buffer` parameter must be a view over an array
// of size 3*nv, and is used for internal computations.
//
// CHECK-fails if either the `jacobian_buffer` or the `jacobian` are not of the
// correct size.
void ComputeContactNormalJacobian(const mjModel& model,
                                  const mjData& data, const mjContact& contact,
                                  absl::Span<double> jacobian_buffer,
                                  absl::Span<double> jacobian);

// Returns the contact with the minimum of the contact distances between two
// geoms detected by MuJoCo. If no collision is detected, this function will
// return a disengaged value.
//
// The computed contact is guaranteed to contain accurate values for the `dist`,
// `pos`, `frame` ([0]-[2] only), `geom1`, and `geom2` fields.
//
// The provided `data` object must have updated kinematic information, i.e. it
// must have called the following MuJoCo routine (either directly or through
// other MuJoCo computations):
// - mj_kinematics
//
// CHECK-fails if:
// - the ID of any geom is invalid for the provided model;
// - both geom IDs are equal;
// - MuJoCo failed to compute contacts for the provided geoms.
absl::optional<mjContact> ComputeContactWithMinimumDistance(
    const mjModel& model, const mjData& data, int geom1_id,
    int geom2_id, double collision_detection_distance);

// Returns the minimum of the contact distances between two geoms detected by
// MuJoCo. If no collision is detected, this function will return a disengaged
// value.
//
// The provided `data` object must have updated kinematic information, i.e. it
// must have called the following MuJoCo routine (either directly or through
// other MuJoCo computations):
// - mj_kinematics
//
// CHECK-fails if:
// - the ID of any geom is invalid for the provided model;
// - both geom IDs are equal;
// - MuJoCo failed to compute contacts for the provided geoms.
absl::optional<double> ComputeMinimumContactDistance(
    const mjModel& model, const mjData& data, int geom1_id,
    int geom2_id, double collision_detection_distance);

// Computes the [6 x model.nv] Jacobian that maps joint velocities to the
// object's Cartesian 6D velocity in the world-frame, and writes it into the
// `jacobian` output parameter as a row-major storage matrix. The linear
// components are written into the first 3 rows, and the angular components are
// written into the last 3 rows.
//
// Only MuJoCo geoms, sites, and bodies are supported. (Precondition
// violation may cause CHECK-failure.)
void ComputeObject6dJacobian(const mjModel& model,
                             const mjData& data, mjtObj object_type,
                             int object_id, absl::Span<double> jacobian);

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_MUJOCO_UTILS_H_
