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

#include "dm_robotics/mujoco/utils.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>

#include "dm_robotics/support/logging.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/types.h"
#include "Eigen/Core"

namespace dm_robotics {
namespace {

// Internally, it is more efficient to use geom IDs instead of the string-based
// GeomGroups/CollisionPairs. The integers in GeomIdGroup are always guaranteed
// to be positive.
using GeomIdGroup = absl::btree_set<int>;
using CollisionIdPair = std::pair<GeomIdGroup, GeomIdGroup>;

// Helper function for re-creating MuJoCo's max contact array. This array maps
// pairs of geom types to the maximum number of contacts that can be detected by
// MuJoCo's collision mechanism between both geoms.
//
// Note that only the upper-right triangle is initialized.
// The returned values are guaranteed to be positive integers or 0.
constexpr std::array<std::array<int, mjNGEOMTYPES>, mjNGEOMTYPES>
GetMujocoMaxContactsArray(bool is_multiccd) {
  std::array<std::array<int, mjNGEOMTYPES>, mjNGEOMTYPES> max_contacts{};

  // Plane
  max_contacts[mjGEOM_PLANE][mjGEOM_PLANE] = 0;
  max_contacts[mjGEOM_PLANE][mjGEOM_HFIELD] = 0;
  max_contacts[mjGEOM_PLANE][mjGEOM_SPHERE] = 1;
  max_contacts[mjGEOM_PLANE][mjGEOM_CAPSULE] = 2;
  max_contacts[mjGEOM_PLANE][mjGEOM_ELLIPSOID] = 1;
  max_contacts[mjGEOM_PLANE][mjGEOM_CYLINDER] = 4;
  max_contacts[mjGEOM_PLANE][mjGEOM_BOX] = 4;
  max_contacts[mjGEOM_PLANE][mjGEOM_MESH] = 3;

  // HField
  max_contacts[mjGEOM_HFIELD][mjGEOM_HFIELD] = 0;
  max_contacts[mjGEOM_HFIELD][mjGEOM_SPHERE] = mjMAXCONPAIR;
  max_contacts[mjGEOM_HFIELD][mjGEOM_CAPSULE] = mjMAXCONPAIR;
  max_contacts[mjGEOM_HFIELD][mjGEOM_ELLIPSOID] = mjMAXCONPAIR;
  max_contacts[mjGEOM_HFIELD][mjGEOM_CYLINDER] = mjMAXCONPAIR;
  max_contacts[mjGEOM_HFIELD][mjGEOM_BOX] = mjMAXCONPAIR;
  max_contacts[mjGEOM_HFIELD][mjGEOM_MESH] = mjMAXCONPAIR;

  // Sphere
  max_contacts[mjGEOM_SPHERE][mjGEOM_SPHERE] = 1;
  max_contacts[mjGEOM_SPHERE][mjGEOM_CAPSULE] = 1;
  max_contacts[mjGEOM_SPHERE][mjGEOM_ELLIPSOID] = 1;
  max_contacts[mjGEOM_SPHERE][mjGEOM_CYLINDER] = 1;
  max_contacts[mjGEOM_SPHERE][mjGEOM_BOX] = 1;
  max_contacts[mjGEOM_SPHERE][mjGEOM_MESH] = 1;

  // Capsule
  max_contacts[mjGEOM_CAPSULE][mjGEOM_CAPSULE] = 2;
  max_contacts[mjGEOM_CAPSULE][mjGEOM_ELLIPSOID] = 1;
  max_contacts[mjGEOM_CAPSULE][mjGEOM_CYLINDER] = is_multiccd ? 5 : 1;
  max_contacts[mjGEOM_CAPSULE][mjGEOM_BOX] = 2;
  max_contacts[mjGEOM_CAPSULE][mjGEOM_MESH] = is_multiccd ? 5 : 1;

  // Ellipsoid
  max_contacts[mjGEOM_ELLIPSOID][mjGEOM_ELLIPSOID] = 1;
  max_contacts[mjGEOM_ELLIPSOID][mjGEOM_CYLINDER] = 1;
  max_contacts[mjGEOM_ELLIPSOID][mjGEOM_BOX] = 1;
  max_contacts[mjGEOM_ELLIPSOID][mjGEOM_MESH] = 1;

  // Cylinder
  max_contacts[mjGEOM_CYLINDER][mjGEOM_CYLINDER] = is_multiccd ? 5 : 1;
  max_contacts[mjGEOM_CYLINDER][mjGEOM_BOX] = is_multiccd ? 5 : 1;
  max_contacts[mjGEOM_CYLINDER][mjGEOM_MESH] = is_multiccd ? 5 : 1;

  // Box
  max_contacts[mjGEOM_BOX][mjGEOM_BOX] = 8;
  max_contacts[mjGEOM_BOX][mjGEOM_MESH] = is_multiccd ? 5 : 1;

  // Mesh
  max_contacts[mjGEOM_MESH][mjGEOM_MESH] = is_multiccd ? 5 : 1;
  return max_contacts;
}
constexpr std::array<std::array<int, mjNGEOMTYPES>, mjNGEOMTYPES>
    kMujocoMaxContacts = GetMujocoMaxContactsArray(false);

constexpr std::array<std::array<int, mjNGEOMTYPES>, mjNGEOMTYPES>
    kMujocoMaxContactsMultiCcd = GetMujocoMaxContactsArray(true);

// Returns all geom names inside `model` as a single string with line endings
// after each name.
std::string GetAllGeomNames(const MjLib& lib, const mjModel& model) {
  std::string names;
  for (int i = 0; i < model.ngeom; i++) {
    absl::StrAppend(&names, lib.mj_id2name(&model, mjOBJ_GEOM, i), "\n");
  }
  return names;
}

// Converts a GeomGroup into a GeomIdGroup.
GeomIdGroup NamedGroupToIdGroup(const MjLib& lib, const mjModel& model,
                                const GeomGroup& named_group) {
  GeomIdGroup id_group;
  for (const auto& geom_name : named_group) {
    const int id = lib.mj_name2id(&model, mjOBJ_GEOM, geom_name.c_str());
    CHECK(id >= 0) << absl::Substitute(
        "NamedGroupToIdGroup: Geom with name [$0] does not exist in model. "
        "Please find the full list of geoms below:\n$1",
        geom_name, GetAllGeomNames(lib, model));
    id_group.insert(id);
  }
  return id_group;
}

// Converts a CollisionPair into a CollisionIdPair.
absl::btree_set<CollisionIdPair> NamedPairsToIdPairs(
    const MjLib& lib, const mjModel& model,
    const absl::btree_set<CollisionPair>& named_pairs) {
  absl::btree_set<CollisionIdPair> id_pairs;
  for (const auto& named_pair : named_pairs) {
    GeomIdGroup id_group_first =
        NamedGroupToIdGroup(lib, model, named_pair.first);
    GeomIdGroup id_group_second =
        NamedGroupToIdGroup(lib, model, named_pair.second);
    id_pairs.emplace(std::move(id_group_first), std::move(id_group_second));
  }
  return id_pairs;
}

// Returns true if geom1 and geom2 are part of the same body, or of bodies that
// are welded together.
//
// It is the caller's responsibility to ensure that the provided IDs are valid.
bool IsWeldedTogether(const mjModel& model, const int geom_id1,
                      const int geom_id2) {
  const int body1 = model.geom_bodyid[geom_id1];
  const int body2 = model.geom_bodyid[geom_id2];
  const int weld1 = model.body_weldid[body1];
  const int weld2 = model.body_weldid[body2];
  return weld1 == weld2;
}

// Returns true iff the provided geom_id is attached to the worldbody or to a
// body that is welded to the worldbody.
//
// It is the caller's responsibility to ensure that the provided ID is valid.
bool IsWeldedToWorldbody(const mjModel& model, const int geom_id) {
  const int body_id = model.geom_bodyid[geom_id];
  return model.body_weldid[body_id] == 0;
}

// Returns true iff the bodies have a parent-child relationship, treating all
// welded bodies as a single body. In other words, returns true if any body
// welded to the first geom's body is a child of any body welded to the second
// geom's body, or vice versa. Note that if both geoms are in the worldbody,
// this function returns true, as MuJoCo's convention is that the worldbody is
// its own parent.
//
// Example:
// (0)worldbody->(1)grandparent->(2)parent->(3)welded_to_parent->(4)child
// ->(5)welded_to_child
//
// AreGeomBodiesParentChild(parent, welded_to_child) -> true.
// parent (body_weldid 2, weld_parent_weldid 1);
// welded_to_child (body_weldid 4, weld_parent_weldid 2).
//
// AreGeomBodiesParentChild(parent, welded_to_parent) -> false.
// parent (body_weldid 2, weld_parent_weldid 1);
// welded_to_parent(body_weldid 2, weld_parent_weldid 1).
//
// AreGeomBodiesParentChild(grandparent, worldbody) -> true.
// grandparent(body_weldid 1, weld_parent_weldid 0);
// worldbody(body_weldid 0, weld_parent_weldid 0)
// will return true
//
// It is the caller's responsibility to ensure that the provided IDs are valid.
bool AreGeomBodiesParentChild(const mjModel& model, const int geom_id1,
                              const int geom_id2) {
  const int body_id1 = model.geom_bodyid[geom_id1];
  const int body_id2 = model.geom_bodyid[geom_id2];

  // body_weldid is the ID of the body's weld.
  const int body_weldid1 = model.body_weldid[body_id1];
  const int body_weldid2 = model.body_weldid[body_id2];

  // weld_parent_id is the ID of the parent of the body's weld.
  const int weld_parent_id1 = model.body_parentid[body_weldid1];
  const int weld_parent_id2 = model.body_parentid[body_weldid2];

  // weld_parent_weldid is the weld ID of the parent of the body's weld.
  const int weld_parent_weldid1 = model.body_weldid[weld_parent_id1];
  const int weld_parent_weldid2 = model.body_weldid[weld_parent_id2];

  return body_weldid1 == weld_parent_weldid2 ||
         body_weldid2 == weld_parent_weldid1;
}

// Returns true if the bounding spheres of the geoms are closer than
// collision_detection_distance, or if any of the geoms is a plane.
//
// If true, other procedures are necessary to determine if the two geoms are
// colliding. If false, we can assume that the objects are not colliding.
bool AreBoundingSpheresInCollision(const MjLib& lib, const mjModel& model,
                                   const mjData& data, const int geom1_id,
                                   const int geom2_id,
                                   const double collision_detection_distance) {
  // Note that in MuJoCo's implementation planes always have an rbound field of
  // 0.0, but this function always returns true in this case.
  if (model.geom_type[geom1_id] == mjtGeom::mjGEOM_PLANE ||
      model.geom_type[geom2_id] == mjtGeom::mjGEOM_PLANE) {
    return true;
  }

  // geom_rbound defines the radius of each bounding sphere centered at the
  // geom_xpos for each geom.
  const double geom1_rbound = model.geom_rbound[geom1_id];
  const double geom2_rbound = model.geom_rbound[geom2_id];

  // Compute the center-to-center distance.
  const double geom_dist = lib.mju_dist3(&data.geom_xpos[3 * geom1_id],
                                         &data.geom_xpos[3 * geom2_id]);

  // The distance between the spheres is computed by subtracting each of the
  // spheres radii from their center-to-center distance.
  const double sphere_dist = geom_dist - geom1_rbound - geom2_rbound;
  return sphere_dist < collision_detection_distance;
}

// Runs MuJoCo's collision detection function and returns the number of
// collisions between geoms with IDs `geom1_id` and `geom2_id` and fills the
// `contacts` array with one contact for each collision detected.
//
// The `contacts` array is always of a fixed size equal to mjMAXCONPAIR, to
// ensure that all possible contacts can be detected, but only the first
// elements of the array are filled. This is because if the contact buffer is
// full, MuJoCo logs a warning but does not signal this through a returned
// value.
//
// Check-fails if:
// - the geoms have invalid IDs;
// - both geoms have the same ID;
// - MuJoCo collision detection mechanism failed.
int ComputeContactsBetweenGeoms(const MjLib& lib, const mjModel& model,
                                const mjData& data, int geom1_id, int geom2_id,
                                double collision_detection_distance,
                                std::array<mjContact, mjMAXCONPAIR>* contacts) {
  // Ensure geom pair is valid.
  bool is_id_invalid = geom1_id < 0 || geom1_id >= model.ngeom ||
                       geom2_id >= model.ngeom || geom2_id < 0;
  CHECK(!is_id_invalid) << absl::Substitute(
      "ComputeContactsBetweenGeoms: Invalid geom ID. First geom ID [$0], "
      "second geom ID [$1]. Model number of geoms: [$2].",
      geom1_id, geom2_id, model.ngeom);
  CHECK(geom1_id != geom2_id) << absl::Substitute(
      "ComputeContactsBetweenGeoms: Both geom IDs cannot be the same. First "
      "geom ID [$0], second geom ID [$1].",
      geom1_id, geom2_id);

  // Swap order to ensure we only use upper-right triangular elements of
  // mjCOLLISIONFUNC. Note that model.geom_type is guaranteed to have valid
  // geom type elements, i.e. geom_type < mjtGeom::mjNGEOMTYPES.
  int geom1_type = model.geom_type[geom1_id];
  int geom2_type = model.geom_type[geom2_id];
  if (geom1_type > geom2_type) {
    std::swap(geom1_type, geom2_type);
    std::swap(geom1_id, geom2_id);
  }

  int num_collisions;
  if (!AreBoundingSpheresInCollision(lib, model, data, geom1_id, geom2_id,
                                     collision_detection_distance)) {
    num_collisions = 0;
  } else {
    num_collisions = lib.mjCOLLISIONFUNC[geom1_type][geom2_type](
        &model, &data, contacts->data(), geom1_id, geom2_id,
        collision_detection_distance);
  }

  // Check-fail if MuJoCo failed. This should never happen.
  CHECK(num_collisions >= 0) << absl::Substitute(
      "ComputeContactsBetweenGeoms: MuJoCo failed with return "
      "value [$0] when computing contacts between geom [$1] with ID "
      "[$2] and geom [$3] with ID [$4] with collision detection "
      "distance [$5].",
      num_collisions, lib.mj_id2name(&model, mjOBJ_GEOM, geom1_id), geom1_id,
      lib.mj_id2name(&model, mjOBJ_GEOM, geom2_id), geom2_id,
      collision_detection_distance);

  // Check-fail if there are more contacts than the maximum allowed for the geom
  // types.
  bool is_multiccd = model.opt.enableflags & mjENBL_MULTICCD;
  int max_contacts;
  if (is_multiccd) {
    max_contacts = kMujocoMaxContactsMultiCcd[geom1_type][geom2_type];
  } else {
    max_contacts = kMujocoMaxContacts[geom1_type][geom2_type];
  }
  CHECK(num_collisions <= max_contacts) << absl::Substitute(
      "ComputeContactsBetweenGeoms: Unexpected number of collisions [$0] "
      "between geom of type [$1] and geom of type [$2]. Please contact the "
      "developers for more information.",
      num_collisions, geom1_type, geom2_type);

  // Fill geom IDs.
  for (int i = 0; i < num_collisions; ++i) {
    contacts->at(i).geom1 = geom1_id;
    contacts->at(i).geom2 = geom2_id;
  }
  return num_collisions;
}

}  // namespace

int GetJointDofSize(const mjModel& model, int joint_id) {
  CHECK(0 <= joint_id && joint_id < model.njnt) << absl::Substitute(
      "GetJointDofSize: `joint_id` [$0] is invalid for the provided model, "
      "which has [$1] joints.",
      joint_id, model.njnt);

  const mjtJoint type = static_cast<mjtJoint>(model.jnt_type[joint_id]);
  switch (type) {
    case mjtJoint::mjJNT_SLIDE:
    case mjtJoint::mjJNT_HINGE:
      return 1;
    case mjtJoint::mjJNT_BALL:
      return 3;
    case mjtJoint::mjJNT_FREE:
      return 6;
  }

  LOG(FATAL) << absl::Substitute(
      "GetJointDofSize: `joint_id` [$0] corresponds to an invalid joint type "
      "[$1] in the provided model.",
      joint_id, model.jnt_type[joint_id]);
}

int GetJointQposSize(const mjModel& model, int joint_id) {
  CHECK(0 <= joint_id && joint_id < model.njnt) << absl::Substitute(
      "GetJointQposSize: `joint_id` [$0] is invalid for the provided model, "
      "which has [$1] joints.",
      joint_id, model.njnt);

  const mjtJoint type = static_cast<mjtJoint>(model.jnt_type[joint_id]);
  switch (type) {
    case mjtJoint::mjJNT_SLIDE:
    case mjtJoint::mjJNT_HINGE:
      return 1;
    case mjtJoint::mjJNT_BALL:
      return 4;
    case mjtJoint::mjJNT_FREE:
      return 7;
  }

  LOG(FATAL) << absl::Substitute(
      "GetJointQposSize: `joint_id` [$0] corresponds to an invalid joint type "
      "[$1] in the provided model.",
      joint_id, model.jnt_type[joint_id]);
}

absl::btree_set<int> JointIdsToDofIds(const mjModel& model,
                                      const absl::btree_set<int>& joint_ids) {
  absl::btree_set<int> dof_ids;
  for (int joint_id : joint_ids) {
    for (int i = 0; i < GetJointDofSize(model, joint_id); ++i) {
      // In MuJoCo models, the jnt_dofadr for a specific joint represents the
      // first DoF ID for that joint. If the joint has more than 1 DoF, the
      // remaining DoF IDs can be obtained as increments from the jnt_dofadr.
      dof_ids.insert(model.jnt_dofadr[joint_id] + i);
    }
  }
  return dof_ids;
}

int ComputeMaximumNumberOfContacts(
    const mjModel& model,
    const absl::btree_set<std::pair<int, int>>& geom_pairs) {
  bool is_multiccd = model.opt.enableflags & mjENBL_MULTICCD;
  int max_num_contacts = 0;
  for (const auto [geom_a, geom_b] : geom_pairs) {
    // Ensure only the upper-triangle of kMujocoMaxContacts is used.
    const int first_type =
        std::min(model.geom_type[geom_a], model.geom_type[geom_b]);
    const int second_type =
        std::max(model.geom_type[geom_a], model.geom_type[geom_b]);
    if (is_multiccd) {
      max_num_contacts += kMujocoMaxContactsMultiCcd[first_type][second_type];
    } else {
      max_num_contacts += kMujocoMaxContacts[first_type][second_type];
    }
  }
  return max_num_contacts;
}

absl::btree_set<std::pair<int, int>> CollisionPairsToGeomIdPairs(
    const MjLib& lib, const mjModel& model,
    const absl::btree_set<CollisionPair>& collision_pairs,
    bool allow_parent_child_collisions, bool allow_worldbody_collisions) {
  // Loop for every pair of geom groups.
  absl::btree_set<std::pair<int, int>> geom_id_pairs;
  for (const auto& id_pair : NamedPairsToIdPairs(lib, model, collision_pairs)) {
    // Look at every possible geom pair, and add to the list if they pass all
    // tests.
    for (const auto& geomid_a : id_pair.first) {
      for (const auto& geomid_b : id_pair.second) {
        // Condition 1: Geoms are not part of the same weld.
        const bool is_pass_weld_body_condition =
            !IsWeldedTogether(model, geomid_a, geomid_b);

        // Condition 2:
        // Parent-child filter collisions are allowed
        // OR geoms do not have a parent-child relationship.
        const bool is_pass_parent_child_condition =
            allow_parent_child_collisions ||
            !AreGeomBodiesParentChild(model, geomid_a, geomid_b);

        // Condition 3:
        // Worlbody collision is enabled
        // AND exactly one of the geoms is welded to the worldbody.
        const bool is_pass_worldbody_condition =
            allow_worldbody_collisions &&
            IsWeldedToWorldbody(model, geomid_a) !=
                IsWeldedToWorldbody(model, geomid_b);

        if (is_pass_weld_body_condition &&
            (is_pass_parent_child_condition || is_pass_worldbody_condition)) {
          geom_id_pairs.insert(std::make_pair(std::min(geomid_a, geomid_b),
                                              std::max(geomid_a, geomid_b)));
        }
      }
    }
  }
  return geom_id_pairs;
}

absl::StatusOr<int> ComputeContactsForGeomPairs(
    const MjLib& lib, const mjModel& model, const mjData& data,
    const absl::btree_set<std::pair<int, int>>& geom_pairs,
    double collision_detection_distance, absl::Span<mjContact> contacts) {
  // We make a buffer so that we can detect if we are out of memory and return
  // an appropriate error.
  std::array<mjContact, mjMAXCONPAIR> contact_buffer;

  int contacts_counter = 0;
  for (auto [geom1_id, geom2_id] : geom_pairs) {
    const int num_collisions = ComputeContactsBetweenGeoms(
        lib, model, data, geom1_id, geom2_id, collision_detection_distance,
        &contact_buffer);

    // Return if size is not enough.
    const int new_contacts_size = contacts_counter + num_collisions;
    if (new_contacts_size > contacts.size()) {
      return absl::OutOfRangeError(absl::Substitute(
          "ComputeContactsForCollisionPairs: Provided `contacts` parameter of "
          "size [$0] is too small to hold all the detected contacts. Failed "
          "when requesting a size of [$1], but note that the necessary total "
          "size may be much larger since not all geoms may have been checked "
          "yet.",
          contacts.size(), new_contacts_size));
    }

    // Copy contact information and continue.
    for (int i = contacts_counter, j = 0; i < new_contacts_size; ++i, ++j) {
      const mjContact& from_contact = contact_buffer[j];
      mjContact& to_contact = contacts[i];

      to_contact.dist = from_contact.dist;
      std::copy_n(&from_contact.pos[0], 3, &to_contact.pos[0]);
      std::copy_n(&from_contact.frame[0], 3, &to_contact.frame[0]);
      to_contact.geom1 = from_contact.geom1;
      to_contact.geom2 = from_contact.geom2;
    }
    contacts_counter += num_collisions;
  }
  return contacts_counter;
}

void ComputeContactNormalJacobian(const MjLib& lib, const mjModel& model,
                                  const mjData& data, const mjContact& contact,
                                  absl::Span<double> jacobian_buffer,
                                  absl::Span<double> jacobian) {
  CHECK(jacobian_buffer.size() == 3 * model.nv) << absl::Substitute(
      "ComputeContactNormalJacobian: Provided `jacobian_buffer` size [$0] is "
      "not the correct size. Expected 3*model.nv=[$1] elements.",
      jacobian_buffer.size(), 3 * model.nv);
  CHECK(jacobian.size() == model.nv) << absl::Substitute(
      "ComputeContactNormalJacobian: Provided `jacobian` size [$0] is not the "
      "correct size. Expected model.nv=[$1] elements.",
      jacobian.size(), model.nv);

  // The normal always points geom1 -> geom2.
  const int geom1_body = model.geom_bodyid[contact.geom1];
  const int geom2_body = model.geom_bodyid[contact.geom2];
  const double dist = contact.dist;
  Eigen::Map<const Eigen::Vector3d> pos_map(contact.pos);
  Eigen::Map<const Eigen::Vector3d> normal_map(contact.frame);
  Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic>> jacobian_map(
      jacobian.data(), 1, model.nv);
  Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>>
      jacobian_buffer_map(jacobian_buffer.data(), 3, model.nv);

  // Compute the contact position of each geom. This is done by using the normal
  // from the mid-point to move towards the geom with a magnitude dist/2.
  const Eigen::Vector3d geom1_contact_pos = pos_map - 0.5 * dist * normal_map;
  const Eigen::Vector3d geom2_contact_pos = pos_map + 0.5 * dist * normal_map;

  // Compute the Jacobian for the point in geom2, and project it into the
  // normal. Eigen's noalias is necessary to prevent dynamic memory allocation.
  lib.mj_jac(&model, &data, jacobian_buffer_map.data(), nullptr,
             geom2_contact_pos.data(), geom2_body);
  jacobian_map.noalias() = normal_map.transpose() * jacobian_buffer_map;

  // Compute the Jacobian for the point in geom1, project it into the normal,
  // and subtract from the Jacobian for the point in geom2. This is the
  // resulting normal contact Jacobian.
  lib.mj_jac(&model, &data, jacobian_buffer_map.data(), nullptr,
             geom1_contact_pos.data(), geom1_body);
  jacobian_map.noalias() -= normal_map.transpose() * jacobian_buffer_map;
}

absl::optional<mjContact> ComputeContactWithMinimumDistance(
    const MjLib& lib, const mjModel& model, const mjData& data, int geom1_id,
    int geom2_id, double collision_detection_distance) {
  std::array<mjContact, mjMAXCONPAIR> contact_buffer;
  int num_collisions = ComputeContactsBetweenGeoms(
      lib, model, data, geom1_id, geom2_id, collision_detection_distance,
      &contact_buffer);

  // If no collision are detected, do not return a value.
  if (num_collisions == 0) {
    return absl::nullopt;
  }

  // Return the contact with the minimum distance of all the contacts detected.
  double minimum_contact_distance = std::numeric_limits<double>::infinity();
  absl::optional<int> minimum_contact_idx;
  for (int i = 0; i < num_collisions; ++i) {
    if (minimum_contact_distance > contact_buffer[i].dist) {
      minimum_contact_distance = contact_buffer[i].dist;
      minimum_contact_idx = i;
    }
  }
  CHECK(minimum_contact_idx.has_value())
      << "ComputeContactWithMinimumDistance: Internal error. Please contact "
         "the developers for more information.";
  return contact_buffer[*minimum_contact_idx];
}

absl::optional<double> ComputeMinimumContactDistance(
    const MjLib& lib, const mjModel& model, const mjData& data, int geom1_id,
    int geom2_id, double collision_detection_distance) {
  absl::optional<mjContact> min_contact = ComputeContactWithMinimumDistance(
      lib, model, data, geom1_id, geom2_id, collision_detection_distance);
  if (!min_contact.has_value()) {
    return absl::nullopt;
  }

  return min_contact->dist;
}

void ComputeObject6dJacobian(const MjLib& lib, const mjModel& model,
                             const mjData& data, mjtObj object_type,
                             int object_id, absl::Span<double> jacobian) {
  switch (object_type) {
    case mjtObj::mjOBJ_BODY:
      lib.mj_jacBody(&model, &data, &jacobian[0], &jacobian[3 * model.nv],
                     object_id);
      break;
    case mjtObj::mjOBJ_GEOM:
      lib.mj_jacGeom(&model, &data, &jacobian[0], &jacobian[3 * model.nv],
                     object_id);
      break;
    case mjtObj::mjOBJ_SITE:
      lib.mj_jacSite(&model, &data, &jacobian[0], &jacobian[3 * model.nv],
                     object_id);
      break;
    default:
      LOG(FATAL) << absl::Substitute(
          "Compute6dVelocityJacobian: Invalid object_type [$0]. Only bodies, "
          "geoms, and sites are supported.",
          lib.mju_type2Str(object_type));
  }
}

}  // namespace dm_robotics
