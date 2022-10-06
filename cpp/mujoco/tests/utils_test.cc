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
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "dm_robotics/support/logging.h"
#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/mock_distributions.h"
#include "absl/random/mocking_bit_gen.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dm_robotics/mujoco/defs.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/test_with_mujoco_model.h"
#include "dm_robotics/mujoco/types.h"

namespace dm_robotics {
namespace {

using UtilsTest = ::dm_robotics::testing::TestWithMujocoModel;
using ::testing::_;
using ::testing::DoubleNear;
using ::testing::Pointwise;

constexpr char kFloorGeomName[] = "floor";
constexpr char kRightHandGeomName[] = "right_hand";
constexpr char kLeftHandGeomName[] = "left_hand";
constexpr char kRightFootGeomName[] = "right_right_foot";
constexpr char kLeftFootGeomName[] = "left_left_foot";
constexpr char kUWaistGeomName[] = "upper_waist";
constexpr char kLWaistGeomName[] = "lower_waist";

constexpr int kDefaultNconmax = 100;

// Returns a pair with the parameters as elements sorted in ascending order.
std::pair<int, int> MakeSortedPair(int a, int b) {
  return std::make_pair(std::min(a, b), std::max(a, b));
}

// Compares two contacts to ensure that they are kinematically equivalent,
// within a certain tolerance.
bool AreContactsKinematicallyEqual(const mjContact& contact1,
                                   const mjContact& contact2,
                                   double tolerance) {
  if (!(contact1.geom1 == contact2.geom1 && contact1.geom2 == contact2.geom2)) {
    return false;
  }

  // Check contact distances.
  if (std::abs(contact1.dist - contact2.dist) > tolerance) {
    return false;
  }

  // Check contact mid-point positions.
  for (int i = 0; i < 3; ++i) {
    if (std::abs(contact1.pos[i] - contact2.pos[i]) > tolerance) {
      return false;
    }
  }

  // Check normals.
  for (int i = 0; i < 3; ++i) {
    if (std::abs(contact1.frame[i] - contact2.frame[i]) > tolerance) {
      return false;
    }
  }

  return true;
}

// Returns the number of Qpos values for a specific joint type.
int JointTypeToQposSize(mjtJoint type) {
  switch (type) {
    case mjtJoint::mjJNT_SLIDE:
      [[fallthrough]];
    case mjtJoint::mjJNT_HINGE:
      return 1;
    case mjtJoint::mjJNT_BALL:
      return 3;
    case mjtJoint::mjJNT_FREE:
      return 7;
  }
  LOG(FATAL) << "JointTypeToQposSize: Invalid joint type: " << type;
}

// Samples Qpos values and runs forward kinematics on data.
//
// If the joint is 1 DoF and has limits, it samples within the joint limits;
// otherwise it samples a value between (-1.0, 1.0) for every qpos element in
// the joint. For ball joints, this results in a random quaternion after
// normalization.
// Note that the linear component of free joints never have limits in MuJoCo,
// and thus we set it to (-1.0, 1.0) for simplicity and to still allow some
// freedom during sampling.
void SetRandomQpos(const MjLib& lib, const mjModel& model, absl::BitGenRef gen,
                   mjData* data) {
  for (int joint_id = 0; joint_id < model.njnt; ++joint_id) {
    const mjtJoint type = static_cast<mjtJoint>(model.jnt_type[joint_id]);
    int qpos_adr = model.jnt_qposadr[joint_id];
    if (model.jnt_limited[joint_id] &&
        (type == mjtJoint::mjJNT_SLIDE || type == mjtJoint::mjJNT_HINGE)) {
      const double hi_lim = model.jnt_range[2 * joint_id + 1];
      const double low_lim = model.jnt_range[2 * joint_id];
      data->qpos[qpos_adr] =
          absl::Uniform<double>(absl::IntervalClosed, gen, low_lim, hi_lim);
    } else {
      for (int qpos_offset = 0; qpos_offset < JointTypeToQposSize(type);
           ++qpos_offset) {
        data->qpos[qpos_adr + qpos_offset] =
            absl::Uniform<double>(absl::IntervalClosed, gen, -1.0, 1.0);
      }
    }
  }
  lib.mj_normalizeQuat(&model, data->qpos);
  lib.mj_fwdPosition(&model, data);
}

// Deterministic call that ensures qpos is in collision.
void SetQposInCollision(const MjLib& lib, const mjModel& model, mjData* data) {
  absl::MockingBitGen gen;
  ON_CALL(absl::MockUniform<double>(), Call(absl::IntervalClosed, gen, _, _))
      .WillByDefault([](absl::IntervalClosedClosedTag, double low,
                        double high) { return 0.1 * high + 0.9 * low; });
  SetRandomQpos(lib, model, gen, data);
  ASSERT_NE(data->ncon, 0);
}

// Returns the contact with the smallest distance between the two geoms, or null
// if no contact between the two geoms can be found.
const mjContact* GetMinimumDistanceContact(
    int geomid_a, int geomid_b, absl::Span<const mjContact> contacts) {
  const mjContact* ptr = nullptr;
  double min_distance = std::numeric_limits<double>::infinity();
  for (const auto& contact : contacts) {
    bool has_same_geoms =
        (contact.geom1 == geomid_a && contact.geom2 == geomid_b) ||
        (contact.geom2 == geomid_a && contact.geom1 == geomid_b);
    if (has_same_geoms && min_distance > contact.dist) {
      ptr = &contact;
      min_distance = contact.dist;
    }
  }
  return ptr;
}

TEST_F(UtilsTest, GetJointDofSizeReturnsCorrectNumberOfDof) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
    const mjtJoint type = static_cast<mjtJoint>(model_->jnt_type[joint_id]);
    if (type == mjtJoint::mjJNT_SLIDE || type == mjtJoint::mjJNT_HINGE) {
      EXPECT_EQ(GetJointDofSize(*model_, joint_id), 1);
    } else if (type == mjtJoint::mjJNT_BALL) {
      EXPECT_EQ(GetJointDofSize(*model_, joint_id), 3);
    } else {
      EXPECT_EQ(GetJointDofSize(*model_, joint_id), 6);
    }
  }
}

TEST_F(UtilsTest, GetJointQposSizeReturnsCorrectNumberOfDof) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
    const mjtJoint type = static_cast<mjtJoint>(model_->jnt_type[joint_id]);
    if (type == mjtJoint::mjJNT_SLIDE || type == mjtJoint::mjJNT_HINGE) {
      EXPECT_EQ(GetJointQposSize(*model_, joint_id), 1);
    } else if (type == mjtJoint::mjJNT_BALL) {
      EXPECT_EQ(GetJointQposSize(*model_, joint_id), 4);
    } else {
      EXPECT_EQ(GetJointQposSize(*model_, joint_id), 7);
    }
  }
}

TEST_F(UtilsTest, JointIdsToDofIdsReturnsCorrectDofIds) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  absl::btree_set<int> joint_ids{0, 3, 6, 7, 9};
  absl::btree_set<int> expected_dof_ids{0, 1, 2, 3, 4, 5, 8, 11, 12, 14};

  EXPECT_EQ(JointIdsToDofIds(*model_, joint_ids), expected_dof_ids);
}

TEST_F(UtilsTest, ComputeMaximumNumberOfContactsReturnsCorrectNumber) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  const int floor_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kFloorGeomName);
  const int right_hand_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kRightHandGeomName);
  const int left_hand_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kLeftHandGeomName);
  const int right_foot_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kRightFootGeomName);
  const int left_foot_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kLeftFootGeomName);
  const int lwaist_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kLWaistGeomName);
  const int uwaist_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kUWaistGeomName);
  auto right_hand_left_hand = std::make_pair(right_hand_id, left_hand_id);
  auto right_hand_left_foot = std::make_pair(right_hand_id, left_foot_id);
  auto right_foot_left_hand = std::make_pair(right_foot_id, left_hand_id);
  auto right_foot_left_foot = std::make_pair(right_foot_id, left_foot_id);
  auto right_hand_right_foot = std::make_pair(right_hand_id, right_foot_id);
  auto lwaist_uwaist = std::make_pair(lwaist_id, uwaist_id);
  auto floor_right_hand = std::make_pair(floor_id, right_hand_id);
  auto floor_lwaist = std::make_pair(floor_id, lwaist_id);

  // Sphere-sphere = 1.
  EXPECT_EQ(ComputeMaximumNumberOfContacts(*model_, {right_hand_left_hand}), 1);

  // Sphere-capsule = 1.
  EXPECT_EQ(ComputeMaximumNumberOfContacts(*model_, {right_hand_left_foot}), 1);

  // Capsule-capsule = 2.
  EXPECT_EQ(ComputeMaximumNumberOfContacts(*model_, {lwaist_uwaist}), 2);

  // Plane-sphere = 1.
  EXPECT_EQ(ComputeMaximumNumberOfContacts(*model_, {floor_right_hand}), 1);

  // Plane-capsule = 2.
  EXPECT_EQ(ComputeMaximumNumberOfContacts(*model_, {floor_lwaist}), 2);

  // All.
  EXPECT_EQ(ComputeMaximumNumberOfContacts(
                *model_, {right_hand_left_hand, right_hand_left_foot,
                          right_foot_left_hand, right_foot_left_foot,
                          right_hand_right_foot, lwaist_uwaist,
                          floor_right_hand, floor_lwaist}),
            11);
}

TEST_F(UtilsTest, CollisionPairsToGeomIdPairsReturnsCorrectIds) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  GeomGroup right_geoms{kRightHandGeomName, kRightFootGeomName};
  GeomGroup left_geoms{kLeftHandGeomName, kLeftFootGeomName};
  GeomGroup waist_geoms{kLWaistGeomName, kUWaistGeomName};
  absl::btree_set<CollisionPair> pairs{
      std::make_pair(right_geoms, left_geoms),
      std::make_pair(right_geoms, right_geoms),
      std::make_pair(waist_geoms, waist_geoms)};

  absl::btree_set<std::pair<int, int>> geom_id_pairs_parent_filter =
      CollisionPairsToGeomIdPairs(*mjlib_, *model_, pairs, false, true);
  absl::btree_set<std::pair<int, int>> geom_id_pairs_no_parent_filter =
      CollisionPairsToGeomIdPairs(*mjlib_, *model_, pairs, true, true);

  const int right_hand_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kRightHandGeomName);
  const int left_hand_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kLeftHandGeomName);
  const int right_foot_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kRightFootGeomName);
  const int left_foot_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kLeftFootGeomName);
  const int lwaist_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kLWaistGeomName);
  const int uwaist_id =
      mjlib_->mj_name2id(model_.get(), mjOBJ_GEOM, kUWaistGeomName);

  auto right_hand_left_hand = MakeSortedPair(right_hand_id, left_hand_id);
  auto right_hand_left_foot = MakeSortedPair(right_hand_id, left_foot_id);
  auto right_foot_left_hand = MakeSortedPair(right_foot_id, left_hand_id);
  auto right_foot_left_foot = MakeSortedPair(right_foot_id, left_foot_id);
  auto right_hand_right_foot = MakeSortedPair(right_hand_id, right_foot_id);
  auto lwaist_uwaist = MakeSortedPair(lwaist_id, uwaist_id);

  // Returned set with parent filter should not contain the waist geoms.
  absl::btree_set<std::pair<int, int>> geom_id_pairs_parent_filter_expected{
      right_hand_left_hand, right_hand_left_foot, right_foot_left_hand,
      right_foot_left_foot, right_hand_right_foot};
  EXPECT_EQ(geom_id_pairs_parent_filter, geom_id_pairs_parent_filter_expected);

  // Returned set without parent filter should contain the waist geoms.
  absl::btree_set<std::pair<int, int>> geom_id_pairs_no_parent_filter_expected{
      right_hand_left_hand, right_hand_left_foot,  right_foot_left_hand,
      right_foot_left_foot, right_hand_right_foot, lwaist_uwaist};
  EXPECT_EQ(geom_id_pairs_no_parent_filter,
            geom_id_pairs_no_parent_filter_expected);
}

TEST_F(UtilsTest, ComputeContactsForGeomPairsDetectsSameContactsAsMujoco) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);
  SetQposInCollision(*mjlib_, *model_, data_.get());

  // Create a CollisionPair with all geoms.
  GeomGroup all_geoms;
  for (int i = 0; i < model_->ngeom; ++i) {
    all_geoms.insert(mjlib_->mj_id2name(model_.get(), mjOBJ_GEOM, i));
  }
  absl::btree_set<CollisionPair> collision_pairs{
      std::make_pair(all_geoms, all_geoms)};

  absl::btree_set<std::pair<int, int>> geom_pairs = CollisionPairsToGeomIdPairs(
      *mjlib_, *model_, collision_pairs, false, true);

  std::vector<mjContact> contacts(
      model_->nconmax > 0 ? model_->nconmax : kDefaultNconmax);
  ASSERT_OK_AND_ASSIGN(int ncon, ComputeContactsForGeomPairs(
                                     *mjlib_, *model_, *data_, geom_pairs, 0.0,
                                     absl::MakeSpan(contacts)));
  EXPECT_EQ(ncon, data_->ncon);

  for (int state_con_idx = 0; state_con_idx < data_->ncon; ++state_con_idx) {
    bool same_contact_exists = false;
    for (int contact_idx = 0; contact_idx < ncon; ++contact_idx) {
      if (AreContactsKinematicallyEqual(data_->contact[state_con_idx],
                                        contacts[contact_idx], 1.0e-6)) {
        same_contact_exists = true;

        // Remove contact and quit inner loop. This is to ensure that one
        // contact in data_ does not match against two contacts in `contacts`.
        contacts.erase(contacts.begin() + contact_idx);
        --ncon;
        contact_idx = ncon;
      }
    }
    EXPECT_TRUE(same_contact_exists);
  }
}

TEST_F(UtilsTest, ComputeContactNormalJacobianIsSameAsMujoco) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);

  // Options necessary to obtain separation normal + dense matrices
  // from collision computations.
  model_->opt.cone = mjtCone::mjCONE_ELLIPTIC;
  model_->opt.jacobian = mjtJacobian::mjJAC_DENSE;

  // Remove MuJoCo unnecessary explicit constraints.
  model_->opt.disableflags |= mjtDisableBit::mjDSBL_EQUALITY;
  model_->opt.disableflags |= mjtDisableBit::mjDSBL_FRICTIONLOSS;
  model_->opt.disableflags |= mjtDisableBit::mjDSBL_LIMIT;

  // Set contact dimensionality to 1 (normals only).
  for (int i = 0; i < model_->ngeom; i++) {
    model_->geom_condim[i] = 1;
  }

  // Create custom data for our purposes and sample Qpos in collision.
  std::unique_ptr<mjData, void (*)(mjData*)> custom_data_(
      mjlib_->mj_makeData(model_.get()), mjlib_->mj_deleteData);
  SetQposInCollision(*mjlib_, *model_, custom_data_.get());

  // Ensure the computed Jacobian and MuJoCo's contact Jacobian are the same.
  std::vector<double> jacobian_buffer(3 * model_->nv);
  std::vector<double> jacobian(model_->nv);
  for (int state_con_idx = 0; state_con_idx < custom_data_->ncon;
       ++state_con_idx) {
    // Get MuJoCo's contact normal Jacobian.
    const int efc_adr = custom_data_->contact[state_con_idx].efc_address;
    absl::Span<const double> efc_j(&custom_data_->efc_J[efc_adr * model_->nv],
                                   model_->nv);

    // Compute the contact normal Jacobian manually.
    ComputeContactNormalJacobian(
        *mjlib_, *model_, *custom_data_, custom_data_->contact[state_con_idx],
        absl::MakeSpan(jacobian_buffer), absl::MakeSpan(jacobian));

    // Ensure that both quantities are the same.
    EXPECT_THAT(jacobian, Pointwise(DoubleNear(1.0e-15), efc_j));
  }
}

TEST_F(UtilsTest, ComputeMinimumContactDistanceReturnValueIsCorrect) {
  LoadModelFromXmlPath(kDmControlSuiteHumanoidXmlPath);
  SetQposInCollision(*mjlib_, *model_, data_.get());

  // Create a CollisionPair with all geoms.
  GeomGroup all_geoms;
  for (int i = 0; i < model_->ngeom; ++i) {
    all_geoms.insert(mjlib_->mj_id2name(model_.get(), mjOBJ_GEOM, i));
  }
  absl::btree_set<CollisionPair> collision_pairs{
      std::make_pair(all_geoms, all_geoms)};

  absl::btree_set<std::pair<int, int>> geom_pairs = CollisionPairsToGeomIdPairs(
      *mjlib_, *model_, collision_pairs, false, true);

  std::vector<mjContact> contacts(
      model_->nconmax > 0 ? model_->nconmax : kDefaultNconmax);
  ASSERT_OK_AND_ASSIGN(int ncon, ComputeContactsForGeomPairs(
                                     *mjlib_, *model_, *data_, geom_pairs, 0.0,
                                     absl::MakeSpan(contacts)));

  // Ensure that the minimum distance computed and the distance of the most
  // negative contact match for a given pair of geom IDs.
  for (const auto& pair : geom_pairs) {
    absl::optional<double> maybe_dist = ComputeMinimumContactDistance(
        *mjlib_, *model_, *data_, pair.first, pair.second, 0.0);
    const mjContact* min_dist_contact = GetMinimumDistanceContact(
        pair.first, pair.second, absl::MakeSpan(contacts.data(), ncon));

    if (min_dist_contact == nullptr) {
      EXPECT_FALSE(maybe_dist.has_value());
    } else {
      EXPECT_EQ(maybe_dist.value(), min_dist_contact->dist);
    }
  }
}

}  // namespace
}  // namespace dm_robotics
