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

#include "dm_robotics/controllers/lsqp/collision_avoidance_constraint.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "dm_robotics/support/logging.h"
#include "absl/container/btree_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/utils.h"
#include "Eigen/Core"

namespace dm_robotics {
namespace {

// Computes the contacts for the geom pairs using `buffer`, and returns a view
// over the contacts for the provided set of geom pairs.
absl::Span<const mjContact> ComputeContacts(
    const MjLib& lib, const mjModel& model, const mjData& data,
    absl::btree_set<std::pair<int, int>>& geom_pairs,
    const double collision_detection_distance, absl::Span<mjContact> buffer) {
  absl::StatusOr<int> num_contacts_or = ComputeContactsForGeomPairs(
      lib, model, data, geom_pairs, collision_detection_distance, buffer);
  CHECK_EQ(num_contacts_or.status(), absl::OkStatus()) << absl::Substitute(
      "ComputeContacts: Internal error [$0]. Please contact the developers.",
      num_contacts_or.status().ToString());
  return absl::MakeSpan(buffer.data(), *num_contacts_or);
}

}  // namespace

CollisionAvoidanceConstraint::CollisionAvoidanceConstraint(
    const Parameters& params, const mjData& data)
    : lib_(*DieIfNull(params.lib)),
      model_(*DieIfNull(params.model)),
      collision_detection_distance_(params.collision_detection_distance),
      minimum_normal_distance_(params.minimum_normal_distance),
      gain_(params.gain),
      bound_relaxation_(params.bound_relaxation),
      joint_dof_ids_(JointIdsToDofIds(model_, params.joint_ids)),
      geom_pairs_(params.geom_pairs),
      linear_jacobian_buffer_(3 * model_.nv),
      normal_jacobian_buffer_(model_.nv) {
  CHECK(params.joint_ids.size() == joint_dof_ids_.size()) << absl::Substitute(
      "CollisionAvoidanceConstraint: One or more joints have more than 1 DoF. "
      "Only 1 DoF joints are supported at the moment. Number of joint IDs "
      "provided was [$0], but total number of DoF resulted in [$1].",
      params.joint_ids.size(), joint_dof_ids_.size());

  // Compute maximum number of contacts if all the geom pairs are in collision
  // at the same time.
  CHECK(!params.geom_pairs.empty())
      << "CollisionAvoidanceConstraint: `geom_pairs` cannot be empty.";
  int max_num_contacts =
      ComputeMaximumNumberOfContacts(model_, params.geom_pairs);

  // Initialize buffers.
  contacts_buffer_.resize(max_num_contacts);
  lower_bound_ = std::vector<double>(max_num_contacts,
                                     -std::numeric_limits<double>::infinity());
  upper_bound_ = std::vector<double>(max_num_contacts);
  coefficient_matrix_ =
      std::vector<double>(max_num_contacts * joint_dof_ids_.size());
  UpdateCoefficientsAndBounds(data);
}

void CollisionAvoidanceConstraint::UpdateCoefficientsAndBounds(
    const mjData& data) {
  detected_contacts_ = ComputeContacts(lib_, model_, data, geom_pairs_,
                                       collision_detection_distance_,
                                       absl::MakeSpan(contacts_buffer_));

  // Reset upper bound and coefficients.
  const int max_num_contacts = contacts_buffer_.size();
  Eigen::Map<Eigen::VectorXd> upper_bound_map(upper_bound_.data(),
                                              max_num_contacts);
  Eigen::Map<Eigen::MatrixXd> coefficient_matrix_map(
      coefficient_matrix_.data(), max_num_contacts, joint_dof_ids_.size());
  upper_bound_map.setConstant(std::numeric_limits<double>::infinity());
  coefficient_matrix_map.setZero();

  // For each contact, compute the upper bound and coefficient matrix row.
  for (int contact_idx = 0; contact_idx < detected_contacts_.size();
       ++contact_idx) {
    const mjContact& contact = detected_contacts_[contact_idx];

    // Compute upper bound value.
    const double hi_bound_dist = contact.dist;
    if (hi_bound_dist > minimum_normal_distance_) {
      upper_bound_map[contact_idx] =
          gain_ * (hi_bound_dist - minimum_normal_distance_) +
          bound_relaxation_;
    } else {
      upper_bound_map[contact_idx] = bound_relaxation_;
    }

    // Compute Jacobian, and set constraint row. Note that Eigen::Map does not
    // support indexing with an array.
    ComputeContactNormalJacobian(lib_, model_, data, contact,
                                 absl::MakeSpan(linear_jacobian_buffer_),
                                 absl::MakeSpan(normal_jacobian_buffer_));
    Eigen::Map<Eigen::VectorXd> normal_jacobian_map(
        normal_jacobian_buffer_.data(), normal_jacobian_buffer_.size());
    int joint_dof_idx = 0;
    for (int joint_dof_id : joint_dof_ids_) {
      coefficient_matrix_map(contact_idx, joint_dof_idx) =
          -normal_jacobian_map[joint_dof_id];
      ++joint_dof_idx;
    }
  }
}

std::string CollisionAvoidanceConstraint::GetContactDebugString(
    const mjData& data) const {
  // Iterate through all geom pair contacts and add a line for each pair that
  // has a normal distance below the `minimum_normal_distance` threshold.
  std::string debug_string;
  for (const auto& contact : detected_contacts_) {
    const double dist = contact.dist;
    if (dist <= minimum_normal_distance_) {
      const int geom1_id = contact.geom1;
      const int geom2_id = contact.geom2;
      const char* geom1_name = lib_.mj_id2name(&model_, mjOBJ_GEOM, geom1_id);
      const char* geom2_name = lib_.mj_id2name(&model_, mjOBJ_GEOM, geom2_id);
      absl::StrAppend(&debug_string, "Geoms [", geom1_name, "] and [",
                      geom2_name, "] detected at a distance of [", dist,
                      "] of each other. This is lower than the defined "
                      "`minimum_normal_distance` of [",
                      minimum_normal_distance_, "].\n");
    }
  }
  return debug_string;
}

absl::Span<const double> CollisionAvoidanceConstraint::GetCoefficientMatrix()
    const {
  return coefficient_matrix_;
}

absl::Span<const double> CollisionAvoidanceConstraint::GetLowerBound() const {
  return lower_bound_;
}

absl::Span<const double> CollisionAvoidanceConstraint::GetUpperBound() const {
  return upper_bound_;
}

int CollisionAvoidanceConstraint::GetNumberOfDof() const {
  return joint_dof_ids_.size();
}

int CollisionAvoidanceConstraint::GetBoundsLength() const {
  return contacts_buffer_.size();
}

}  // namespace dm_robotics
