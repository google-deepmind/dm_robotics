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

#ifndef DM_ROBOTICS_CONTROLLERS_LSQP_COLLISION_AVOIDANCE_CONSTRAINT_H_
#define DM_ROBOTICS_CONTROLLERS_LSQP_COLLISION_AVOIDANCE_CONSTRAINT_H_

#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include "dm_robotics/mujoco/mjlib.h"

namespace dm_robotics {

// One-sided inequality constraint on the normal velocity between two MuJoCo
// geoms such that:
//   V_n < R_b                              if dist_c < dist_m
//   V_n < G (dist_c - dist_m) + R_b        if dist_m <= dist_c <= dist_d
//   V_n < infinity                         otherwise
// where:
//   * V_n is the normal velocity between both geoms;
//   * R_b is the bound relaxation;
//   * G is the gain;
//   * dist_c is the current normal distance between both geoms;
//   * dist_m is the minimum allowed distance between both geoms;
//   * dist_d is the collision detection distance.
// The decision variables are assumed to be the joint velocities.
//
// The minimum distance between geoms can be negative, which would mean that the
// objects are allowed to penetrate by |min_distance|. The bound relaxation is
// a constant that can be used to relax or tighten the resulting constraint,
// noting that a negative value may result in a negative upper bound.
//
// The gain parameter G is usually set to:
//  G = K/T
// where K is a positive real number (0,1] that determines how fast geoms are
// allowed to move towards each other at each iteration(the smaller the value
// the more conservative the constraint will be); and T is a positive real
// number that determines how long the velocity will be executed for, (a.k.a.
// integration timestep).
//
// If the normal distance between any two geoms is already less than the
// `minimum_normal_distance`, the normal velocity will be limited to be less or
// equal to the `bound_relaxation`. This is to prevent spring-like velocities in
// the case that the current configuration is in collision.
//
// The number of DoF and bounds length for this constraint is always constant.
// The bounds length is always the maximum number of constraint rows for the
// provided `geom_pairs`, noting that if no contacts are detected the
// corresponding coefficients will be zero and the upper bound +infinity.
//
// References:
// F. Kanehiro, F. Lamiraux, O. Kanoun, E. Yoshida, J.P. Laumond, "A Local
// Collision Avoidance Method for Non-strictly Convex Polyhedra", Robotics:
// Science and Systems (2008).
//
// C. Fang, A. Rocchi, E. M. Hoffman, N. G. Tsagarakis and D. G. Caldwell,
// "Efficient self-collision avoidance based on focus of interest for humanoid
// robots," 2015 IEEE-RAS 15th International Conference on Humanoid Robots
// (Humanoids), Seoul, 2015, pp. 1060-1066, doi: 10.1109/HUMANOIDS.2015.7363500.
class CollisionAvoidanceConstraint : public LsqpConstraint {
 public:
  // Initialization parameters for CollisionAvoidanceConstraint.
  // Only 1 DoF joints are supported at the moment.
  //
  // The `geom_pairs` field defines the geoms that should avoid each other. Each
  // element in the set must be a pair of two different MuJoCo geom IDs. Note
  // that `exclude` and `pair` elements in the MuJoCo model will be ignored, and
  // MuJoCo's broad-phase collision-detection filters will ignored, i.e.
  // conaffinity/contype, parent-child, and/or same-body filtering. It is the
  // user's responsibility to ensure that geom pairs are not provided for
  // unnecessary contacts, e.g. between two geoms corresponding to the same
  // body, or providing the same pair twice in different order.
  //
  // The caller retains ownership of lib and model.
  // It is the caller's responsibility to ensure that the *lib and *model
  // objects outlive any CollisionAvoidanceConstraint instances created with
  // this object.
  struct Parameters {
    const MjLib* lib;
    const mjModel* model;
    double collision_detection_distance;
    double minimum_normal_distance;
    double gain;
    double bound_relaxation = 0.0;
    absl::btree_set<int> joint_ids;
    absl::btree_set<std::pair<int, int>> geom_pairs;
  };

  // Constructs a collision avoidance constraint.
  CollisionAvoidanceConstraint(const Parameters& params, const mjData& data);

  CollisionAvoidanceConstraint(const CollisionAvoidanceConstraint&) = delete;
  CollisionAvoidanceConstraint& operator=(const CollisionAvoidanceConstraint&) =
      delete;

  // Updates the coefficients and bounds for the current Qpos configuration in
  // `data`. This function does not perform dynamic memory allocation.
  void UpdateCoefficientsAndBounds(const mjData& data);

  // Returns a message describing all detected geom pair contacts during the
  // last call to `UpdateCoefficentsAndBounds` that have normal distances below
  // the `minimum_normal_distance` threshold.
  //
  // If no contacts were found with normal distances below
  // `minimum_normal_distance`, the returned string will be empty.
  std::string GetContactDebugString(const mjData& data) const;

  // LsqpConstraint virtual members.
  absl::Span<const double> GetCoefficientMatrix() const override;
  absl::Span<const double> GetLowerBound() const override;
  absl::Span<const double> GetUpperBound() const override;
  int GetNumberOfDof() const override;
  int GetBoundsLength() const override;

 private:
  const MjLib& lib_;
  const mjModel& model_;
  double collision_detection_distance_;
  double minimum_normal_distance_;
  double gain_;
  double bound_relaxation_;
  absl::btree_set<int> joint_dof_ids_;
  absl::btree_set<std::pair<int, int>> geom_pairs_;
  std::vector<mjContact> contacts_buffer_;
  std::vector<double> linear_jacobian_buffer_;
  std::vector<double> normal_jacobian_buffer_;

  absl::Span<const mjContact> detected_contacts_;  // # detected contacts
  std::vector<double> lower_bound_;                // # detected contacts
  std::vector<double> upper_bound_;                // # detected contacts
  std::vector<double> coefficient_matrix_;  // # detected contacts x num DoF
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_CONTROLLERS_LSQP_COLLISION_AVOIDANCE_CONSTRAINT_H_
