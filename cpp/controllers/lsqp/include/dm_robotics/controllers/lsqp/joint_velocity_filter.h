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

#ifndef DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_VELOCITY_FILTER_H_
#define DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_VELOCITY_FILTER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "dm_robotics/controllers/lsqp/collision_avoidance_constraint.h"
#include "dm_robotics/controllers/lsqp/joint_position_limit_constraint.h"
#include "dm_robotics/least_squares_qp/common/box_constraint.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint_union.h"
#include "dm_robotics/least_squares_qp/common/identity_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h"
#include "dm_robotics/mujoco/types.h"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {

// Filters desired joint velocities through the use of an LSQP Stack-of-tasks
// problem solved at every call to `FilterJointVelocities`. The target joint
// velocities are specified as a single vector.
//
// In its most basic configuration, it computes the joint velocities that are as
// close as possible to the desired joint velocities while also supporting the
// following functionality:
// * Collision avoidance can be enabled for a set of CollisionPair objects,
//   which defines which geoms should avoid each other.
// * Limits on the joint positions and velocities can be defined to ensure that
//   the computed joint velocities do not result in limit violations.
// An error is returned if the filter failed to compute the velocities.
//
// Refer to the documentation of `JointVelocityFilter::Parameters` for more
// information on the available configuration options.
//
// This class requires an updated mjData object at every call to
// `FilterJointVelocities`. It is the user's responsibility to ensure that the
// mjData object has a consistent and accurate `qpos` field.
class JointVelocityFilter {
 public:
  // Initialization parameters for JointVelocityFilter.
  //
  // The caller retains ownership of `model`.
  // It is the caller's responsibility to ensure the *model object outlives any
  // `JointVelocityFilter` instances created with this object.
  struct Parameters {
    // Pointer to a MuJoCo model.
    const mjModel* model;

    // MuJoCo joint IDs of the joints to be controlled. Only 1 DoF joints are
    // allowed at the moment.
    absl::btree_set<int> joint_ids;

    // Amount of time that the joint velocities will be executed for, A.K.A.
    // 'dt'. If unsure, higher values are more conservative. This timestep will
    // be used when integrating the joint velocities to ensure that safety
    // constraints are not violated.
    absl::Duration integration_timestep;

    // Whether to enable joint limit avoidance. Joint limits are deduced from
    // the MuJoCo model.
    bool enable_joint_position_limits = true;

    // Value (0,1] that defines how fast each joint is allowed to move towards
    // the joint limits in each iteration. Values lower than 1 are safer but may
    // make the joints move slowly. 0.95 is usually enough since it is not
    // affected by Jacobian linearization. Ignored if
    // `enable_joint_position_limits` is `false`.
    double joint_position_limit_velocity_scale = 0.95;

    // Offset in meters (slide joints) or radians (hinge joints) to be added to
    // the limits. Positive values decrease the range of motion, negative values
    // increase it (i.e. negative values allow penetration). Ignored if
    // `enable_joint_position_limits` is `false`.
    double minimum_distance_from_joint_position_limit = 0.01;  // 1cm; ~0.5deg.

    // Whether to enable joint velocity limits.
    bool enable_joint_velocity_limits = false;

    // Array of maximum allowed magnitudes of joint velocities for each joint,
    // in m/s (slide joints) or rad/s (hinge joints). Must be ordered according
    // to the `joint_ids` field. Ignored if `enable_joint_velocity_limits` is
    // `false`.
    std::vector<double> joint_velocity_magnitude_limits;

    // Whether to enable active collision avoidance.
    bool enable_collision_avoidance = false;

    // If `use_minimum_distance_contacts_only` is true, it will only create one
    // inequality constraint per geom pair, corresponding to the MuJoCo contact
    // with the minimum distance. Otherwise, it will create one inequality
    // constraint for each of the MuJoCo contacts detected per geom pair.
    // Ignored if `enable_collision_avoidance` is `false`.
    //
    // In problems where many geoms are avoiding each other, setting this option
    // to `true` will considerably speed up solve times, but the solution is
    // more likely to result in penetration at high speeds.
    bool use_minimum_distance_contacts_only = false;

    // Value between (0, 1] that defines how fast each geom is allowed to move
    // towards another in each iteration. Values lower than 1 are safer but may
    // make the geoms move slower towards each other. In the literature, a
    // common starting value is 0.85. Ignored if `enable_collision_avoidance` is
    // `false`.
    double collision_avoidance_normal_velocity_scale = 0.85;

    // Defines the minimum distance that the solver will attempt to leave
    // between any two geoms. A negative distance would allow the geoms to
    // penetrate by the specified amount. Ignored if
    // `enable_collision_avoidance` is `false`.
    double minimum_distance_from_collisions = 0.005;  // 5mm.

    // Defines the distance between two geoms at which the active collision
    // avoidance behaviour will start. A large value will cause collisions to be
    // detected early, but may incur high computational costs. A negative value
    // will cause the geoms to be detected only after they penetrate by the
    // specified amount.
    double collision_detection_distance = 0.1;  // 10cm.

    // Set of collision pairs in which to perform active collision avoidance. A
    // collision pair is defined as a pair of geom groups. A geom group is a set
    // of geom names. For each collision pair, the filter will attempt to
    // compute joint velocities that avoid collisions between every geom in the
    // first geom group with every geom in the second geom group. Self collision
    // is achieved by adding a collision pair with the same geom group in both
    // pair fields.
    absl::btree_set<CollisionPair> collision_pairs;

    // If `true`, an extra validity check will be performed on the computed
    // velocities to ensure it does not violate any constraints. At the moment,
    // this checks whether the computed velocities would result in increased
    // penetration when integrated. If the computed velocities are found to be
    // invalid, an error will be returned with a description of why the validity
    // check failed.
    bool check_solution_validity = true;

    // Maximum number of iterations that the internal LSQP solver is allowed to
    // spend on the joint velocity optimization problem.
    // If the internal solver is unable to find a feasible solution within the
    // specified number of iterations, it will return an error.
    int max_qp_solver_iterations = 300;

    // Absolute tolerance for the internal LSQP solver. A smaller tolerance may
    // be more accurate but may require a large number of iterations. This
    // tolerance affects the optimality and validity (i.e. constraint violation)
    // of the joint velocity control optimization problem. The physical
    // interpretation of the tolerance is different depending on the task or
    // constraint being considered. For example, when considering the validity
    // of the solution with respect to the collision avoidance constraint, this
    // value represents the tolerance of the maximum normal velocity between any
    // two geoms, in m/s.
    double solution_tolerance = 1.0e-3;

    // If `true`, the internal LSQP solver will use an adaptive step size when
    // solving the resultant QP problem. Note that setting this to `true` can
    // greatly speed up the convergence of the algorithm, but the solution will
    // no longer be numerically deterministic.
    bool use_adaptive_step_size = false;

    // If `true`, a warning will be logged during a call to
    // `FilterJointVelocities` if `data` describes a joint configuration
    // in which one or more collision pairs are closer than the
    // `minimum_normal_distance` threshold. Ignored if
    // `enable_collision_avoidance` is `false`.
    bool log_collision_warnings = false;
  };

  // Returns an OK-status if the parameters are valid, or an error describing
  // why they are not valid.
  static absl::Status ValidateParameters(const Parameters& params);

  // Constructs a filter for the parameters provided.
  // This is guaranteed to succeed if `ValidateParameters` returns an OK-status.
  explicit JointVelocityFilter(const Parameters& params);

  JointVelocityFilter(const JointVelocityFilter&) = delete;
  JointVelocityFilter& operator=(const JointVelocityFilter&) = delete;

  // Computes the array of joint velocities that realizes the desired target
  // joint velocities, and returns a view over the computed array. The
  // returned view is valid until the next call to `FilterJointVelocities`.
  //
  // The computed velocities are guaranteed to be valid and respect the
  // constraints defined by the user, e.g. collision avoidance constraint and
  // joint limits.
  //
  // Returns an error if:
  // * The internal solver was unable to find a valid solution control
  //   optimization problem.
  // * The internal solver found a solution to the optimization problem(s), but
  //   the computed velocities violate the constraints defined by the user.
  //   This may happen if, for example, MuJoCo collision detection was not
  //   accurate or if errors due to the local Jacobian linearization about the
  //   current configuration cause the computed velocities to violate the
  //   constraints when integrated over the user-defined integration timestep.
  absl::StatusOr<absl::Span<const double>> FilterJointVelocities(
      const mjData& data,
      absl::Span<const double> target_joint_velocities);

 private:
  const mjModel& model_;
  std::unique_ptr<mjData, void (*)(mjData*)> data_;
  absl::btree_set<int> joint_dof_ids_;
  absl::Duration integration_timestep_;
  double minimum_distance_from_collisions_;

  LsqpStackOfTasksSolver qp_solver_;
  std::vector<double> solution_;
  bool check_solution_validity_;
  bool log_collision_warnings_;

  // Owned by qp_solver_ if not null.
  IdentityTask* joint_velocity_task_;
  IdentityConstraintUnion* joint_kinematic_constraints_;
  CollisionAvoidanceConstraint* collision_avoidance_constraint_;

  // Used by joint_kinematic_constraints_ if not empty.
  std::vector<IdentityConstraint*> enabled_joint_kinematic_constraints_;

  // Added to enabled_joint_kinematic_constraints_ if not null, and used by
  // joint_kinematic_constraints_.
  std::unique_ptr<JointPositionLimitConstraint>
      joint_position_limit_constraint_;
  std::unique_ptr<BoxConstraint> joint_velocity_limit_constraint_;

  // For collision sanity-checking of solution.
  absl::btree_map<std::pair<int, int>, absl::optional<double>>
      geom_pair_to_dist_curr_;
  absl::btree_map<std::pair<int, int>, absl::optional<double>>
      geom_pair_to_dist_after_;

  // Helper functions.
  absl::Status UpdateTasks(
      absl::Span<const double> target_joint_velocities);
  absl::Status UpdateConstraints();
  absl::Status CheckSolutionValidity();
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_CONTROLLERS_LSQP_JOINT_VELOCITY_FILTER_H_
