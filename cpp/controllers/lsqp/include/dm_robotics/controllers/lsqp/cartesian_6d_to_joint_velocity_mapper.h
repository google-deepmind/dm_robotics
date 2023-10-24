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

#ifndef DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_TO_JOINT_VELOCITY_MAPPER_H_
#define DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_TO_JOINT_VELOCITY_MAPPER_H_

#include <utility>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_direction_constraint.h"
#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_direction_task.h"
#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_task.h"
#include "dm_robotics/controllers/lsqp/collision_avoidance_constraint.h"
#include "dm_robotics/controllers/lsqp/joint_acceleration_constraint.h"
#include "dm_robotics/controllers/lsqp/joint_position_limit_constraint.h"
#include "dm_robotics/least_squares_qp/common/box_constraint.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint_union.h"
#include "dm_robotics/least_squares_qp/common/identity_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/types.h"

namespace dm_robotics {

// Maps a Cartesian 6D velocity about the global frame to joint velocities
// through the use of an LSQP Stack-of-tasks problem solved at every call to
// `ComputeJointVelocities`. The target Cartesian 6D velocity is specified as a
// single vector, with the 3D linear velocity term followed by the 3D angular
// velocity term.
//
// At every call to `ComputeJointVelocities`, the mapper will attempt to compute
// joint velocities that realize the target Cartesian velocity on the frame
// attached to the MuJoCo object defined by the `object_type` and `object_name`
// parameters passed on construction. The target Cartesian velocity must be
// expressed about the MuJoCo object's origin, in world orientation. An error is
// returned if the computed velocities do not exist (i.e. problem is
// infeasible), or if the mapper failed to compute the velocities.
//
// In its most basic configuration, it computes the joint velocities that
// achieve the desired Cartesian 6d velocity with singularity robustness.
// In addition, this mapper also supports the following functionality:
// * Nullspace control can be enabled to bias the joint velocities to a desired
//   value without affecting the accuracy of the resultant Cartesian velocity.
// * Collision avoidance can be enabled for a set of CollisionPair objects,
//   which defines which geoms should avoid each other.
// * Limits on the joint positions, velocities, and accelerations can be defined
//   to ensure that the computed joint velocities do not result in limit
//   violations.
// Refer to the documentation of `Cartesian6dToJointVelocityMapper::Parameters`
// for more information on the available configuration options.
//
// This class requires an updated mjData object at every call to
// `ComputeJointVelocities`. It is the user's responsibility to ensure that the
// mjData object has consistent and accurate `qpos` and `qvel` fields.
class Cartesian6dToJointVelocityMapper {
 public:
  // Initialization parameters for Cartesian6dToJointVelocityMapper.
  //
  // The caller retains ownership of lib and model.
  // It is the caller's responsibility to ensure the *lib and *model objects
  // outlive any Cartesian6dToJointVelocityMapper instances created with this
  // object.
  struct Parameters {
    // Pointer to a MuJoCo model.
    const mjModel* model;

    // MuJoCo joint IDs of the joints to be controlled. Only 1 DoF joints are
    // allowed at the moment.
    absl::btree_set<int> joint_ids;

    // Type of the MuJoCo object that defines the Cartesian frame being
    // controlled. Only geoms, sites, and bodies are allowed.
    mjtObj object_type;

    // Name of the MuJoCo object that defines the Cartesian frame being
    // controlled.
    std::string object_name;

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

    // Whether to enable joint acceleration limits. Note that enabling joint
    // acceleration limits may reduce the size of the feasible space
    // considerably, and due to this a vector of joint velocities satisfying all
    // the constraints at the same time may not always exist. In this case, the
    // mapper will return an error when attempting to compute the joint
    // velocities. It is the user's responsibility to tune the parameters
    // accordingly, and to ensure that a failure to compute valid joint
    // velocities does not result in unexpected behaviour. In our experiments,
    // we found that enabling acceleration limits required us to enable
    // `remove_joint_acceleration_limits_if_in_conflict` and reduce the
    // `collision_avoidance_normal_velocity_scale` parameter considerably.
    bool enable_joint_acceleration_limits = false;

    // If `true`, the acceleration limits constraint will be checked for
    // feasibility in each iteration and removed if in conflict with the joint
    // position or joint velocity limit constraints. This will result in the
    // joint position and joint velocity limit constraints taking prescedence
    // over the joint acceleration constraint. Ignored if
    // `enable_joint_acceleration_limits` is `false`.
    bool remove_joint_acceleration_limits_if_in_conflict = true;

    // Array of maximum allowed magnitudes of joint acceleration for each joint,
    // in m/s^2 (slide joints) or rad/s^2 (hinge joints). Must be ordered
    // according to the `joint_ids` field. Ignored if
    // `enable_joint_acceleration_limits` is `false`.
    std::vector<double> joint_acceleration_magnitude_limits;

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
    // detected early, but may incure high computational costs. A negative value
    // will cause the geoms to be detected only after they penetrate by the
    // specified amount.
    double collision_detection_distance = 0.1;  // 10cm.

    // Set of collision pairs in which to perform active collision avoidance. A
    // collision pair is defined as a pair of geom groups. A geom group is a set
    // of geom names. For each collision pair, the mapper will attempt to
    // compute joint velocities that avoid collisions between every geom in the
    // first geom group with every geom in the second geom group. Self collision
    // is achieved by adding a collision pair with the same geom group in both
    // pair fields.
    absl::btree_set<CollisionPair> collision_pairs;

    // 6x6 matrix (in column-major ordering) containing the weights for each
    // component of the Cartesian 6D velocity being controlled by the Cartesian
    // velocity task, such that the quadratic cost term of this task is
    // defined as:
    //   || W (C q_dot - b) ||^2
    // where `W` is the weighting matrix; `C` is the coefficient matrix; `q_dot`
    // are the joint velocities; and `b` is the bias.
    std::array<double, 36> cartesian_velocity_task_weighting_matrix = {
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // Linear velocity - X
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,  // Linear velocity - Y
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  // Linear velocity - Z
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  // Angular velocity - X
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  // Angular velocity - Y
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0   // Angular velocity - Z
    };

    // Weight of the Cartesian velocity direction task.
    //
    // This task attempts to minimize the component of the
    // realized Cartesian velocity that is perpendicular to the target Cartesian
    // velocity direction. Setting this weight to zero disables the task.
    double cartesian_velocity_direction_task_weight = 0.0;

    // 6x6 matrix (in column-major ordering) containing the weights for each
    // component of the Cartesian 6D velocity direction being controlled by the
    // Cartesian velocity direction task, such that the quadratic cost term of
    // this task is defined as:
    //   || W (C q_dot - b) ||^2
    // where `W` is the weighting matrix; `C` is the coefficient matrix; `q_dot`
    // are the joint velocities; and `b` is the bias.
    std::array<double, 36> cartesian_velocity_direction_task_weighting_matrix =
        {
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // Linear velocity - X
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,  // Linear velocity - Y
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  // Linear velocity - Z
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  // Angular velocity - X
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  // Angular velocity - Y
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0   // Angular velocity - Z
    };

    // Whether to enable the Cartesian velocity direction constraint.
    //
    // This constraint task limits the realized Cartesian velocity direction to
    // be within a 180 degree shift from the target Cartesian velocity
    // direction.
    bool enable_cartesian_velocity_direction_constraint = false;

    // Array of flags defining which components of the velocity should be
    // constrained in the following order: [Vx, Vy, Vz, Wx, Wy, Wz].
    std::array<bool, 6> cartesian_velocity_direction_constraint_axes = {
        true, true, true, true, true, true};

    // If `true`, an extra validity check will be performed on the computed
    // velocities to ensure it does not violate any constraints. At the moment,
    // this checks whether the computed velocities would result in increased
    // penetration when integrated. If the computed velocities are found to be
    // invalid, an error will be returned with a description of why the validity
    // check failed.
    bool check_solution_validity = true;

    // Maximum number of iterations that the internal LSQP solver is allowed to
    // spend on the Cartesian velocity optimization problem (first hierarchy).
    // If the internal solver is unable to find a feasible solution to the first
    // hierarchy (i.e. without nullspace) within the specified number of
    // iterations, it will return an error.
    int max_cartesian_velocity_control_iterations = 1000;

    // Weight of the regularization task for singularity robustness on the
    // Cartesian velocity control optimization problem.
    double regularization_weight = 1.0e-3;

    // Absolute tolerance for the internal LSQP solver. A smaller tolerance may
    // be more accurate but may require a large number of iterations. This
    // tolerance affects the optimality and validity (i.e. constraint violation)
    // of the solution for both, the Cartesian control optimization problem and
    // of the nullspace projection optimization problem, if enabled. The
    // physical interpretation of the tolerance is different depending on the
    // task or constraint being considered. For example, when considering the
    // validity of the solution with respect to the collision avoidance
    // constraint, this value represents the tolerance of the maximum normal
    // velocity between any two geoms, in m/s.
    double solution_tolerance = 1.0e-3;

    // Whether to enable joint space velocity nullspace control.
    bool enable_nullspace_control = false;

    // If `false`, `ComputeJointVelocities` will return the minimum norm least
    // squares solution to the Cartesian velocity optimization problem if the
    // internal LSQP solver is unable to solve the nullspace optimization
    // problem (second hierarchy). If `true`, an error will be returned. Ignored
    // if `enable_nullspace_control` is `false`.
    bool return_error_on_nullspace_failure = false;

    // If `true`, the `nullspace_bias` argument to `ComputeJointVelocities` will
    // be clamped to the feasible space of the joint position limits, joint
    // velocity limits, and joint acceleration limits constraints. This improves
    // stability, but the solution to the nullspace optimization problem will
    // no-longer be as close as possible to the nullspace bias in the
    // least-squares sense, but instead to the clipped nullspace bias. Ignored
    // if `enable_nullspace_control` is `false`.
    bool clamp_nullspace_bias_to_feasible_space = true;

    //  Maximum number of iterations that the internal LSQP solver is allowed to
    //  spend on the nullspace optimization problem (second hierarchy). If the
    //  internal solver is unable to find a feasible solution to the second
    //  hierarchy within the specified number of iterations, it will return
    //  either the minimum norm least squares solution to the first hierarchy,
    //  or an error depending on the value of
    //  `return_error_on_nullspace_failure`. Ignored if
    //  `enable_nullspace_control` is `false`.
    int max_nullspace_control_iterations = 300;

    // Hierarchical projection slack for the internal LSQP solver. A smaller
    // value will result in a stricter nullspace projection constraint, but may
    // be numerically unstable. This parameter and the `solution_tolerance` both
    // affect the accuracy of the nullspace projection. As a general rule, the
    // solution to the nullspace projection optimization problem will at most
    // decrease the accuracy of the Cartesian velocity by `solution_tolerance` +
    // `nullspace_projection_slack`.
    double nullspace_projection_slack = 1.0e-4;

    // If `true`, the internal LSQP solver will use an adaptive step size when
    // solving the resultant QP problem. Note that setting this to `true` can
    // greatly speed up the convergence of the algorithm, but the solution will
    // no longer be numerically deterministic.
    bool use_adaptive_step_size = false;

    // If `true`, a warning will be logged if the internal LSQP solver is unable
    // to solve the nullspace optimization problem (second hierarchy). Ignored
    // if `return_error_on_nullspace_failure` is `true`.
    bool log_nullspace_failure_warnings = true;

    // If `true`, a warning will be logged during a call to
    // `ComputeJointVelocities` if `data` describes a joint configuration
    // in which one or more collision pairs are closer than the
    // `minimum_normal_distance` threshold. Ignored if
    // `enable_collision_avoidance` is `false`.
    bool log_collision_warnings = false;
  };

  // Returns an OK-status if the parameters are valid, or an error describing
  // why they are not valid.
  static absl::Status ValidateParameters(const Parameters& params);

  // Constructs a mapper for the parameters provided.
  // This is guaranteed to succeed if `ValidateParameters` returns an OK-status.
  explicit Cartesian6dToJointVelocityMapper(const Parameters& params);

  Cartesian6dToJointVelocityMapper(const Cartesian6dToJointVelocityMapper&) =
      delete;
  Cartesian6dToJointVelocityMapper& operator=(
      const Cartesian6dToJointVelocityMapper&) = delete;

  // Computes the array of joint velocities that realizes the target 6D
  // Cartesian velocity, and returns a view over the computed array. The
  // returned view is valid until the next call to `ComputeJointVelocities`.
  //
  // The `nullspace_bias` parameter defines the target joint velocities that the
  // nullspace optimization problem will attempt to realize. If
  // `return_error_on_nullspace_failure` is `false`, failures when solving the
  // nullspace optimization problem will be ignored, and the solution to the
  // Cartesian control optimization problem will be returned instead.
  //
  // The computed velocities are guaranteed to be valid and respect the
  // constraints defined by the user, e.g. collision avoidance constraint and
  // joint limits.
  //
  // Returns an error if:
  // * The internal solver was unable to find a valid solution to the Cartesian
  //   control optimization problem.
  // * The internal solver was unable to find a valid solution to the nullspace
  //   optimization problem, and `return_error_on_nullspace_failure` is `true`.
  // * The internal solver found a solution to the optimization problem(s), but
  //   but the computed velocities violate the constraints defined by the user.
  //   This may happen if, for example, MuJoCo collision detection was not
  //   accurate or if errors due to the local Jacobian linearization about the
  //   current configuration cause the computed velocities to violate the
  //   constraints when integrated over the user-defined integration timestep.
  //
  // Must not be called if `enable_nullspace_control` is `false`. The
  // `nullspace_bias` parameter must be the same size as the number of joints if
  // nullspace control is enabled. (Precondition violation may cause
  // CHECK-failure.)
  absl::StatusOr<absl::Span<const double>> ComputeJointVelocities(
      const mjData& data, absl::Span<const double> target_6d_cartesian_velocity,
      absl::Span<const double> nullspace_bias);

  // Equivalent to `ComputeJointVelocities` above without a nullspace_bias.
  //
  // Must not be called if `enable_nullspace_control` is `true`. (Precondition
  // violation may cause CHECK-failure.)
  absl::StatusOr<absl::Span<const double>> ComputeJointVelocities(
      const mjData& data,
      absl::Span<const double> target_6d_cartesian_velocity);

 private:
  const MjLib& lib_;
  const mjModel& model_;
  std::unique_ptr<mjData, void (*)(mjData*)> data_;
  absl::btree_set<int> joint_dof_ids_;
  absl::Duration integration_timestep_;
  bool clamp_nullspace_bias_to_feasible_space_;
  bool remove_joint_acceleration_limits_if_in_conflict_;
  double minimum_distance_from_collisions_;

  LsqpStackOfTasksSolver qp_solver_;
  std::vector<double> solution_;
  bool check_solution_validity_;
  bool log_collision_warnings_;

  // Owned by qp_solver_ if not null.
  Cartesian6dVelocityTask* cartesian_velocity_task_;
  Cartesian6dVelocityDirectionTask* cartesian_velocity_direction_task_;
  Cartesian6dVelocityDirectionConstraint*
      cartesian_velocity_direction_constraint_;
  IdentityTask* nullspace_task_;
  IdentityConstraintUnion* joint_kinematic_constraints_;
  CollisionAvoidanceConstraint* collision_avoidance_constraint_;

  // Used by joint_kinematic_constraints_ if not empty.
  std::vector<IdentityConstraint*> enabled_joint_kinematic_constraints_;

  // Added to enabled_joint_kinematic_constraints_ if not null, and used by
  // joint_kinematic_constraints_.
  std::unique_ptr<JointPositionLimitConstraint>
      joint_position_limit_constraint_;
  std::unique_ptr<BoxConstraint> joint_velocity_limit_constraint_;
  std::unique_ptr<JointAccelerationConstraint>
      joint_acceleration_limit_constraint_;

  // Non-empty if nullspace control is enabled.
  std::vector<double> clipped_nullspace_bias_;

  // For collision sanity-checking of solution.
  absl::btree_map<std::pair<int, int>, absl::optional<double>>
      geom_pair_to_dist_curr_;
  absl::btree_map<std::pair<int, int>, absl::optional<double>>
      geom_pair_to_dist_after_;

  // Helper functions.
  absl::StatusOr<absl::Span<const double>> ComputeJointVelocitiesImpl(
      const mjData& data, absl::Span<const double> target_6d_cartesian_velocity,
      absl::Span<const double> nullspace_bias);
  absl::Status UpdateTasks(
      absl::Span<const double> target_6d_cartesian_velocity,
      absl::Span<const double> nullspace_bias);
  absl::Status UpdateConstraints();
  absl::Status CheckSolutionValidity();
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_TO_JOINT_VELOCITY_MAPPER_H_
