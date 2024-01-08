// Copyright 2022 DeepMind Technologies Limited.
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
#include <array>
#include <optional>
#include <vector>

#include "cartesian_6d_to_joint_velocity_mapper_python_helpers.h"  // controller
#include "pybind11/pybind11.h"  // pybind
#include "pybind11/stl.h"  // pybind
// Internal status include

namespace dm_robotics {
namespace {

namespace py = pybind11;

constexpr char kMapperDocstring[] = R"delim(
Mapper from Cartesian 6D velocities to joint velocities.

Maps a Cartesian 6D velocity about the global frame to joint
velocities through the use of an LSQP Stack-of-tasks problem solved at every
call to `ComputeJointVelocities`. The target Cartesian 6D velocity is specified
as a single vector, with the 3D linear velocity term followed by the 3D angular
velocity term.

At every call to `ComputeJointVelocities`, the mapper will attempt to compute
joint velocities that realize the target Cartesian velocity on the frame
attached to the MuJoCo object defined by the `object_type` and `object_name`
parameters passed on construction. The target Cartesian velocity must be
expressed about the MuJoCo object's origin, in world orientation. An exception
is thrown if the computed velocities do not exist (i.e. problem is infeasible),
or if the mapper failed to compute the velocities.

In its most basic configuration, it computes the joint velocities that achieve
the desired Cartesian 6d velocity with singularity robustness. In addition, this
mapper also supports the following functionality:

*   Nullspace control can be enabled to bias the joint velocities to a desired
    value without affecting the accuracy of the resultant Cartesian velocity.
*   Collision avoidance can be enabled for a set of CollisionPair objects, which
    defines which geoms should avoid each other.
*   Limits on the joint positions, velocities, and accelerations can be defined
    to ensure that the computed joint velocities do not result in limit
    violations.
)delim";

constexpr char kParametersDocstring[] = R"delim(
Initialization parameters for cartesian_6d_to_joint_velocity_mapper.Mapper.

Attributes:
  model: (`dm_control.MjModel`) MuJoCo model describing the scene.
  joint_ids: (`Sequence[int]`) MuJoCo joint IDs of the joints to be controlled.
    Only 1 DoF joints are allowed at the moment.
  object_type: (`dm_control.mujoco.wrappers.mjbindings.enums.mjtObj`) type of
    the MuJoCo object that defines the Cartesian frame being controlled. Only
    geoms, sites, and bodies are allowed.
  object_name: (`str`) name of the MuJoCo object that defines the Cartesian
    frame being controlled.
  integration_timestep: (`float` or `datetime.timedelta`) amount of time that
    the joint velocities will be executed for, A.K.A. 'dt'. If unsure, higher
    values are more conservative. This timestep will be used when integrating
    the joint velocities to ensure that safety constraints are not violated.
    If a `float` is provided, it will be interpreted in seconds.
  enable_joint_position_limits: (`bool`) whether to enable joint limit
    avoidance. Joint limits are deduced from the MuJoCo model.
  joint_position_limit_velocity_scale: (`float`) value (0,1] that defines how
    fast each joint is allowed to move towards the joint limits in each
    iteration. Values lower than 1 are safer but may make the joints move
    slowly. 0.95 is usually enough since it is not affected by Jacobian
    linearization. Ignored if `enable_joint_position_limits` is false.
  minimum_distance_from_joint_position_limit: (`float`) offset in meters
    (slide joints) or radians (hinge joints) to be added to the limits.
    Positive values decrease the range of motion, negative values increase it
    (i.e. negative values allow penetration). Ignored if
    `enable_joint_position_limits` is false.
  enable_joint_velocity_limits: (`bool`) whether to enable joint velocity
    limits.
  joint_velocity_magnitude_limits: array of maximum allowed magnitudes of
    joint velocities for each joint, in m/s (slide joints) or rad/s (hinge
    joints). Must be ordered according to the `joint_ids` field. Ignored if
    `enable_joint_velocity_limits` is false.
  enable_joint_acceleration_limits: (`bool`) whether to enable joint
    acceleration limits. Note that enabling joint acceleration limits may reduce
    the size of the feasible space considerably, and due to this a vector of
    joint velocities satisfying all the constraints at the same time may not
    always exist. In this case, the mapper will throw an exception when
    attempting to compute the joint velocities. It is the user's responsibility
    to tune the parameters accordingly, and to ensure that a failure to compute
    valid joint velocities does not result in unexpected behaviour. In our
    experiments, we found that enabling acceleration limits required us to
    enable `remove_joint_acceleration_limits_if_in_conflict` and reduce the
    `collision_avoidance_normal_velocity_scale` parameter considerably.
  remove_joint_acceleration_limits_if_in_conflict: (`bool`) if true, the
    acceleration limits constraint will be checked for feasibility in each
    iteration and removed if in conflict with the joint position or joint
    velocity limit constraints. This will result in the joint position and joint
    velocity limit constraints taking prescedence over the joint acceleration
    constraint. Ignored if `enable_joint_acceleration_limits` is false.
  joint_acceleration_magnitude_limits: array of maximum allowed magnitudes of
    joint acceleration for each joint, in m/s^2 (slide joints) or rad/s^2
    (hinge joints). Must be ordered according to the `joint_ids` field.
    Ignored if `enable_joint_acceleration_limits` is false.
  enable_collision_avoidance: (`bool`) whether to enable active collision
    avoidance.
  use_minimum_distance_contacts_only: ('bool') if true, it will only create one
    inequality constraint per geom pair, corresponding to the MuJoCo contact
    with the minimum distance. Otherwise, it will create one inequality
    constraint for each of the MuJoCo contacts detected per geom pair.
    Ignored if `enable_collision_avoidance` is `false`. In problems where many
    geoms are avoiding each other, setting this option to `true` will
    considerably speed up solve times, but the solution is more likely to
    result in penetration at high speeds.
  collision_avoidance_normal_velocity_scale: (`float`) value between (0, 1] that
    defines how fast each geom is allowed to move towards another in each
    iteration. Values lower than 1 are safer but may make the geoms move
    slower towards each other. In the literature, a common starting value is
    0.85. Ignored if `enable_collision_avoidance` is false.
  minimum_distance_from_collisions: (`float`) defines the minimum distance that
    the solver will attempt to leave between any two geoms. A negative distance
    would allow the geoms to penetrate by the specified amount. Ignored if
    `enable_collision_avoidance` is false.
  collision_detection_distance: (`float`) defines the distance between two geoms
    at which the active collision avoidance behaviour will start. A large value
    will cause collisions to be detected early, but may incure high
    computational costs. A negative value will cause the geoms to be detected
    only after they penetrate by the specified amount.
  collision_pairs: (`Sequence[Tuple[Sequence[str], Sequence[str]]]`) set of
    collision pairs in which to perform active collision avoidance. A collision
    pair is defined as a pair of geom groups. A geom group is a set of geom
    names. For each collision pair, the mapper will attempt to compute joint
    velocities that avoid collisions between every geom in the first geom group
    with every geom in the second geom group. Self collision is achieved by
    adding a collision pair with the same geom group in both pair fields.
  cartesian_velocity_task_weighting_matrix: (`Sequence[float]`) 6x6 matrix, in
    column-major ordering, containing the weights for each component of the
    Cartesian 6D velocity being controlled by the Cartesian velocity task.
  cartesian_velocity_direction_task_weight: (`float`) weight of the Cartesian
    velocity direction task. This task attempts to minimize the component of
    the realized Cartesian velocity that is perpendicular to the target
    Cartesian velocity direction. A weight of 0.0 disables this task.
  cartesian_velocity_direction_task_weighting_matrix: (`Sequence[float]`) 6x6
    matrix, in column-major ordering, containing the weights for each component
    of the Cartesian 6D velocity direction being controlled by the Cartesian
    velocity direction task.
  enable_cartesian_velocity_direction_constraint: (`bool`) whether to enable the
    Cartesian velocity direction constraint. This constraint limits the
    realized Cartesian velocity direction to be within 180 degrees from the
    target Cartesian velocity direction.
  cartesian_velocity_direction_constraint_axes: (`Sequence[bool]`) sequence of
    flags defining which components of the velocity should be constrained by
    the Cartesian velocity direction constraint. The flags correspond to each
    of the velocity axes in the order [Vx, Vy, Vz, Wx, Wy, Wz].
  check_solution_validity: (`bool`) if true, an extra validity check will be
    performed on the computed velocities to ensure it does not violate any
    constraints. At the moment, this checks whether the computed velocities
    would result in increased penetration when integrated. If the computed
    velocities are found to be invalid, an exception will be thrown with a
    description of why the validity check failed.
  max_cartesian_velocity_control_iterations: (`int`) maximum number of
    iterations that the internal LSQP solver is allowed to spend on the
    Cartesian velocity optimization problem (first hierarchy). If the internal
    solver is unable to find a feasible solution to the first hierarchy (i.e.
    without nullspace) within the specified number of iterations, it will throw
    an exception.
  regularization_weight: (`float`) weight of the regularization task for
    singularity robustness on the Cartesian velocity control optimization
    problem.
  solution_tolerance: (`float`) absolute tolerance for the internal LSQP solver.
    A smaller tolerance may be more accurate but may require a large number of
    iterations. This tolerance affects the optimality and validity (i.e.
    constraint violation) of the solution for both, the Cartesian control
    optimization problem and of the nullspace projection optimization
    problem, if enabled. The physical interpretation of the tolerance is
    different depending on the task or constraint being considered. For
    example, when considering the validity of the solution with respect to
    the collision avoidance constraint, this value represents the tolerance
    of the maximum normal velocity between any two geoms, in m/s.
  enable_nullspace_control: (`bool`) whether to enable joint space velocity
    nullspace control.
  return_error_on_nullspace_failure: (`bool`) if false, `ComputeJointVelocities`
    will return the minimum norm least squares solution to the Cartesian
    velocity optimization problem if the internal LSQP solver is unable to
    solve the nullspace optimization problem (second hierarchy). If true,
    an exception will be thrown. Ignored if `enable_nullspace_control` is
    false.
  clamp_nullspace_bias_to_feasible_space: (`bool`) if true, the `nullspace_bias`
    argument to `ComputeJointVelocities` will be clamped to the feasible
    space of the joint position limits, joint velocity limits, and joint
    acceleration limits constraints. This improves stability, but the
    solution to the nullspace optimization problem will no-longer be as close
    as possible to the nullspace bias in the least-squares sense, but
    instead to the clipped nullspace bias. Ignored if
    `enable_nullspace_control` is false.
  max_nullspace_control_iterations: (`int`) maximum number of iterations that
    the internal LSQP solver is allowed to spend on the nullspace optimization
    problem (second hierarchy). If the internal solver is unable to find a
    feasible solution to the second hierarchy within the specified number of
    iterations, it will return either the minimum norm least squares solution
    to the first hierarchy, or throw an exception depending on the value of
    `return_error_on_nullspace_failure`. Ignored if
    `enable_nullspace_control` is false.
  nullspace_projection_slack: (`float`) hierarchical projection slack for the
    internal LSQP solver. A smaller value will result in a stricter nullspace
    projection constraint, but may be numerically unstable. This parameter
    and the `solution_tolerance` both affect the accuracy of the nullspace
    projection. As a general rule, the solution to the nullspace projection
    optimization problem will at most decrease the accuracy of the Cartesian
    velocity by `solution_tolerance` + `nullspace_projection_slack`.
  use_adaptive_step_size: (`bool`) if true, the internal LSQP solver will use
    an adaptive step size when solving the resultant QP problem. Note that
    setting this to true can greatly speed up the convergence of the algorithm,
    but the solution will no longer be numerically deterministic.
  log_nullspace_failure_warnings: (`bool`) if true, a warning will be logged
    if the internal LSQP solver is unable to solve the nullspace optimization
    problem (second hierarchy). Ignored if `return_error_on_nullspace_failure`
    is true.
  log_collision_warnings: (`bool`) If true, a warning will be logged during
    a call to `ComputeJointVelocities` if `data` describes a joint configuration
    in which one or more collision pairs are closer than the
    `minimum_normal_distance` threshold. Ignored if `enable_collision_avoidance`
    is false.

)delim";

constexpr char kComputeJointVelocitiesOptionalNullspaceDoscstring[] = R"delim(
Computes the joint velocities that realize the target 6D velocity (1).

Args:
  data: (`dm_control.MjData`) MuJoCo data with the current scene configuration.
  target_6d_cartesian_velocity: (`Sequence[float]`) 6D array representing the
    target Cartesian velocity. Must be expressed as [(lin_vel), (ang_vel)].
  nullspace_bias: (`Optional[Sequence[float]]`) target joint velocities for the
    secondary nullspace objective. If `return_error_on_nullspace_failure` is
    false, failures when solving the nullspace optimization problem will be
    ignored, and the solution to the Cartesian control optimization problem will
    be returned instead. Must be the same size as the number of DoF being
    controlled. If `None`, equivalent to `compute_joint_velocities` (2).
)delim";

constexpr char kComputeJointVelocitiesNoNullspaceDocstring[] = R"delim(
Computes the joint velocities that realize the target 6D velocity (2).

Equivalent to `compute_joint_velocities` (1) when nullspace control is not
enabled. This overload should only be used if nullspace control is not enabled.

Args:
  data: (`dm_control.MjData`) MuJoCo data with the current scene configuration.
  target_6d_cartesian_velocity: (`Sequence[float]`) 6D array representing the
    target Cartesian velocity. Must be expressed as [(lin_vel), (ang_vel)].
)delim";

}  // namespace

PYBIND11_MODULE(cartesian_6d_to_joint_velocity_mapper, m) {
  // Internal status module placeholder.
  using dm_robotics::internal::PyCartesian6dToJointVelocityMapper;
  using dm_robotics::internal::PyCartesian6dToJointVelocityMapperParameters;

  // Cartesian6dToJointVelocityMapper::Parameters.
  py::class_<PyCartesian6dToJointVelocityMapperParameters>(m, "Parameters",
                                                           kParametersDocstring)
      .def(py::init<>())
      .def_property("model",
                    &PyCartesian6dToJointVelocityMapperParameters::GetModel,
                    &PyCartesian6dToJointVelocityMapperParameters::SetModel)
      .def_property("joint_ids",
                    &PyCartesian6dToJointVelocityMapperParameters::GetJointIds,
                    &PyCartesian6dToJointVelocityMapperParameters::SetJointIds)
      .def_property(
          "object_type",
          &PyCartesian6dToJointVelocityMapperParameters::GetObjectType,
          &PyCartesian6dToJointVelocityMapperParameters::SetObjectType)
      .def_property(
          "object_name",
          &PyCartesian6dToJointVelocityMapperParameters::GetObjectName,
          &PyCartesian6dToJointVelocityMapperParameters::SetObjectName)
      .def_property(
          "integration_timestep",
          &PyCartesian6dToJointVelocityMapperParameters::GetIntegrationTimestep,
          &PyCartesian6dToJointVelocityMapperParameters::SetIntegrationTimestep)

      .def_property("enable_joint_position_limits",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetEnableJointPositionLimits,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetEnableJointPositionLimits)
      .def_property("joint_position_limit_velocity_scale",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetJointPositionLimitVelocityScale,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetJointPositionLimitVelocityScale)
      .def_property("minimum_distance_from_joint_position_limit",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetMinimumDistanceFromJointPositionLimit,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetMinimumDistanceFromJointPositionLimit)

      .def_property("enable_joint_velocity_limits",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetEnableJointVelocityLimits,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetEnableJointVelocityLimits)
      .def_property("joint_velocity_magnitude_limits",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetJointVelocityMagnitudeLimits,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetJointVelocityMagnitudeLimits)

      .def_property("enable_joint_acceleration_limits",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetEnableJointAccelerationLimits,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetEnableJointAccelerationLimits)
      .def_property("remove_joint_acceleration_limits_if_in_conflict",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetRemoveJointAccelerationLimitsIfInConflict,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetRemoveJointAccelerationLimitsIfInConflict)
      .def_property("joint_acceleration_magnitude_limits",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetJointAccelerationMagnitudeLimits,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetJointAccelerationMagnitudeLimits)

      .def_property("enable_collision_avoidance",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetEnableCollisionAvoidance,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetEnableCollisionAvoidance)
      .def_property("use_minimum_distance_contacts_only",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetUseMinimumDistanceContactsOnly,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetUseMinimumDistanceContactsOnly)
      .def_property("collision_avoidance_normal_velocity_scale",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetCollisionAvoidanceNormalVelocityScale,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetCollisionAvoidanceNormalVelocityScale)
      .def_property("minimum_distance_from_collisions",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetMinimumDistanceFromCollisions,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetMinimumDistanceFromCollisions)
      .def_property("collision_detection_distance",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetCollisionDetectionDistance,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetCollisionDetectionDistance)
      .def_property(
          "collision_pairs",
          &PyCartesian6dToJointVelocityMapperParameters::GetCollisionPairs,
          &PyCartesian6dToJointVelocityMapperParameters::SetCollisionPairs)

      .def_property("cartesian_velocity_task_weighting_matrix",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetCartesianVelocityTaskWeightingMatrix,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetCartesianVelocityTaskWeightingMatrix)
      .def_property("cartesian_velocity_direction_task_weight",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetCartesianVelocityDirectionTaskWeight,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetCartesianVelocityDirectionTaskWeight)
      .def_property("cartesian_velocity_direction_task_weighting_matrix",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetCartesianVelocityDirectionTaskWeightingMatrix,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetCartesianVelocityDirectionTaskWeightingMatrix)
      .def_property("enable_cartesian_velocity_direction_constraint",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetEnableCartesianVelocityDirectionConstraint,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetEnableCartesianVelocityDirectionConstraint)
      .def_property("cartesian_velocity_direction_constraint_axes",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetCartesianVelocityDirectionConstraintAxes,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetCartesianVelocityDirectionConstraintAxes)

      .def_property("check_solution_validity",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetCheckSolutionValidity,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetCheckSolutionValidity)
      .def_property("max_cartesian_velocity_control_iterations",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetMaxCartesianVelocityControlIterations,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetMaxCartesianVelocityControlIterations)
      .def_property("regularization_weight",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetRegularizationWeight,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetRegularizationWeight)
      .def_property(
          "solution_tolerance",
          &PyCartesian6dToJointVelocityMapperParameters::GetSolutionTolerance,
          &PyCartesian6dToJointVelocityMapperParameters::SetSolutionTolerance)

      .def_property("enable_nullspace_control",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetEnableNullspaceControl,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetEnableNullspaceControl)
      .def_property("return_error_on_nullspace_failure",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetReturnErrorOnNullspaceFailure,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetReturnErrorOnNullspaceFailure)
      .def_property("clamp_nullspace_bias_to_feasible_space",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetClampNullspaceBiasToFeasibleSpace,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetClampNullspaceBiasToFeasibleSpace)
      .def_property("max_nullspace_control_iterations",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetMaxNullspaceControlIterations,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetMaxNullspaceControlIterations)
      .def_property("nullspace_projection_slack",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetNullspaceProjectionSlack,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetNullspaceProjectionSlack)
      .def_property(
          "use_adaptive_step_size",
          &PyCartesian6dToJointVelocityMapperParameters::GetUseAdaptiveStepSize,
          &PyCartesian6dToJointVelocityMapperParameters::SetUseAdaptiveStepSize)
      .def_property("log_nullspace_failure_warnings",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetLogNullspaceFailureWarnings,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetLogNullspaceFailureWarnings)
      .def_property("log_collision_warnings",
                    &PyCartesian6dToJointVelocityMapperParameters::
                        GetLogCollisionWarnings,
                    &PyCartesian6dToJointVelocityMapperParameters::
                        SetLogCollisionWarnings);

  // Cartesian6dToJointVelocityMapper.
  py::class_<PyCartesian6dToJointVelocityMapper>(m, "Mapper", kMapperDocstring)
      .def(py::init<const PyCartesian6dToJointVelocityMapperParameters&>(),
           py::arg("params"),
           "Initializes a Cartesian 6D to Joint Velocity Mapper.")
      .def("compute_joint_velocities",
           py::overload_cast<py::handle, const std::array<double, 6>&,
                             const std::optional<std::vector<double>>&>(
               &PyCartesian6dToJointVelocityMapper::ComputeJointVelocities),
           py::arg("data"), py::arg("target_6d_cartesian_velocity"),
           py::arg("nullspace_bias").none(true),
           kComputeJointVelocitiesOptionalNullspaceDoscstring)
      .def("compute_joint_velocities",
           py::overload_cast<py::handle, const std::array<double, 6>&>(
               &PyCartesian6dToJointVelocityMapper::ComputeJointVelocities),
           py::arg("data"), py::arg("target_6d_cartesian_velocity"),
           kComputeJointVelocitiesNoNullspaceDocstring);
}
}  // namespace dm_robotics
