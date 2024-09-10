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
#include <vector>

#include "joint_velocity_filter_python_helpers.h"  // controller
#include "pybind11/pybind11.h"  // pybind
#include "pybind11/stl.h"  // pybind
// Internal status include

namespace dm_robotics {
namespace {

namespace py = pybind11;

constexpr char kFilterDocstring[] = R"delim(
Filters joint velocities subject to constraints.

Filters joint velocities through the use of an LSQP Stack-of-tasks problem
solved at every call to `FilterJointVelocities`. The target joint velocities
are specified as a single vector.

In its most basic configuration, it computes the joint velocities that are as
close as possible to the desired joint velocities while also supporting the
following functionality:
*   Collision avoidance can be enabled for a set of CollisionPair objects,
    which defines which geoms should avoid each other.
*   Limits on the joint positions and velocities can be defined to ensure that
    the computed joint velocities do not result in limit violations.
An error is returned if the filter failed to compute the velocities.
)delim";

constexpr char kParametersDocstring[] = R"delim(
Initialization parameters for joint_velocity_filter.JointVelocityFilter.

Attributes:
  model: (`dm_control.MjModel`) MuJoCo model describing the scene.
  joint_ids: (`Sequence[int]`) MuJoCo joint IDs of the joints to be controlled.
    Only 1 DoF joints are allowed at the moment.
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
  check_solution_validity: (`bool`) if true, an extra validity check will be
    performed on the computed velocities to ensure it does not violate any
    constraints. At the moment, this checks whether the computed velocities
    would result in increased penetration when integrated. If the computed
    velocities are found to be invalid, an exception will be thrown with a
    description of why the validity check failed.
  max_qp_solver_iterations: (`int`) maximum number of iterations that the
    internal LSQP solver is allowed to spend on the joint velocity optimization
    problem. If the internal solver is unable to find a feasible solution within
    the specified number of iterations, it will throw an exception.
  solution_tolerance: (`float`) absolute tolerance for the internal LSQP solver.
    A smaller tolerance may be more accurate but may require a large number of
    iterations. This tolerance affects the optimality and validity (i.e.
    constraint violation) of the joint velocity control optimization problem.
    The physical interpretation of the tolerance is different depending on the
    task or constraint being considered. For example, when considering the
    validity of the solution with respect to the collision avoidance constraint,
    this value represents the tolerance of the maximum normal velocity between
    any two geoms, in m/s.
  use_adaptive_step_size: (`bool`) if true, the internal LSQP solver will use
    an adaptive step size when solving the resultant QP problem. Note that
    setting this to true can greatly speed up the convergence of the algorithm,
    but the solution will no longer be numerically deterministic.
  log_collision_warnings: (`bool`) If true, a warning will be logged during
    a call to `FilterJointVelocities` if `data` describes a joint configuration
    in which one or more collision pairs are closer than the
    `minimum_normal_distance` threshold. Ignored if `enable_collision_avoidance`
    is false.

)delim";

constexpr char kFilterJointVelocitiesDocstring[] = R"delim(
Filters joint velocities subject to the configured constraints.

The computed velocities are guaranteed to be valid and respect the
constraints defined by the user, e.g. collision avoidance constraint and
joint limits.

Returns an error if:
* The internal solver was unable to find a valid solution to the control
  optimization problem.
* The internal solver found a solution to the optimization problem(s), but
  the computed velocities violate the constraints defined by the user.
  This may happen if, for example, MuJoCo collision detection was not
  accurate or if errors due to the local Jacobian linearization about the
  current configuration cause the computed velocities to violate the
  constraints when integrated over the user-defined integration timestep.

Args:
  data: (`dm_control.MjData`) MuJoCo data with the current scene configuration.
  target_joint_velocities: (`Sequence[float]`) Array representing the target
    joint velocities.
)delim";

}  // namespace

PYBIND11_MODULE(joint_velocity_filter, m) {
  // Internal status module placeholder.
  using dm_robotics::internal::PyJointVelocityFilter;
  using dm_robotics::internal::PyJointVelocityFilterParameters;

  // JointVelocityFilter::Parameters.
  py::class_<PyJointVelocityFilterParameters>(m, "Parameters",
                                              kParametersDocstring)
      .def(py::init<>())
      .def_property("model", &PyJointVelocityFilterParameters::GetModel,
                    &PyJointVelocityFilterParameters::SetModel)
      .def_property("joint_ids", &PyJointVelocityFilterParameters::GetJointIds,
                    &PyJointVelocityFilterParameters::SetJointIds)
      .def_property("integration_timestep",
                    &PyJointVelocityFilterParameters::GetIntegrationTimestep,
                    &PyJointVelocityFilterParameters::SetIntegrationTimestep)

      .def_property(
          "enable_joint_position_limits",
          &PyJointVelocityFilterParameters::GetEnableJointPositionLimits,
          &PyJointVelocityFilterParameters::SetEnableJointPositionLimits)
      .def_property(
          "joint_position_limit_velocity_scale",
          &PyJointVelocityFilterParameters::GetJointPositionLimitVelocityScale,
          &PyJointVelocityFilterParameters::SetJointPositionLimitVelocityScale)
      .def_property("minimum_distance_from_joint_position_limit",
                    &PyJointVelocityFilterParameters::
                        GetMinimumDistanceFromJointPositionLimit,
                    &PyJointVelocityFilterParameters::
                        SetMinimumDistanceFromJointPositionLimit)

      .def_property(
          "enable_joint_velocity_limits",
          &PyJointVelocityFilterParameters::GetEnableJointVelocityLimits,
          &PyJointVelocityFilterParameters::SetEnableJointVelocityLimits)
      .def_property(
          "joint_velocity_magnitude_limits",
          &PyJointVelocityFilterParameters::GetJointVelocityMagnitudeLimits,
          &PyJointVelocityFilterParameters::SetJointVelocityMagnitudeLimits)

      .def_property(
          "enable_collision_avoidance",
          &PyJointVelocityFilterParameters::GetEnableCollisionAvoidance,
          &PyJointVelocityFilterParameters::SetEnableCollisionAvoidance)
      .def_property(
          "use_minimum_distance_contacts_only",
          &PyJointVelocityFilterParameters::GetUseMinimumDistanceContactsOnly,
          &PyJointVelocityFilterParameters::SetUseMinimumDistanceContactsOnly)
      .def_property("collision_avoidance_normal_velocity_scale",
                    &PyJointVelocityFilterParameters::
                        GetCollisionAvoidanceNormalVelocityScale,
                    &PyJointVelocityFilterParameters::
                        SetCollisionAvoidanceNormalVelocityScale)
      .def_property(
          "minimum_distance_from_collisions",
          &PyJointVelocityFilterParameters::GetMinimumDistanceFromCollisions,
          &PyJointVelocityFilterParameters::SetMinimumDistanceFromCollisions)
      .def_property(
          "collision_detection_distance",
          &PyJointVelocityFilterParameters::GetCollisionDetectionDistance,
          &PyJointVelocityFilterParameters::SetCollisionDetectionDistance)
      .def_property("collision_pairs",
                    &PyJointVelocityFilterParameters::GetCollisionPairs,
                    &PyJointVelocityFilterParameters::SetCollisionPairs)

      .def_property("check_solution_validity",
                    &PyJointVelocityFilterParameters::GetCheckSolutionValidity,
                    &PyJointVelocityFilterParameters::SetCheckSolutionValidity)
      .def_property("max_qp_solver_iterations",
                    &PyJointVelocityFilterParameters::GetMaxQpSolverIterations,
                    &PyJointVelocityFilterParameters::SetMaxQpSolverIterations)
      .def_property("solution_tolerance",
                    &PyJointVelocityFilterParameters::GetSolutionTolerance,
                    &PyJointVelocityFilterParameters::SetSolutionTolerance)

      .def_property("use_adaptive_step_size",
                    &PyJointVelocityFilterParameters::GetUseAdaptiveStepSize,
                    &PyJointVelocityFilterParameters::SetUseAdaptiveStepSize)
      .def_property("log_collision_warnings",
                    &PyJointVelocityFilterParameters::GetLogCollisionWarnings,
                    &PyJointVelocityFilterParameters::SetLogCollisionWarnings);

  // JointVelocityFilter.
  py::class_<PyJointVelocityFilter>(m, "JointVelocityFilter", kFilterDocstring)
      .def(py::init<const PyJointVelocityFilterParameters&>(),
           py::arg("params"), "Initializes a joint velocity filter.")
      .def("filter_joint_velocities",
           py::overload_cast<py::handle, const std::vector<double>&>(
               &PyJointVelocityFilter::FilterJointVelocities),
           py::arg("data"), py::arg("target_joint_velocities"),
           kFilterJointVelocitiesDocstring);
}
}  // namespace dm_robotics
