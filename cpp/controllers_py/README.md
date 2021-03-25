# DM Robotics: Controllers Library (Python)

Contents:

-   [Cartesian 6D to Joint Velocity Mapper](#Cartesian-6D-to-Joint-Velocity-Mapper)

## Cartesian 6D to Joint Velocity Mapper

Python bindings for
`dm_robotics/controllers/lsqp/cartesian_6d_to_joint_velocity_mapper`.

This module consists of two classes:

*   `cartesian_6d_to_joint_velocity_mapper.Parameters`
*   `cartesian_6d_to_joint_velocity_mapper.Mapper`

The mapper solves a constrained linear least-squares optimization problem at
every iteration to compute the joint velocities that best achieve the desired
Cartesian 6D velocity of an object.

In its most basic configuration, it computes the joint velocities that achieve
the desired Cartesian 6d velocity with singularity robustness. In addition, this
mapper also supports the following functionality:

*   Nullspace control can be enabled to bias the joint velocities to a desired
    value without affecting the accuracy of the resultant Cartesian velocity;
*   Collision avoidance can be enabled between any two MuJoCo geoms;
*   Limits on the joint positions, velocities, and accelerations can be defined
    to ensure that the computed joint velocities do not result in limit
    violations.

Please refer to
`dm_robotics/controllers/lsqp/cartesian_6d_to_joint_velocity_mapper.h` or the
class docstrings for more information.

Dependencies:

-   dm_robotics/least_squares_qp
-   dm_robotics/controllers
-   [dm_control](https://github.com/deepmind/dm_control)

### Usage

```python
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.controllers import cartesian_6d_to_joint_velocity_mapper

# Initialize simulation. Assumes velocity controlled robot.
# physics.data.ctrl[:] is an array of size 7 that corresponds to the commanded
# velocities of the joints with IDs 7, 8, 9, 10, 12, 13, 14.
physics = mujoco.Physics(...) # Create MuJoCo physics.

# Create mapper parameters.
params = cartesian_6d_to_joint_velocity_mapper.Parameters()
#
# Set model parameters.
params.model = physics.model
params.joint_ids = [7, 8, 9, 10, 12, 13, 14]  # MuJoCo joint IDs being controlled.
params.object_type = enums.mjtObj.mjOBJ_SITE  # MuJoCo object being controlled.
params.object_name = "end_effector"  # name of MuJoCo object being controlled.
params.integration_timestep = 0.005  # Amount of time the joint velocities will be executed for.
#
# Enable joint position limit constraint. Limits are read automatically from the
# model.
params.enable_joint_position_limits = True
params.joint_position_limit_velocity_scale = 0.95
params.minimum_distance_from_joint_position_limit = 0.01  # ~0.5deg.
#
# Enable joint velocity limits.
params.enable_joint_velocity_limits = True
params.joint_velocity_magnitude_limits = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#
# Enable joint acceleration limits.
params.enable_joint_acceleration_limits = True
params.remove_joint_acceleration_limits_if_in_conflict = True
params.joint_acceleration_magnitude_limits = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
#
# Enable collision avoidance between the following geoms:
#   * "gripper" and "base_link"
#   * "base_link" and "floor"
#   * "link1" and "floor"
#   * "gripper" and "floor"
#   * "link1" and "link4"
#   * "link1" and "link5"
#   * "link1" and "link6"
#   * "link2" and "link4"
#   * "link2" and "link5"
#   * "link2" and "link6"
# Note that collision avoidance will not be enabled for a pair of geoms if they
# are attached to the same body or are attached to bodies that have a
# parent-child relationship.
params.enable_collision_avoidance = True
params.collision_avoidance_normal_velocity_scale = 0.5
params.minimum_distance_from_collisions = 0.01
params.collision_detection_distance = 0.3
params.collision_pairs = [(["gripper"], ["base_link"]),
                          (["base_link", "link1", "gripper"], ["floor"]),
                          (["link1", "link2"], ["link4", "link5", "link6"])]
#
# Numerical stability parameters.
params.check_solution_validity = True
params.solution_tolerance = 1e-3
params.regularization_weight = 1e-2
params.enable_nullspace_control = True
params.return_error_on_nullspace_failure = False
params.nullspace_projection_slack = 1e-7

# Create mapper.
mapper = cartesian_6d_to_joint_velocity_mapper.Mapper(params)

# Compute joint velocities and apply them to the joint velocity actuator
# commands at every step.
while True:
  # The nullspace bias is often chosen to be a velocity towards the mid-range of
  # the joints, but can be chosen to be any 7D joint velocity vector.
  nullspace_joint_velocity_bias = get_nullspace_bias()
  target_cartesian_velocity = get_end_effector_target_velocity()
  solution = mapper.compute_joint_velocities(physics.data, target_velocity,
                                             nullspace_bias)
  physics.data.ctrl[:] = solution
  physics.step()
```
