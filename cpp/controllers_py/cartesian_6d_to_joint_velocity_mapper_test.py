# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for cartesian_6d_to_joint_velocity_mapper PyBind11 module."""

from absl.testing import absltest
from dm_control import mujoco
from dm_control.suite import humanoid
from dm_robotics.controllers import cartesian_6d_to_joint_velocity_mapper
import numpy as np

_MjGeom = mujoco.wrapper.mjbindings.enums.mjtObj.mjOBJ_GEOM
_MjBody = mujoco.wrapper.mjbindings.enums.mjtObj.mjOBJ_BODY
_MjSite = mujoco.wrapper.mjbindings.enums.mjtObj.mjOBJ_SITE
mjlib = mujoco.wrapper.mjbindings.mjlib


def _set_joint_velocities(physics, joint_ids, joint_velocities):
  """Sets the joint velocities in physics for a subset of joints."""
  for i in range(0, physics.model.nv):
    physics.data.qvel[i] = 0.0
  for joint_id, velocity in zip(sorted(joint_ids), joint_velocities):
    dof_adr = physics.model.jnt_dofadr[joint_id]
    physics.data.qvel[dof_adr] = velocity


def _compute_object_jacobian(physics, object_name, object_type):
  """Computes an object's Jacobian."""
  mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)
  mjlib.mj_fwdVelocity(physics.model.ptr, physics.data.ptr)

  jacobian = np.empty((6, physics.model.nv), dtype=physics.data.qpos.dtype)
  if object_type == _MjGeom or object_type == "geom":
    mjlib.mj_jacGeom(physics.model.ptr, physics.data.ptr, jacobian[:3],
                     jacobian[3:], physics.model.name2id(object_name, _MjGeom))
  elif object_type == _MjSite or object_type == "site":
    mjlib.mj_jacSite(physics.model.ptr, physics.data.ptr, jacobian[:3],
                     jacobian[3:], physics.model.name2id(object_name, _MjSite))
  elif object_type == _MjBody or object_type == "body":
    mjlib.mj_jacBody(physics.model.ptr, physics.data.ptr, jacobian[:3],
                     jacobian[3:], physics.model.name2id(object_name, _MjBody))
  else:
    raise ValueError("Invalid object type.")
  return jacobian


def _compute_object_jacobian_for_joints(physics, object_name, object_type,
                                        joint_ids):
  """Computes an object's Jacobian for a subset of joints."""
  jacobian = _compute_object_jacobian(physics, object_name, object_type)
  dof_ids = [physics.model.jnt_dofadr[joint_id] for joint_id in joint_ids]
  return jacobian[:, dof_ids]


# This is necessary because MuJoCo's object_velocity is computationally
# different than J*qvel. Since the QP uses the Jacobian, the solution should be
# compared to J*qvel to ensure it matches the expected tolerances.
def _compute_object_velocity_with_jacobian(physics, object_name, object_type):
  """Computes an object velocity by evaluating J*qvel."""
  qvel = np.array(physics.data.qvel, dtype=physics.data.qpos.dtype)
  return _compute_object_jacobian(physics, object_name, object_type).dot(qvel)


class Cartesian6DToJointVelocityMapperTest(absltest.TestCase):

  def test_parameters_attributes(self):
    params = cartesian_6d_to_joint_velocity_mapper.Parameters()

    attributes = sorted(
        [attr for attr in dir(params) if not attr.startswith("_")])
    expected_attributes = sorted([
        "model",
        "joint_ids",
        "object_type",
        "object_name",
        "integration_timestep",
        "enable_joint_position_limits",
        "joint_position_limit_velocity_scale",
        "minimum_distance_from_joint_position_limit",
        "enable_joint_velocity_limits",
        "joint_velocity_magnitude_limits",
        "enable_joint_acceleration_limits",
        "remove_joint_acceleration_limits_if_in_conflict",
        "joint_acceleration_magnitude_limits",
        "enable_collision_avoidance",
        "collision_avoidance_normal_velocity_scale",
        "minimum_distance_from_collisions",
        "collision_detection_distance",
        "collision_pairs",
        "cartesian_velocity_task_weighting_matrix",
        "check_solution_validity",
        "max_cartesian_velocity_control_iterations",
        "regularization_weight",
        "solution_tolerance",
        "enable_nullspace_control",
        "return_error_on_nullspace_failure",
        "clamp_nullspace_bias_to_feasible_space",
        "max_nullspace_control_iterations",
        "nullspace_projection_slack",
        "log_nullspace_failure_warnings",
        "log_collision_warnings",
    ])
    self.assertEqual(expected_attributes, attributes)

  def test_mapper_attributes(self):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    params = cartesian_6d_to_joint_velocity_mapper.Parameters()
    params.model = physics.model
    params.joint_ids = [19, 20, 21]
    params.object_type = _MjGeom
    params.object_name = "left_hand"
    params.integration_timestep = 1.0
    params.enable_nullspace_control = True
    mapper = cartesian_6d_to_joint_velocity_mapper.Mapper(params)

    self.assertTrue(hasattr(mapper, "compute_joint_velocities"))

  def test_solution_without_nullspace_realizes_target(self):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    params = cartesian_6d_to_joint_velocity_mapper.Parameters()
    params.model = physics.model
    params.joint_ids = [16, 17, 18]
    params.object_type = _MjGeom
    params.object_name = "right_hand"
    params.integration_timestep = 1.0
    params.solution_tolerance = 1.0e-15
    params.regularization_weight = 0.0
    mapper = cartesian_6d_to_joint_velocity_mapper.Mapper(params)

    # Set target to a realizable velocity and solve.
    target_velocity = [
        0.0450566, 0.0199436, 0.0199436, 0, 0.0071797, -0.0071797
    ]
    solution = mapper.compute_joint_velocities(physics.data, target_velocity,
                                               None)
    _set_joint_velocities(physics, params.joint_ids, solution)

    # Realized Cartesian velocity must be within the specified tolerance of the
    # target velocity.
    realized_velocity = _compute_object_velocity_with_jacobian(
        physics, params.object_name, params.object_type)

    # Ensure the realized Cartesian velocity is within tolerance of the target
    # velocity.
    # Note that for an unconstrained stack-of-tasks problem with one task that
    # is realizable, the `absolute_tolerance` represents how far from optimality
    # the solution is, measured by:
    #   e_dual = W ||J^T J qvel - (xdot_target^T J)^T||
    #   e_dual = W ||J^T xdot_target - J^T xdot_realized||
    jacobian = _compute_object_jacobian_for_joints(physics, params.object_name,
                                                   params.object_type,
                                                   params.joint_ids)
    e_dual = np.linalg.norm(
        np.transpose(jacobian).dot(realized_velocity -
                                   np.array(target_velocity)),
        ord=np.inf)
    self.assertLess(e_dual, params.solution_tolerance)

  def test_solution_with_nullspace_realizes_target(self):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    target_velocity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nullspace_bias = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]

    # Shared parameters for optimization problem with and without nullspace
    # hierarchy.
    params = cartesian_6d_to_joint_velocity_mapper.Parameters()
    params.model = physics.model
    params.joint_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    params.object_type = _MjBody
    params.object_name = "right_foot"
    params.integration_timestep = 1.0
    params.solution_tolerance = 1.0e-6
    params.regularization_weight = 1.0e-3

    # Compute solution without nullspace.
    no_nullspace_mapper = cartesian_6d_to_joint_velocity_mapper.Mapper(params)
    no_nullspace_solution = no_nullspace_mapper.compute_joint_velocities(
        physics.data, target_velocity)
    _set_joint_velocities(physics, params.joint_ids, no_nullspace_solution)
    no_nullspace_cartesian_vel = _compute_object_velocity_with_jacobian(
        physics, params.object_name, params.object_type)

    # Reuse the same parameters but add nullspace projection, and compute the
    # solution to the optimization problem with nullspace.
    params.enable_nullspace_control = True
    params.return_error_on_nullspace_failure = True
    params.nullspace_projection_slack = 1.0e-7
    nullspace_mapper = cartesian_6d_to_joint_velocity_mapper.Mapper(params)
    nullspace_solution = nullspace_mapper.compute_joint_velocities(
        physics.data, target_velocity, nullspace_bias)
    _set_joint_velocities(physics, params.joint_ids, nullspace_solution)
    nullspace_cartesian_vel = _compute_object_velocity_with_jacobian(
        physics, params.object_name, params.object_type)

    # The nullspace solution should be different than the no-nullspace solution.
    # For this problem, we computed the Euclidean distance of both solutions to
    # be around ~0.85; this is expected since there's 10 DoF and only 6 DoF are
    # being used for Cartesian control. Test that the solutions differ by at
    # least 0.8, and that the nullspace solution is closer to the nullspace
    # target.
    solution_diff = np.linalg.norm(
        np.array(nullspace_solution) - np.array(no_nullspace_solution))
    no_nullspace_sol_bias_error = np.linalg.norm(
        np.array(nullspace_bias) - np.array(no_nullspace_solution))
    nullspace_sol_bias_error = np.linalg.norm(
        np.array(nullspace_bias) - np.array(nullspace_solution))
    self.assertGreater(solution_diff, 0.8)
    self.assertLess(nullspace_sol_bias_error, no_nullspace_sol_bias_error)

    # The nullspace solution should respect the nullspace inequality constraint:
    #  xdot_first - slack  <= xdot <= xdot_first + slack,
    # with a tolerance of kSolution tolerance (and thus increasing the slack).
    # This means that it must be within solution_tolerance + slack from the
    # Cartesian velocity achieved without nullspace.
    self.assertLess(
        np.linalg.norm(
            np.array(nullspace_cartesian_vel) -
            np.array(no_nullspace_cartesian_vel),
            ord=np.inf),
        params.solution_tolerance + params.nullspace_projection_slack)

  def test_solution_with_all_constraints_and_nullspace_not_in_collision(self):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    # Increase collision detection margin for all geoms.
    for i in range(0, physics.model.ngeom):
      physics.model.geom_margin[i] = 0.01

    # Place the humanoid in a position where the left hand can collide with the
    # floor if it moves down.
    physics.data.qpos[2] = 0.3
    mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

    # Set params with collision avoidance and full list of constraints.
    params = cartesian_6d_to_joint_velocity_mapper.Parameters()
    params.model = physics.model
    params.joint_ids = [19, 20, 21]
    params.object_type = _MjSite
    params.object_name = "left_hand"
    params.integration_timestep = 0.005  # 5ms

    params.enable_joint_position_limits = True
    params.joint_position_limit_velocity_scale = 0.95
    params.minimum_distance_from_joint_position_limit = 0.01  # ~0.5deg.

    params.enable_joint_velocity_limits = True
    params.joint_velocity_magnitude_limits = [0.5, 0.5, 0.5]

    params.enable_joint_acceleration_limits = True
    params.remove_joint_acceleration_limits_if_in_conflict = True
    params.joint_acceleration_magnitude_limits = [1.0, 1.0, 1.0]

    params.enable_collision_avoidance = True
    params.collision_avoidance_normal_velocity_scale = 0.01
    params.minimum_distance_from_collisions = 0.005
    params.collision_detection_distance = 10.0
    params.collision_pairs = [
        (["left_upper_arm", "left_lower_arm", "left_hand"], ["floor"])
    ]

    params.check_solution_validity = True
    params.solution_tolerance = 1e-6
    params.regularization_weight = 1e-3
    params.enable_nullspace_control = True
    params.return_error_on_nullspace_failure = False
    params.nullspace_projection_slack = 1e-7
    mapper = cartesian_6d_to_joint_velocity_mapper.Mapper(params)

    # Approximate the distance of the left hand and floor geoms by the
    # difference in Z components minus the radius.
    lhand_radius = physics.named.model.geom_size["left_hand"][0]
    lhand_floor_dist = (
        physics.named.data.geom_xpos["left_hand"][2] -
        physics.named.data.geom_xpos["floor"][2] - lhand_radius)

    nullspace_bias = [-1.0, 0.0, 1.0]
    target_velocity = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]

    # Compute velocities and integrate, for 5000 steps.
    for _ in range(0, 5000):
      # Compute joint velocities.
      solution = mapper.compute_joint_velocities(physics.data, target_velocity,
                                                 nullspace_bias)

      # Set joint velocities, integrate, and run MuJoCo kinematics.
      _set_joint_velocities(physics, params.joint_ids, solution)
      mjlib.mj_integratePos(physics.model.ptr, physics.data.qpos,
                            physics.data.qvel,
                            params.integration_timestep.total_seconds())
      mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

      # Compute the new distance between the floor and the left hand.
      # We expect the left hand site to get closer to the floor and settle at
      # around <0.006.
      new_lhand_floor_dist = (
          physics.named.data.geom_xpos["left_hand"][2] -
          physics.named.data.geom_xpos["floor"][2] - lhand_radius)
      self.assertLess(new_lhand_floor_dist, max(0.006, lhand_floor_dist))
      lhand_floor_dist = new_lhand_floor_dist

      # Ensure there is no contact between any left arm geom and the floor.
      for contact in physics.data.contact:
        geom1_name = physics.model.id2name(contact.geom1, _MjGeom)
        geom2_name = physics.model.id2name(contact.geom2, _MjGeom)
        if contact.dist < params.minimum_distance_from_collisions:
          is_any_left_hand = (
              geom1_name == "left_hand" or geom2_name == "left_hand")
          is_any_left_upperarm = (
              geom1_name == "left_upper_arm" or geom2_name == "left_upper_arm")
          is_any_left_lowerarm = (
              geom1_name == "left_lower_arm" or geom2_name == "left_lower_arm")
          is_any_left_arm = (
              is_any_left_hand or is_any_left_upperarm or is_any_left_lowerarm)
          is_any_floor = (geom1_name == "floor" or geom2_name == "floor")

          self.assertFalse(is_any_left_arm and is_any_floor)

  def test_invalid_parameters_throws(self):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    params = cartesian_6d_to_joint_velocity_mapper.Parameters()
    params.model = physics.model
    params.joint_ids = [19, 20, 21]
    params.object_type = _MjGeom
    params.object_name = "invalid_geom_name"
    params.integration_timestep = 1.0
    params.enable_nullspace_control = True
    with self.assertRaises(Exception):
      _ = cartesian_6d_to_joint_velocity_mapper.Mapper(params)


if __name__ == "__main__":
  absltest.main()
