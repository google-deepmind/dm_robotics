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
"""Tests for joint_velocity_filter PyBind11 module."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mujoco
from dm_control.suite import humanoid
from dm_robotics.controllers import joint_velocity_filter
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


class JointVelocityFilterTest(absltest.TestCase):

  def test_parameters_attributes(self):
    params = joint_velocity_filter.Parameters()

    attributes = sorted(
        [attr for attr in dir(params) if not attr.startswith("_")]
    )
    expected_attributes = sorted([
        "model",
        "joint_ids",
        "integration_timestep",
        "enable_joint_position_limits",
        "joint_position_limit_velocity_scale",
        "minimum_distance_from_joint_position_limit",
        "enable_joint_velocity_limits",
        "joint_velocity_magnitude_limits",
        "enable_collision_avoidance",
        "use_minimum_distance_contacts_only",
        "collision_avoidance_normal_velocity_scale",
        "minimum_distance_from_collisions",
        "collision_detection_distance",
        "collision_pairs",
        "check_solution_validity",
        "max_qp_solver_iterations",
        "solution_tolerance",
        "use_adaptive_step_size",
        "log_collision_warnings",
    ])
    self.assertEqual(expected_attributes, attributes)


@parameterized.named_parameters(
    ("use_adaptive_step_size", True),
    ("do_not_use_adaptive_step_size", False),
)
class JointVelocityFilterParameterizedTest(absltest.TestCase):

  def test_filter_attributes(self, use_adaptive_step_size):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    params = joint_velocity_filter.Parameters()
    params.model = physics.model
    params.joint_ids = [19, 20, 21]
    params.integration_timestep = 1.0
    params.use_adaptive_step_size = use_adaptive_step_size
    joint_vel_filter = joint_velocity_filter.JointVelocityFilter(params)

    self.assertTrue(hasattr(joint_vel_filter, "filter_joint_velocities"))

  def test_solution_realizes_target(self, use_adaptive_step_size):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    params = joint_velocity_filter.Parameters()
    params.model = physics.model
    params.joint_ids = [16, 17, 18]
    params.integration_timestep = 1.0
    params.solution_tolerance = 1.0e-15
    params.max_qp_solver_iterations = 300
    params.use_adaptive_step_size = use_adaptive_step_size
    joint_vel_filter = joint_velocity_filter.JointVelocityFilter(params)

    # Set target to a realizable velocity and solve.
    target_velocity = [
        0.0450566,
        0.0199436,
        0.0199436,
    ]
    solution = joint_vel_filter.filter_joint_velocities(
        physics.data, target_velocity
    )
    _set_joint_velocities(physics, params.joint_ids, solution)

    # Realized joint velocity must be within the specified tolerance of the
    # target velocity.
    diff_norm = np.linalg.norm(solution - np.array(target_velocity))
    self.assertLess(diff_norm, params.solution_tolerance)

  def test_solution_with_all_constraints_not_in_collision(
      self, use_adaptive_step_size
  ):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    # Increase collision detection margin for all geoms.
    for i in range(0, physics.model.ngeom):
      physics.model.geom_margin[i] = 0.01

    # Place the humanoid in a position where the left hand can collide with the
    # floor if it moves down.
    physics.data.qpos[2] = 0.3
    mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

    # Set params with collision avoidance and full list of constraints.
    params = joint_velocity_filter.Parameters()
    params.model = physics.model
    params.joint_ids = [19, 20, 21]
    params.integration_timestep = 0.005  # 5ms

    params.enable_joint_position_limits = True
    params.joint_position_limit_velocity_scale = 0.95
    params.minimum_distance_from_joint_position_limit = 0.01  # ~0.5deg.

    params.enable_joint_velocity_limits = True
    params.joint_velocity_magnitude_limits = [0.5, 0.5, 0.5]

    params.enable_collision_avoidance = True
    params.collision_avoidance_normal_velocity_scale = 0.01
    params.minimum_distance_from_collisions = 0.005
    params.collision_detection_distance = 10.0
    params.collision_pairs = [(
        ["left_upper_arm", "left_lower_arm", "left_hand"],
        ["floor"],
    )]

    params.check_solution_validity = True
    params.solution_tolerance = 1e-6
    params.max_qp_solver_iterations = 300
    params.use_adaptive_step_size = use_adaptive_step_size
    joint_vel_filter = joint_velocity_filter.JointVelocityFilter(params)

    # Approximate the distance of the left hand and floor geoms by the
    # difference in Z components minus the radius.
    lhand_radius = physics.named.model.geom_size["left_hand"][0]
    lhand_floor_dist = (
        physics.named.data.geom_xpos["left_hand"][2]
        - physics.named.data.geom_xpos["floor"][2]
        - lhand_radius
    )

    # We make the left hand move down towards the plane.
    # These values were chosen to produce an initial downward motion that later
    # isn't purely downwards but would still hit the floor.
    target_velocity = [-0.02, 0.02, -0.02]

    # Compute velocities and integrate, for 5000 steps.
    for _ in range(0, 5000):
      # Compute joint velocities.
      solution = joint_vel_filter.filter_joint_velocities(
          physics.data, target_velocity
      )

      # Set joint velocities, integrate, and run MuJoCo kinematics.
      _set_joint_velocities(physics, params.joint_ids, solution)
      mjlib.mj_integratePos(
          physics.model.ptr,
          physics.data.qpos,
          physics.data.qvel,
          params.integration_timestep.total_seconds(),
      )
      mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

      # Compute the new distance between the floor and the left hand.
      # We expect the left hand site to get closer to the floor and settle at
      # around <0.006.
      new_lhand_floor_dist = (
          physics.named.data.geom_xpos["left_hand"][2]
          - physics.named.data.geom_xpos["floor"][2]
          - lhand_radius
      )
      self.assertLess(new_lhand_floor_dist, max(0.006, lhand_floor_dist))
      lhand_floor_dist = new_lhand_floor_dist

      # Ensure there is no contact between any left arm geom and the floor.
      for contact in physics.data.contact:
        geom1_name = physics.model.id2name(contact.geom1, _MjGeom)
        geom2_name = physics.model.id2name(contact.geom2, _MjGeom)
        if contact.dist < params.minimum_distance_from_collisions:
          is_any_left_hand = (
              geom1_name == "left_hand" or geom2_name == "left_hand"
          )
          is_any_left_upperarm = (
              geom1_name == "left_upper_arm" or geom2_name == "left_upper_arm"
          )
          is_any_left_lowerarm = (
              geom1_name == "left_lower_arm" or geom2_name == "left_lower_arm"
          )
          is_any_left_arm = (
              is_any_left_hand or is_any_left_upperarm or is_any_left_lowerarm
          )
          is_any_floor = geom1_name == "floor" or geom2_name == "floor"

          self.assertFalse(is_any_left_arm and is_any_floor)

  def test_invalid_parameters_throws(self, use_adaptive_step_size):
    physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())

    params = joint_velocity_filter.Parameters()
    params.model = physics.model
    params.joint_ids = [19, 20, 21]
    params.integration_timestep = -1.0  # Not valid.
    params.use_adaptive_step_size = use_adaptive_step_size
    with self.assertRaises(Exception):
      _ = joint_velocity_filter.JointVelocityFilter(params)


if __name__ == "__main__":
  absltest.main()
