# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for cartesian_6d_velocity_effector."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
from dm_robotics.moma.effectors import arm_effector
from dm_robotics.moma.effectors import cartesian_6d_velocity_effector
from dm_robotics.moma.effectors import test_utils
from dm_robotics.moma.models.robots.robot_arms import sawyer
import numpy as np


@parameterized.named_parameters(
    ('use_adaptive_step_size', True),
    ('do_not_use_adaptive_step_size', False),
)
class Cartesian6dVelocityEffectorTest(parameterized.TestCase):

  def test_setting_control(self, use_adaptive_qp_step_size):
    arm = sawyer.Sawyer(with_pedestal=False)
    joints = arm.joints
    element = arm.wrist_site
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    sawyer_effector = arm_effector.ArmEffector(
        arm=arm, action_range_override=None, robot_name='sawyer')

    cartesian_effector = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
        'robot0',
        sawyer_effector,
        cartesian_6d_velocity_effector.ModelParams(element, joints),
        cartesian_6d_velocity_effector.ControlParams(
            control_timestep_seconds=1.0, nullspace_gain=0.0),
        use_adaptive_qp_step_size=use_adaptive_qp_step_size)
    cartesian_effector.after_compile(arm.mjcf_model, physics)

    # Set a Cartesian command that can be tracked at the initial Sawyer
    # configuration, and step physics.
    cartesian_command = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                                 dtype=np.float32)
    # Let the actuators to catch-up to the requested velocity, after which the
    # resultant Cartesian velocity should be within expected tolerances.
    for _ in range(300):
      # Apply gravity compensation at every timestep; the velocity actuators
      # do not work well under gravity.
      _compensate_gravity(physics, arm.mjcf_model)
      cartesian_effector.set_control(physics, cartesian_command)
      physics.step()

    for _ in range(5):
      _compensate_gravity(physics, arm.mjcf_model)
      cartesian_effector.set_control(physics, cartesian_command)
      physics.step()

      twist = physics.data.object_velocity(
          element.full_identifier, 'site', local_frame=False)
      np.testing.assert_allclose(twist[0], cartesian_command[:3], atol=5e-2)
      np.testing.assert_allclose(twist[1], cartesian_command[3:], atol=5e-2)

  def test_control_frame(self, use_adaptive_qp_step_size):
    arm = sawyer.Sawyer(with_pedestal=False)
    joints = arm.joints
    element = arm.wrist_site
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    sawyer_effector = arm_effector.ArmEffector(
        arm=arm, action_range_override=None, robot_name='sawyer')

    # Run the test 10 times with random poses.
    np.random.seed(42)
    for _ in range(10):
      control_frame = geometry.PoseStamped(
          pose=geometry.Pose.from_poseuler(np.random.rand(6)), frame=element)
      cartesian_effector = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
          'robot0',
          sawyer_effector,
          cartesian_6d_velocity_effector.ModelParams(element, joints,
                                                     control_frame),
          cartesian_6d_velocity_effector.ControlParams(
              control_timestep_seconds=1.0,
              max_lin_vel=10.0,
              max_rot_vel=10.0,
              nullspace_gain=0.0),
          use_adaptive_qp_step_size=use_adaptive_qp_step_size)
      cartesian_effector.after_compile(arm.mjcf_model, physics)

      # Create a cartesian command stamped on the control frame, and compute the
      # expected 6D target that is passed to the mapper.
      cartesian_command = np.array([0.3, 0.1, -0.2, 0.4, 0.2, 0.5],
                                   dtype=np.float32)
      stamped_command = geometry.TwistStamped(cartesian_command, control_frame)
      qp_frame = geometry.HybridPoseStamped(
          pose=None,
          frame=element,
          quaternion_override=geometry.PoseStamped(None, None))
      cartesian_6d_target = stamped_command.get_relative_twist(
          qp_frame, mujoco_physics.wrap(physics)).full

      # Wrap the `_compute_joint_velocities` function and ensure that it is
      # called with a Cartesian 6D velocity expressed about the element's
      # origin in world-orientation.
      with mock.patch.object(
          cartesian_6d_velocity_effector.Cartesian6dVelocityEffector,
          '_compute_joint_velocities',
          wraps=cartesian_effector._compute_joint_velocities) as mock_fn:
        cartesian_effector.set_control(physics, cartesian_command)

        # We test the input Cartesian 6D target with a very low tolerance
        # instead of equality, as internally the transformation may be done
        # differently and result in numerically similar but different values.
        self.assertEqual(1, mock_fn.call_count)
        args_kwargs = mock_fn.call_args[1]
        self.assertEqual(args_kwargs['physics'], physics)
        np.testing.assert_allclose(
            args_kwargs['cartesian_6d_target'],
            cartesian_6d_target,
            atol=1.0e-7)

  def test_joint_velocity_limits(self, use_adaptive_qp_step_size):
    arm = sawyer.Sawyer(with_pedestal=False)
    joints = arm.joints
    element = arm.wrist_site
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    sawyer_effector = arm_effector.ArmEffector(
        arm=arm, action_range_override=None, robot_name='sawyer')

    joint_vel_limits = np.ones(7) * 1e-2
    cartesian_effector = (
        cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
            'robot0',
            sawyer_effector,
            cartesian_6d_velocity_effector.ModelParams(element, joints),
            cartesian_6d_velocity_effector.ControlParams(
                control_timestep_seconds=1.0,
                nullspace_gain=0.0,
                max_lin_vel=1e3,
                max_rot_vel=1e3,
                joint_velocity_limits=joint_vel_limits),
            use_adaptive_qp_step_size=use_adaptive_qp_step_size))
    cartesian_effector.after_compile(arm.mjcf_model, physics)

    # Set a very large Cartesian command, and ensure that joint velocity limits
    # are never violated.
    cartesian_command = np.ones(6) * 1e3
    for _ in range(1000):
      _compensate_gravity(physics, arm.mjcf_model)
      cartesian_effector.set_control(physics, cartesian_command)
      joint_vel_ctrls = np.absolute(physics.bind(arm.actuators).ctrl)
      self.assertTrue(np.less_equal(joint_vel_ctrls, joint_vel_limits).all())
      physics.step()

  def test_limiting_to_workspace(self, use_adaptive_qp_step_size):
    arm = sawyer.Sawyer(with_pedestal=False)
    joints = arm.joints
    element = arm.wrist_site
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    effector_6d = test_utils.SpyEffectorWithControlFrame(element, dofs=6)
    sawyer_effector = arm_effector.ArmEffector(
        arm=arm, action_range_override=None, robot_name='sawyer')

    joint_vel_limits = np.ones(7) * 1e-2
    cartesian_effector = (
        cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
            'robot0',
            sawyer_effector,
            cartesian_6d_velocity_effector.ModelParams(element, joints),
            cartesian_6d_velocity_effector.ControlParams(
                control_timestep_seconds=1.0,
                nullspace_gain=0.0,
                max_lin_vel=1e3,
                max_rot_vel=1e3,
                joint_velocity_limits=joint_vel_limits),
            use_adaptive_qp_step_size=use_adaptive_qp_step_size))
    cartesian_effector.after_compile(arm.mjcf_model, physics)
    arm.set_joint_angles(
        physics, joint_angles=test_utils.SAFE_SAWYER_JOINTS_POS)
    # Propagate the changes to the rest of the physics.
    physics.step()

    # The arm is pointing down in front of the base. Create a
    # workspace that encompasses it, and check that all commands are
    # valid.
    min_workspace_limits = np.array([0.0, -0.5, 0.0])
    max_workspace_limits = np.array([0.9, 0.5, 0.5])
    effector_with_limits = (
        cartesian_6d_velocity_effector.limit_to_workspace(
            effector_6d, arm.wrist_site, min_workspace_limits,
            max_workspace_limits))
    effector_with_limits.set_control(physics, command=np.ones(6) * 0.1)
    np.testing.assert_allclose(
        effector_6d.previous_action, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        atol=1e-3,
        rtol=0.0)

    # The arm is pointing down in front of the base. Create a
    # workspace where the X position is in bounds, but Y and Z are out
    # of bounds.
    min_workspace_limits = np.array([0.0, -0.9, 0.5])
    max_workspace_limits = np.array([0.9, -0.5, 0.9])
    effector_with_limits = (
        cartesian_6d_velocity_effector.limit_to_workspace(
            effector_6d, arm.wrist_site, min_workspace_limits,
            max_workspace_limits))
    # The action should only affect DOFs that are out of bounds and
    # are moving away from where they should.
    effector_with_limits.set_control(physics, command=np.ones(6) * 0.1)
    np.testing.assert_allclose(
        effector_6d.previous_action, [0.1, 0.0, 0.1, 0.1, 0.1, 0.1],
        atol=1e-3,
        rtol=0.0)

  def test_limiting_orientation_to_workspace(self, use_adaptive_qp_step_size):
    arm = sawyer.Sawyer(with_pedestal=False)
    joints = arm.joints
    element = arm.wrist_site
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    effector_6d = test_utils.SpyEffectorWithControlFrame(element, dofs=6)
    sawyer_effector = arm_effector.ArmEffector(
        arm=arm, action_range_override=None, robot_name='sawyer')

    joint_vel_limits = np.ones(7) * 1e-2
    cartesian_effector = (
        cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
            'robot0',
            sawyer_effector,
            cartesian_6d_velocity_effector.ModelParams(element, joints),
            cartesian_6d_velocity_effector.ControlParams(
                control_timestep_seconds=1.0,
                nullspace_gain=0.0,
                max_lin_vel=1e3,
                max_rot_vel=1e3,
                joint_velocity_limits=joint_vel_limits),
            use_adaptive_qp_step_size=use_adaptive_qp_step_size))
    cartesian_effector.after_compile(arm.mjcf_model, physics)
    arm.set_joint_angles(
        physics, joint_angles=test_utils.SAFE_SAWYER_JOINTS_POS)
    # Propagate the changes to the rest of the physics.
    physics.step()

    # The arm is pointing down in front of the base. Create a
    # workspace that encompasses it, and check that all commands are
    # valid.
    min_workspace_limits = np.array([0.0, -0.5, 0.0, 0.0, -np.pi, -np.pi])
    max_workspace_limits = np.array([0.9, 0.5, 0.5, 2 * np.pi, np.pi, np.pi])
    effector_with_limits = (
        cartesian_6d_velocity_effector.limit_to_workspace(
            effector_6d, arm.wrist_site, min_workspace_limits,
            max_workspace_limits))
    effector_with_limits.set_control(physics, command=np.ones(6) * 0.1)
    np.testing.assert_allclose(
        effector_6d.previous_action, np.ones(6) * 0.1, atol=1e-3, rtol=0.0)

    # The arm is pointing down in front of the base (x_rot = np.pi). Create a
    # workspace where the Y and Z orientations are in bounds, but X is out of
    # bounds.
    arm.set_joint_angles(
        physics, joint_angles=test_utils.SAFE_SAWYER_JOINTS_POS)
    # Propagate the changes to the rest of the physics.
    physics.step()
    min_workspace_limits = np.array([0., -0.5, 0., -np.pi / 2, -np.pi, -np.pi])
    max_workspace_limits = np.array([0.9, 0.5, 0.5, 0.0, np.pi, np.pi])
    effector_with_limits = (
        cartesian_6d_velocity_effector.limit_to_workspace(
            effector_6d, arm.wrist_site, min_workspace_limits,
            max_workspace_limits))
    # The action should only affect DOFs that are out of bounds and
    # are moving away from where they should.
    effector_with_limits.set_control(physics, command=np.ones(6) * 0.1)
    np.testing.assert_allclose(
        effector_6d.previous_action, [0.1, 0.1, 0.1, 0., 0.1, 0.1],
        atol=1e-3,
        rtol=0.0)

  def test_collision_avoidance(self, use_adaptive_qp_step_size):
    # Add a sphere above the sawyer that it would collide with if it moves up.
    arm = sawyer.Sawyer(with_pedestal=False)
    obstacle = arm.mjcf_model.worldbody.add(
        'geom', type='sphere', pos='0.7 0 0.8', size='0.3')
    move_up_cmd = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    # Make an effector without collision avoidance, and initialize a physics
    # object to be used with this effector.
    unsafe_physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    unsafe_effector = _create_cartesian_effector(
        unsafe_physics.model.opt.timestep,
        arm,
        None,
        use_adaptive_qp_step_size,
        physics=unsafe_physics)

    # Make an effector with collision avoidance, and initialize a physics object
    # to be used with this effector.
    collision_pairs = _self_collision(arm) + _collisions_between(arm, obstacle)
    collision_params = cartesian_6d_velocity_effector.CollisionParams(
        collision_pairs)
    safe_physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    safe_effector = _create_cartesian_effector(safe_physics.model.opt.timestep,
                                               arm, collision_params,
                                               use_adaptive_qp_step_size,
                                               safe_physics)

    # Assert the no-collision avoidance setup collides.
    # This should happen between iterations 250 & 300.
    collided_per_step = _get_collisions_over_n_steps(300, unsafe_physics,
                                                     unsafe_effector,
                                                     move_up_cmd, arm)
    self.assertTrue(all(collided_per_step[250:]))

    # Assert the collision avoidance setup never collides for the same command.
    collided_per_step = _get_collisions_over_n_steps(1000, safe_physics,
                                                     safe_effector, move_up_cmd,
                                                     arm)
    self.assertFalse(any(collided_per_step))


def _compensate_gravity(physics, mjcf_model):
  """Adds fake forces to bodies in physics to compensate gravity."""
  gravity = np.hstack([physics.model.opt.gravity, [0, 0, 0]])
  bodies = physics.bind(mjcf_model.find_all('body'))
  bodies.xfrc_applied = -gravity * bodies.mass[..., None]


def _step_and_check_collisions(physics, mjcf_model, cartesian_effector,
                               cartesian_command):
  """Steps the physics with grav comp. Returns True if in collision."""
  _compensate_gravity(physics, mjcf_model)
  cartesian_effector.set_control(physics, cartesian_command)
  physics.step()
  for contact in physics.data.contact:
    if contact.dist < 0.0:
      return True
  return False


def _get_collisions_over_n_steps(n, physics, effector, command, arm):
  """Returns the result of `_step_and_check_collisions` over `n` steps."""
  collided_per_step = []
  for _ in range(n):
    collided_per_step.append(
        _step_and_check_collisions(physics, arm.mjcf_model, effector, command))
  return collided_per_step


def _self_collision(entity):
  return [(entity.collision_geom_group, entity.collision_geom_group)]


def _collisions_between(lhs, rhs: mjcf.Element):
  return [(lhs.collision_geom_group, [rhs.full_identifier])]


def _create_cartesian_effector(timestep, arm, collision_params,
                               use_adaptive_qp_step_size, physics):
  joint_effector = arm_effector.ArmEffector(
      arm=arm, action_range_override=None, robot_name='sawyer')
  cartesian_effector = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
      'robot0',
      joint_effector,
      cartesian_6d_velocity_effector.ModelParams(arm.wrist_site, arm.joints),
      cartesian_6d_velocity_effector.ControlParams(
          control_timestep_seconds=timestep, nullspace_gain=0.0),
      collision_params,
      use_adaptive_qp_step_size=use_adaptive_qp_step_size)
  cartesian_effector.after_compile(arm.mjcf_model, physics)
  return cartesian_effector


if __name__ == '__main__':
  absltest.main()
