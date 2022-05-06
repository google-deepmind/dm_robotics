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

"""Tests for cartesian_4d_velocity_effector.py."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
from dm_robotics.moma.effectors import cartesian_4d_velocity_effector
from dm_robotics.moma.effectors import test_utils
from dm_robotics.moma.models.robots.robot_arms import sawyer
import numpy as np


class Cartesian4DVelocityEffectorTest(parameterized.TestCase):

  def test_zero_xy_rot_vel_pointing_down(self):
    # When the arm is pointing straight down, the default effector shouldn't
    # apply any X or Y rotations.
    arm = sawyer.Sawyer(with_pedestal=False)
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    effector_6d = test_utils.SpyEffectorWithControlFrame(arm.wrist_site, dofs=6)
    effector_4d = cartesian_4d_velocity_effector.Cartesian4dVelocityEffector(
        effector_6d, element=arm.wrist_site, effector_prefix='sawyer_4d')
    arm.set_joint_angles(
        physics, joint_angles=test_utils.SAFE_SAWYER_JOINTS_POS)
    physics.step()  # propagate the changes to the rest of the physics.

    # Send an XYZ + Z rot command. We shouldn't see any XY rotation components.
    effector_4d.set_control(physics, command=np.ones(4) * 0.1)
    np.testing.assert_allclose(effector_6d.previous_action,
                               [0.1, 0.1, 0.1, 0.0, 0.0, 0.1],
                               atol=1e-3, rtol=0.0)

  def test_nonzero_xy_rot_vel_not_pointing_down(self):
    # When the arm is NOT pointing straight down, the default effector should
    # apply X and Y rotations to push it back to the desired quat.
    arm = sawyer.Sawyer(with_pedestal=False)
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    effector_6d = test_utils.SpyEffectorWithControlFrame(arm.wrist_site, dofs=6)
    effector_4d = cartesian_4d_velocity_effector.Cartesian4dVelocityEffector(
        effector_6d, element=arm.wrist_site, effector_prefix='sawyer_4d')
    # random offset to all joints.
    joint_angles = test_utils.SAFE_SAWYER_JOINTS_POS + 0.1
    arm.set_joint_angles(physics, joint_angles=joint_angles)
    physics.step()  # propagate the changes to the rest of the physics.

    # Send an XYZ + Z rot command. We SHOULD see XY rotation components.
    effector_4d.set_control(physics, command=np.ones(4) * 0.1)
    xy_rot_components = effector_6d.previous_action[3:5]
    self.assertFalse(np.any(np.isclose(xy_rot_components, np.zeros(2))))

  def test_limiting_to_workspace(self):
    arm = sawyer.Sawyer(with_pedestal=False)
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    effector_6d = test_utils.SpyEffectorWithControlFrame(arm.wrist_site, dofs=6)
    effector_4d = cartesian_4d_velocity_effector.Cartesian4dVelocityEffector(
        effector_6d, element=arm.wrist_site, effector_prefix='sawyer_4d')
    arm.set_joint_angles(
        physics, joint_angles=test_utils.SAFE_SAWYER_JOINTS_POS)
    physics.step()  # propagate the changes to the rest of the physics.

    # The arm is pointing down in front of the base. Create a workspace
    # that encompasses it, and check that all commands are valid.
    min_workspace_limits = np.array([0.0, -0.5, 0.0])
    max_workspace_limits = np.array([0.9, 0.5, 0.5])
    effector_with_limits = cartesian_4d_velocity_effector.limit_to_workspace(
        effector_4d, arm.wrist_site, min_workspace_limits, max_workspace_limits)
    effector_with_limits.set_control(physics, command=np.ones(4) * 0.1)
    np.testing.assert_allclose(effector_6d.previous_action,
                               [0.1, 0.1, 0.1, 0.0, 0.0, 0.1],
                               atol=1e-3, rtol=0.0)

    # The arm is pointing down in front of the base. Create a workspace
    # where the X position is in bounds, but Y and Z are out of bounds.
    min_workspace_limits = np.array([0.0, -0.9, 0.5])
    max_workspace_limits = np.array([0.9, -0.5, 0.9])
    effector_with_limits = cartesian_4d_velocity_effector.limit_to_workspace(
        effector_4d, arm.wrist_site, min_workspace_limits, max_workspace_limits)
    # The action should only affect DOFs that are out of bounds and are moving
    # away from where they should.
    effector_with_limits.set_control(physics, command=np.ones(4) * 0.1)
    np.testing.assert_allclose(effector_6d.previous_action,
                               [0.1, 0.0, 0.1, 0.0, 0.0, 0.1],
                               atol=1e-3, rtol=0.0)

  def test_limiting_wrist_rotation(self):
    arm = sawyer.Sawyer(with_pedestal=False)
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    effector_6d = test_utils.SpyEffectorWithControlFrame(arm.wrist_site, dofs=6)
    effector_4d = cartesian_4d_velocity_effector.Cartesian4dVelocityEffector(
        effector_6d, element=arm.wrist_site, effector_prefix='sawyer_4d')
    arm.set_joint_angles(
        physics, joint_angles=test_utils.SAFE_SAWYER_JOINTS_POS)
    physics.step()  # propagate the changes to the rest of the physics.

    # The arm is pointing down in front of the base. Create a workspace
    # that encompasses it.
    min_workspace_limits = np.array([0.0, -0.5, 0.0])
    max_workspace_limits = np.array([0.9, 0.5, 0.5])
    # Provide wrist limits which are outside where the wrist is currently.
    wrist_limits = np.array([
        test_utils.SAFE_SAWYER_JOINTS_POS[-1] + 0.1,
        test_utils.SAFE_SAWYER_JOINTS_POS[-1] + 0.2
    ])
    effector_with_limits = cartesian_4d_velocity_effector.limit_to_workspace(
        effector_4d, arm.wrist_site, min_workspace_limits, max_workspace_limits,
        wrist_joint=arm.joints[-1], wrist_limits=wrist_limits,
        reverse_wrist_range=True)  # For the Sawyer, pos wrist -> neg Z rot.
    effector_with_limits.set_control(physics, command=np.ones(4) * 0.1)
    np.testing.assert_allclose(effector_6d.previous_action,
                               [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                               atol=1e-3, rtol=0.0)

  def test_changing_control_frame(self):
    arm = sawyer.Sawyer(with_pedestal=False)
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    effector_6d = test_utils.SpyEffectorWithControlFrame(arm.wrist_site, dofs=6)
    arm.set_joint_angles(
        physics, joint_angles=test_utils.SAFE_SAWYER_JOINTS_POS)
    physics.step()  # propagate the changes to the rest of the physics

    # The frame the we want to align the z axis of the arm wrist site to.
    target_frame = self._target_frame = geometry.HybridPoseStamped(
        pose=None,
        frame=arm.wrist_site,
        quaternion_override=geometry.PoseStamped(
            pose=geometry.Pose(
                position=None, quaternion=(
                    cartesian_4d_velocity_effector.DOWNFACING_EE_QUAT_WXYZ))))

    # The frame in which the 6d effector expects to receive the velocity
    # command.
    world_orientation_frame = geometry.HybridPoseStamped(
        pose=None,
        frame=arm.wrist_site,
        quaternion_override=geometry.PoseStamped(None, None))

    # Run the test 10 times with random poses.
    np.random.seed(42)
    for _ in range(10):

      # Build the effector with a random control frame.
      control_frame = geometry.PoseStamped(
          pose=geometry.Pose.from_poseuler(np.random.rand(6)), frame=None)
      cartesian_effector = (
          cartesian_4d_velocity_effector.Cartesian4dVelocityEffector(
              effector_6d,
              element=arm.wrist_site,
              control_frame=control_frame,
              effector_prefix='sawyer_4d'))
      cartesian_effector.after_compile(arm.mjcf_model, physics)

      # Create a cartesian command, this command is expressed in the
      # control frame.
      cartesian_command = np.array([0.3, 0.1, -0.2, 0.0, 0.0, 0.5],
                                   dtype=np.float32)

      # Send the command to the effector.
      cartesian_effector.set_control(
          physics=physics, command=np.append(
              cartesian_command[0:3], cartesian_command[5]))

      # Create the twist stamped command
      stamped_command_control_frame = geometry.TwistStamped(
          cartesian_command, control_frame)

      # Get the twist in target frame and remove the x and y rotations.
      # The robot is already pointing downwards, and is therefore aligned with
      # the target frame. This means that we do not expect to have any rotation
      # along the x or y axis of the target frame.
      stamped_command_target_frame = stamped_command_control_frame.to_frame(
          target_frame, mujoco_physics.wrap(physics))
      target_frame_command = copy.copy(stamped_command_target_frame.twist.full)
      target_frame_command[3:5] = np.zeros(2)
      stamped_command_target_frame = stamped_command_target_frame.with_twist(
          target_frame_command)

      # Change the command to the world orientation frame.
      stamped_command_world_orientation_frame = (
          stamped_command_target_frame.to_frame(
              world_orientation_frame, mujoco_physics.wrap(physics)))

      np.testing.assert_allclose(
          effector_6d.previous_action,
          stamped_command_world_orientation_frame.twist.full,
          atol=1e-3)

if __name__ == '__main__':
  absltest.main()
