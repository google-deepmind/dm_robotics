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

"""Tests for constrained_actions_effectors.py."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_env import specs
from dm_robotics.moma import effector
from dm_robotics.moma.effectors import constrained_actions_effectors
from dm_robotics.moma.models.robots.robot_arms import sawyer
import numpy as np


class _SpyEffector(effector.Effector):

  def __init__(self, dofs: int = 7):
    self._previous_action = np.zeros(dofs)

  def after_compile(self, mjcf_model) -> None:
    pass

  def initialize_episode(self, physics, random_state) -> None:
    pass

  def action_spec(self, physics) -> specs.BoundedArray:
    return specs.BoundedArray(
        self._previous_action.shape, self._previous_action.dtype,
        minimum=-1.0, maximum=1.0)

  def set_control(self, physics, command: np.ndarray) -> None:
    self._previous_action = command[:]

  @property
  def prefix(self) -> str:
    return 'spy'

  @property
  def previous_action(self) -> np.ndarray:
    return self._previous_action


class ConstrainedActionsEffectorsTest(parameterized.TestCase):

  def test_joint_position_limits(self):
    fake_joint_effector = _SpyEffector()
    min_joint_limits = -0.1 * np.ones(7)
    max_joint_limits = 0.1 * np.ones(7)
    arm = sawyer.Sawyer(with_pedestal=False)
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    effector_with_limits = constrained_actions_effectors.LimitJointPositions(
        joint_effector=fake_joint_effector, min_joint_limits=min_joint_limits,
        max_joint_limits=max_joint_limits, arm=arm)

    # Set the arm to a valid position and actuate the joints.
    valid_joint_pos = [0., 0., 0., 0., 0., 0., 0.]
    physics.bind(arm.joints).qpos = valid_joint_pos
    expected_action = np.ones(7)
    effector_with_limits.set_control(physics, expected_action)
    np.testing.assert_allclose(fake_joint_effector.previous_action,
                               expected_action)

    # Set the arm below the min limits. The limiter should only clip negative
    # actions. Positive actions are unchanged.
    below_limits_joint_pos = -0.2 * np.ones(7)
    physics.bind(arm.joints).qpos = below_limits_joint_pos
    input_action = np.asarray([-1., -1., -1., -1., 1., 1., 1.])
    expected_action = np.asarray([0., 0., 0., 0., 1., 1., 1.])
    effector_with_limits.set_control(physics, input_action)
    np.testing.assert_allclose(fake_joint_effector.previous_action,
                               expected_action)

    # Set the arm above the min limits. The limiter should only clip positive
    # actions. Negative actions are unchanged.
    above_limits_joint_pos = 0.2 * np.ones(7)
    physics.bind(arm.joints).qpos = above_limits_joint_pos
    input_action = np.asarray([-1., -1., -1., -1., 1., 1., 1.])
    expected_action = np.asarray([-1., -1., -1., -1., 0., 0., 0.])
    effector_with_limits.set_control(physics, input_action)
    np.testing.assert_allclose(fake_joint_effector.previous_action,
                               expected_action)

  def test_state_and_command_have_different_shapes(self):
    # Imagine a joint torque effector that you want to limit based on both
    # the current position AND velocity.
    fake_joint_effector = _SpyEffector()
    min_joint_pos_limits = -0.2 * np.ones(7)
    max_joint_pos_limits = 0.2 * np.ones(7)
    min_joint_vel_limits = -0.1 * np.ones(7)
    max_joint_vel_limits = 0.1 * np.ones(7)
    arm = sawyer.Sawyer(with_pedestal=False)
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)

    # The "state" in this case consists of both the joint position AND velocity.
    def state_getter(p):
      return np.stack((p.bind(arm.joints).qpos,
                       p.bind(arm.joints).qvel), axis=1)
    # The limits should have the same shape and include both pos and vel.
    min_limits = np.stack((min_joint_pos_limits, min_joint_vel_limits), axis=1)
    max_limits = np.stack((max_joint_pos_limits, max_joint_vel_limits), axis=1)
    # And our state checkers need to handle the (pos, vel) states.
    min_checker = lambda st, lim, cmd: np.any(st < lim, axis=1) & (cmd < 0.)
    max_checker = lambda st, lim, cmd: np.any(st > lim, axis=1) & (cmd > 0.)
    constrained_effector = (
        constrained_actions_effectors.ConstrainedActionEffector(
            fake_joint_effector, min_limits, max_limits, state_getter,
            min_checker, max_checker))

    # Set the arm to a valid position and vel and actuate the joints.
    valid_joint_pos = [0., 0., 0., 0., 0., 0., 0.]
    valid_joint_vel = [0., 0., 0., 0., 0., 0., 0.]
    physics.bind(arm.joints).qpos = valid_joint_pos
    physics.bind(arm.joints).qvel = valid_joint_vel
    expected_action = np.ones(7)
    constrained_effector.set_control(physics, expected_action)
    np.testing.assert_allclose(fake_joint_effector.previous_action,
                               expected_action)

    # Set some joints below the min limits. The limiter should only clip
    # negative actions. Positive actions are unchanged.
    joints_pos = [-0.3, -0.3, -0.3, -0.3, 0.0, 0.0, 0.0]
    joints_vel = [-0.2, -0.2, 0.0, 0.0, -0.2, -0.2, 0.0]
    physics.bind(arm.joints).qpos = joints_pos
    physics.bind(arm.joints).qvel = joints_vel
    input_action = np.asarray([-1., 1., -1., 1., -1., 1., -1.])
    expected_action = np.asarray([0., 1., 0., 1., 0., 1., -1.])
    constrained_effector.set_control(physics, input_action)
    np.testing.assert_allclose(fake_joint_effector.previous_action,
                               expected_action)

    # Set some joints above the max limits. The limiter should only clip
    # positive actions. Negative actions are unchanged.
    joints_pos = [0.3, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0]
    joints_vel = [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0]
    physics.bind(arm.joints).qpos = joints_pos
    physics.bind(arm.joints).qvel = joints_vel
    input_action = np.asarray([1., -1., 1., -1., 1., -1., 1.])
    expected_action = np.asarray([0., -1., 0., -1., 0., -1., 1.])
    constrained_effector.set_control(physics, input_action)
    np.testing.assert_allclose(fake_joint_effector.previous_action,
                               expected_action)


if __name__ == '__main__':
  absltest.main()
