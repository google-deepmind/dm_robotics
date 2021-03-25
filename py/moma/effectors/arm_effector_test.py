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

"""Tests for arm_effector."""

from absl.testing import absltest
from dm_control import mjcf
from dm_robotics.moma.effectors import arm_effector
from dm_robotics.moma.models.robots.robot_arms import sawyer
import numpy as np


class ArmEffectorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._arm = sawyer.Sawyer(with_pedestal=False)
    self._physics = mjcf.Physics.from_mjcf_model(self._arm.mjcf_model)

  def test_setting_control(self):
    effector = arm_effector.ArmEffector(
        arm=self._arm, action_range_override=None, robot_name='sawyer')
    joint_command = np.ones(7, dtype=np.float32) * 0.02
    effector.set_control(self._physics, joint_command)
    np.testing.assert_allclose(
        self._physics.bind(self._arm.actuators).ctrl, joint_command)

  def test_action_range_override_affects_action_spec(self):
    effector = arm_effector.ArmEffector(
        arm=self._arm, action_range_override=[(-0.1, 0.1)], robot_name='sawyer')
    action_spec = effector.action_spec(self._physics)
    np.testing.assert_allclose(action_spec.minimum, np.ones(7) * -0.1)
    np.testing.assert_allclose(action_spec.maximum, np.ones(7) * 0.1)


if __name__ == '__main__':
  absltest.main()
