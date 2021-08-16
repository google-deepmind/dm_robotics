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

"""Tests for the sawyer.Sawyer class."""
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_robotics.moma.models import types
from dm_robotics.moma.models.robots.robot_arms import sawyer
from dm_robotics.moma.models.robots.robot_arms import sawyer_constants as consts
import numpy as np


_JOINT_ANGLES = np.array(consts.NUM_DOFS * [np.pi])


@parameterized.named_parameters(
    {'testcase_name': 'Integrated_velocity_with_pedestal',
     'actuation': consts.Actuation.INTEGRATED_VELOCITY,
     'with_pedestal': True},
    {'testcase_name': 'Integrated_velocity_without_pedestal',
     'actuation': consts.Actuation.INTEGRATED_VELOCITY,
     'with_pedestal': False},
    )
class SawyerTest(parameterized.TestCase):

  def test_physics_step(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
    physics.step()

  def test_initialize_episode(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
    robot.initialize_episode(physics, np.random.RandomState(3))

  def test_joints(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    self.assertLen(robot.joints, consts.NUM_DOFS)
    for joint in robot.joints:
      self.assertEqual(joint.tag, 'joint')

  def test_actuators(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    self.assertLen(robot.actuators, consts.NUM_DOFS)
    for actuator in robot.actuators:
      if actuation == consts.Actuation.INTEGRATED_VELOCITY:
        self.assertEqual(actuator.tag, 'general')

  def test_mjcf_model(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    self.assertIsInstance(robot.mjcf_model, mjcf.RootElement)

  def test_wrist_site(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    self.assertIsInstance(robot.wrist_site, types.MjcfElement)
    self.assertEqual(robot.wrist_site.tag, 'site')

  def test_joint_torque_sensors(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    self.assertLen(robot.joint_torque_sensors, 7)
    for sensor in robot.joint_torque_sensors:
      self.assertEqual(sensor.tag, 'torque')

  def test_set_joint_angles(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
    robot.set_joint_angles(physics, _JOINT_ANGLES)
    physics_joints_qpos = physics.bind(robot.joints).qpos[:]
    np.testing.assert_array_equal(physics_joints_qpos, _JOINT_ANGLES)

    if actuation == consts.Actuation.INTEGRATED_VELOCITY:
      physics_actuator_act = physics.bind(robot.actuators).act[:]
      np.testing.assert_array_equal(physics_actuator_act, physics_joints_qpos)

  def test_after_substep(self, actuation, with_pedestal):
    robot1 = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    robot2 = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    arena = composer.Arena()

    arena.attach(robot1)
    frame = arena.attach(robot2)
    frame.pos = (1, 0, 0)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    act_limits = consts.JOINT_LIMITS

    actuators1 = physics.bind(robot1.actuators)
    actuators2 = physics.bind(robot2.actuators)

    invalid_range_values = [10., 10., 10., 10., 10., 10., 10.]
    actuators1.act = invalid_range_values
    valid_range_values = [3., 2., 0., 1.7, -1.0, 0.1, -3.2]
    actuators2.act = valid_range_values

    robot1.after_substep(physics, None)
    robot2.after_substep(physics, None)

    # For robot1, actuator state should be clipped.
    np.testing.assert_array_equal(actuators1.act, act_limits['max'])
    # Whilst for robot2, actuator state should not have changed.
    np.testing.assert_array_equal(actuators2.act, valid_range_values)

  def test_collision_geom_group(self, actuation, with_pedestal):
    robot = sawyer.Sawyer(actuation=actuation, with_pedestal=with_pedestal)
    self.assertNotEmpty(robot.collision_geom_group)

if __name__ == '__main__':
  absltest.main()
