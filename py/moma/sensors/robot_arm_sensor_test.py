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

"""Tests for robot_arm_sensor.py."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_robotics.moma.models.robots.robot_arms import sawyer
from dm_robotics.moma.sensors import joint_observations
from dm_robotics.moma.sensors import robot_arm_sensor
import numpy as np


# Joint configuration that places each joint in the center of its range
_JOINT_QPOS = [0., 0., 0., -1.5, 0., 1.9, 0.]
# Joint velocities that are safe.
_JOINT_QVEL = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# Absolute tolerance parameter.
_A_TOL = 5e-03
# Relative tolerance parameter.
_R_TOL = 0.01


class RobotArmSensorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._arm = sawyer.Sawyer(with_pedestal=False)
    self._name = 'sawyer'
    self._physics = mjcf.Physics.from_mjcf_model(self._arm.mjcf_model)

  def _verify_observable(self, sensor, obs_enum, expected_value):
    observable = sensor.observables[sensor.get_obs_key(obs_enum)]
    np.testing.assert_allclose(observable(self._physics), expected_value,
                               rtol=_R_TOL, atol=_A_TOL)

  @parameterized.named_parameters(('_with_torques', True),
                                  ('_without_torques', False),)
  def test_sensor_has_working_observables(self, torques):
    sensor = robot_arm_sensor.RobotArmSensor(arm=self._arm, name=self._name,
                                             have_torque_sensors=torques)
    self.assertIn(sensor.get_obs_key(joint_observations.Observations.JOINT_POS),
                  sensor.observables)
    self.assertIn(sensor.get_obs_key(joint_observations.Observations.JOINT_VEL),
                  sensor.observables)

    # Check that the observations are giving the correct values.
    expected_joint_angles = _JOINT_QPOS
    self._physics.bind(self._arm.joints).qpos = expected_joint_angles
    self._verify_observable(sensor, joint_observations.Observations.JOINT_POS,
                            expected_joint_angles)
    expected_joint_vel = _JOINT_QVEL
    self._physics.bind(self._arm.joints).qvel = expected_joint_vel
    self._verify_observable(sensor, joint_observations.Observations.JOINT_VEL,
                            expected_joint_vel)

    if torques:
      self.assertIn(
          sensor.get_obs_key(joint_observations.Observations.JOINT_TORQUES),
          sensor.observables)
    else:
      self.assertNotIn(
          sensor.get_obs_key(joint_observations.Observations.JOINT_TORQUES),
          sensor.observables)


if __name__ == '__main__':
  absltest.main()
