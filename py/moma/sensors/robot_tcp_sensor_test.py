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

# Lint as: python3
"""Tests for robot_tcp_sensor."""

from absl.testing import absltest
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.sensors import robot_tcp_sensor


class RobotArmSensorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._gripper = robotiq_2f85.Robotiq2F85(name='gripper')
    self._name = 'gripper'

  def test_sensor_has_all_observables(self):
    sensor = robot_tcp_sensor.RobotTCPSensor(self._gripper, self._name)
    for obs in robot_tcp_sensor.Observations:
      self.assertIn(sensor.get_obs_key(obs), sensor.observables)


if __name__ == '__main__':
  absltest.main()
