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

"""Tests for robotiq_gripper_sensor."""

from absl.testing import absltest
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.sensors import robotiq_gripper_observations
from dm_robotics.moma.sensors import robotiq_gripper_sensor

# Absolute tolerance parameter.
_A_TOL = 5e-03
# Relative tolerance parameter.
_R_TOL = 0.01


class RobotiqGripperSensorTest(absltest.TestCase):

  def test_sensor_has_all_observables(self):
    name = 'gripper'
    gripper = robotiq_2f85.Robotiq2F85(name=name)
    sensor = robotiq_gripper_sensor.RobotiqGripperSensor(
        gripper=gripper, name=name)
    sensor.initialize_for_task(0.1, 0.001, 100)
    expected_observable_names = set(
        sensor.get_obs_key(obs)
        for obs in robotiq_gripper_observations.Observations)
    actual_observable_names = set(sensor.observables.keys())
    self.assertSameElements(expected_observable_names, actual_observable_names)


if __name__ == '__main__':
  absltest.main()
