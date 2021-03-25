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

"""Tests for robot_wrist_ft_sensor."""

from absl.testing import absltest
from dm_robotics.moma.models.end_effectors.wrist_sensors import robotiq_fts300
from dm_robotics.moma.sensors import robot_wrist_ft_sensor
from dm_robotics.moma.sensors import wrench_observations


class RobotWristFTSensorTest(absltest.TestCase):

  def test_sensor_has_all_observables(self):
    wrist_ft_sensor = robotiq_fts300.RobotiqFTS300()
    sensor = robot_wrist_ft_sensor.RobotWristFTSensor(
        wrist_ft_sensor=wrist_ft_sensor, name='ft_sensor')
    for obs in wrench_observations.Observations:
      self.assertIn(sensor.get_obs_key(obs), sensor.observables)


if __name__ == '__main__':
  absltest.main()
