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

"""Tests for generic_pose_sensor."""

from absl.testing import absltest
from dm_robotics.geometry import geometry
from dm_robotics.moma.sensors import generic_pose_sensor
import numpy as np


class GenericPoseSensorTest(absltest.TestCase):

  def test_sensor(self):
    pos = [1.0, 2.0, 3.0]
    quat = [0.0, 1.0, 0.0, 0.1]
    pose_fn = lambda _: geometry.Pose(position=pos, quaternion=quat)
    sensor = generic_pose_sensor.GenericPoseSensor(pose_fn, name='generic')

    key = 'generic_pose'
    # Check that the observables is added.
    self.assertIn(key, sensor.observables)

    # Check that the pose is returned.
    sensor_pose = sensor.observables[key](None)
    np.testing.assert_equal(sensor_pose, np.concatenate((pos, quat)))

if __name__ == '__main__':
  absltest.main()
