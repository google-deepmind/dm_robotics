# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for observation_physics."""

from absl.testing import absltest
from dm_robotics.geometry import geometry
from dm_robotics.geometry import observation_physics
import numpy as np


class ObservationPhysicsTest(absltest.TestCase):

  def test_happy_path(self):
    physics = observation_physics.ObservationPhysics(
        geometry.Pose.from_poseuler)
    raw_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    physics.set_observation({'pose1': np.asarray(raw_data)})
    self.assertEqual(
        physics.world_pose('pose1'), geometry.Pose.from_poseuler(raw_data))

  def test_missing(self):
    physics = observation_physics.ObservationPhysics(
        geometry.Pose.from_poseuler)
    raw_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    physics.set_observation({'pose1': np.asarray(raw_data)})
    with self.assertRaises(ValueError):
      physics.world_pose('pose2')


if __name__ == '__main__':
  absltest.main()
