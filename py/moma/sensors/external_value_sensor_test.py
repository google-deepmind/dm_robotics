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

"""Tests for external_value_sensor."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.sensors import external_value_sensor
import numpy as np


class ExternalValueSensorTest(parameterized.TestCase):

  @parameterized.parameters([
      [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
  ])
  def test_external_value_set(self, init_val, set_val):
    arena = empty.Arena()
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    sensor = external_value_sensor.ExternalValueSensor(
        name='test_sensor', initial_value=init_val)

    self.assertEqual('test_sensor', sensor.get_obs_key(None))
    self.assertIn(sensor.get_obs_key(None), sensor.observables)
    cur_value = sensor.observables[sensor.get_obs_key(None)](physics)
    self.assertTrue(np.allclose(cur_value, init_val))

    sensor.set_value(set_val)
    cur_value = sensor.observables[sensor.get_obs_key(None)](physics)
    self.assertTrue(np.allclose(cur_value, set_val))


if __name__ == '__main__':
  absltest.main()
