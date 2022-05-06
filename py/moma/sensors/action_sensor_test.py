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

"""Tests for action_sensor."""

from absl.testing import absltest
from dm_env import specs
from dm_robotics.moma import effector as moma_effector
from dm_robotics.moma.sensors import action_sensor
import numpy as np

_DOFS = 7


class _FooEffector(moma_effector.Effector):

  def __init__(self):
    pass

  def initialize_episode(self, physics, random_state) -> None:
    pass

  def action_spec(self, physics) -> specs.BoundedArray:
    return specs.BoundedArray(
        shape=(_DOFS,), dtype=np.float64,
        minimum=np.ones((_DOFS,)) * -1.0, maximum=np.ones((_DOFS,)))

  def set_control(self, physics, command: np.ndarray) -> None:
    pass

  @property
  def prefix(self) -> str:
    return 'foo_effector'

  @property
  def previous_action(self) -> np.ndarray:
    return self._previous_action


class ActionSensorTest(absltest.TestCase):

  def test_observation_name(self):
    effector = _FooEffector()
    effector, sensor = action_sensor.create_sensed_effector(effector)
    expected_obs_key = 'foo_effector_previous_action'
    obs_key = sensor.get_obs_key(
        action_sensor.Observations.PREVIOUS_ACTION)
    self.assertEqual(expected_obs_key, obs_key)

  def test_read_previous_action(self):
    effector = _FooEffector()
    effector, sensor = action_sensor.create_sensed_effector(effector)
    command = np.ones(_DOFS) * 0.3
    effector.set_control(None, command)
    obs_key = 'foo_effector_previous_action'

    np.testing.assert_allclose(command, sensor.observables[obs_key](None))


if __name__ == '__main__':
  absltest.main()
