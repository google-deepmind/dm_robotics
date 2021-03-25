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

"""Sensor to expose some externally defined value."""

from typing import Dict

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.moma import sensor as moma_sensor
import numpy as np


class ExternalValueSensor(moma_sensor.Sensor):
  """Sensor to expose some externally defined value.

  This can be useful when we need to expose to the agent some value that is
  determined by a non-physical process (eg: the progress of a curriculum).
  """

  def __init__(self, name: str, initial_value: np.ndarray):
    self._name = name
    self._value = initial_value

    self._observables = {
        self._name: observable.Generic(lambda _: self._value),
    }

    for obs in self._observables.values():
      obs.enabled = True

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    return self._observables

  @property
  def name(self) -> str:
    return self._name

  def get_obs_key(self, obs) -> str:
    return self._name

  def set_value(self, value: np.ndarray) -> None:
    if value.shape != self._value.shape:
      raise ValueError('Incompatible value shape')
    self._value = value
