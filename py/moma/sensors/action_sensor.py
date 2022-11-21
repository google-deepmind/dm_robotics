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

"""Sensor to measure the previous command received by an effector."""

import enum
from typing import Dict, Tuple, Optional

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from dm_robotics.moma import effector as moma_effector
from dm_robotics.moma import sensor
import numpy as np


class SpyEffector(moma_effector.Effector):
  """Effector used to cache the last command used by the delegate effector."""

  def __init__(self, effector: moma_effector.Effector):
    self._previous_action: Optional[np.ndarray] = None
    self._delegate_effector = effector

  def after_compile(self, mjcf_model: mjcf.RootElement,
                    physics: mjcf.Physics) -> None:
    self._delegate_effector.after_compile(mjcf_model, physics)

  def initialize_episode(self, physics, random_state) -> None:
    self._previous_action = self.action_spec(physics).minimum
    self._delegate_effector.initialize_episode(physics, random_state)

  def action_spec(self, physics) -> specs.BoundedArray:
    return self._delegate_effector.action_spec(physics)

  def set_control(self, physics, command: np.ndarray) -> None:
    self._previous_action = command[:]
    self._delegate_effector.set_control(physics, command)

  @property
  def prefix(self) -> str:
    return self._delegate_effector.prefix

  def close(self):
    self._delegate_effector.close()

  @property
  def previous_action(self) -> Optional[np.ndarray]:
    return self._previous_action


@enum.unique
class Observations(enum.Enum):
  """Observations exposed by this sensor."""
  # The previous command that was used by the effector.
  PREVIOUS_ACTION = '{}_previous_action'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)


class ActionSensor(sensor.Sensor[Observations]):
  """Tracks the previous command that was received by an effector.

  In most cases, we chain several effectors when controlling our robots. For
  example, we expose a cartesian effector to the agent. This cartesian command
  is changed to a joint command and is passed to a joint effector that actuates
  the robot. This sensor is used to capture the command that is received by the
  joint effector.
  """

  def __init__(self, effector: SpyEffector, name: str):
    """Constructor.

    The ActionSensor should be built using the create_sensed_effector function
    below.

    Args:
      effector: Effector that exposes a `previous_action` property that is read
        by the sensor.
      name: Name of the sensor.
    """
    self._effector = effector
    self._name = name
    self._observables = {
        self.get_obs_key(Observations.PREVIOUS_ACTION): observable.Generic(
            self._previous_action),
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

  def get_obs_key(self, obs: Observations) -> str:
    return obs.get_obs_key(self._name)

  def _previous_action(self, physics: mjcf.Physics):
    if self._effector.previous_action is None:
      self._effector.initialize_episode(physics, None)
    return self._effector.previous_action


def create_sensed_effector(
    effector: moma_effector.Effector
) -> Tuple[moma_effector.Effector, ActionSensor]:
  """Returns the effector and sensor to measure the last command received."""
  new_effector = SpyEffector(effector)
  action_sensor = ActionSensor(new_effector, name=effector.prefix)
  return new_effector, action_sensor
