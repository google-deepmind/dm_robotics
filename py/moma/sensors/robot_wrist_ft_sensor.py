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

"""Sensor for measuring the wrist force and torque."""

from typing import Dict

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.moma import sensor as moma_sensor
from dm_robotics.moma.sensors import wrench_observations
import numpy as np


class RobotWristFTSensor(moma_sensor.Sensor):
  """Sensor providing force and torque observations of a mujoco ft sensor."""

  def __init__(self, wrist_ft_sensor, name: str):
    """Constructor.

    Args:
      wrist_ft_sensor: Object that exposes:
        - A `force_sensor` property that returns a mujoco force sensor.
        - A `torque_sensor` property that returns a mujoco torque sensor.
        MuJoCo force and torque sensors report forces in the child->parent
        direction. However, the real F/T sensors may report forces with
        Z pointing outwards, that is, in the parent->child direction. These
        should be transformed to a canonical coordinate system either by a
        bespoke Sensor or a TimestepPreprocessor.
      name: The name of the sensor.
    """
    self._wrist_ft_sensor = wrist_ft_sensor
    self._name = name

    self._observables = {
        self.get_obs_key(wrench_observations.Observations.FORCE):
            observable.Generic(self._wrist_force),
        self.get_obs_key(wrench_observations.Observations.TORQUE):
            observable.Generic(self._wrist_torque),
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

  def get_obs_key(self, obs: wrench_observations.Observations) -> str:
    return obs.get_obs_key(self._name)

  def _wrist_force(self, physics: mjcf.Physics) -> np.ndarray:
    force_sensor = self._wrist_ft_sensor.force_sensor
    return physics.bind(force_sensor).sensordata  # pytype: disable=attribute-error

  def _wrist_torque(self, physics: mjcf.Physics) -> np.ndarray:
    torque_sensor = self._wrist_ft_sensor.torque_sensor
    return physics.bind(torque_sensor).sensordata  # pytype: disable=attribute-error
