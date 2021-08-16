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

"""Sensor for measuring pose and vel of TCP."""

from typing import Dict

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.moma import sensor as moma_sensor
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand
from dm_robotics.moma.sensors import site_sensor
import numpy as np

# Observation enum listing all the observations exposed by the sensor.
Observations = site_sensor.Observations


class RobotTCPSensor(moma_sensor.Sensor):
  """Robot tcp sensor providing measurements of the Tool Center Point."""

  def __init__(self, gripper: robot_hand.AnyRobotHand, name: str):
    self._name = f'{name}_tcp'
    self._gripper_site_sensor = site_sensor.SiteSensor(
        site=gripper.tool_center_point, name=self._name)

    self._observables = self._gripper_site_sensor.observables

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    self._gripper_site_sensor.initialize_episode(physics, random_state)

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    return self._observables

  @property
  def name(self) -> str:
    return self._name

  def get_obs_key(self, obs: Observations) -> str:
    return obs.get_obs_key(self._name)
