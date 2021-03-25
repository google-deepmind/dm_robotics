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

"""Sensor for a robot arm."""

from typing import Dict

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.moma import sensor as moma_sensor
from dm_robotics.moma.models.robots.robot_arms import robot_arm
from dm_robotics.moma.sensors import joint_observations
import numpy as np


class RobotArmSensor(moma_sensor.Sensor):
  """Robot arm sensor providing joint-related observations."""

  def __init__(self, arm: robot_arm.RobotArm, name: str,
               have_torque_sensors: bool = True):
    self._arm = arm
    self._name = name

    self._observables = {
        self.get_obs_key(joint_observations.Observations.JOINT_POS):
            observable.Generic(self._joint_pos),
        self.get_obs_key(joint_observations.Observations.JOINT_VEL):
            observable.Generic(self._joint_vel),
    }

    if have_torque_sensors:
      obs_key = self.get_obs_key(joint_observations.Observations.JOINT_TORQUES)
      self._observables[obs_key] = observable.Generic(self._joint_torques)

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

  def get_obs_key(self, obs: joint_observations.Observations) -> str:
    return obs.get_obs_key(self._name)

  def _joint_pos(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self._arm.joints).qpos  # pytype: disable=attribute-error

  def _joint_vel(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self._arm.joints).qvel  # pytype: disable=attribute-error

  def _joint_torques(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self._arm.joint_torque_sensors).sensordata[2::3]  # pytype: disable=attribute-error
