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

"""Enum to help key into a timestep's observations for the Robotiq gripper."""

import abc
import enum
from typing import Optional

from dm_control import mjcf
from dm_robotics.moma import sensor as moma_sensor
import numpy as np


@enum.unique
class Observations(enum.Enum):
  """Observations exposed by a Robotiq gripper sensor."""
  # The finger position of the gripper.
  POS = '{}_pos'
  # The finger velocity of the gripper.
  VEL = '{}_vel'
  # Whether an object is grasped by the gripper.
  GRASP = '{}_grasp'
  # Health status of the gripper.
  HEALTH_STATUS = '{}_health_status'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)


@enum.unique
class HealthStatus(enum.Enum):
  """Health status reported by the Robotiq gripper.

  In sim, we can assume the gripper is always `READY`, but for real Robotiq
  grippers, we get the health status from the hardware.
  """
  UNKNOWN = 0
  READY = 1
  WARNING = 2
  ERROR = 3
  STALE = 4


class RobotiqGripperSensor(
    moma_sensor.Sensor[Observations], metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def health_status(self, physics: Optional[mjcf.Physics] = None) -> np.ndarray:
    """Returns the health status observation for the robotiq sensor."""
    pass
