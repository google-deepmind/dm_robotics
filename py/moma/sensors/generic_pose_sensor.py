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

"""Sensor for exposing a pose."""

import enum
from typing import Callable, Dict

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.geometry import geometry
from dm_robotics.moma import sensor as moma_sensor
from dm_robotics.moma.utils import pose_utils
import numpy as np


@enum.unique
class Observations(enum.Enum):
  """Observations exposed by this sensor."""
  # The pose of the prop, given as [x, y, z, quat_w, quat_x, quat_y, quat_z].
  POSE = '{}_pose'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)


class GenericPoseSensor(moma_sensor.Sensor):
  """Pose sensor to expose a geometry pose.

  This sensor is used to expose an arbitrary pose as an observable of the
  environment.

  An toy example use case would be:
    pos = [1.0, 2.0, 3.0]
    quat = [0.0, 1.0, 0.0, 0.1]
    pose_fn = lambda _: geometry.Pose(position=pos, quaternion=quat)
    sensor = generic_pose_sensor.GenericPoseSensor(pose_fn, name='generic')
  """

  def __init__(self,
               pose_fn: Callable[[mjcf.Physics], geometry.Pose],
               name: str):
    """Initialization.

    Args:
      pose_fn: Callable that takes the physics and returns a geometry.Pose. The
        pose returned by the callable is the observation provided by this
        sensor.
      name: Name of the observed element.
    """
    self._pose_fn = pose_fn
    self._name = name
    self._observables = {
        self.get_obs_key(Observations.POSE): observable.Generic(self._pose),
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

  def get_obs_key(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, obs: Observations) -> str:
    return obs.get_obs_key(self._name)

  def _pose(self, physics: mjcf.Physics) -> np.ndarray:
    pose = self._pose_fn(physics)
    pos, quat = pose.position, pose.quaternion
    return np.append(pos, pose_utils.positive_leading_quat(quat))
