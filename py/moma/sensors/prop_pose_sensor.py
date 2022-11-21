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

"""Sensor for tracking the pose of props in the arena."""

import enum
from typing import Dict, Iterable, List

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.moma import prop as moma_prop
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


class PropPoseSensor(moma_sensor.Sensor[Observations]):
  """Tracks the world pose of an object in the arena."""

  def __init__(self, prop: moma_prop.Prop, name: str):
    self._prop = prop
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

  def get_obs_key(self, obs: Observations) -> str:
    return obs.get_obs_key(self._name)

  def _pose(self, physics: mjcf.Physics) -> np.ndarray:
    pos, quat = self._prop.get_pose(physics)
    return np.append(pos, pose_utils.positive_leading_quat(quat))


def build_prop_pose_sensors(
    props: Iterable[moma_prop.Prop]) -> List[PropPoseSensor]:
  return [PropPoseSensor(prop=p, name=p.name) for p in props]
