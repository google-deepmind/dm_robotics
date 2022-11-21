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

"""Sensor used for measuring positions and velocities of a mujoco site."""

import enum
from typing import Dict

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.moma import sensor as moma_sensor
from dm_robotics.moma.sensors import mujoco_utils
from dm_robotics.transformations import transformations as tr
import numpy as np

_MjcfElement = mjcf.element._ElementImpl  # pylint: disable=protected-access


@enum.unique
class Observations(enum.Enum):
  """Observations exposed by this sensor."""
  # The world x,y,z position of the site.
  POS = '{}_pos'
  # The world orientation quaternion of the site.
  QUAT = '{}_quat'
  # The concatenated world pos and quat x,y,z,w,i,j,k of the site.
  POSE = '{}_pose'
  # The world rotation matrix of the site.
  RMAT = '{}_rmat'
  # The Linear+Euler world velocity of the site.
  VEL_WORLD = '{}_vel_world'
  # The local velocity (as a twist) of the site in its own frame.
  VEL_RELATIVE = '{}_vel_relative'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)


class SiteSensor(moma_sensor.Sensor[Observations]):
  """Sensor to measure position and velocity of a mujoco site."""

  def __init__(self, site: _MjcfElement, name: str):

    if site.tag != 'site':
      raise ValueError(
          f'The provided mjcf element: {site} is not a mujoco site it is a '
          f'{site.tag}.')
    self._site = site
    self._name = name

    self._observables = {
        self.get_obs_key(Observations.POS):
            observable.Generic(self._site_pos),
        self.get_obs_key(Observations.QUAT):
            observable.Generic(self._site_quat),
        self.get_obs_key(Observations.RMAT):
            observable.Generic(self._site_rmat),
        self.get_obs_key(Observations.POSE):
            observable.Generic(self._site_pose),
        self.get_obs_key(Observations.VEL_WORLD):
            observable.Generic(self._site_vel_world),
        self.get_obs_key(Observations.VEL_RELATIVE):
            observable.Generic(self._site_vel_relative),
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

  def _site_pos(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self._site).xpos  # pytype: disable=attribute-error

  def _site_quat(self, physics: mjcf.Physics) -> np.ndarray:
    quat = tr.mat_to_quat(np.reshape(self._site_rmat(physics), [3, 3]))
    return tr.positive_leading_quat(quat)

  def _site_rmat(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self._site).xmat  # pytype: disable=attribute-error

  def _site_pose(self, physics: mjcf.Physics) -> np.ndarray:
    # TODO(jscholz): rendundant with pos & quat; remove pos & quat?
    return np.concatenate((self._site_pos(physics), self._site_quat(physics)),
                          axis=0)

  def _site_vel_world(self, physics: mjcf.Physics) -> np.ndarray:
    return mujoco_utils.get_site_vel(
        physics, self._site, world_frame=True)  # pytype: disable=attribute-error

  def _site_vel_relative(self, physics: mjcf.Physics) -> np.ndarray:
    return mujoco_utils.get_site_vel(
        physics, self._site, world_frame=False)  # pytype: disable=attribute-error
