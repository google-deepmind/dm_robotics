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

"""Module containing Robotiq FTS300 Sensor."""
import collections

from dm_control import composer
from dm_control import mjcf
from dm_robotics.moma.models import types
from dm_robotics.moma.models import utils as models_utils
from dm_robotics.moma.models.end_effectors.wrist_sensors import robotiq_fts300_constants as consts
import numpy as np

_ROBOTIQ_ASSETS_PATH = 'robots/robotiq/assets'

_ATTACHMENT_SITE = 'ft_sensor_attachment_site'
_FRAME_SITE = 'ft_sensor_frame_site'
_FORCE_SENSOR_NAME = 'ft_sensor_force'
_TORQUE_SENSOR_NAME = 'ft_sensor_torque'

_SensorParams = collections.namedtuple(
    'SensorParams',
    ['force_std', 'torque_std', 'max_abs_force', 'max_abs_torque'])


_COLLISION_KWARGS = [{
    'name': 'base_mount_CollisionGeom',
    'type': 'sphere',
    'pos': '0 0.0 0.015',
    'size': '0.05'
}]

# Dictionary mapping body names to a list of their collision geoms
_COLLISION_GEOMS_DICT = {
    'base_mount': _COLLISION_KWARGS,
}


class RobotiqFTS300(composer.Entity):
  """A class representing Robotiq FTS300 force/torque sensor."""

  _mjcf_root: mjcf.RootElement

  def _build(
      self,
      name: str = 'robotiq_fts300',
  ) -> None:
    """Initializes RobotiqFTS300.

    Args:
      name: The name of this sensor. Used as a prefix in the MJCF name
        attributes.
    """
    self._mjcf_root = mjcf.from_path(consts.XML_PATH)
    self._mjcf_root.model = name

    self._attachment_site = self._mjcf_root.find('site', _ATTACHMENT_SITE)
    self._sensor_frame_site = self._mjcf_root.find('site', _FRAME_SITE)
    self._force_sensor = self._mjcf_root.find('sensor', _FORCE_SENSOR_NAME)
    self._torque_sensor = self._mjcf_root.find('sensor', _TORQUE_SENSOR_NAME)
    self._add_collision_geoms()

  def _add_collision_geoms(self):
    """Add collision geoms."""
    self._collision_geoms = models_utils.attach_collision_geoms(
        self.mjcf_model, _COLLISION_GEOMS_DICT)

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState):
    """Function called at the beginning of every episode."""
    del random_state  # Unused.

    # Apply gravity compensation
    body_elements = self.mjcf_model.find_all('body')
    gravity = np.hstack([physics.model.opt.gravity, [0, 0, 0]])
    physics_bodies = physics.bind(body_elements)
    if physics_bodies is None:
      raise ValueError('Calling physics.bind with bodies returns None.')
    physics_bodies.xfrc_applied[:] = -gravity * physics_bodies.mass[..., None]

  @property
  def force_sensor(self) -> types.MjcfElement:
    return self._force_sensor

  @property
  def torque_sensor(self) -> types.MjcfElement:
    return self._torque_sensor

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    return self._mjcf_root

  @property
  def attachment_site(self) -> types.MjcfElement:
    return self._attachment_site

  @property
  def frame_site(self) -> types.MjcfElement:
    return self._sensor_frame_site

  @property
  def sensor_params(self):
    """`_SensorParams` namedtuple specifying noise and clipping parameters."""
    return _SensorParams(
        # The noise values (zero-mean standard deviation) below were extracted
        # from the manufacturer's datasheet. Whilst torque drift is non-
        # significant as per the manual, force drift (+/-3N over 24h) is not
        # currently modelled.
        force_std=(1.2, 1.2, 0.5),
        torque_std=(0.02, 0.02, 0.12),
        # The absolute force/torque range values below were also extracted from
        # the manufacturer's datasheet.
        max_abs_force=300.,
        max_abs_torque=30.)

  @property
  def collision_geom_group(self):
    collision_geom_group = [
        geom.full_identifier for geom in self._collision_geoms
    ]
    return collision_geom_group
