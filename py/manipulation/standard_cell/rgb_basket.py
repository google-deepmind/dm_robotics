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
"""RGB basket."""

import os
from typing import List, Optional

from dm_control import composer
from dm_control import mjcf
from dm_robotics.moma.models import types
from dm_robotics.moma.models import utils as models_utils

RESOURCES_ROOT_DIR = (
    os.path.join(os.path.dirname(__file__), 'rgb_basket_assets')
)
RGB_BASKET_XML_PATH = os.path.join(RESOURCES_ROOT_DIR, 'rgb_basket.xml')

DEFAULT_CAMERA_KWARGS = {
    'basket_front_left':
        dict(
            fovy=35.,
            pos=(0.970, -0.375, 0.235),
            quat=(0.7815, 0.4900, 0.2050, 0.3272),
        ),
    'basket_front_right':
        dict(
            fovy=35.,
            pos=(0.970, 0.375, 0.235),
            quat=(0.3272, 0.2050, 0.4900, 0.7815),
        ),
    'basket_back_left':
        dict(
            fovy=30.,
            pos=(0.060, -0.260, 0.390),
            quat=(0.754, 0.373, -0.250, -0.480)),
}

# Define simplified collision geometry for better contacts and motion planning.
_CAMERAS_AND_CAMERA_STRUTS_GEOMS_COLLISIONS_KWARGS = [{
    'name': 'basket_front_left_camera_strut_CollisionGeom',
    'type': 'capsule',
    'fromto': '0.42 -0.42 0.15 0.42 -0.42 0.25',
    'size': '0.06'
}, {
    'name': 'basket_front_left_camera_CollisionGeom',
    'type': 'capsule',
    'fromto': '0.43 -0.43 0.28 0.36 -0.36 0.24',
    'size': '0.04'
}, {
    'name': 'basket_front_right_camera_strut_CollisionGeom',
    'type': 'capsule',
    'fromto': '-0.42 -0.42 0.15 -0.42 -0.42 0.25',
    'size': '0.06'
}, {
    'name': 'basket_front_right_camera_CollisionGeom',
    'type': 'capsule',
    'fromto': '-0.43 -0.43 0.28 -0.36 -0.36 0.24',
    'size': '0.04'
}, {
    'name': 'basket_back_camera_vertical_strut_CollisionGeom',
    'type': 'capsule',
    'fromto': '-0.28 0.57 0.10 -0.28 0.57 0.35',
    'size': '0.06'
}, {
    'name': 'basket_back_camera_horizontal_strut_CollisionGeom',
    'type': 'capsule',
    'fromto': '-0.28 0.57 0.15 -0.28 0.43 0.15',
    'size': '0.06'
}, {
    'name': 'basket_back_camera_CollisionGeom',
    'type': 'capsule',
    'fromto': '-0.285 0.6 0.42 -0.255 0.53 0.36',
    'size': '0.04'
}]

_BASKET_STRUTS_GEOMS_COLLISIONS_KWARGS = [{
    'name': 'basket_back_strut_CollisionGeom',
    'type': 'capsule',
    'fromto': '0.4 0.42 0.15 -0.4 0.42 0.15',
    'size': '0.05'
}, {
    'name': 'basket_front_strut_CollisionGeom',
    'type': 'capsule',
    'fromto': '0.4 -0.42 0.15 -0.4 -0.42 0.15',
    'size': '0.05'
}, {
    'name': 'basket_right_strut_CollisionGeom',
    'type': 'capsule',
    'fromto': '-0.42 0.4 0.15 -0.42 -0.4 0.15',
    'size': '0.05'
}, {
    'name': 'basket_left_strut_CollisionGeom',
    'type': 'capsule',
    'fromto': '0.42 0.4 0.15 0.42 -0.4 0.15',
    'size': '0.05'
}]

_BASKET_SURFACE_GEOMS_COLLISIONS_KWARGS = [
    {
        'name': 'basket_base_surface_CollisionGeom',
        'type': 'box',
        'pos': '0.0 0.0 -0.02',
        'size': '0.25 0.25 0.02'
    },
    {
        'name': 'basket_sloped_side_surfaces_CollisionGeom1',
        'type': 'box',
        'pos': '-0.28 0 0.07',
        'size': '0.20 0.40 0.005',
        'axisangle': '0 1 0 0.44506'  # slope angle 25.5 deg
    },
    {
        'name': 'basket_sloped_side_surfaces_CollisionGeom2',
        'type': 'box',
        'pos': '0.28 0.0 0.07',
        'size': '0.20 0.40 0.005',
        'axisangle': '0 1 0 2.69653'  # slope angle 154.5 deg
    },
    {
        'name': 'basket_sloped_side_surfaces_CollisionGeom3',
        'type': 'box',
        'pos': '0.0 -0.28 0.07',
        'size': '0.40 0.20 0.005',
        'axisangle': '1 0 0 2.69653'  # slope angle 154.5 deg
    },
    {
        'name': 'basket_sloped_side_surfaces_CollisionGeom4',
        'type': 'box',
        'pos': '0.0 0.28 0.07',
        'size': '0.40 0.20 0.005',
        'axisangle': '1 0 0 0.44506'  # slope angle 25.5 deg
    }
]

# Dictionary mapping body names to a list of their collision geoms
_CAMERAS_AND_CAMERA_STRUTS_COLLISION_GEOMS_DICT = {
    'collision_basket': _CAMERAS_AND_CAMERA_STRUTS_GEOMS_COLLISIONS_KWARGS,
}
_BASKET_STRUTS_COLLISION_GEOMS_DICT = {
    'collision_basket': _BASKET_STRUTS_GEOMS_COLLISIONS_KWARGS,
}
_BASKET_SURFACE_COLLISION_GEOMS_DICT = {
    'collision_basket': _BASKET_SURFACE_GEOMS_COLLISIONS_KWARGS,
}


class RGBBasket(composer.Arena):
  """An arena corresponding to the basket used in the RGB setup."""

  def _build(self, name: Optional[str] = None):
    """Initializes this arena.

    Args:
      name: (optional) A string, the name of this arena. If `None`, use the
        model name defined in the MJCF file.
    """
    super()._build(name)
    self._mjcf_root.include_copy(
        mjcf.from_path(RGB_BASKET_XML_PATH), override_attributes=True)
    self._set_to_non_colliding_geoms()
    self._add_cameras()
    self._add_collision_geoms()

  def _set_to_non_colliding_geoms(self):
    """Set mesh geoms of the basket to be non-colliding."""
    for geom in self._mjcf_root.find_all('geom'):
      geom.contype = 0
      geom.conaffinity = 0

  def _add_cameras(self):
    """Add basket cameras."""
    camera_body = self.mjcf_model.find('body', 'rgb_basket').add(
        'body', name='camera_ref', pos='0 0.6 0', quat='0.707 0 0 -0.707')
    self._cameras = []
    for camera_name, mjcf_camera_kwargs in DEFAULT_CAMERA_KWARGS.items():
      self._cameras.append(
          camera_body.add('camera', name=camera_name, **mjcf_camera_kwargs))

  def _add_collision_geoms(self):
    """Add collision geoms."""
    self.mjcf_model.worldbody.add(
        'body', name='collision_basket', pos='0 0 0', quat='0.707 0 0 0.707')
    self._cameras_and_camera_struts_collision_geoms = (
        models_utils.attach_collision_geoms(
            self.mjcf_model, _CAMERAS_AND_CAMERA_STRUTS_COLLISION_GEOMS_DICT))
    self._basket_struts_collision_geoms = models_utils.attach_collision_geoms(
        self.mjcf_model, _BASKET_STRUTS_COLLISION_GEOMS_DICT)
    # Enable collision with the surface collision geoms of the basket.
    collision_geoms_kwargs = models_utils.default_collision_geoms_kwargs()
    collision_geoms_kwargs['contype'] = 1
    collision_geoms_kwargs['conaffinity'] = 1
    collision_geoms_kwargs['rgba'] = '1 1 0 0.3'
    self._basket_surface_collision_geoms = models_utils.attach_collision_geoms(
        self.mjcf_model, _BASKET_SURFACE_COLLISION_GEOMS_DICT,
        collision_geoms_kwargs)

  @property
  def collision_geom_group(self):
    collision_geom_group = (
        self.cameras_and_camera_struts_collision_geom_group +
        self.basket_struts_collision_geom_group +
        self.basket_surface_collision_geom_group)
    return collision_geom_group

  @property
  def cameras_and_camera_struts_collision_geom_group(self):
    collision_geom_group = [
        geom.full_identifier
        for geom in self._cameras_and_camera_struts_collision_geoms
    ]
    return collision_geom_group

  @property
  def basket_struts_collision_geom_group(self):
    collision_geom_group = [
        geom.full_identifier for geom in self._basket_struts_collision_geoms
    ]
    return collision_geom_group

  @property
  def basket_surface_collision_geom_group(self):
    collision_geom_group = [
        geom.full_identifier for geom in self._basket_surface_collision_geoms
    ]
    return collision_geom_group

  @property
  def cameras(self) -> List[types.MjcfElement]:
    """Basket cameras."""
    return self._cameras

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to this basket."""
    return self._mjcf_root
