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

"""MoMA Models utils."""

from typing import Any, Dict, List, Optional, Union, Sequence

from dm_control import mjcf
from dm_robotics.moma.models import types

# Parameters that are added to every
# collision geom to ensure they are all
# uniform in color, contype/conaffinity, and
# group.
_COLLISION_GEOMS_KWARGS = {
    'element_name': 'geom',
    'contype': '0',
    'conaffinity': '0',
    'rgba': '1 1 1 0.3',
    'group': '5'
}


def default_collision_geoms_kwargs() -> Dict[str, Any]:
  return _COLLISION_GEOMS_KWARGS.copy()


def attach_collision_geoms(
    mjcf_model: mjcf.RootElement,
    collision_geoms_dict: Dict[str, List[Dict[str, str]]],
    shared_kwargs: Optional[Dict[str, str]] = None):
  """Attaches primitive collision geoms as defined by collision_geoms_dict.

  Args:
    mjcf_model: MJCF model on which to attach the collision geoms.
    collision_geoms_dict: Dictionary mapping body names to primitive geom
      parameters. For every collision geom parameter, the shared collision
      parameters specified by shared_kwargs will be added, and a new geom will
      be attached to the respective body name.
    shared_kwargs: Parameters to be shared between all collision geoms.
      If this is None (which is the default value), then the values from
      `default_collision_geoms_kwargs` are used.  An empty dict will result in
      no shared kwargs.

  Returns:
    List of collision geoms attached to their respective bodies.
  """
  if shared_kwargs is None:
    shared_kwargs = default_collision_geoms_kwargs()

  attached_collision_geoms = []
  for body_name, collision_geoms in collision_geoms_dict.items():
    for kwargs in collision_geoms:
      merged_kwargs = kwargs.copy()
      merged_kwargs.update(shared_kwargs)
      attached_collision_geoms.append(
          mjcf_model.find('body', body_name).add(**merged_kwargs))
  return attached_collision_geoms


def binding(physics: mjcf.Physics,
            elements: Union[Sequence[mjcf.Element], mjcf.Element]
            ) -> types.Binding:
  """Binds the elements with physics and returns a non optional object.

  The goal of this function is to return a non optional element so that when
  the physics_elements object is used, there is no pytype error.

  Args:
    physics: The mujoco physics instance.
    elements: The mjcf elements to bind.

  Returns:
    The non optional binding of the elements.
  """
  physics_elements = physics.bind(elements)
  if physics_elements is None:
    raise ValueError(f'Calling physics.bind with {elements} returns None.')
  return physics_elements
