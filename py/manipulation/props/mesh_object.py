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
"""A prop originated in mesh files.

A "prop" is any object within an environment which the robot can manipulate. A
`MeshProp` allows users to use objects modeled in CAD within their Mujoco
simulations.
"""
import os
from typing import List, Optional, Sequence, Union

from absl import logging
from dm_control import mjcf
from dm_robotics.manipulation.props.utils import mesh_formats_utils
from dm_robotics.moma import prop

# Internal file import.

# The default value of '1 1 1 1' has the effect of leaving the texture
# unchanged. Refer to http://www.mujoco.org/book/XMLreference.html#material .
DEFAULT_COLOR_RGBA = '1 1 1 1'
MIN_MASS = 0.001
_MUJOCO_SUPPORTED_MESH_TYPES = ('.stl', '.msh')
_DEFAULT_SIZE = 0.005
_DEFAULT_POS = 0
_DEFAULT_FRICTION = (0.5, 0.005, 0.0001)


class MeshProp(prop.Prop):
  """Represents an object originated in XML and meshes."""

  def _build_meshes_from_list(self,
                              mesh_list: Union[List[str],
                                               Sequence[Sequence[float]]],
                              mesh_prefix: str = 'visual') -> int:
    """Creates mesh assets from mesh files.

    Args:
      mesh_list: list of mesh files or pre-loaded meshes.
      mesh_prefix: prefix for asset names.

    Returns:
      Number of processed meshes.
    """
    mesh_idx = 0
    for mesh_source in mesh_list:
      name = 'mesh_%s_%s_%02d' % (mesh_prefix, self.name, mesh_idx)
      if isinstance(mesh_source, str):
        logging.debug('Loading mesh file %s', mesh_source)
        extension = os.path.splitext(mesh_source)[1]
        if extension in _MUJOCO_SUPPORTED_MESH_TYPES:
          with open(mesh_source, 'rb') as f:
            self._mjcf_root.asset.add(
                'mesh',
                name=name,
                scale=self._size,
                file=mjcf.Asset(f.read(), extension))
          mesh_idx += 1
        elif extension == '.obj':
          msh_strings = mesh_formats_utils.obj_file_to_mujoco_msh(mesh_source)
          for msh_string in msh_strings:
            self._mjcf_root.asset.add(
                'mesh',
                name=name,
                scale=self._size,
                file=mjcf.Asset(msh_string, '.msh'))
            mesh_idx += 1
        else:
          raise ValueError(f'Unsupported object extension: {extension}')
      else:  # TODO(b/195733842): add tests.
        meshes, faces = mesh_source
        for vertices, face in zip(meshes, faces):
          self._mjcf_root.asset.add(
              'mesh', name=name, scale=self._size, vertex=vertices, face=face)
        mesh_idx += 1
    return mesh_idx

  def _build(self,
             visual_meshes: List[str],
             collision_meshes: Optional[List[str]] = None,
             texture_file: Optional[str] = None,
             name: Optional[str] = 'mesh_object',
             size: Optional[List[float]] = None,
             color: Optional[str] = None,
             pos: Optional[List[float]] = None,
             masses: Optional[List[float]] = None,
             mjcf_model_export_dir: Optional[str] = None) -> None:
    """Creates mesh assets from mesh files.

    Args:
      visual_meshes: list of paths to mesh files for a single asset.
      collision_meshes: list of mesh files to use as collision volumes.
      texture_file: path to the texture file of the mesh.
      name: name of the mesh in MuJoCo.
      size: scaling value for the object size.
      color: an RGBA color in `str` format (from MuJoCo, for example '1 0 0 1'
        for red). A color will overwrite any object texture. `None` (default)
        will either use the texture, if provided, or the default color defined
        in DEFAULT_COLOR_RGBA.
      pos: initial position of the mesh. If not set, defaults to the origin.
      masses: masses of the mesh files.
      mjcf_model_export_dir: directory path where to save the mjcf.model in MJCF
        (XML) format.
    """

    root = mjcf.element.RootElement(model=name)
    root.worldbody.add('body', name='prop_root')
    super()._build(name=name, mjcf_root=root, prop_root='prop_root')

    collision_meshes = collision_meshes or visual_meshes

    self._size = size or [_DEFAULT_SIZE] * 3
    self._pos = pos or [_DEFAULT_POS] * 3
    self._color_to_replace_texture = color
    self._mjcf_model_export_dir = mjcf_model_export_dir

    self._visual_mesh_count = self._build_meshes_from_list(
        visual_meshes, mesh_prefix='visual')
    self._collision_mesh_count = self._build_meshes_from_list(
        collision_meshes, mesh_prefix='collision')

    self._visual_dclass = self._mjcf_root.default.add(
        'default', dclass='visual')
    self._visual_dclass.geom.group = 1
    self._visual_dclass.geom.conaffinity = 0
    self._visual_dclass.geom.contype = 0
    if not masses:
      self._visual_dclass.geom.mass = MIN_MASS
    self._visual_dclass.geom.rgba = (
        DEFAULT_COLOR_RGBA if color is None else list(color))

    self._collision_dclass = self.mjcf_model.default.add(
        'default', dclass='collision')
    self._collision_dclass.geom.group = 4
    self._collision_dclass.geom.conaffinity = 1
    self._collision_dclass.geom.contype = 1
    self._collision_dclass.geom.solref = (.004, 1)
    self._collision_dclass.geom.condim = 6
    self._collision_dclass.geom.friction = _DEFAULT_FRICTION
    if not masses:
      self._collision_dclass.geom.mass = 0.2 / self._collision_mesh_count
    self._masses = masses

    self._bbox_coords_axisp = [[
        -self._size[0] * 0.5, -self._size[0] * 0.5, -self._size[0] * 0.5
    ], [self._size[0] * 0.5, self._size[0] * 0.5, self._size[0] * 0.5]]

    if texture_file:
      with open(texture_file, 'rb') as f:
        self._mjcf_root.asset.add(
            'texture',
            name='tex_object',
            type='2d',
            file=mjcf.Asset(f.read(), '.png'))
        self._main_mat = self._mjcf_root.asset.add(
            'material', name='mat_texture', texture='tex_object')

    self._make_model()

  def _add(self, kind, parent=None, **kwargs):
    parent = parent or self._mjcf_root.worldbody
    result = parent.add(kind, **kwargs)
    return result

  def _add_geom(self, parent=None, **kwargs):
    return self._add(kind='geom', parent=parent, **kwargs)

  def _make_model(self):
    # make visual geoms
    for i in range(self._visual_mesh_count):
      geom_name = 'mesh_%s_%02d_visual' % (self.name, i)
      mesh_ref = 'mesh_visual_%s_%02d' % (self.name, i)
      if self._color_to_replace_texture:
        self._add_geom(  # 'color' is used for visual mesh.
            name=geom_name,
            type='mesh',
            mesh=mesh_ref,
            pos=self._pos,
            dclass=self._visual_dclass,
            rgba=self._color_to_replace_texture)
      else:  # textured material will be used instead of color.
        self._add_geom(
            name=geom_name,
            type='mesh',
            mesh=mesh_ref,
            pos=self._pos,
            dclass=self._visual_dclass,
            material='mat_texture')

    # make collision geoms
    for i in range(self._collision_mesh_count):
      geom = self._add_geom(
          name='mesh_%s_%02d_collision' % (self.name, i),
          type='mesh',
          mesh='mesh_collision_%s_%02d' % (self.name, i),
          pos=self._pos,
          dclass=self._collision_dclass)
      if self._masses:
        geom.mass = self._masses[i]

    if self._mjcf_model_export_dir:
      mjcf.export_with_assets(self.mjcf_model, self._mjcf_model_export_dir)

  @property
  def color(self):
    return self._visual_dclass.geom.rgba
