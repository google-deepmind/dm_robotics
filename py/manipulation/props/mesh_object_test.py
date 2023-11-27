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
"""Tests for mesh_object.py."""

import os
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.composer import arena
from dm_robotics.manipulation.props import mesh_object
import numpy as np
from PIL import Image

# Internal resources import.

_TEST_ASSETS_PATH = os.path.join(
    os.path.dirname(__file__),
    'utils/test_assets')


def _create_texture(img_size):
  img = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
  return Image.fromarray(img)


def _create_texture_file(texture_filename):
  # create custom texture and save it.
  img_size = [4, 4, 3]
  texture = _create_texture(img_size=img_size)
  texture.save(texture_filename, 'PNG')


class MeshObjectTest(parameterized.TestCase):

  def test_create_object(self):
    mesh_file = os.path.join(_TEST_ASSETS_PATH, 'octahedron.obj')
    prop_name = 'p1'
    prop = mesh_object.MeshProp(name=prop_name, visual_meshes=[mesh_file])
    self.assertEqual(prop.name, prop_name)
    self.assertEmpty(prop.textures)
    _compile_prop(prop)

  def test_create_with_custom_texture(self):
    mesh_file = os.path.join(_TEST_ASSETS_PATH, 'octahedron.obj')

    out_dir = self.create_tempdir().full_path
    texture_file = os.path.join(out_dir, 'tmp.png')
    _create_texture_file(texture_file)

    prop_name_1 = 'p1_custom_texture'
    prop_1 = mesh_object.MeshProp(
        name=prop_name_1, visual_meshes=[mesh_file], texture_file=texture_file)
    self.assertEqual(prop_1.name, prop_name_1)
    texture_1 = prop_1.textures[0]
    self.assertNotEmpty(texture_1)

    prop_name_2 = 'p2_custom_texture'
    prop_2 = mesh_object.MeshProp(
        name=prop_name_2, visual_meshes=[mesh_file], texture_file=texture_file)
    self.assertEqual(prop_2.name, prop_name_2)
    texture_2 = prop_2.textures[0]
    self.assertNotEmpty(texture_2)
    self.assertSequenceEqual(texture_1, texture_2)

    for p in [prop_1, prop_2]:
      _compile_prop(p)


def _compile_prop(prop: mesh_object.MeshProp):
  # Create an empty arena, otherwise the entity becomes the worldbody and
  # has no inertial properties.
  empty_arena = arena.Arena()
  empty_arena.attach(prop)

  # Compile the model.
  return mjcf.Physics.from_mjcf_model(empty_arena.mjcf_model)


if __name__ == '__main__':
  absltest.main()
