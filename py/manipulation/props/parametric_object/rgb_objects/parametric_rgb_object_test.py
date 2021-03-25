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

"""Tests for parametric_rgb_object.py."""
import logging

from absl.testing import absltest
from absl.testing import parameterized

from dm_robotics.manipulation.props.parametric_object.rgb_objects import parametric_rgb_object
from dm_robotics.manipulation.props.parametric_object.rgb_objects import rgb_object_names

RgbVersion = parametric_rgb_object.RgbVersion


class RgbObjectTest(parameterized.TestCase):
  @parameterized.named_parameters(
      ('cylinder', {'sds': 2, 'scx': 50, 'scy': 50, 'scz': 50, 'shr': 0,
                    'shx': 0, 'shy': 0, 'hlw': 0, 'drf': 0}),
      ('cube', {'sds': 4, 'scx': 50, 'scy': 50, 'scz': 50, 'shr': 0,
                'shx': 0, 'shy': 0, 'hlw': 0, 'drf': 0}),
      ('triangle', {'sds': 3, 'scx': 50, 'scy': 50, 'scz': 50, 'shr': 0,
                    'shx': 0, 'shy': 0, 'hlw': 0, 'drf': 0}))

  def test_rgb_shape_init_and_instances(self, shape):
    my_rgb = parametric_rgb_object.RgbObject()
    if not my_rgb.shape.check_instance(shape):
      raise ValueError(f'The provided RGB-object is invalid {shape}.')

  def test_rgb_object(self):
    # Retrieve the RGB-objects specifications creating an instance of the class
    my_rgb = parametric_rgb_object.RgbObject()

    # Checking the shape tuple
    rgb_shape_names = ('sds', 'shr', 'drf', 'hlw', 'shx', 'shy', 'scx', 'scy',
                       'scz')
    self.assertTupleEqual(my_rgb.shape.param_names, rgb_shape_names)
    # Checking the texture tuple
    rgb_texture_tuple = ('r', 'g', 'b')
    self.assertTupleEqual(my_rgb.texture.param_names, rgb_texture_tuple)

    # Checking the shape tuple with iterators
    names = iter(my_rgb.shape.param_names)
    for ni, si in zip(names, rgb_shape_names):
      self.assertEqual(ni, si)

    with self.assertRaises(StopIteration):
      next(names)

  @parameterized.named_parameters(
      ('octagon', {'sds': 8, 'shr': 0, 'drf': 0, 'hlw': 0,
                   'shx': 0, 'shy': 0, 'scx': 50, 'scy': 50, 'scz': 50}),
      ('cube', {'sds': 4, 'shr': 0, 'drf': 0, 'hlw': 0,
                'shx': 0, 'shy': 0, 'scx': 50, 'scy': 50, 'scz': 50}),
      ('triangle', {'sds': 3, 'shr': 0, 'drf': 0, 'hlw': 0,
                    'shx': 0, 'shy': 0, 'scx': 50, 'scy': 50, 'scz': 50}))
  def test_check_instance(self, shape):
    my_rgb = parametric_rgb_object.RgbObject()
    texture = {'r': 0, 'g': 122, 'b': 255}
    if not my_rgb.check_instance(shape, texture):
      raise ValueError(f'The provided RGB-object is invalid {shape}, {texture}')

  def test_param_names(self):
    my_rgb = parametric_rgb_object.RgbObject()
    # Checking the shape tuple
    rgb_shape_tuple = ('sds', 'shr', 'drf', 'hlw', 'shx', 'shy', 'scx', 'scy',
                       'scz')
    self.assertTupleEqual(my_rgb.shape.param_names, rgb_shape_tuple)
    # Checking the texture tuple
    rgb_texture_tuple = ('r', 'g', 'b')
    self.assertTupleEqual(my_rgb.texture.param_names, rgb_texture_tuple)

    # Checking the shape tuple with iterators
    names = iter(my_rgb.shape.param_names)
    for ni, si in zip(names, rgb_shape_tuple):
      self.assertEqual(ni, si)

    with self.assertRaises(StopIteration):
      next(names)

  def test_param_types(self):
    my_rgb = parametric_rgb_object.RgbObject()
    # Checking the types tuple
    reply = my_rgb.shape.check_instance({'sds': 4, 'shr': 0, 'drf': 0,
                                         'hlw': 20, 'shx': 0, 'shy': 0,
                                         'scx': 50, 'scy': 50, 'scz': 50})
    self.assertEqual(True, reply)
    reply = my_rgb.shape.check_instance({'sds': 4.0, 'shr': 0.0, 'drf': 0.0,
                                         'hlw': 20.0, 'shx': 0.0, 'shy': 0.0,
                                         'scx': 50.0, 'scy': 50.0, 'scz': 50.0})
    self.assertEqual(False, reply)

  @parameterized.parameters(
      ('b1', 'sds4_shr0_drf0_hlw20_shx0_shy0_scx50_scy50_scz50'),
      ('b2', 'sds8_shr0_drf0_hlw0_shx0_shy0_scx45_scy45_scz50'),
      ('b3', 'sds4_shr48_drf0_hlw0_shx0_shy0_scx46_scy49_scz63'),
      ('b5', 'sds4_shr0_drf0_hlw0_shx0_shy31_scx50_scy50_scz50'),
      ('b6', 'sds4_shr0_drf0_hlw0_shx0_shy0_scx32_scy48_scz96'),
      ('g1', 'sds4_shr0_drf0_hlw15_shx0_shy0_scx50_scy50_scz50'),
      ('g2', 'sds6_shr0_drf0_hlw0_shx0_shy0_scx46_scy46_scz50'),
      ('g3', 'sds4_shr25_drf0_hlw0_shx0_shy0_scx51_scy51_scz60'),
      ('g5', 'sds4_shr0_drf0_hlw0_shx0_shy20_scx50_scy50_scz50'),
      ('g6', 'sds4_shr0_drf0_hlw0_shx0_shy0_scx40_scy56_scz80'),
      ('r1', 'sds4_shr0_drf0_hlw35_shx0_shy0_scx50_scy50_scz50'),
      ('r2', 'sds10_shr0_drf0_hlw0_shx0_shy0_scx45_scy45_scz50'),
      ('r3', 'sds4_shr75_drf0_hlw0_shx0_shy0_scx41_scy49_scz71'),
      ('r5', 'sds4_shr0_drf0_hlw0_shx0_shy42_scx50_scy50_scz50'),
      ('r6', 'sds4_shr0_drf0_hlw0_shx0_shy0_scx29_scy29_scz150'))
  def test_get_name_v1_0(self, nickname, full_name):
    my_rgb = parametric_rgb_object.RgbObject()
    rgb_objects_versioned_names = rgb_object_names.RgbObjectsNames()
    shape_params = rgb_objects_versioned_names.nicknames[nickname]
    self.assertEqual(my_rgb.shape.get_name(shape_params), full_name)

  def test_v1_3(self):
    my_rgb = parametric_rgb_object.RgbObject(version=RgbVersion.v1_3)
    rgb_objects_versioned_names = rgb_object_names.RgbObjectsNames(
        version=RgbVersion.v1_3)

    for params in rgb_objects_versioned_names.nicknames.values():
      self.assertTrue(my_rgb.shape.check_instance(params))

  def test_versions(self):
    _ = parametric_rgb_object.RgbObject(version=RgbVersion.v1_0)
    _ = parametric_rgb_object.RgbObject(version=RgbVersion.v1_3)

    v1_0 = rgb_object_names.RgbObjectsNames(version=RgbVersion.v1_0)
    v1_3 = rgb_object_names.RgbObjectsNames(version=RgbVersion.v1_3)
    logging.info('Version 1.0 has %s objects', len(v1_0.nicknames))
    logging.info('Version 1.3 has %s objects', len(v1_3.nicknames))

if __name__ == '__main__':
  absltest.main()
