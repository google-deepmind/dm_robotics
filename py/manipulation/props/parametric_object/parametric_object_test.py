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

"""Tests for parametric_object.py."""
from absl.testing import absltest
from absl.testing import parameterized

from dm_robotics.manipulation.props.parametric_object import parametric_object


class PropertyTest(parameterized.TestCase):

  def test_size_mismatch_in_init(self):
    with self.assertRaises(ValueError):
      _ = parametric_object.ParametricMinMaxBounds({
          'p': [[0, 255]], 'q': [[0, 255]], 'r': [[1, 2, 3]]})

    with self.assertRaises(ValueError):
      _ = parametric_object.ParametricMinMaxBounds({
          'p': [[0, 255]], 'q': [[0, 255]], 'r': [[1]]})

    with self.assertRaises(ValueError):
      _ = parametric_object.ParametricMinMaxBounds({
          'p': [[0, 255]], 'q': [[0, 255]], 'r': [[]]})

    with self.assertRaises(ValueError):
      _ = parametric_object.ParametricMinMaxBounds({
          'p': [[0, 255]], 'q': [[0, 255]], 'r': []})

  def test_check_instance_assertions(self):
    param_names = ('p', 'q', 'r')
    param_check = parametric_object.ParametricMinMaxBounds({
        'p': [[0, 255]], 'q': [[0, 255]], 'r': [[0, 255]]})
    prop = parametric_object.ParametricProperties(param_names, param_check)
    prop.check_instance({'p': 122, 'q': 122, 'r': 122})
    self.assertEqual(prop.param_names, ('p', 'q', 'r'))
    self.assertEqual(len(prop.param_units), len(prop.param_names))
    self.assertEqual(len(prop.param_types), len(prop.param_names))

    reply = prop.check_instance({'p': 500, 'q': 0, 'r': 0})
    self.assertEqual(False, reply)

    reply = prop.check_instance({'p': 0, 'q': -500, 'r': 0})
    self.assertEqual(False, reply)

    param_check = parametric_object.ParametricMinMaxBounds({
        'p': [[0, 255]], 'q': [['p', 'r']], 'r': [[0, 255]]})
    prop = parametric_object.ParametricProperties(param_names, param_check)
    prop.check_instance({'p': 0, 'q': 122, 'r': 255})
    reply = prop.check_instance({'p': 0, 'q': 255, 'r': 122})
    self.assertEqual(False, reply)

    reply = prop.check_instance({'p': 122, 'q': 0, 'r': 255})
    self.assertEqual(False, reply)

    with self.assertRaises(ValueError):
      prop.check_instance({'p': 0, 'q': 255})

    param_names = ('p0', 'p1', 'p2')
    with self.assertRaises(ValueError):
      param_check = parametric_object.ParametricMinMaxBounds({
          'p0': [[0, 255]], 'p1': [[0, 255]], 'p2': [[0, 255]]}).check_instance

  def test_get_dict(self):
    names = ('first', 'second', 'third')
    checks = parametric_object.ParametricMinMaxBounds({
        'first': [[0, 255]],
        'second': [[0, 255]],
        'third': [[0, 255]]})
    prop = parametric_object.ParametricProperties(names, checks)
    _ = prop.get_dict('first0_second0_third0')
    with self.assertRaises(ValueError):
      _ = prop.get_dict('first0_second0')
    with self.assertRaises(ValueError):
      _ = prop.get_dict('first0_second0_fourth0')
    with self.assertRaises(ValueError):
      _ = prop.get_dict('first0_second0_')

  def test_set_types(self):
    names = ('first', 'second', 'third')
    types = {'first': parametric_object.ParametersTypes.INTEGER,
             'second': parametric_object.ParametersTypes.INTEGER,
             'third': parametric_object.ParametersTypes.INTEGER}
    checks = parametric_object.ParametricMinMaxBounds({
        'first': [[0, 255]],
        'second': [[0, 255]],
        'third': [[0, 255]]}, types)
    prop = parametric_object.ParametricProperties(names, checks)
    reply = prop.check_instance({'first': 0, 'second': 255, 'third': 122})
    self.assertEqual(True, reply)
    reply = prop.check_instance({'first': 0.0, 'second': 0.0, 'third': 0.0})
    self.assertEqual(False, reply)

    prop_shape = parametric_object.ParametricProperties(names, checks)
    prop_texture = parametric_object.ParametricProperties(names, checks)
    prop = parametric_object.ParametricObject(prop_shape, prop_texture)
    reply = prop.check_instance({'first': 0, 'second': 255, 'third': 122},
                                {'first': 0, 'second': 255, 'third': 122})
    self.assertEqual(True, reply)

    names = ('first', 'second', 'third')
    types = {'first': parametric_object.ParametersTypes.FLOAT,
             'second': parametric_object.ParametersTypes.FLOAT,
             'third': parametric_object.ParametersTypes.FLOAT}
    checks = parametric_object.ParametricMinMaxBounds({
        'first': [[0, 255]],
        'second': [[0, 255]],
        'third': [[0, 255]]}, types)
    prop = parametric_object.ParametricProperties(names, checks)
    _ = prop.check_instance({'first': 0.0, 'second': 0.0, 'third': 0.0})
    reply = prop.check_instance({'first': 0, 'second': 255, 'third': 122})
    self.assertEqual(False, reply)

    types = {'first': parametric_object.ParametersTypes.FLOAT,
             'second': parametric_object.ParametersTypes.INTEGER,
             'third': parametric_object.ParametersTypes.FLOAT}
    checks = parametric_object.ParametricMinMaxBounds({
        'first': [[0, 255]],
        'second': [[0, 255]],
        'third': [[0, 255]]}, types)
    prop = parametric_object.ParametricProperties(names, checks)
    _ = prop.check_instance({'first': 0.0, 'second': 0, 'third': 0.0})
    reply = prop.check_instance({'first': 0, 'second': 255, 'third': 122})
    self.assertEqual(False, reply)

    names = ('p', 'q', 'r')
    checks = parametric_object.ParametricMinMaxBounds({
        'p': [[0, 255]], 'q': [[0, 255]], 'r': [[0, 255]]})
    prop = parametric_object.ParametricProperties(names, checks)
    _ = prop.check_instance({'p': 0.0, 'q': 0, 'r': 0.0})
    _ = prop.check_instance({'p': 0, 'q': 255.0, 'r': 122})

  def test_parameters_min_max_tuples(self):
    # 0 <= p0, p1, p2 <=1
    # 3 <= p0, p1, p2 <=4
    names = ('first', 'second', 'third')
    checks = parametric_object.ParametricMinMaxBounds({
        'first': [[0, 1], [3, 4]],
        'second': [[0, 1], [3, 4]],
        'third': [[0, 1], [3, 4]]})
    prop = parametric_object.ParametricProperties(names, checks)

    _ = prop.check_instance({'first': 0, 'second': 0, 'third': 0})
    _ = prop.check_instance({'first': 1, 'second': 1, 'third': 1})
    _ = prop.check_instance({'first': 3, 'second': 3, 'third': 3})
    _ = prop.check_instance({'first': 4, 'second': 4, 'third': 4})

    reply = prop.check_instance({'first': 2, 'second': 2, 'third': 2})
    self.assertEqual(False, reply)
    reply = prop.check_instance({'first': 2, 'second': 3, 'third': 3})
    self.assertEqual(False, reply)
    reply = prop.check_instance({'first': 2, 'second': 3, 'third': 2})
    self.assertEqual(False, reply)
    reply = prop.check_instance({'first': 3, 'second': 3, 'third': 2})
    self.assertEqual(False, reply)
    reply = prop.check_instance({'first': 5, 'second': 3, 'third': 3})
    self.assertEqual(False, reply)
    reply = prop.check_instance({'first': 1, 'second': 3, 'third': 3})
    self.assertEqual(False, reply)

    # if a == 2, 0 <= b <= c, 0 <= c <=10
    # if 3 <= a <= 10, 0 <= b <= 10, 0 <= c <=10
    names = ('a', 'b', 'c')
    checks = parametric_object.ParametricMinMaxBounds({
        'a': [[2, 2], [3, 10]],
        'b': [[0, 'c'], [0, 10]],
        'c': [[0, 10], [0, 10]]})
    prop = parametric_object.ParametricProperties(names, checks)

    # if a == 2, 0 <= b <= c, 0 <= c <=10
    # if 3 <= a <= 10, 0 <= b <= 10, 0 <= c <=10
    with self.assertRaises(ValueError):
      checks = parametric_object.ParametricMinMaxBounds({
          'a': [[2, 2], [3, 10]],
          'b': [[0, 'c'], [0, 10]],
          'c': [[0, 10]]})

    _ = prop.check_instance({'a': 2, 'b': 2, 'c': 10})
    _ = prop.check_instance({'a': 3, 'b': 5, 'c': 2})
    reply = prop.check_instance({'a': 2, 'b': 5, 'c': 2})
    self.assertEqual(False, reply)

  def test_add_parametric_dict(self):
    a = parametric_object.ParametersDict({'k1': 1, 'k2': 2})
    b = parametric_object.ParametersDict({'k1': 3, 'k2': 4})
    c = parametric_object.ParametersDict({'k3': 5, 'k4': 6})
    d = parametric_object.ParametersDict({'k1': 7, 'k4': 8})
    r = a + b
    self.assertEqual(r['k1'], 4)
    self.assertEqual(r['k2'], 6)
    with self.assertRaises(TypeError):
      r = a + 1
    with self.assertRaises(ValueError):
      r = a + c
    with self.assertRaises(ValueError):
      r = a + d

  def test_sub_parametric_dict(self):
    a = parametric_object.ParametersDict({'k1': 1, 'k2': 2})
    b = parametric_object.ParametersDict({'k1': 3, 'k2': 4})
    c = parametric_object.ParametersDict({'k3': 5, 'k4': 6})
    d = parametric_object.ParametersDict({'k1': 7, 'k4': 8})
    r = a - b
    self.assertEqual(r['k1'], -2)
    self.assertEqual(r['k2'], -2)
    with self.assertRaises(TypeError):
      r = a - 1
    with self.assertRaises(ValueError):
      r = a - c
    with self.assertRaises(ValueError):
      r = a - d

  def test_mult_parametric_dict(self):
    a = parametric_object.ParametersDict({'k1': 1, 'k2': 2})
    b = parametric_object.ParametersDict({'k1': 3, 'k2': 4})
    r = a * 0.5
    self.assertEqual(r['k1'], int(a['k1']*1/2))
    self.assertEqual(r['k2'], int(a['k2']*1/2))
    with self.assertRaises(TypeError):
      r = a * b
    with self.assertRaises(TypeError):
      r = 0.5 * b

  def test_truediv_parametric_dict(self):
    a = parametric_object.ParametersDict({'k1': 1, 'k2': 2})
    b = parametric_object.ParametersDict({'k1': 3, 'k2': 4})
    r = a // 2
    self.assertEqual(r['k1'], int(a['k1'] // 2))
    self.assertEqual(r['k2'], int(a['k2'] // 2))
    with self.assertRaises(TypeError):
      r = a // b
    with self.assertRaises(TypeError):
      r = 0.5 // b

  def test_types_conversion(self):
    names = ('first', 'second', 'third')
    checks = parametric_object.ParametricMinMaxBounds({
        'first': [[0, 255]],
        'second': [[0, 255]],
        'third': [[0, 255]]})
    prop = parametric_object.ParametricProperties(names, checks)
    dictionary = {'first': 0, 'second': 3, 'third': 2}
    param_dict = parametric_object.ParametersDict(dictionary)
    name = prop.get_name(param_dict)
    self.assertEqual(name, 'first0_second3_third2')

    dictionary = {'first': 0.0, 'second': 0.1, 'third': 2.0}
    param_dict = parametric_object.ParametersDict(dictionary)
    name = prop.get_name(param_dict)
    self.assertEqual(name, 'first0.0_second0.1_third2.0')

    dictionary = {'first': 1.0, 'second': 3.0, 'third': 4}
    param_dict = parametric_object.ParametersDict(dictionary)
    name = prop.get_name(param_dict)
    reconstruction = prop.get_dict(name)
    self.assertEqual(dictionary, reconstruction)

  def test_types_algebra(self):
    dictionary = {'first': 0, 'second': 3, 'third': 2}
    types = (parametric_object.ParametersTypes.INTEGER,)*3
    param_dict = parametric_object.ParametersDict(dictionary, param_types=types)
    param_half = param_dict * 1.1
    self.assertAlmostEqual(param_half['first'], 0)
    self.assertAlmostEqual(param_half['second'], 3)
    self.assertAlmostEqual(param_half['third'], 2)

    types = (parametric_object.ParametersTypes.FLOAT,)*3
    param_dict = parametric_object.ParametersDict(dictionary, param_types=types)
    param_half = param_dict * 1.1
    self.assertAlmostEqual(param_half['first'], 0)
    self.assertAlmostEqual(param_half['second'], 3.3)
    self.assertAlmostEqual(param_half['third'], 2.2)

    dictionary = {'first': 0, 'second': 3, 'third': 2}
    types = (parametric_object.ParametersTypes.INTEGER,)*3
    param_dict = parametric_object.ParametersDict(dictionary, param_types=types)
    param_half = param_dict / 3
    self.assertAlmostEqual(param_half['first'], 0)
    self.assertAlmostEqual(param_half['second'], int(3/3))
    self.assertAlmostEqual(param_half['third'], int(2/3))

    types = (parametric_object.ParametersTypes.FLOAT,)*3
    param_dict = parametric_object.ParametersDict(dictionary, param_types=types)
    param_half = param_dict / 3
    self.assertAlmostEqual(param_half['first'], 0)
    self.assertAlmostEqual(param_half['second'], float(3/3))
    self.assertAlmostEqual(param_half['third'], float(2/3))

if __name__ == '__main__':
  absltest.main()
