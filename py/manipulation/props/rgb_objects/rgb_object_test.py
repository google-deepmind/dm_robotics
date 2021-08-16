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
"""Tests for rgb_object.py."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.manipulation.props import mesh_object
from dm_robotics.manipulation.props.rgb_objects import rgb_object
import numpy as np


class RgbObjectTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("rgb_v1", rgb_object.PropsVersion.RGB_OBJECTS_V1, 152))
  def test_all_rgb_objects_creation(self, rgb_version, num_objects):
    self.assertLen(rgb_object.PROP_FEATURES[rgb_version].ids_list, num_objects)
    colors = list(rgb_object.DEFAULT_COLOR_SET.values())
    for (i,
         obj_id) in enumerate(rgb_object.PROP_FEATURES[rgb_version].ids_list):
      color = colors[i % len(colors)]
      prop = rgb_object.RgbObjectProp(
          rgb_version=rgb_version, obj_id=obj_id, name=obj_id, color=color)
      self.assertEqual(prop.name, obj_id)
      np.testing.assert_array_equal(prop.color, color)

  @parameterized.named_parameters(
      ("rgb_v1", rgb_object.PropsVersion.RGB_OBJECTS_V1))
  def test_random_props_creation(self, rgb_version):
    for i in range(40):
      obj_id = str(i)
      prop = rgb_object.RandomRgbObjectProp(
          rgb_version=rgb_version, name=obj_id)
      self.assertEqual(prop.name, obj_id)

  def test_triplets_creation(self):
    for (rgb_version, obj_ids) in rgb_object.PROP_SETS.values():
      for obj_id in obj_ids:
        prop = rgb_object.RgbObjectProp(
            rgb_version=rgb_version, obj_id=obj_id, name=obj_id)
        self.assertEqual(prop.name, obj_id)

  @parameterized.named_parameters(
      ("rgb_v1", rgb_object.PropsVersion.RGB_OBJECTS_V1))
  def test_dynamic_triplets_creation(self, rgb_version):
    # Test on a new dictionary.
    names = ["a1", "a2", "a3"]
    (id_list_red, id_list_green, id_list_blue) = [[x] for x in names]
    d = mesh_object.PropSetDict({
        "s1_tuple": (rgb_version, tuple(names)),
        "s2_list": (rgb_version, list(names)),
        "s3_dynamic":
            functools.partial(
                rgb_object.random_set,
                rgb_version=rgb_object.PropsVersion.RGB_OBJECTS_V1,
                id_list_red=id_list_red,
                id_list_green=id_list_green,
                id_list_blue=id_list_blue),
    })
    for v in d.values():
      self.assertSequenceEqual(v[1], names)

  def test_random_prop_sets(self):
    for (rgb_version,
         obj_ids) in rgb_object.RANDOM_PROP_SETS_FUNCTIONS.values():
      for obj_id in obj_ids:
        prop = rgb_object.RgbObjectProp(
            rgb_version=rgb_version, obj_id=obj_id, name=obj_id)
        self.assertEqual(prop.name, obj_id)

  @parameterized.named_parameters(
      ("rgb_v1", rgb_object.PropsVersion.RGB_OBJECTS_V1))
  def test_random_set(self, rgb_version):
    for _ in range(20):
      prop_set = rgb_object.random_set(rgb_version)[1]
      self.assertIsInstance(prop_set, list)
      self.assertLen(prop_set, 3)
    for _ in range(10):
      prop_set = rgb_object.random_set()[1]  # default value of `rgb_version`
      self.assertIsInstance(prop_set, list)
      self.assertLen(prop_set, 3)

  @parameterized.named_parameters(
      ("rgb_v1", rgb_object.PropsVersion.RGB_OBJECTS_V1))
  def test_generated_params(self, rgb_version):
    for obj_id in rgb_object.PROP_FEATURES[rgb_version].ids_list:
      prop = rgb_object.RgbObjectProp(rgb_version=rgb_version, obj_id=obj_id)
      self.assertEqual(prop.object_params.rgb_version, rgb_version)
      generated_params = prop.object_params.generated_params
      generated_param_names = (
          prop.object_params.parametric_object.shape.param_names)
      self.assertEqual(len(generated_params), len(generated_param_names))
      self.assertEqual(
          list(generated_params.keys()), list(generated_param_names))

      # Validity of shape bounds. 2 available methods.
      self.assertTrue(
          prop.object_params.parametric_object.shape.check_instance(
              generated_params))
      param_bounds_validator = prop.object_params.parametric_object.shape_bounds
      self.assertTrue(param_bounds_validator(generated_params))

  @parameterized.named_parameters(
      ("rgb_v1", rgb_object.PropsVersion.RGB_OBJECTS_V1))
  def test_min_max(self, rgb_version):
    (params_min,
     params_max) = rgb_object.RgbObjectParameters.min_max(rgb_version)
    self.assertEqual(params_min["sds"], 4)
    self.assertEqual(params_max["sds"], 10)

  def test_random_set_id_not_in_list_error(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "id_list includes g13 which is not part of "
        "PropsVersion.RGB_OBJECTS_V1"):
      _ = rgb_object.random_set(
          rgb_version=rgb_object.PropsVersion.RGB_OBJECTS_V1,
          id_list=["x2", "m3", "g13"])[1]

  def test_random_set_id_list(self):
    for _ in range(20):
      prop_set = rgb_object.random_set(
          rgb_version=rgb_object.PropsVersion.RGB_OBJECTS_V1,
          id_list_red=["x5"],
          id_list_green=["y3"],
          id_list_blue=["v23"],
      )[1]
      self.assertEqual(set(prop_set), set(["x5", "y3", "v23"]))
    for _ in range(20):
      prop_set = rgb_object.random_set(
          rgb_version=rgb_object.PropsVersion.RGB_OBJECTS_V1,
          id_list_red=rgb_object.RGB_OBJECTS_DIM["6"],
          id_list_green=rgb_object.RGB_OBJECTS_DIM["23"],
          id_list_blue=rgb_object.RGB_OBJECTS_DIM["67"],
      )[1]
      self.assertIn(prop_set[0], rgb_object.RGB_OBJECTS_DIM["6"])
      self.assertIn(prop_set[1], rgb_object.RGB_OBJECTS_DIM["23"])
      self.assertIn(prop_set[2], rgb_object.RGB_OBJECTS_DIM["67"])
    for _ in range(20):
      prop_set = rgb_object.random_set(
          rgb_version=rgb_object.PropsVersion.RGB_OBJECTS_V1,
          id_list=rgb_object.RGB_OBJECTS_DIM["3"],
      )[1]
      self.assertIn(prop_set[0], rgb_object.RGB_OBJECTS_DIM["3"])
      self.assertIn(prop_set[1], rgb_object.RGB_OBJECTS_DIM["3"])
      self.assertIn(prop_set[2], rgb_object.RGB_OBJECTS_DIM["3"])

  def test_random_set_id_list_only_blue(self):
    outside_56_once = False
    for _ in range(20):
      prop_set = rgb_object.random_set(
          rgb_version=rgb_object.PropsVersion.RGB_OBJECTS_V1,
          id_list_blue=rgb_object.RGB_OBJECTS_DIM["56"],
      )[1]
      if prop_set[0] not in rgb_object.RGB_OBJECTS_DIM["56"]:
        outside_56_once = True
      self.assertIn(prop_set[0], rgb_object.RGB_OBJECTS_FULL_SET)
      self.assertIn(prop_set[1], rgb_object.RGB_OBJECTS_FULL_SET)
      self.assertIn(prop_set[2], rgb_object.RGB_OBJECTS_DIM["56"])

    # If the red object is always from dim 56, there is a bug.
    self.assertTrue(outside_56_once)

  def test_object_set_is_locked(self):
    self.assertEqual(
        set(rgb_object.RGB_OBJECTS_FULL_SET),
        set([
            "h57", "e57", "l6", "x58", "f58", "v6", "h26", "l26", "v25", "y58",
            "u56", "l56", "r37", "x5", "l67", "u38", "m35", "u26", "e2", "m25",
            "h3", "m26", "r58", "f38", "e6", "x6", "x3", "y37", "h36", "h5",
            "m2", "v58", "u35", "f6", "x23", "x56", "r2", "f56", "f2", "u57",
            "v35", "h58", "l37", "m38", "r56", "l38", "y5", "e35", "v38", "e5",
            "v3", "v56", "v37", "x35", "h6", "x36", "u37", "u25", "r25", "r5",
            "u6", "y35", "e26", "r3", "x2", "f26", "v5", "l23", "x67", "e3",
            "r6", "u58", "m67", "u67", "y3", "y67", "h2", "l3", "m56", "e36",
            "x37", "u36", "r57", "l35", "h56", "r23", "u5", "e25", "l58", "f67",
            "m6", "l25", "m57", "e58", "h35", "x25", "y6", "l5", "x57", "m58",
            "m23", "h37", "u3", "y56", "v57", "x38", "h25", "u2", "x26", "y26",
            "f23", "e38", "f35", "e37", "v23", "e67", "v2", "y57", "f37", "h23",
            "v26", "y2", "m3", "e23", "f36", "y25", "h67", "r35", "f5", "f57",
            "l57", "h38", "v67", "m37", "y23", "r26", "l2", "y38", "e56", "u23",
            "r38", "f3", "m5", "d2", "b6", "b2", "b3", "b5", "g5", "g6", "g2",
            "g3"
        ]))
    self.assertEqual(
        set(rgb_object.RGB_OBJECTS_TRAIN_SET),
        set([
            "h57",
            "e57",
            "x58",
            "f58",
            "h26",
            "l26",
            "v25",
            "y58",
            "u56",
            "l56",
            "r37",
            "l67",
            "u38",
            "m35",
            "u26",
            "m25",
            "m26",
            "r58",
            "f38",
            "y37",
            "h36",
            "v58",
            "u35",
            "x23",
            "x56",
            "f56",
            "u57",
            "v35",
            "h58",
            "l37",
            "m38",
            "r56",
            "l38",
            "e35",
            "v38",
            "v56",
            "v37",
            "x35",
            "x36",
            "u37",
            "u25",
            "r25",
            "y35",
            "e26",
            "f26",
            "l23",
            "x67",
            "u58",
            "m67",
            "u67",
            "y67",
            "m56",
            "e36",
            "x37",
            "u36",
            "r57",
            "l35",
            "h56",
            "r23",
            "e25",
            "l58",
            "f67",
            "l25",
            "m57",
            "e58",
            "h35",
            "x25",
            "x57",
            "m58",
            "m23",
            "h37",
            "y56",
            "v57",
            "x38",
            "h25",
            "x26",
            "y26",
            "f23",
            "e38",
            "f35",
            "e37",
            "v23",
            "e67",
            "y57",
            "f37",
            "h23",
            "v26",
            "e23",
            "f36",
            "y25",
            "h67",
            "r35",
            "f57",
            "l57",
            "h38",
            "v67",
            "m37",
            "y23",
            "r26",
            "y38",
            "e56",
            "u23",
            "r38",
        ]))


if __name__ == "__main__":
  absltest.main()
