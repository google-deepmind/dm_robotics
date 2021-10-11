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
from dm_robotics.manipulation.props import object_collection
from dm_robotics.manipulation.props.rgb_objects import rgb_object
import numpy as np


class RgbObjectTest(parameterized.TestCase):

  @parameterized.named_parameters(("rgb_v1", rgb_object.V1, 152))
  def test_all_rgb_objects_creation(self, rgb_version, num_objects):
    self.assertLen(rgb_object.PROP_FEATURES[rgb_version].ids, num_objects)
    colors = list(rgb_object.DEFAULT_COLOR_SET.values())
    for (i, obj_id) in enumerate(rgb_object.PROP_FEATURES[rgb_version].ids):
      color = colors[i % len(colors)]
      prop = rgb_object.RgbObjectProp(
          rgb_version=rgb_version, obj_id=obj_id, name=obj_id, color=color)
      self.assertEqual(prop.name, obj_id)
      np.testing.assert_array_equal(prop.color, color)

  @parameterized.named_parameters(("rgb_v1", rgb_object.V1))
  def test_random_props_creation(self, rgb_version):
    for i in range(40):
      obj_id = str(i)
      prop = rgb_object.RandomRgbObjectProp(
          rgb_version=rgb_version, name=obj_id)
      self.assertEqual(prop.name, obj_id)

  def test_triplets_creation(self):
    for prop_triplet in rgb_object.PROP_TRIPLETS.values():
      for obj_id in prop_triplet.ids:
        prop = rgb_object.RgbObjectProp(
            rgb_version=prop_triplet.version, obj_id=obj_id, name=obj_id)
        self.assertEqual(prop.name, obj_id)

  @parameterized.named_parameters(("rgb_v1", rgb_object.V1))
  def test_dynamic_triplets_creation(self, rgb_version):
    # Test on a new dictionary.
    names = ["a1", "a2", "a3"]
    (id_list_red, id_list_green, id_list_blue) = [[x] for x in names]
    d = object_collection.PropSetDict({
        "s1_tuple":
            rgb_object.PropsSetType(rgb_version, tuple(names)),
        "s2_list":
            rgb_object.PropsSetType(rgb_version, list(names)),
        "s3_dynamic":
            functools.partial(
                rgb_object.random_triplet,
                rgb_version=rgb_object.V1,
                id_list_red=id_list_red,
                id_list_green=id_list_green,
                id_list_blue=id_list_blue),
    })
    for v in d.values():
      self.assertSequenceEqual(v.ids, names)

  def test_random_prop_triplet(self):
    for triplet in rgb_object.RANDOM_PROP_TRIPLETS_FUNCTIONS.values():
      for obj_id in triplet.ids:
        prop = rgb_object.RgbObjectProp(
            rgb_version=triplet.version, obj_id=obj_id, name=obj_id)
        self.assertEqual(prop.name, obj_id)

  @parameterized.named_parameters(("rgb_v1", rgb_object.V1))
  def test_random_triplet(self, rgb_version):
    for _ in range(20):
      prop_triplet = rgb_object.random_triplet(rgb_version).ids
      self.assertIsInstance(prop_triplet, list)
      self.assertLen(prop_triplet, 3)
    for _ in range(10):
      prop_triplet = rgb_object.random_triplet().ids
      self.assertIsInstance(prop_triplet, list)
      self.assertLen(prop_triplet, 3)

  @parameterized.named_parameters(("rgb_v1", rgb_object.V1))
  def test_generated_params(self, rgb_version):
    for obj_id in rgb_object.PROP_FEATURES[rgb_version].ids:
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

  @parameterized.named_parameters(("rgb_v1", rgb_object.V1))
  def test_min_max(self, rgb_version):
    (params_min,
     params_max) = rgb_object.RgbObjectParameters.min_max(rgb_version)
    self.assertEqual(params_min["sds"], 4)
    self.assertEqual(params_max["sds"], 10)

  def test_random_set_id_not_in_list_error(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "id_list includes g13 which is not part of "
        "PropsVersion.RGB_OBJECTS_V1"):
      _ = rgb_object.random_triplet(
          rgb_version=rgb_object.V1, id_list=["x2", "m3", "g13"]).ids

  def test_random_triplet_id_list(self):
    for _ in range(20):
      prop_triplet = rgb_object.random_triplet(
          rgb_version=rgb_object.V1,
          id_list_red=["x5"],
          id_list_green=["y3"],
          id_list_blue=["v23"],
      ).ids
      self.assertEqual(set(prop_triplet), set(["x5", "y3", "v23"]))
    for _ in range(20):
      prop_triplet = rgb_object.random_triplet(
          rgb_version=rgb_object.V1,
          id_list_red=rgb_object.RGB_OBJECTS_DIM["6"],
          id_list_green=rgb_object.RGB_OBJECTS_DIM["23"],
          id_list_blue=rgb_object.RGB_OBJECTS_DIM["67"],
      ).ids
      self.assertIn(prop_triplet[0], rgb_object.RGB_OBJECTS_DIM["6"])
      self.assertIn(prop_triplet[1], rgb_object.RGB_OBJECTS_DIM["23"])
      self.assertIn(prop_triplet[2], rgb_object.RGB_OBJECTS_DIM["67"])
    for _ in range(20):
      prop_triplet = rgb_object.random_triplet(
          rgb_version=rgb_object.V1,
          id_list=rgb_object.RGB_OBJECTS_DIM["3"],
      ).ids
      self.assertIn(prop_triplet[0], rgb_object.RGB_OBJECTS_DIM["3"])
      self.assertIn(prop_triplet[1], rgb_object.RGB_OBJECTS_DIM["3"])
      self.assertIn(prop_triplet[2], rgb_object.RGB_OBJECTS_DIM["3"])

  def test_random_triplet_id_list_only_blue(self):
    outside_56_once = False
    for _ in range(20):
      prop_triplet = rgb_object.random_triplet(
          rgb_version=rgb_object.V1,
          id_list_blue=rgb_object.RGB_OBJECTS_DIM["56"],
      ).ids
      if prop_triplet[0] not in rgb_object.RGB_OBJECTS_DIM["56"]:
        outside_56_once = True
      self.assertIn(prop_triplet[0], rgb_object.RGB_OBJECTS_FULL_SET)
      self.assertIn(prop_triplet[1], rgb_object.RGB_OBJECTS_FULL_SET)
      self.assertIn(prop_triplet[2], rgb_object.RGB_OBJECTS_DIM["56"])

    # If the red object is always from dim 56, there is a bug.
    self.assertTrue(outside_56_once)

  def test_object_set_is_locked(self):
    self.assertEqual(
        set(rgb_object.RGB_OBJECTS_FULL_SET),
        set([  # sorted alphabetically.
            "b2", "b3", "b5", "b6",
            "e2", "e23", "e25", "e26", "e3", "e35", "e36", "e37", "e38", "e5",
            "e56", "e57", "e58", "e6", "e67",
            "f2", "f23", "f26", "f3", "f35", "f36", "f37", "f38", "f5", "f56",
            "f57", "f58", "f6", "f67",
            "g2", "g3", "g5", "g6",
            "h2", "h23", "h25", "h26", "h3", "h35", "h36", "h37", "h38", "h5",
            "h56", "h57", "h58", "h6", "h67",
            "l2", "l23", "l25", "l26", "l3", "l35", "l37", "l38", "l5", "l56",
            "l57", "l58", "l6", "l67",
            "m2", "m23", "m25", "m26", "m3", "m35", "m37", "m38", "m5", "m56",
            "m57", "m58", "m6", "m67",
            "r2", "r23", "r25", "r26", "r3", "r35", "r37", "r38", "r5", "r56",
            "r57", "r58", "r6",
            "s0",
            "u2", "u23", "u25", "u26", "u3", "u35", "u36", "u37", "u38", "u5",
            "u56", "u57", "u58", "u6", "u67",
            "v2", "v23", "v25", "v26", "v3", "v35", "v37", "v38", "v5", "v56",
            "v57", "v58", "v6", "v67",
            "x2", "x23", "x25", "x26", "x3", "x35", "x36", "x37", "x38", "x5",
            "x56", "x57", "x58", "x6", "x67",
            "y2", "y23", "y25", "y26", "y3", "y35", "y37", "y38", "y5", "y56",
            "y57", "y58", "y6", "y67"
        ]))
    self.assertEqual(
        set(rgb_object.RGB_OBJECTS_TRAIN_SET),
        set([  # sorted alphabetically.
            "e23", "e25", "e26", "e35", "e36", "e37", "e38", "e56", "e57",
            "e58", "e67",
            "f23", "f26", "f35", "f36", "f37", "f38", "f56", "f57", "f58",
            "f67",
            "h23", "h25", "h26", "h35", "h36", "h37", "h38", "h56", "h57",
            "h58", "h67",
            "l23", "l25", "l26", "l35", "l37", "l38", "l56", "l57", "l58",
            "l67",
            "m23", "m25", "m26", "m35", "m37", "m38", "m56", "m57", "m58",
            "m67",
            "r23", "r25", "r26", "r35", "r37", "r38", "r56", "r57", "r58",
            "u23", "u25", "u26", "u35", "u36", "u37", "u38", "u56", "u57",
            "u58", "u67",
            "v23", "v25", "v26", "v35", "v37", "v38", "v56", "v57", "v58",
            "v67",
            "x23", "x25", "x26", "x35", "x36", "x37", "x38", "x56",
            "x57", "x58", "x67",
            "y23", "y25", "y26", "y35", "y37", "y38", "y56", "y57", "y58",
            "y67",
        ]))
    self.assertEqual(
        set(rgb_object.RGB_OBJECTS_TEST_SET),
        set([  # sorted alphabetically.
            "b2", "b3", "b5", "b6", "g2", "g3", "g5", "g6", "r2", "r3",
            "r5", "r6", "s0"
        ]))
    self.assertEqual(
        set(rgb_object.RGB_OBJECTS_TEST_SET),
        set([  # sorted alphabetically.
            "b2", "b3", "b5", "b6", "g2", "g3", "g5", "g6", "r2", "r3",
            "r5", "r6", "s0",
        ]))


if __name__ == "__main__":
  absltest.main()
