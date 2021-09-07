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
"""Tests for ycb_object.py."""

import os
from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.manipulation.props.ycb_objects import ycb_object
import numpy as np

# Internal imports.

_TEST_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    'test_assets')


_COLOR_SET = {
    'RED': [1, 0, 0, 1],
    'GREEN': [0, 1, 0, 1],
    'BLUE': [0, 0, 1, 1]
}


class YcbObjectTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('g3_object_google16k', '077_rubiks_cube'),
      ('g3_object_tsdf', '041_small_marker'))
  def test_ycb_object_creation(self, obj_id):
    obj_name = f'ycb_{obj_id}'
    prop = ycb_object.YcbProp(
        dataset_path=_TEST_DATASET_PATH,
        obj_id=obj_id, name=obj_name)
    self.assertEqual(prop.name, obj_name)

  @parameterized.named_parameters(
      ('g3_object_google16k', '077_rubiks_cube'),
      ('g3_object_tsdf', '041_small_marker'))
  def test_colored_object_creation(self, obj_id):
    colors = list(_COLOR_SET.values())
    for color in colors:
      prop = ycb_object.YcbProp(
          dataset_path=_TEST_DATASET_PATH,
          obj_id=obj_id, name=obj_id, color=color)
      self.assertEqual(prop.name, obj_id)
      np.testing.assert_array_equal(prop.color, color)

  def test_metadata(self):
    # Test that some relevant sets are subsets of bigger ones.
    all_ids = set(ycb_object._ALL_OBJECT_PROPERTIES.keys())
    has_google16k_ids = set(ycb_object._HAS_GOOGLE_16K)
    supported_ids = set(ycb_object.OBJECT_IDS)
    self.assertTrue(supported_ids.issubset(all_ids))
    self.assertTrue(has_google16k_ids.issubset(supported_ids))

  def test_static_sets(self):
    for v in ycb_object.PROP_SETS.values():
      obj_list = v[1]
      for obj_id in obj_list:
        is_supported = obj_id in ycb_object.OBJECT_PROPERTIES
        self.assertTrue(is_supported)


if __name__ == '__main__':
  absltest.main()
