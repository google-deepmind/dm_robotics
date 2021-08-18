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
"""Tests for dm_robotics.manipulation.standard_cell.rgb_basket."""

from absl.testing import absltest

from dm_control import mjcf

from dm_robotics.manipulation.standard_cell import rgb_basket


class RGBBasketTest(absltest.TestCase):

  def test_initialize(self):

    basket = rgb_basket.RGBBasket()

    physics = mjcf.Physics.from_mjcf_model(basket.mjcf_model)
    # Check if we can call step the basket.
    physics.step()

  def test_collision_geom_group_with_primitive_collisions_enabled(self):
    basket = rgb_basket.RGBBasket()
    self.assertNotEmpty(basket.collision_geom_group)


if __name__ == '__main__':
  absltest.main()
