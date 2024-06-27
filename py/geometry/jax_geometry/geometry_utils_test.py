# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for geometry.jax.geometry_utils."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_robotics.geometry.jax_geometry import geometry_utils
from jax import numpy as jnp


class GeometryUtilsTest(parameterized.TestCase):

  @chex.all_variants
  def test_bilinear_interpolate(self):
    img = jnp.array([[0., 1.],
                     [1., 2.]])
    interpolate = functools.partial(
        geometry_utils.interpolate, method='bilinear')
    chex.assert_trees_all_close(
        self.variant(interpolate)(img, [0.0, 0.5]).squeeze(), 0.5)
    chex.assert_trees_all_close(
        self.variant(interpolate)(img, [0.0, 0.9]).squeeze(), 0.9)
    chex.assert_trees_all_close(
        self.variant(interpolate)(img, [1.0, 0.5]).squeeze(), 1.5)
    chex.assert_trees_all_close(
        self.variant(interpolate)(img, [0.5, 0.5]).squeeze(), 1.)

  @chex.all_variants
  def test_nearest_interpolate(self):
    img = jnp.array([[0., 1.],
                     [1., 2.]])
    interpolate = functools.partial(
        geometry_utils.interpolate, method='nearest')
    chex.assert_trees_all_close(
        self.variant(interpolate)(img, [0.0, 0.4]).squeeze(), 0.0)
    chex.assert_trees_all_close(
        self.variant(interpolate)(img, [0.0, 0.9]).squeeze(), 1.0)
    chex.assert_trees_all_close(
        self.variant(interpolate)(img, [0.9, 0.1]).squeeze(), 1.0)
    chex.assert_trees_all_close(
        self.variant(interpolate)(img, [0.5, 0.5]).squeeze(), 2.0)

  @chex.all_variants
  def test_interpolate_mask(self):
    mask = jnp.array([[False, False, True, True],
                      [False, False, True, True],
                      [False, False, True, True],
                      [False, False, True, True]])

    max_interpolate = functools.partial(
        geometry_utils.map_neighborhood, fn=jnp.max, size=2)
    min_interpolate = functools.partial(
        geometry_utils.map_neighborhood, fn=jnp.min, size=2)
    or_interpolate = functools.partial(
        geometry_utils.map_neighborhood, fn=jnp.any, size=2)
    and_interpolate = functools.partial(
        geometry_utils.map_neighborhood, fn=jnp.all, size=2)

    self.assertFalse(self.variant(max_interpolate)(mask, [0.1, 0.0]).squeeze())
    self.assertFalse(self.variant(min_interpolate)(mask, [0.1, 0.0]).squeeze())
    self.assertFalse(self.variant(or_interpolate)(mask, [0.1, 0.0]).squeeze())
    self.assertFalse(self.variant(and_interpolate)(mask, [0.1, 0.0]).squeeze())

    self.assertTrue(self.variant(max_interpolate)(mask, [1.1, 1.0]).squeeze())
    self.assertFalse(self.variant(min_interpolate)(mask, [1.1, 1.0]).squeeze())
    self.assertTrue(self.variant(or_interpolate)(mask, [1.1, 1.0]).squeeze())
    self.assertFalse(self.variant(and_interpolate)(mask, [1.1, 1.0]).squeeze())

    self.assertTrue(self.variant(max_interpolate)(mask, [2.1, 1.0]).squeeze())
    self.assertTrue(self.variant(min_interpolate)(mask, [2.1, 1.0]).squeeze())
    self.assertTrue(self.variant(or_interpolate)(mask, [2.1, 1.0]).squeeze())
    self.assertTrue(self.variant(and_interpolate)(mask, [2.1, 1.0]).squeeze())

    # These "interpolation" methods operate over a neighborhood even when
    # queried with an exact integer-valued coordinate. This will cause counter-
    # intuitive behavior in which a query of an exact coordinate won't
    # necessarily give you that value back. We include these tests to highlight
    # this behavior.
    # NOTE: the pixel at 1,1 is False, so all of these "should" be False.
    self.assertTrue(self.variant(max_interpolate)(mask, [1, 1]).squeeze())
    self.assertFalse(self.variant(min_interpolate)(mask, [1, 1]).squeeze())
    self.assertTrue(self.variant(or_interpolate)(mask, [1, 1]).squeeze())
    self.assertFalse(self.variant(and_interpolate)(mask, [1, 1]).squeeze())

if __name__ == '__main__':
  absltest.main()
