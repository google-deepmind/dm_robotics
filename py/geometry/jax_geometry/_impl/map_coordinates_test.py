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

"""Tests for geometry._impl.map_coordinates."""

import functools
from typing import Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_robotics.geometry.jax_geometry import projective_geometry as pg
from dm_robotics.geometry.jax_geometry._impl import map_coordinates
import jax
import jax.numpy as jnp
import numpy as np


def get_source_image(image_shape):
  return jax.random.uniform(
      jax.random.PRNGKey(42), image_shape
  )


def _padded_slice(
    operand: chex.Array,
    start_indices: Sequence[chex.Array],
    slice_sizes: Tuple[int, ...],
    cval: chex.Scalar = 0.,
) -> chex.Array:
  """Returns a slice, padding `operand` as necessary."""
  # NOTE: this function is a memory-inefficient alternative to
  #   `map_coordinates.dynamic_slice_with_padding`, since it pads `operand`
  #   rather than the sliced result. Intended usage is only for testing, and
  #   should not be branched to non-test applications.
  pad_config = [
      (np.maximum(0 - start, 0), np.maximum(start + size - shape, 0))
      for start, size, shape in zip(start_indices, slice_sizes, operand.shape)
  ]
  padded = jnp.pad(operand, pad_config, mode='constant', constant_values=cval)
  offset_coord = start_indices + jnp.array([c[0] for c in pad_config])

  return padded[tuple(
      [slice(c, c + l) for c, l in zip(offset_coord, slice_sizes)])]


class DynamicSliceWithPaddingTest(chex.TestCase, parameterized.TestCase):

  @chex.all_variants
  def test_in_bounds_various_shapes(self):
    with self.subTest('Simple'):
      # Simple example for readability.
      image = jnp.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
      expected_result = jnp.array(
          [[0, 0, 0],
           [0, 1, 2]])
      actual_result = map_coordinates.dynamic_slice_with_padding(
          image, (-1, -1), (2, 3), cval=0)
      chex.assert_trees_all_equal(actual_result, expected_result)

    with self.subTest('2DInBounds'):
      self._test_in_bounds_slice(
          operand_shape=(5, 5), slice_sizes=(2, 2), coord=jnp.array([0, 0]))
      self._test_in_bounds_slice(
          operand_shape=(5, 5), slice_sizes=(3, 2), coord=jnp.array([2, 2]))
      self._test_in_bounds_slice(
          operand_shape=(16, 24), slice_sizes=(5, 10), coord=jnp.array([4, 6]))

    with self.subTest('2DOutOfBounds_Underflow'):
      self._test_out_of_bounds_slice(
          operand_shape=(5, 5),
          slice_sizes=(2, 2),
          coord=jnp.array([-1, -1]),
          cval=0.)

    with self.subTest('2DOutOfBounds_Overflow'):
      self._test_out_of_bounds_slice(
          operand_shape=(5, 5),
          slice_sizes=(2, 2),
          coord=jnp.array([4, 4]),
          cval=0.)

    with self.subTest('3DInBounds'):
      self._test_in_bounds_slice(
          operand_shape=(5, 5, 3),
          slice_sizes=(2, 2, 1),
          coord=jnp.array([0, 0, 0]))
      self._test_in_bounds_slice(
          operand_shape=(5, 5, 4),
          slice_sizes=(1, 2, 3),
          coord=jnp.array([2, 2, 1]))
      self._test_in_bounds_slice(
          operand_shape=(6, 6, 2),
          slice_sizes=(3, 2, 0),
          coord=jnp.array([2, 2, 2]))

    with self.subTest('3DOutOfBounds_Underflow'):
      self._test_out_of_bounds_slice(
          operand_shape=(5, 5, 3),
          slice_sizes=(2, 2, 2),
          coord=jnp.array([-1, -1, -1]),
          cval=1)

    with self.subTest('3DOutOfBounds_Overflow'):
      self._test_out_of_bounds_slice(
          operand_shape=(5, 5, 3),
          slice_sizes=(4, 3, 2),
          coord=jnp.array([2, 3, 4]),
          cval=-1)

  def _test_in_bounds_slice(self, operand_shape, slice_sizes, coord):
    # Tests `dynamic_slice_with_padding` against manual slice. Only valid
    # in-bounds.
    operand = jnp.arange(1, 1 + np.prod(operand_shape)).reshape(operand_shape)
    slice_fn = functools.partial(
        map_coordinates.dynamic_slice_with_padding,
        slice_sizes=slice_sizes)
    actual_result = self.variant(slice_fn)(operand, coord)
    expected_result = operand[tuple(
        [slice(c, c + l) for c, l in zip(coord, slice_sizes)])]
    chex.assert_trees_all_equal(actual_result, expected_result)

  def _test_out_of_bounds_slice(self, operand_shape, slice_sizes, coord, cval):
    # Tests `dynamic_slice_with_padding` against `_padded_slice`.
    operand = jnp.arange(1, 1 + np.prod(operand_shape)).reshape(operand_shape)
    slice_fn = functools.partial(
        map_coordinates.dynamic_slice_with_padding,
        slice_sizes=slice_sizes,
        cval=cval)
    actual_result = self.variant(slice_fn)(operand, coord)
    expected_result = _padded_slice(operand, coord, slice_sizes, cval=cval)
    chex.assert_trees_all_equal(actual_result, expected_result)


class MapCoordinatesTest(chex.TestCase, parameterized.TestCase):

  @chex.all_variants
  def test_identity(self):
    height, width = 32, 64

    coords = jnp.flip(pg.raster_coordinates(height, width), axis=-1)
    image = get_source_image((height, width))

    result = self.variant(map_coordinates.map_coordinates)(image, coords)

    chex.assert_trees_all_equal_shapes(image, result)
    chex.assert_trees_all_close(image, result, rtol=1e-3)
    self.assertFalse(jnp.any(jnp.isnan(image)))

  @chex.all_variants
  def test_edges(self):
    height, width = 32, 32

    # (H, W, YX)
    coords = jnp.asarray([[
        # top edge
        [-2., 16.],
        [-0.5, 16.],
        [0., 16.],
        # left edge
        [16., -2.],
        [16., -0.5],
        [16., 0.],
        # bottom edge
        [16., 34.],
        [16., 32.5],
        [16., 32.],
        # right edge
        [16., 34.],
        [16., 32.5],
        [16., 32.],
    ]])

    for method, order in [('nearest', 0), ('bilinear', 1)]:
      map_coordinates_fn = functools.partial(
          map_coordinates.map_coordinates, method=method)

      with self.subTest(f'IntegerType_{method}'):
        image = jnp.arange(1, height * width + 1).reshape((height, width))
        result = self.variant(map_coordinates_fn)(image, coords)
        self.assertFalse(jnp.any(jnp.isnan(result)))
        expected_result = jax.scipy.ndimage.map_coordinates(
            image, coords[0].T, order=order, mode='constant', cval=0.0
        )
        chex.assert_trees_all_close(expected_result.ravel(), result.ravel())

      with self.subTest(f'FloatType_{method}'):
        image = image.astype(jnp.float32)
        result = self.variant(map_coordinates_fn)(image, coords)
        self.assertFalse(jnp.any(jnp.isnan(result)))
        expected_result = jax.scipy.ndimage.map_coordinates(
            image, coords[0].T, order=order, mode='constant', cval=0.0
        )
        chex.assert_trees_all_close(expected_result.ravel(), result.ravel())


if __name__ == '__main__':
  absltest.main()
