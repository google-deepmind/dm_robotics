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

"""JAX normal utilities."""

import dataclasses
import functools
from typing import Union

from dm_robotics.geometry.jax_geometry import basic_types
import jax
from jax import numpy as jnp


@dataclasses.dataclass
class _DelayedPrecision:
  """Workaround to set precision based on platform."""
  precision: float = -1.

  @property
  def value(self):
    if self.precision < 0:
      self.precision = self._get_value()
    return self.precision

  def _get_value(self):
    if jax.devices()[0].platform == 'tpu':
      return 1e-4
    else:
      return 1e-10

# TODO(mdenil): Is this complication really needed?
# The issue is that jax calls cannot be made until after app.run() is called.
# The _DelayedPrecision class delays checking the platform until the first time
# `value` is called, which will be after all the necessary setup has happened.
_EPS = _DelayedPrecision()


def _difference_weight(
    target: jnp.ndarray, ref: jnp.ndarray, alpha: Union[float, jnp.ndarray]
) -> jnp.ndarray:
  """Compute the weighting for a target given a reference."""
  return jnp.exp(-alpha * jnp.mean(jnp.abs(target - ref)))


def points_to_normals(
    point_map: basic_types.Raster,  # (H, W, 3)
    ref: basic_types.Raster,  # (H, W, C)
    alpha: Union[float, jnp.ndarray] = 1.0  # ()
) -> basic_types.Raster:  # (H, W, 3)
  """Estimate normal vectors from pointcloud images.

  Implements the depth-to-normal transform from:

    Yang et al. Unsupervised Learning of Geometry with Edge-aware Depth-Normal
    Consistency. 2018.

  Args:
    point_map: A size `[H, W, 3]` array containing 3d positions for each pixel
      in the image.
    ref: A `[H, W, ...]` array containing per-pixel weighting
      features for the contribution of each pixel pair to the normal estimate.
      For example, this could be RGB values for each pixel in point_map.
      (optional; if `ref` are provided then weight_fn must also be
      specified).
    alpha: A scalar smoothing parameter.

  Returns:
    A size `[H, W, 3]` array of estimated normal vectors for each point.
  """
  # This implementation is slightly different from the formula in Eq. 6 of the
  # reference paper. Their formula normalizes the computed normal only once at
  # the end of the summation, but this implementation normalizes each term
  # individually in addition to normalizing at the end. This makes it so the
  # estimated normal does not depend on the magnitude of the tangent vectors
  # used to compute the component normals.

  # Cut off the outer edge of pixels, where we don't have 8 neighbours.
  p = point_map[1:-1, 1:-1]

  # yx offsets for each pair of points to cross product
  # the second point should be oriented counterclockwise from the first
  offset_pairs = jnp.asarray([
      [(+0, +1), (-1, +0)],
      [(+1, +1), (-1, +1)],
      [(+0, -1), (+1, +0)],
      [(-1, -1), (+1, -1)],
      [(+0, -1), (-1, +0)],
      [(+0, +1), (+1, +0)],
  ], dtype=jnp.int32)

  def _offset_slice(array, offset):
    start_indices = [1 + offset[0], 1 + offset[1], 0]
    slice_sizes = [array.shape[0] - 2, array.shape[1] - 2, array.shape[-1]]
    return jax.lax.dynamic_slice(array, start_indices, slice_sizes)

  weight_fn = jax.vmap(
      jax.vmap(functools.partial(_difference_weight, alpha=alpha))
  )

  def _normal_from_pair(normals, offset_pair):
    p0_offset, p1_offset = offset_pair
    # TODO(mdenil): check sign convention (order of args to cross)
    n = jnp.cross(
        _offset_slice(point_map, p0_offset) - p,
        _offset_slice(point_map, p1_offset) - p)
    n /= (jnp.linalg.norm(n, axis=-1, keepdims=True) + _EPS.value)

    # Apply reference weights.
    p_features = ref[1:-1, 1:-1]
    p0_weights = weight_fn(p_features, _offset_slice(ref, p0_offset))
    p1_weights = weight_fn(p_features, _offset_slice(ref, p1_offset))
    n *= p1_weights[..., jnp.newaxis] * p0_weights[..., jnp.newaxis]

    return normals + n, ()

  normals, _ = jax.lax.scan(
      _normal_from_pair, jnp.zeros_like(point_map[1:-1, 1:-1]), offset_pairs
  )
  normals /= (jnp.linalg.norm(normals, axis=-1, keepdims=True) + _EPS.value)

  # TODO(mdenil): Revisit edge handing.
  # We lost a 1-pixel wide border where there aren't 8 neighbours for computing
  # normals.  This adds back the border by simply copying the outer edge of the
  # valid area.  This ensures the output has the same shape as the input, but
  # copying may not be the best thing to do.
  return jnp.pad(normals, pad_width=((1, 1), (1, 1), (0, 0)), mode='edge')
