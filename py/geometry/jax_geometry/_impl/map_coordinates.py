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

"""Custom implementation of map_coordinates.

This is a custom implementation of jax.scipy.ndimage.map_coordinates. It is less
functional than the scipy version, but also faster.
"""

import functools
from typing import Callable, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp


def dynamic_slice_with_padding(
    operand: chex.Array,
    start_indices: Sequence[chex.Array],
    slice_sizes: Tuple[int, ...],
    cval: chex.Scalar = 0.,
) -> chex.Array:
  """Slice array, masking out-of-bounds indices with `cval`.

  This function accounts for the potentially surprising behavior of
  `jax.lax.dynamic_slice` when the requested slice overruns the bounds of the
  source array: in this case the start index is adjusted to return a slice of
  the requested size:
  https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_slice.html
  See also:
  https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing

  This function returns the slice that would be obtained if the input were
  padded to prevent over/underflow, and query adjusted accordingly.

  Args:
    operand: an array to slice.
    start_indices: a list of scalar indices, one per dimension. These values
      may be dynamic.
    slice_sizes: the size of the slice. Must be a sequence of non-negative
      integers with length equal to `ndim(operand)`. Inside a JIT compiled
      function, only static values are supported (all JAX arrays inside JIT
      must have statically known size).
    cval: A scalar value to set for padded entries.

  Returns:
    An array containing the slice.

  Examples:
    Here is a simple two-dimensional dynamic slice:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    DeviceArray([[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]], dtype=int32)

    Note the differing behaviour for out-of-bounds queries:

    >>> jax.lax.dynamic_slice(x, (1, 1), (2, 4))
    DeviceArray([[ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_slice_with_padding(x, (1, 1), (2, 4), cval=0)
    DeviceArray([[ 5,  6,  7,  0],
                 [ 9, 10, 11,  0]], dtype=int32)
  """
  # NOTE: For memory-efficiency we perform the padding on the resulting slice
  #   rather than the input array, at the cost of some additional complexity.

  # Call dynamic_slice, which will shift the query to fit in-bounds.
  input_shape = jnp.asarray(operand.shape)
  output_shape = jnp.asarray(slice_sizes)
  start_indices = jnp.asarray(start_indices)
  clipped_indices = jnp.clip(start_indices,
                             jnp.zeros_like(input_shape),
                             input_shape - output_shape)
  raw_slice = jax.lax.dynamic_slice(operand, clipped_indices, slice_sizes)

  # Shift the slice indices to match the values that would have been obtained
  # by padding `operand` to allow the query to be in-bounds.
  offset = clipped_indices - start_indices
  rolled_slice = jnp.roll(raw_slice, offset, axis=range(raw_slice.ndim))

  # Compute mask in which `True` indicates out-of-bounds coordinates.
  grid = jnp.mgrid[tuple(slice(0, x) for x in slice_sizes)]
  mask = jnp.stack(grid) - jnp.expand_dims(offset, range(1, grid.ndim))
  mask = (mask < 0) | (
      mask >= jnp.expand_dims(output_shape, range(1, grid.ndim)))
  mask = jnp.any(mask, axis=0)  # OR along coordinate dimension.

  # Set masked values to `cval`.
  # NOTE: we achieve this using a dense multiplication rather than an indexing
  # operation `rolled_slice.at[mask].set(cval)` to avoid the following error:
  # >>> NonConcreteBooleanIndexError: Array boolean indices must be concrete
  # Dense multiplication also tends to be faster on TPU than gather/scatter
  # ops, however this was not profiled.
  return (mask * cval + rolled_slice * ~mask).astype(raw_slice.dtype)


def map_neighborhood(
    image: chex.Array,  # (H, W)
    coordinate: chex.Array,  # (2,)
    fn: Callable[[chex.Array], float],
    size: Union[int, Tuple[int, int]] = 2,
) -> chex.Array:  # ())
  """Applies a function to the neighbourhood of a specified coordinate.

  Args:
    image: (H, W) The (single channel) image to sample from.
    coordinate: (2,) The (y, x) location to sample.
    fn: A callable taking a `(size, size)` array containing the neighboring
      slice of `image` and returning an interpolated value.
    size: Integer or Tuple[y, x] representing the number of pixels in the
      neighborhood of `coordinate`. This neighborhood is centered on
      `coordinate`.

  Returns:
    The sampled value.
  """
  # Obtain neighborhood centered on the query point.
  size = (size, size) if isinstance(size, int) else size
  half_size = (jnp.ceil(jnp.asarray(size) / 2) - 1).astype(jnp.int32)
  lo = jnp.floor(coordinate - half_size).astype(jnp.int32)

  neighboring_vals = dynamic_slice_with_padding(image, lo, size, cval=0.)

  return fn(neighboring_vals)  # pytype: disable=bad-return-type  # numpy-scalars


def bilinear_interpolate(
    coordinate: chex.Array,  # (2,)
    image: chex.Array,  # (H, W)
) -> chex.Array:  # ()
  """Bilinearly interpolate an image at the specified coordinate.

  Args:
    coordinate: The (y, x) location to sample.
    image: The (single channel) image to sample from, in (y, x) order

  Returns:
    The sampled value.
  """

  def _bilinear_interpolate_2x2(
      neighborhood: chex.Array,  # (2, 2)
      coordinate: chex.Array,  # (2,)
  ) -> chex.Array:  # ()
    """Bilinear interpolation on a 2x2 neighborhood."""
    weights = coordinate - jnp.floor(coordinate)
    weights = jnp.stack([1 - weights, weights], axis=1)
    weights = jnp.outer(weights[0], weights[1])
    return (weights * neighborhood).sum()

  fn = functools.partial(_bilinear_interpolate_2x2, coordinate=coordinate)
  return map_neighborhood(image, coordinate, fn=fn, size=(2, 2))


def nearest_interpolate(
    coordinate: chex.Array,  # (2,)
    image: chex.Array,  # (H, W)
) -> chex.Array:  # ()
  """Perform nearest interpolation on an image at the specified coordinate.

  Args:
    coordinate: The location to sample, in (y, x) order.
    image: The (single channel) image to sample from.

  Returns:
    The sampled value.
  """

  def _nearest_interpolate_2x2(
      neighborhood: chex.Array,  # (2, 2)
      coordinate: chex.Array,  # (2,)
  ) -> chex.Array:  # ()
    """Nearest-neighbor interpolation on a 2x2 neighborhood."""
    coordinate = (jax.lax.round(coordinate) - jnp.floor(coordinate))
    coordinate = coordinate.astype(jnp.int32)
    return neighborhood[coordinate[0], coordinate[1]]

  fn = functools.partial(_nearest_interpolate_2x2, coordinate=coordinate)
  return map_neighborhood(image, coordinate, fn=fn, size=(2, 2))


def map_coordinates(
    image: chex.Array,  # (H, W)
    coordinates: chex.Array,  # (H', W', 2)
    method: str = 'bilinear',
) -> chex.Array:  # (H', W')
  """Map the input array to new coordinates by interpolation.

  This function expects coordinates to be in the last dimension of the input-
  tensor, rather than the first, so it corresponds to:
    ```
    jax.scipy.ndimage.map_coordinates(
        array=image,
        coordinates=coordinates.transpose((2, 0, 1)),
        order=1,
        mode='constant',
        cval=0.0)
    ```

  See:
  https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.ndimage.map_coordinates.html

  Like `jax.scipy.ndimage.map_coordinates`, this function operates on flat
  rasters, i.e. images without a channel dimension, and should be vmapped to
  handle full images.

  Args:
    image: (H, W) The image (without a channel dimension) to be interpolated.
    coordinates: (H', W', 2) The coordinates at which `image` is interpolated.
      Note this axis ordering differs from `jax.scipy.ndimage.map_coordinates`,
      which uses axis 0 as the coordinate dimension. In both cases coordinates
      are represented in (y, x) order, i.e. coord[:, :, 0] indexes the height-
      dimension and coord[:, :, 1] indexs the width dimension.
    method: A string indicating which type of interpolation to perform. Can be
      'bilinear' or 'nearest'.


  Returns:
    (H', W') The interpolated image (again without a channel dimension).
  """
  if method == 'bilinear':
    interp_fn = bilinear_interpolate
  elif method == 'nearest':
    interp_fn = nearest_interpolate
  else:
    raise ValueError(f'Invalid method: {method} for interpolation.')

  def _interpolate_image_at(coordinate: chex.Array) -> chex.Array:
    return interp_fn(coordinate, image)
  result = jax.vmap(jax.vmap(_interpolate_image_at))(coordinates)

  # `jax.scipy.ndimage.map_coordinates` casts result to input dtype, so we do
  # the same here. NOTE: rounding with jax.lax, not jax.numpy, since the former
  # always rounds up at the midpoint between integers, whereas the latter rounds
  # to the nearest even value. This choice was made for consistency with
  # `jax.scipy.ndimage.map_coordinates`.
  if (jnp.issubdtype(image.dtype, jnp.integer) and
      not jnp.issubdtype(result.dtype, jnp.integer)):
    result = jax.lax.round(result)
  return result.astype(image.dtype)
