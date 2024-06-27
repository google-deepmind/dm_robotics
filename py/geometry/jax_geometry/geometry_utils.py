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

"""JAX projection utilities."""

import functools
from typing import Callable, Tuple, Union

from dm_robotics.geometry.jax_geometry import basic_types
from dm_robotics.geometry.jax_geometry._impl import map_coordinates as map_coordinates_impl
import jax
from jax import numpy as jnp


def to_homogeneous(
    points: Union[basic_types.Raster, basic_types.Cloud]
) -> Union[basic_types.Raster, basic_types.Cloud]:
  """Converts points to homogeneous by appending a `1`.

  Args:
    points: `(..., D)` `D`-dimensional points.

  Returns:
    `(..., D + 1)`-dimensional array points containing `points` with a
      trailing `1`.
  """
  return jnp.concatenate((points, jnp.ones_like(points[..., :1])), axis=-1)


def from_homogeneous(
    points: Union[basic_types.Raster, basic_types.Cloud]
) -> Union[basic_types.Raster, basic_types.Cloud]:
  """Converts points from homogenous coordinates by normalizing the last dim.

  Args:
    points: `(..., D + 1)` `D`-dimensional homogenous points.

  Returns:
    `D`-dimensional array containing `points` normalized by the last value.
  """
  return points[..., :-1] / (points[..., -1:] + 1e-12)


def bilinear_interpolate(
    image: basic_types.Raster,  # (H, W)
    coordinate: basic_types.Point,  # (2,)
) -> basic_types.Point:  # ())
  """Bilinearly interpolate an image at the specified coordinate.

  Args:
    image: (H, W) The (single channel) image to sample from.
    coordinate: (2,) The (x, y) location to sample.

  Returns:
    The sampled value.
  """
  # Flip coordinate since map_coordinates_impl expects in (y, x) order.
  coordinate = jnp.flip(coordinate, axis=0)
  return map_coordinates_impl.bilinear_interpolate(coordinate, image)


def nearest_interpolate(
    image: basic_types.Raster,  # (H, W)
    coordinate: basic_types.Point,  # (2,)
) -> basic_types.Point:  # ())
  """Performs nearest interpolation on an image at the specified coordinate.

  Args:
    image: (H, W) The (single channel) image to sample from.
    coordinate: (2,) The (x, y) location to sample.

  Returns:
    The sampled value.
  """
  # Flip coordinate since map_coordinates_impl expects in (y, x) order.
  coordinate = jnp.flip(coordinate, axis=0)
  return map_coordinates_impl.nearest_interpolate(coordinate, image)


def map_neighborhood(
    image: basic_types.Raster,  # (H, W)
    coordinate: basic_types.Point,  # (2,)
    fn: Callable[[basic_types.Raster], jnp.ndarray],
    size: Union[int, Tuple[int, int]] = 2,
) -> basic_types.Point:  # ())
  """Applies a function to the neighbourhood of a specified coordinate.

  Args:
    image: (H, W) The (single channel) image to sample from.
    coordinate: (2,) The (x, y) location to sample.
    fn: A callable taking a `(size, size)` array containing the neighboring
      slice of `image` and returning an array as output.
    size: Integer or Tuple[x, y] representing the number of pixels in the
      neighborhood of `coordinate`. This neighborhood is centered on
      `coordinate`.

  Returns:
    The sampled value.
  """
  # Obtain neighborhood centered on the query point.
  size = (size, size) if isinstance(size, int) else size

  # Flip coordinate and size since dynamic_slice expects in (y, x) order.
  coordinate = jnp.flip(jnp.asarray(coordinate), axis=0)
  size = size[::-1]
  return map_coordinates_impl.map_neighborhood(image, coordinate, fn, size)  # pytype: disable=wrong-arg-types  # jax-ndarray


def _interpolate(
    image: basic_types.Raster,  # (H, W)
    img_pt: basic_types.Point,  # (2,)
    method: str = 'bilinear') -> basic_types.Point:  # ())
  """Interpolates `image` at `img_pt` using the specified method."""
  if method == 'bilinear':
    return bilinear_interpolate(image, img_pt)
  elif method == 'nearest':
    return nearest_interpolate(image, img_pt)
  else:
    raise ValueError(f'Invalid interpolation method {method}')


def interpolate(
    image: basic_types.Raster,  # (H, W, [C])
    img_pts: basic_types.Point,  # ([N], 2)
    method: str = 'bilinear'
) -> Union[basic_types.Point, basic_types.Cloud]:  # (N, C)
  """Interpolates `image` at `img_pts` using the specified method.

  Args:
    image: An array with shape (H, W) or (H, W, C) to be interpolated.
    img_pts: An array with shape (2,) or (N, 2) containing the (possibly
      real-valued) image-coordinate at which to interpolate `image`, in (x, y)
      order.
    method: A string indicating which type of interpolation to perform. Can be
      'bilinear' or 'nearest'.

  Returns:
    An array with shape (N, C) containing the channel-wise interpolation of
    `image` using the indicated method. Note the output will have point and
    channel dimensions even if the input omits them.
  """
  image = jnp.atleast_3d(image)  # (H, W, C)
  img_pts = jnp.atleast_2d(img_pts)  # (N, 2)

  if jnp.issubdtype(img_pts, jnp.integer):
    # If query is integer-valued then no interpolation is required.
    return image[img_pts[:, 1], img_pts[:, 0], :]  # (N, C)

  else:
    interp_fn = functools.partial(_interpolate, method=method)

    # Vmap over the channel dimension.
    interp_fn = jax.vmap(interp_fn, in_axes=[-1, None])

    # Vmap over the points dimension.
    interp_fn = jax.vmap(interp_fn, in_axes=[None, 0])

    return interp_fn(image, img_pts)


def map_coordinates(
    image: basic_types.Raster,  # (H, W)
    coordinates: basic_types.Point,  # (H', W', 2)
    method: str = 'bilinear',
) -> basic_types.Raster:  # (H', W')
  """Map the input array to new coordinates by interpolation.

  This function expects coordinates to be in the last dimension of the input-
  tensor, rather than the first, so it corresponds to:
    ```
    jax.scipy.ndimage.map_coordinates(
        array=image,
        coordinates=np.flip(coordinates, axis=-1).transpose((2, 0, 1)),
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
      which uses axis 0 as the coordinate dimension. This function expects
      coordinates in (x, y) order, i.e. coord[:, :, 0] indexes the width-
      dimension and coord[:, :, 1] indexes the height-dimension.
    method: A string indicating which type of interpolation to perform. Can be
      'bilinear' or 'nearest'.

  Returns:
    (H', W') The interpolated image (again without a channel dimension).
  """
  # Flip coordinate since map_coordinates_impl expects in (y, x) order.
  coordinates = jnp.flip(coordinates, axis=-1)
  return map_coordinates_impl.map_coordinates(image, coordinates, method=method)
