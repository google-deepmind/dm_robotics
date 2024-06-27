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

"""Model of a finite projective camera."""

import functools
from typing import Sequence

import chex
from dm_robotics.geometry.jax_geometry import auto_vectorize
from dm_robotics.geometry.jax_geometry import basic_types
import jax
import jax.numpy as jnp


# Internal dot product clamped to highest-precision.
_dot = functools.partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)


@auto_vectorize.indexable
@chex.dataclass(frozen=True)
class Camera:
  """Provides a model of a finite projective camera.

  This class is based on Chapter 6 of:

    R. Hartley and A. Zisserman, "Multiple view geometry in computer vision",
    Cambridge University Press (Second Edition), ISBN: 0521540518, 2004.

  This class is composed of primary properties, which are stored explicitly as
  fields, and derived properties, which are computed on demand from the primary
  properties. Derived properties are exposed as `@property`s so from an
  interface perspective there is no difference between them.

  This class can represent batches of Cameras, with arbitrarily many batch
  dimensions. Documented shapes indicate the shape for a scalar Camera; in a
  batched Camera they will be prefixed with the batch_shape.

  Fields:
  [Primary] Properties:
    extrinsics: (4, 4) Camera extrinsics matrix. Represented as a homogeneous
      world->camera frame transformation.
    intrinsics: (3, 3) Camera intrinsics matrix.

  [Derived] Properties:
    projection: (3, 4) The camera projection matrix.
    frame: The camera frame as a camera->world transform.
    center: (3,) The camera center in world coordinates.
    principal_plane: (4,) The principal plane of the camera.
    principal_ray: (3,) The principal ray of the camera.

    The remaining attributes follow the notation from the reference book.

    P: (3, 4) The camera matrix.
    M: The leftmost (3, 3) submatrix of the camera matrix.
    C: The camera center in world coordinates.
    p1: First column of P.
    p2: Second column of P.
    p3: Third column of P.
    p4: Fourth column of P.
    P1: First row of P.
    P2: Second row of P.
    P3: Third row of P.
    m1: First column of M.
    m2: Second column of M.
    m3: Third column of M.
  """

  extrinsics: chex.Array  # (4, 4) = R[ I | -C ]
  intrinsics: chex.Array  # (3, 3) = K

  @classmethod
  def new(cls, extrinsics: chex.Array, intrinsics: chex.Array) -> 'Camera':
    """Broadcasting constructor for Camera.

    Operates similar to the default class constructor, but also verifies shapes
    and broadcasts compatible batch dimensions.

    Args:
      extrinsics: Extrinsics for the camera.
      intrinsics: Intrinsics for the camera.

    Returns:
      A Camera.
    """
    if extrinsics.shape[-2:] != (4, 4):
      raise ValueError(
          f'Expected extrinsics of shape (..., 4, 4). Got {extrinsics.shape}.')
    if intrinsics.shape[-2:] != (3, 3):
      raise ValueError(
          f'Expected intrinsics of shape (..., 3, 3). Got {intrinsics.shape}.')

    batch_shape = jnp.broadcast_shapes(extrinsics.shape[:-2],
                                       intrinsics.shape[:-2])

    return cls(
        extrinsics=jnp.broadcast_to(extrinsics, batch_shape + (4, 4)),
        intrinsics=jnp.broadcast_to(intrinsics, batch_shape + (3, 3)))

  @classmethod
  def at_origin(cls, intrinsics: chex.Array) -> 'Camera':
    """Create a Camera where the camera and world frame coincide.

    A convenience constructor that sets the extrinsics to identity.

    Args:
      intrinsics: Intrinics for the camera.

    Returns:
      A Camera at the world origin with the specified intrinsics.
    """
    return cls.new(extrinsics=jnp.eye(4), intrinsics=intrinsics)

  @classmethod
  def in_frame(cls, frame: basic_types.Frame,
               intrinsics: chex.Array) -> 'Camera':
    """Create a Camera in the specified frame.

    Args:
      frame: The desired camera frame.
      intrinsics: Intrinsics for the camera.

    Returns:
      A Camera in the specified frame with the specified intrinsics.
    """
    return cls.new(extrinsics=frame.inv().to_hmat(), intrinsics=intrinsics)

  @property
  def batch_shape(self) -> Sequence[int]:
    """Shape of the batch dimensions for this Camera."""
    return self.extrinsics.shape[:-2]

  @auto_vectorize.batched_property
  def projection(self) -> chex.Array:
    """Camera projection matrix."""
    return _dot(self.intrinsics, self.extrinsics[:3, :])

  @auto_vectorize.batched_property
  def frame(self) -> basic_types.Frame:
    # NB: The order of operations here is important for stability on TPU.
    # inv(from_hmat(extrinsics)) inverts the Frame. Frames internally use a
    # position/quaternion representation, which is stable to invert.
    # from_hmat(inv(extrinsics)) is mathematically the same, but instead inverts
    # the extrinsics matrix, and this operation is less numerically stable.
    return basic_types.Frame.from_hmat(self.extrinsics).inv()

  @auto_vectorize.batched_property
  def center(self) -> chex.Array:
    """Camera center in world coordinates."""
    return self.C

  @auto_vectorize.batched_property
  def principal_plane(self) -> chex.Array:
    """Principal plane of the Camera."""
    return self.P3

  @auto_vectorize.batched_property
  def principal_ray(self) -> chex.Array:
    """Principal ray of the Camera."""
    return jnp.linalg.det(self.M) * self.m3

  @auto_vectorize.batched_property
  def P(self) -> chex.Array:  # pylint: disable=invalid-name
    """Camera projection matrix."""
    return self.projection

  @auto_vectorize.batched_property
  def M(self) -> chex.Array:  # pylint: disable=invalid-name
    """Leftmost (3, 3) submatrix of the camera matrix."""
    return self.P[:3, :3]

  @auto_vectorize.batched_property
  def C(self) -> chex.Array:  # pylint: disable=invalid-name
    """Camera center in world coordinates."""
    return -jnp.linalg.solve(self.M, self.p4)

  @auto_vectorize.batched_property
  def p1(self) -> chex.Array:
    """First column of P."""
    return self.P[:, 0]

  @auto_vectorize.batched_property
  def p2(self) -> chex.Array:
    """Second column of P."""
    return self.P[:, 1]

  @auto_vectorize.batched_property
  def p3(self) -> chex.Array:
    """Third column of P."""
    return self.P[:, 2]

  @auto_vectorize.batched_property
  def p4(self) -> chex.Array:
    """Fourth column of P."""
    return self.P[:, 3]

  @auto_vectorize.batched_property
  def P1(self) -> chex.Array:  # pylint: disable=invalid-name
    """First row of P."""
    return self.P[0, :]

  @auto_vectorize.batched_property
  def P2(self) -> chex.Array:  # pylint: disable=invalid-name
    """Second row of P."""
    return self.P[1, :]

  @auto_vectorize.batched_property
  def P3(self) -> chex.Array:  # pylint: disable=invalid-name
    """Third row of P."""
    return self.P[2, :]

  @auto_vectorize.batched_property
  def m1(self) -> chex.Array:
    """First column of M."""
    return self.M[0, :]

  @auto_vectorize.batched_property
  def m2(self) -> chex.Array:
    """Second column of M."""
    return self.M[1, :]

  @auto_vectorize.batched_property
  def m3(self) -> chex.Array:
    """Third column of M."""
    return self.M[2, :]
