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

"""Jax implementations of frame geometry.

This module is a jax-counterpart to the python-based frame system here:
third_party/py/dm_robotics/geometry/geometry.py

It's purpose is to provide a (batchable, differentiable) abstraction for Pose
and other physical quantities (e.g. velocity, force) with operators for
transforming them between frames.

See usage examples below.
"""

from typing import Sequence, Union

import chex
from dm_robotics.geometry.jax_geometry import auto_vectorize
from dm_robotics.transformations.jax_transformations import transformations as jtr
import jax.numpy as jnp
import numpy as np


_IDENTITY_QUATERNION = np.array([1, 0, 0, 0], dtype=np.float32)
_ZERO_POSITION = np.zeros(3, dtype=np.float32)


@auto_vectorize.indexable
@chex.dataclass(frozen=True)  # Like `geometry.Pose` this class is immutable.
class Pose:
  """A container representing a 3D pose.

  This class is a Jax-counterpart to `dm_robotics/geometry/geometry.py:Pose`.
  It supports (mostly) the same operations, as well as seamlessly working with
  vmap and jit on batches of `Pose`.

  Examples:
  1) Computing a relative pose, non-batch case:
      pose1 = Pose(position=[1., 2., 3.],
                   quaternion=[1., 0., 0., 0.])
      pose2 = Pose(position=[3., 2., 1.],
                   quaternion=[0.8660254, 0.5, 0., 0.])
      pose_1_2 = pose1.inv().mul(pose2)
      >>> Pose(position=DeviceArray([ 2.,  0., -2.], dtype=float32),
               quaternion=DeviceArray([0.8660254, 0.5, 0., 0.], dtype=float32))

  2) Creating a batch of poses from a 2-dim array of position and axixangle:
      Pose.from_posaxisangle([[1., 2., 3., np.pi/2, 0., 0.],
                              [3., 2., 1., 0., np.pi/3, 0.]])
      >>> Pose(position=DeviceArray([[1., 2., 3.],
                                    [3., 2., 1.]], dtype=float32),
               quaternion=DeviceArray(
                   [[0.70710677, 0.70710677, 0., 0.],
                    [0.8660254 , 0., 0.5, 0.]], dtype=float32))

  3) Converting a batch of homogeneous poses to `Pose`:
    Pose.from_hmat(jnp.stack((jnp.eye(4), jnp.eye(4))))
    >>> Pose(position=DeviceArray([[0., 0., 0.],
                                   [0., 0., 0.]], dtype=float32),
             quaternion=DeviceArray([[1., 0., 0., 0.],
                                     [1., 0., 0., 0.]], dtype=float32))

  4) Computing relative pose from model outputs works seamlessly on batches
     without explicit vmap:
    posrot6_1 = hk.Linear(9)(inputs)
    posrot6_2 = hk.Linear(9)(inputs)
    pose1 = Pose.from_posrot6(posrot6_1)
    pose2 = Pose.from_posrot6(posrot6_2)
    pose_1_2 = pose1.inv().mul(pose2)
    ...
  """

  position: chex.Array  # ([B], 3,)
  quaternion: chex.Array  # ([B], 4,)

  def __eq__(self, other: object) -> bool:
    if isinstance(other, Pose):
      return jnp.allclose(self.position, other.position) and (
          jnp.allclose(self.quaternion, other.quaternion) or
          jnp.allclose(self.quaternion, -1 * other.quaternion))
    else:
      return NotImplemented

  @classmethod
  def new(cls, position: Union[chex.Array, Sequence[chex.Scalar]],
          quaternion: Union[chex.Array, Sequence[chex.Scalar]]) -> "Pose":
    """Construct a new Pose.

    This factory does type normalization and verification, unlike the default
    constructor.

    Args:
      position: Position of the pose.
      quaternion: Orientation of the pose.

    Returns:
      A new Pose with the specified position and orientation.
    """
    return cls(
        position=jnp.asarray(position), quaternion=jnp.asarray(quaternion))

  @classmethod
  def identity(cls) -> "Pose":
    """Helper method for initializing an identity Pose."""
    return cls.new(position=_ZERO_POSITION, quaternion=_IDENTITY_QUATERNION)

  @property
  def batch_shape(self) -> Sequence[int]:
    """Shape of the batch dimensions for this Pose."""
    return self.position.shape[:-1]

  @auto_vectorize.batched_method
  def mul(self, other: "Pose") -> "Pose":
    """Multiplies other pose by this pose.

    Args:
      other: The other Pose to multiply by.

    Returns:
      Resulting pose.
    """
    new_position, new_quaternion = jtr.pos_quat_mul(
        self.position, self.quaternion, other.position, other.quaternion)

    return Pose(position=new_position,
                quaternion=new_quaternion)

  @auto_vectorize.batched_method
  def inv(self) -> "Pose":
    """Returns the inverse of this pose."""
    position = self.position
    quaternion = self.quaternion

    inv_quat = jtr.quat_inv(quaternion)
    inv_pos = jtr.quat_rotate(inv_quat, -1 * position)

    return Pose(position=inv_pos, quaternion=inv_quat)

  @auto_vectorize.batched_method
  def to_posquat(self) -> jnp.ndarray:
    """Converts the pose to a 7-dimensional posquat array."""
    return jnp.concatenate((self.position, self.quaternion), axis=-1)

  @classmethod
  def from_posquat(cls, posquat: chex.Array) -> "Pose":
    """Creates a `Pose` from position and quaternion arrays.

    Args:
      posquat: (7,) array containing position as the first 3 dimensions and an
        orientation expressed as a `quaternion` in the remaining 4 dimensions.

    Returns:
      A `Pose` corresponding to the provided posquat.
    """
    return cls.new(position=posquat[..., :3], quaternion=posquat[..., 3:])

  @auto_vectorize.batched_method
  def to_posaxisangle(self) -> jnp.ndarray:
    """Converts the pose to a 6-dimensional posaxisangle array."""
    axisangle = jtr.quat_to_axisangle(self.quaternion)
    posaxisangle = jnp.concatenate((self.position, axisangle), axis=-1)

    return posaxisangle

  @classmethod
  def from_posaxisangle(cls, posaxisangle: chex.Array) -> "Pose":
    """Creates a `Pose` from position and axisangle arrays.

    Args:
      posaxisangle: (6,) array containing position as the first 3 dimensions and
        an orientation expressed as a `axisangle` in the remaining 3 dimensions.

    Returns:
      A `Pose` corresponding to the provided posaxisangle.
    """
    position = posaxisangle[..., :3]
    axisangle = posaxisangle[..., 3:]

    batched_aa_to_quat = auto_vectorize.multi_vmap(
        jtr.axisangle_to_quat)
    quaternion = batched_aa_to_quat(axisangle)

    return cls.new(position=position, quaternion=quaternion)

  @auto_vectorize.batched_method
  def to_hmat(self) -> jnp.ndarray:
    """Converts the pose to a [bx]4x4 homogeneous matrix."""
    return jtr.pos_quat_to_hmat(self.position, self.quaternion)

  @classmethod
  def from_hmat(cls, hmat: jnp.ndarray) -> "Pose":
    """Creates a `Pose` from a 4x4 homogeneous matrix.

    Args:
      hmat: (4, 4) array containing a homogeneous transform.

    Returns:
      A `Pose` corresponding to the provided hmat.
    """
    batched_hmat_to_pos_quat = auto_vectorize.multi_vmap(
        jtr.hmat_to_pos_quat, n_times=-2)
    position, quaternion = batched_hmat_to_pos_quat(hmat)

    return cls.new(position=position, quaternion=quaternion)

  @auto_vectorize.batched_method
  def to_posrot6(self) -> jnp.ndarray:
    """Converts the pose to a 9-dimensional posrot6 array."""
    rot6 = jtr.rmat_to_rot6(jtr.quat_to_mat(self.quaternion))
    posrot6 = jnp.concatenate((self.position, rot6), axis=-1)

    return posrot6

  @classmethod
  def from_posrot6(cls, posrot6: chex.Array) -> "Pose":
    """Creates a `Pose` from position and rot6 arrays.

    Args:
      posrot6: (9,) array containing position as the first 3 dimensions and an
        orientation expressed as a `rot6` in the remaining 6 dimensions. See
        `transformations.rot6_to_rmat` for details.

    Returns:
      A `Pose` corresponding to the provided posrot6.
    """
    position, rot6 = posrot6[..., :3], posrot6[..., 3:]

    def rot6_to_quat(rot6: jnp.ndarray) -> jnp.ndarray:
      return jtr.mat_to_quat(jtr.rot6_to_rmat(rot6))

    quaternion = auto_vectorize.multi_vmap(rot6_to_quat)(rot6)

    return cls.new(position=position, quaternion=quaternion)
