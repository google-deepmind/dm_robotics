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

"""A chex.dataclass-based pointcloud class."""

import functools
from typing import Sequence

import chex
from dm_robotics.geometry.jax_geometry import auto_vectorize
from dm_robotics.geometry.jax_geometry import basic_types
from dm_robotics.geometry.jax_geometry import geometry_utils
import jax
from jax import numpy as jnp


# Internal dot product clamped to highest-precision.
_dot = functools.partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)


@chex.dataclass(frozen=True)
class PointCloudStamped:
  """A PointCloudStamped is a PointCloud combined with a Frame.

  PointCloudStamped is a special case of a PointCloud that requires the points
  be 3-dimensional.

  The Frame defines a coordinate system in which coordinates of the points in
  the PointCloud are expressed.
  """
  frame: basic_types.Frame  # This must be a local->world transform.
  points: basic_types.PointCloud  # Must contain 3d points.

  @property
  def batch_shape(self) -> Sequence[int]:
    """Shape of the batch dimensions for this PointCloudStamped."""
    return self.points.shape[:-2]  # All dims beyond two (i.e. (N, 3)) are batch

  @auto_vectorize.batched_method
  def to_world(self) -> basic_types.PointCloud:
    """Returns a PointCloud containing `points` expressed in the world-frame."""
    points_hom = geometry_utils.to_homogeneous(self.points)
    transformed_hom = _dot(points_hom, self.frame.to_hmat().T)
    return geometry_utils.from_homogeneous(transformed_hom)

  @auto_vectorize.batched_method
  def to_frame(self, frame: basic_types.Frame) -> 'PointCloudStamped':
    """Re-expresses the pointcloud in a new frame.

    Args:
      frame: The desired frame in which the points should be expressed.

    Returns:
      A new PointCloudStamped containing `points` expressed in the provided
      frame.
    """
    points_hom = geometry_utils.to_homogeneous(self.points)
    frame_world_old = self.frame
    frame_world_new = frame
    frame_new_old = frame_world_new.inv().mul(frame_world_old)
    transformed_hom = _dot(points_hom, frame_new_old.to_hmat().T)
    new_points = geometry_utils.from_homogeneous(transformed_hom)
    return PointCloudStamped(frame=frame, points=new_points)
