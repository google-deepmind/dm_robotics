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

"""Tests for geometry.jax.basic_types."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_robotics.geometry.jax_geometry import basic_types
from dm_robotics.geometry.jax_geometry import frame_geometry
from dm_robotics.geometry.jax_geometry import geometry_utils
from dm_robotics.geometry.jax_geometry import pointcloud_stamped
import jax
import jax.numpy as jnp


DEFAULT_BATCH_SHAPES = [((),), ((5,),)]


def as_batch(array, batch_shape):
  return jnp.broadcast_to(array, batch_shape + array.shape)


class PointCloudStampedTest(chex.TestCase, parameterized.TestCase):

  @chex.all_variants
  @parameterized.parameters(DEFAULT_BATCH_SHAPES)
  def test_to_world(self, batch_shape):
    points = jnp.arange(27).reshape(9, 3)
    posaxisangle = jnp.array([0., 1., 2., 0.1, 0.2, 0.3])

    # Add leading batch-dims to verify vectorization works.
    points = as_batch(points, batch_shape)
    posaxisangle = as_batch(posaxisangle, batch_shape)

    frame = frame_geometry.Pose.from_posaxisangle(posaxisangle)
    pc = pointcloud_stamped.PointCloudStamped(frame=frame, points=points)

    # Compute world pointcloud using pointcloud method.
    def to_world_member_fn():
      return pc.to_world()

    to_world_member_fn = self.variant(to_world_member_fn)
    actual_world_pts = to_world_member_fn()

    # Compute world pointcloud manually.
    def to_world_manual_fn(
        pointcloud: pointcloud_stamped.PointCloudStamped
    ) -> basic_types.PointCloud:
      return geometry_utils.from_homogeneous(
          geometry_utils.to_homogeneous(pointcloud.points)
          @ pointcloud.frame.to_hmat().T)

    if batch_shape:
      to_world_manual_fn = jax.vmap(to_world_manual_fn)
    to_world_manual_fn = self.variant(to_world_manual_fn)
    expected_world_pts = to_world_manual_fn(pc)

    chex.assert_trees_all_close(
        actual_world_pts, expected_world_pts,
        atol=1e-5)  # <-- shouldn't need this!?

  @chex.all_variants
  @parameterized.parameters(DEFAULT_BATCH_SHAPES)
  def test_to_frame(self, batch_shape):
    points = jnp.arange(27).reshape(9, 3)
    orig_posaxisangle = jnp.array([0., 1., 2., 0.1, 0.2, 0.3])
    new_posaxisangle = jnp.array([3., 4., 5., 0.3, 0.4, 0.5])

    # Add leading batch-dims to verify vectorization works.
    points = as_batch(points, batch_shape)
    orig_posaxisangle = as_batch(orig_posaxisangle, batch_shape)
    new_posaxisangle = as_batch(new_posaxisangle, batch_shape)

    orig_frame = frame_geometry.Pose.from_posaxisangle(orig_posaxisangle)
    pc = pointcloud_stamped.PointCloudStamped(frame=orig_frame, points=points)

    new_frame = frame_geometry.Pose.from_posaxisangle(new_posaxisangle)

    # Compute world pointcloud using pointcloud method.
    def to_frame_member_fn(new_frame):
      return pc.to_frame(new_frame)

    to_frame_member_fn = self.variant(to_frame_member_fn)
    actual_world_pts = to_frame_member_fn(new_frame)

    # Compute world pointcloud manually.
    def to_frame_manual_fn(
        pointcloud: pointcloud_stamped.PointCloudStamped,
        new_frame: basic_types.Frame) -> basic_types.PointCloud:
      points_hom = geometry_utils.to_homogeneous(pointcloud.points)
      frame_new_old = new_frame.inv().mul(pointcloud.frame)
      transformed_hom = points_hom @ frame_new_old.to_hmat().T
      new_points = geometry_utils.from_homogeneous(transformed_hom)
      return pointcloud_stamped.PointCloudStamped(
          frame=new_frame, points=new_points)

    if batch_shape:
      to_frame_manual_fn = jax.vmap(to_frame_manual_fn)
    to_frame_manual_fn = self.variant(to_frame_manual_fn)
    expected_world_pts = to_frame_manual_fn(pc, new_frame)

    chex.assert_trees_all_close(
        actual_world_pts, expected_world_pts,
        atol=1e-5)  # <-- shouldn't need this!?


if __name__ == '__main__':
  absltest.main()
