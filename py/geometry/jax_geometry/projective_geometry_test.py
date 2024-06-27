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

"""Tests for geometry.jax.projective_geometry."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_robotics.geometry.jax_geometry import basic_types
from dm_robotics.geometry.jax_geometry import camera as camera_type
from dm_robotics.geometry.jax_geometry import geometry_utils as gu
from dm_robotics.geometry.jax_geometry import pointcloud_stamped
from dm_robotics.geometry.jax_geometry import projective_geometry as pg
import jax
import jax.numpy as jnp
import numpy as np


def get_source_image(image_shape):
  return jax.random.uniform(
      jax.random.PRNGKey(42), image_shape + (3,)
  )


class ProjectiveGeometryTest(chex.TestCase, parameterized.TestCase):

  @chex.all_variants
  def test_raster_flatten(self):
    raster = jnp.zeros((4, 3, 7))
    cloud = self.variant(pg.raster_flatten)(raster)
    chex.assert_shape(cloud, ((4 * 3, 7)))

  @chex.all_variants
  def test_raster_coordinates_2d(self):
    height, width = 32, 64
    coords = self.variant(
        pg.raster_coordinates, static_argnums=(0, 1))(height, width)

    # Check that coords has the same shape as the source raster.
    chex.assert_shape(coords, (height, width, 2))

    # Check that x counts faster than y
    x, y = coords[..., 0], coords[..., 1]
    np.testing.assert_array_equal(x[0, :3], [0., 1., 2.])
    np.testing.assert_array_equal(y[:3, 0], [0., 1., 2.])

    # Check that flattening results in (x, y) order.
    coords = coords.reshape((-1, 2))
    np.testing.assert_array_equal(coords.min(axis=0), (0, 0))
    np.testing.assert_array_equal(coords.max(axis=0) + 1., (width, height))

  @chex.all_variants
  def test_raster_coordinates_3d(self):
    height, width, depth = 4, 8, 16

    coords = self.variant(
        pg.raster_coordinates, static_argnums=(0, 1, 2))(height, width, depth)

    # Check that coords has the same shape as the source raster.
    chex.assert_shape(coords, (height, width, depth, 3))

    # Check that z counts faster than x counts faster than y.
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    np.testing.assert_array_equal(z[0, 0, :3], [0., 1., 2.])
    np.testing.assert_array_equal(x[0, :3, 0], [0., 1., 2.])
    np.testing.assert_array_equal(y[:3, 0, 0], [0., 1., 2.])

    # Check that flattening results in (x, y, z) order.
    coords = coords.reshape((-1, 3))
    np.testing.assert_array_equal(coords.min(axis=0), (0, 0, 0))
    np.testing.assert_array_equal(
        coords.max(axis=0) + 1., (width, height, depth))

  @chex.all_variants
  @parameterized.named_parameters(
      ('eye', np.eye(4, 4),),
      # poseuler_to_hmat(
      #     [1., 2., 3., np.pi / 2, np.pi / 3, np.pi / 4], ordering='XYZ')
      ('hmat', np.asarray([
          [3.53553391e-01, -3.53553391e-01, 8.66025404e-01, 1.00000000e+00],
          [6.12372436e-01, -6.12372436e-01, -5.00000000e-01, 2.00000000e+00],
          [7.07106781e-01, 7.07106781e-01, 3.06161700e-17, 3.00000000e+00],
          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
      ]),),
  )
  def test_points_to_pixels_to_points(self, hmat_cam_points):
    xs, ys = jnp.meshgrid(jnp.linspace(-1, 1, 10), jnp.linspace(-1, 1, 10))
    zs = 1.0 + xs**2 + ys**2
    points_cam = jnp.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=-1)
    points_world = gu.from_homogeneous(
        gu.to_homogeneous(points_cam) @ jnp.linalg.inv(hmat_cam_points).T)

    intrinsic_matrix = jnp.asarray([
        [10.0, 0.0, 5.0],
        [0.0, 11.0, 6.0],
        [0.0, 0.0, 1.0],
    ])

    camera = camera_type.Camera.new(
        intrinsics=intrinsic_matrix, extrinsics=jnp.eye(4))

    pc = pointcloud_stamped.PointCloudStamped(
        frame=basic_types.Frame.from_hmat(hmat_cam_points), points=points_world)

    pixels = self.variant(pg.project_pointcloud_to_camera)(
        camera=camera, pointcloud=pc)

    pc2 = self.variant(pg.depth_to_points)(camera, pixels, zs.reshape((-1, 1)))

    self.assertTrue(jnp.allclose(pc.to_world(), pc2.to_world()))

  @chex.all_variants
  def test_resample_image_identity(self):
    # Resampling an image at the coordinates provided by get_coordinates should
    # be an identity operation.
    image_shape = (64, 32)

    source_coords = pg.raster_coordinates(*image_shape)
    source_image = get_source_image(image_shape)

    resampled_image, valid_mask = self.variant(pg.resample_image)(source_image,
                                                                  source_coords)

    self.assertFalse(jnp.any(jnp.isnan(resampled_image)))
    self.assertFalse(jnp.any(jnp.isnan(valid_mask)))
    chex.assert_equal_shape_prefix((source_coords, resampled_image), 2)
    chex.assert_equal_shape_prefix((resampled_image, valid_mask), 2)
    chex.assert_shape(valid_mask, resampled_image.shape[:2] + (1,))
    chex.assert_trees_all_close(source_image, resampled_image, atol=1e-5)
    self.assertTrue(jnp.all(valid_mask))

  @chex.all_variants
  @parameterized.parameters(0.5, 2.0)
  def test_resample_image_resize(self, factor):
    source_shape = (64, 32)
    target_shape = tuple(int(factor * d) for d in source_shape)

    # Manually set up sampling coordinates to ensure everything is in bounds.
    source_coords = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, source_shape[1] - 1, num=target_shape[1]),
            jnp.linspace(0, source_shape[0] - 1, num=target_shape[0]),
            indexing='xy'),
        axis=-1)
    source_image = get_source_image(source_shape)

    resampled_image, valid_mask = self.variant(pg.resample_image)(source_image,
                                                                  source_coords)

    self.assertFalse(jnp.any(jnp.isnan(resampled_image)))
    self.assertFalse(jnp.any(jnp.isnan(valid_mask)))
    chex.assert_equal_shape_prefix((source_coords, resampled_image), 2)
    self.assertTrue(jnp.all(valid_mask))

  @chex.all_variants
  @parameterized.named_parameters(
      dict(
          testcase_name='posy',
          shift=(1.5, 0.0),
          valid=((0, -2), (None,)),
          invalid=((-2, None), (None,)),
          source0=((1, -1), (None,)),
          source1=((2, None), (None,)),
      ),
      dict(
          testcase_name='negy',
          shift=(-1.5, 0.0),
          valid=((2, None), (None,)),
          invalid=((0, 2), (None,)),
          source0=((0, -2), (None,)),
          source1=((1, -1), (None,)),
      ),
      dict(
          testcase_name='posx',
          shift=(0.0, 1.5),
          valid=((None,), (0, -2)),
          invalid=((None,), (-2, None)),
          source0=((None,), (1, -1)),
          source1=((None,), (2, None)),
      ),
      dict(
          testcase_name='negx',
          shift=(0.0, -1.5),
          valid=((None,), (2, None)),
          invalid=((None,), (0, 2)),
          source0=((None,), (0, -2)),
          source1=((None,), (1, -1)),
      ))
  def test_resample_image_shift(self, shift, valid, invalid, source0, source1):
    """Check that edge effects are handled correctly by image shifts.

    If source_coords are shifted by `shift` then `resampled_image[valid]` is
    valid in the result and `resampled_image[invalid]` is not. The valid part of
    `resampled_image` is given by `0.5 * source[source0] + source[source1]`.
    (NB: This implies that shfits must be half integers.)

    Args:
      shift: A yx tuple of coordinate shifts.
      valid: A yx tuple of valid slice parameters.
      invalid: A yx tuple of invalid slice parameters.
      source0: A yx tuple of source slice parameters.
      source1: A yx tuple of source slice parameters.
    """
    image_shape = (64, 32)
    yshift, xshift = shift
    valid = tuple(slice(*x) for x in valid)
    invalid = tuple(slice(*x) for x in invalid)
    source0 = tuple(slice(*x) for x in source0)
    source1 = tuple(slice(*x) for x in source1)

    source_coords = pg.raster_coordinates(*image_shape)
    source_coords = source_coords.at[..., 0].add(xshift)
    source_coords = source_coords.at[..., 1].add(yshift)
    source_image = get_source_image(image_shape)

    resampled_image, valid_mask = self.variant(pg.resample_image)(source_image,
                                                                  source_coords)
    self.assertFalse(jnp.any(jnp.isnan(resampled_image)))
    self.assertFalse(jnp.any(jnp.isnan(valid_mask)))

    self.assertTrue(jnp.all(valid_mask[valid] == 1.0))
    self.assertTrue(jnp.all(valid_mask[invalid] == 0.0))

    resampled_image_valid = resampled_image[valid]
    resampled_image_valid_expected = 0.5 * (
        source_image[source0] + source_image[source1])

    chex.assert_trees_all_close(resampled_image_valid,
                                resampled_image_valid_expected,
                                rtol=1e-4)

  @chex.all_variants
  def test_pixelwise_argmin(self):
    # Parameters
    num_sources, height, width, num_channels = 3, 32, 65, 3

    # Set up data
    sources = jax.random.uniform(
        jax.random.PRNGKey(42),
        (num_sources, height, width, num_channels),
    )
    target = jnp.zeros((height, width, num_channels))
    source_masks = jax.random.bernoulli(
        jax.random.PRNGKey(43), 0.9, (num_sources, height, width, 1)
    ).astype(sources.dtype)

    # Call current implementation.
    actual_image, actual_mask = self.variant(pg.pixelwise_argmin)(sources,
                                                                  target,
                                                                  source_masks)

    # Reference implementation using fancy indexing.

    # Obtain a (S, H, W, C) tensor of errors w.r.t. the target.  If a mask was
    # provided, use it to place a large error at the corresponding pixels of the
    # error to ensure they're not selected in the argmin (if both are masked it
    # will get dropped anyway by the final mask at the end).
    # Note: Real `inf` breaks argmin.
    inv_mask = jnp.where(source_masks == 0, 1e5, source_masks)
    # TODO(jscholz) Allow user to pass in photometric loss vs. hard-coding L1.
    err_imgs = jnp.abs(target[jnp.newaxis, ...] - sources) * inv_mask
    err_imgs = jnp.sum(err_imgs, axis=-1)  # sum out channel dim before argmin.
    # Apply argmin to select pixels from source images to generate the best
    # possible reconstruction of the target.
    best_source_idxs = jnp.argmin(err_imgs, axis=0)
    best_source_idxs = best_source_idxs.reshape(height * width)
    flattened_imgs = sources.reshape(
        (num_sources, height * width, num_channels)
    )
    source_union = flattened_imgs[
        best_source_idxs, jnp.arange(height * width), :
    ]
    expected_image = source_union.reshape((height, width, num_channels))
    expected_mask = jnp.max(source_masks, axis=0)

    # Verify current implementation matches reference.
    chex.assert_trees_all_equal_shapes(
        (actual_image, actual_mask), (expected_image, expected_mask)
    )
    chex.assert_trees_all_close(
        (actual_image, actual_mask), (expected_image, expected_mask)
    )

  @chex.all_variants
  def test_resize_image(self):
    # The rationale behind this test is that if you have an image of shape
    # (h, w) and you know the depth at a point (i,j) in the image then you can
    # use the camera intrinsics to back project that point into the camera
    # frame.
    #
    # If you scale the image to have shape (h', w') then the point (i, j) from
    # the original image appears at (i', j') = (i*w'/w, j*h'/h) in the scaled
    # image. If the intrinsics have been scaled correctly then using the scaled
    # intrinsics to back project (i', j') should result in the same point in the
    # camera frame as using the original intrinsics to back project (i, j).

    height, width, num_channels = 16, 32, 3
    image = jnp.ones((height, width, num_channels))
    depths = jnp.ones((height * width, 1))
    intrinsics = jnp.asarray([
        [1 / 32, 0., 16.],
        [0., 1 / 16, 8.],
        [0., 0., 1.],
    ])
    coords = pg.raster_coordinates_for(image).reshape((-1, 2))
    camera = camera_type.Camera.at_origin(intrinsics=intrinsics)
    back_projection = pg.depth_to_points(
        camera=camera, img_points=coords, depths=depths)

    new_height, new_width = 5, 6
    new_camera, new_image = self.variant(
        pg.resize_image_and_camera, static_argnums=2)(camera, image,
                                                      (new_height, new_width))
    new_coords = coords * jnp.asarray((new_width / width, new_height / height))
    new_back_projection = pg.depth_to_points(
        camera=new_camera, img_points=new_coords, depths=depths)

    chex.assert_shape(new_image, (new_height, new_width, num_channels))
    chex.assert_trees_all_close(back_projection, new_back_projection, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
