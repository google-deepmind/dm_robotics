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
import operator
from typing import Sequence, Tuple

from dm_robotics.geometry.jax_geometry import basic_types
from dm_robotics.geometry.jax_geometry import camera as camera_type
from dm_robotics.geometry.jax_geometry import geometry_utils
from dm_robotics.geometry.jax_geometry import pointcloud_stamped
import jax
from jax import numpy as jnp


# Internal dot product clamped to highest-precision.
_dot = functools.partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)


def raster_flatten(raster: basic_types.Raster) -> basic_types.Cloud:
  """Flatten a Raster into a Cloud."""
  *shape, channels = raster.shape
  return raster.reshape((functools.reduce(operator.mul, shape), channels))


def raster_fold(values: basic_types.Cloud,
                shape: Sequence[int]) -> basic_types.Raster:
  """Fold a Cloud into a Raster."""
  return values.reshape(tuple(shape) + (values.shape[-1],))


def raster_fold_like(values: basic_types.Cloud,
                     like: basic_types.Raster) -> basic_types.Raster:
  return raster_fold(values, like.shape[:-1])


def raster_coordinates(*raster_shape: int) -> basic_types.PointRaster:
  """Generate the coordinates for each cell in a Raster of the specified shape.

  E.g., if img is a Raster of shape (H, W, C) then raster_coordinates((H, W))
  returns an array of coordinates (as a PointRaster).

  Args:
    *raster_shape: The shape of the backing raster, excluding the element
      dimension (i.e. `raster.shape[:-1]`).

  Returns:
    A `PointRaster` with shape `raster_shape + (len(raster_shape),)` such that
    `result[i, j, ..., :]` holds the coordinates of the indexed cell.
  """
  # We need to swap the first two elements of raster_shape because:
  # raster_shape is like (H, W, D, ...); ie. (y, x, z, ...) order.
  # extents is like (W, H, D, ...); ie. (x, y, z, ...) order.
  extents = list(raster_shape)
  extents[0], extents[1] = extents[1], extents[0]
  axes = (jnp.linspace(0, d - 1, num=d) for d in extents)
  return jnp.stack(list(c for c in jnp.meshgrid(*axes, indexing='xy')), axis=-1)


def raster_coordinates_for(
    raster: basic_types.Raster) -> basic_types.PointRaster:
  """Generate coordinates for each cell in the provided raster.

  Equivalent to `raster_coordinates(*raster.shape[:-1])`.

  Args:
    raster: The raster to generate coordinates for.

  Returns:
    A coordinate raster for the provided argument.
  """
  return raster_coordinates(*raster.shape[:-1])


def get_depth_at_points(
    depth_map: basic_types.DepthMap,
    img_coords: basic_types.PointRaster,  # Should be PointCloud
    method: str = 'bilinear',
) -> basic_types.Cloud:
  """Extracts the depth values at requested coordinates.

  Args:
    depth_map: A dense (H, W) or (H, W, 1) array of depth values.
    img_coords: An (N, 2) array of 2D pixel coordinates. Can be integer or
      real-valued.
    method: A string indicating which type of interpolation to perform, if
      real-valued coordinates are provided. Can 'bilinear' or 'nearest'.

  Returns:
    An (N, 1) array of depth values for each point.
  """
  depth_map = jnp.atleast_3d(depth_map)  # (H, W, 1)
  if depth_map.shape[-1] != 1:
    raise ValueError('`depth_map` should have a single channel dimension but '
                     f' has shape {depth_map.shape}.')
  return geometry_utils.interpolate(depth_map, img_coords, method=method)


def intrinsic_camera_matrix(
    img_shape: Tuple[int, int],
    fovy: float = 45,
    inverse: bool = False,
    full: bool = False,
) -> jnp.ndarray:
  """Build intrinsic camera matrix for simple pinhole model.

  Args:
    img_shape: Image shape as (height, width)
    fovy: Vertical field of view.
    inverse: If True, return the inverse camera matrix for mapping from image
      coordinates to camera-frame rays.
    full: If True, returns (3, 4) for multiplation with extrinsics. Else just
      return (3, 3)

  Returns:
    A (3, 3) (or (3, 4) if `full`) array containing the (possibly inverse)
    intrinsics.
  """
  height, width = img_shape
  focal_len = height / 2 / jnp.tan(fovy / 2 * jnp.pi / 180)

  # Note: this matrix omits the x-flip in the mujoco camera matrix.
  intrinsic_matrix = jnp.array([[focal_len, 0, (width - 1) / 2],
                                [0, focal_len, (height - 1) / 2],
                                [0, 0, 1]])

  if inverse:
    intrinsic_matrix = jnp.linalg.inv(intrinsic_matrix)
  if full:
    intrinsic_matrix = jnp.hstack((intrinsic_matrix, jnp.zeros((3, 1))))
  return intrinsic_matrix


def project_pointcloud_to_camera(
    camera: camera_type.Camera,
    pointcloud: pointcloud_stamped.PointCloudStamped,
) -> basic_types.PointCloud:
  """Projects provided 3D points to the camera.

  Note: this function may also be invoked on rasterized inputs, in which case
  the image dimensions are preserved.
  I.e., this:
    cloud = depth_to_points(cam1, raster_coordinates_for(depth_map), depth_map)
    coords = project_pointcloud_to_camera(cam2, cloud)  # cloud is (H, W, 3)
  is equivalent to this:
    cloud = depth_image_to_pointcloud(cam1, depth_map)
    coords = raster_fold_like(
        project_pointcloud_to_camera(cam2, cloud),  # cloud is (W * W, 3)
        depth_map)

  Args:
    camera: A camera to project points into.
    pointcloud: The points to be projected.

  Returns:
    An (N, 2) array of 2D points in the desired camera.
  """
  hmat_world_points = pointcloud.frame.to_hmat()

  # Do the projection matmuls at highest precision. Have noticed significant
  # noise in this calculation when point is close to camera presumably do to
  # bad conditioning in the bfloat16 regime.
  projection = _dot(camera.projection, hmat_world_points)  # (3, 4) @ (4, 4)
  points_hom = geometry_utils.to_homogeneous(pointcloud.points)  # (n, 4)
  projected_hom = _dot(points_hom, projection.T)  # (n, 4) @ (4, 3) -> (n, 3)

  return geometry_utils.from_homogeneous(projected_hom)


def depth_to_points(
    camera: camera_type.Camera,
    img_points: basic_types.PointRaster,  # This should be PointCloud
    depths: basic_types.Cloud,
) -> pointcloud_stamped.PointCloudStamped:
  """Un-projects provided 2D points to 3D.

  Note: this function may also be invoked on rasterized inputs, in which case
  the image dimensions are preserved.
  I.e., this:
    depth_to_points(camera, pg.raster_coordinates_for(depth_map), depth_map)
  is equivalent to this:
    raster_fold_like(depth_image_to_pointcloud(camera, depth_map), depth_map)

  This shape polymorphism can be carried through `project_pointcloud_to_camera`,
  see above.

  Args:
    camera: A Camera describing the acquisition of img_points.
    img_points: An (N, 2) array of 2D pixel coordinates.
    depths: An (N, 1) array of depth values.

  Returns:
    A PointCloudStamped containing the back projected points.
  """
  img_points_hom = geometry_utils.to_homogeneous(img_points)
  projected_3d = depths * _dot(img_points_hom,
                               jnp.linalg.inv(camera.intrinsics).T)

  return pointcloud_stamped.PointCloudStamped(
      frame=camera.frame, points=projected_3d)


def depth_image_to_pointcloud(
    camera: camera_type.Camera,
    depth: basic_types.DepthMap,
) -> pointcloud_stamped.PointCloudStamped:
  """Un-projects every pixel in a depth-image to a point-cloud.

  Note: frame of `camera` and `depth` should agree.

  Converts a depth-map (e.g. (H, W, 1)) containing depths at all image
  coordinates to a point-cloud (e.g. (H*W, 3)) containing xyz coordinates
  in the frame of the camera. Note this PointCloud is stamped in with the frame
  of the camera so it can be transformed to world or other frames as desired.

  Args:
    camera: A Camera describing the image acquisition process in the frame in
      which `depth` was acquired.
    depth: A DepthMap providing a depth value for every pixel as seen from
      `camera`.

  Returns:
    A PointCloudStamped containing the back-projected pixels.
  """
  return depth_to_points(camera, raster_flatten(raster_coordinates_for(depth)),
                         raster_flatten(depth))


def rgbd_to_pointcloud(
    camera: camera_type.Camera,
    image: basic_types.Image,
    depth: basic_types.DepthMap,
) -> Tuple[pointcloud_stamped.PointCloudStamped, basic_types.Cloud]:
  """Converts an RGB image and depth-map to a colored pointcloud.

  Args:
    camera: The camera describing the acquisition of `image`.
    image: (height, width, 3) An RGB image.
    depth: (height, width, 1) A depth map.

  Returns:
    xyz: A PointCloudStamped containing the locations of the backprojected
      points.
    rgb: A Cloud of colors, aligned with the points in `xyz`.
  """
  xyz = depth_image_to_pointcloud(camera, depth)
  rgb = raster_flatten(image)
  return xyz, rgb


def resample_image(
    image: basic_types.Image,  # (H, W, C)
    coords: basic_types.PointRaster,  # (H', W', 2)
) -> Tuple[basic_types.Image, basic_types.Raster]:  # (H', W', C), (H', W', 1)
  """Resamples an image at the specified coordinates.

  Args:
    image: (H, W, C) The source image to sample from.
    coords: (H', W', 2) A raster with height and width corresponding to the
      desired output image shape containing source pixel coordinates (x, y) to
      sample in `image`.

  Returns:
    resampled_image: An image of shape `coords.shape[:2] + (image.shape[-1],)`,
      i.e., (H', W', C), whose pixel values are obtained by linearly
      interpolating the source image at the specified points.
    valid_mask: An image of shape `coords.shape[:2] + (1,)` whose value is `1.0`
      at locations where the target image sampled a valid value, and `0.0`
      otherwise.
  """

  def sample_channel(image):
    return geometry_utils.map_coordinates(image, coords)

  sampled_image = jax.vmap(sample_channel, in_axes=0, out_axes=-1)(
      image.transpose((2, 0, 1))
  )

  # Compute a validity mask.
  x = coords[..., 0]
  y = coords[..., 1]
  height, width = image.shape[:2]
  valid_mask = jnp.logical_and(
      jnp.logical_and(0 <= y, y <= height - 1),
      jnp.logical_and(0 <= x, x <= width - 1)).astype(image.dtype)
  valid_mask = valid_mask[..., jnp.newaxis]

  return sampled_image, valid_mask


def reproject_points(
    source_img_coords: basic_types.PointRaster,
    source_camera: camera_type.Camera,
    source_depth: basic_types.DepthMap,
    target_camera: camera_type.Camera,
) -> basic_types.PointCloud:
  """Projects image coordinates from `source` to `target` viewpoint.

  Note the source/target semantics differ from `reproject_image`. Here we "push"
  points defined in the source view to the target view (requires
  `source_depth`), whereas `reproject_image` "pulls" points from the source view
  to the target view (requires `target_depth`). This difference is due to the
  need to interpolate points for image-reprojection -- we effectively "push"
  each pixel from the target view to the source view, and interpolate the source
  image at that point to obtain a pixel value in the target view.

  Args:
    source_img_coords: An (N, 2) array of 2D pixel coordinates in the source
      camera.
    source_camera: A Camera representing the source viewpoint.
    source_depth: A dense (H, W) or (H, W, 1) array of depth values.
    target_camera: A Camera representing the target viewpoint.

  Returns:
    An (N, 2) array of 2D points in the desired camera.
  """
  depth_at_pts = get_depth_at_points(source_depth, source_img_coords)
  source_cloud = depth_to_points(source_camera, source_img_coords, depth_at_pts)
  return project_pointcloud_to_camera(target_camera, source_cloud)


def reproject_image(
    source_camera: camera_type.Camera,
    source_image: basic_types.Image,
    target_camera: camera_type.Camera,
    target_depth: basic_types.DepthMap,
) -> Tuple[basic_types.Image, basic_types.Raster]:
  """Reprojects a `source_image` to a `target_camera` viewpoint.

  Occlusions may cause incorrect colors to be sampled.

  Args:
    source_camera: A Camera describing the acquisition of `source_image`.
    source_image: A source image to warp.
    target_camera: A Camera describing the acquisition of the target image.
    target_depth: A depth map from the target view.

  Returns:
    projected_image: (H, W, C) The result of reprojecting the source image into
      the target viewpoint.
    valid_mask: (H, W, 1) A binary mask indicating which pixels of
      `projected_image` contain valid data.
  """
  # TODO(jscholz) consider re-writing as a single-pixel reproject and vmap?
  height, width, _ = source_image.shape

  # Project coordinates from target image to source image.
  # TODO(jscholz) allow user to provide source depth-map rather than target?
  # Dense version:
  # target_depth = get_depth_at_points(target_depth, coords)
  # Non-dense version:
  target_cloud = depth_image_to_pointcloud(target_camera, target_depth)
  source_coords = project_pointcloud_to_camera(source_camera, target_cloud)

  source_coords = source_coords.reshape((height, width, 2))

  projected_image, valid_mask = resample_image(source_image, source_coords)

  return projected_image, valid_mask


def batched_reproject(
    source_camera: camera_type.Camera,  # (B,)
    source_image: basic_types.Image,  # (B, H, W, C)
    target_camera: camera_type.Camera,  # (B,)
    target_depth: basic_types.DepthMap,
) -> Tuple[basic_types.Image, basic_types.Raster]:
  """Reprojects an image to a new viewpoint using target pose and depth map.

  Args:
    source_camera: (B,) A Camera describing the acquisition of `source_image`.
    source_image: (B, H, W, C) A source image to warp.
    target_camera: (B,) A Camera describing the acquisition of the target image.
    target_depth: (B, H, W, 1) A depth map from the target view.

  Returns:
    projected_image: (B, H, W, C) The result of reprojecting the source image
      into the target viewpoint.
    valid_mask: (B, H, W, 1) A binary mask indicating which pixels of
      `projected_image` contain valid data.
  """
  # TODO(jscholz) this is simple now; move to user-code.
  return jax.vmap(reproject_image)(
      source_camera, source_image, target_camera, target_depth
  )


def _pixel_argmin(
    sources: jnp.ndarray,  # (S, C)
    target: jnp.ndarray,  # (C,)
    source_masks: jnp.ndarray  # (S, 1)
) -> jnp.ndarray:  # (C,)
  """See docstring for pixelwise_argmin."""
  inv_mask = jnp.where(source_masks == 0, 1e5, source_masks)
  errs = jnp.abs(target[jnp.newaxis, ...] - sources) * inv_mask
  errs = jnp.sum(errs, axis=-1)  # (S,)

  best_source_idx = jnp.argmin(errs)
  num_sources = sources.shape[0]

  # Matrix multiplication here is dramatically faster than indexing on TPU.
  selector = (jnp.arange(num_sources) == best_source_idx).astype(sources.dtype)
  source_union = jnp.dot(selector, sources)

  return source_union


def pixelwise_argmin(
    sources: basic_types.Image,  # (S, H, W, C)
    target: basic_types.Image,  # (H, W, C)
    source_masks: basic_types.Raster  # (S, H, W, 1)
) -> jnp.ndarray:  # (H, W, C)
  """A mosaic of `sources` according to pixel-wise distance to `target`.

  Args:
    sources: A tensor of shape (S, H, W, C) containing S source images.
    target: A single target image of shape (H, W, C)
    source_masks: A tensor of shape (S, H, W, 1) containing masks for each
      source image.  The masks can be arbitrary weights, but only 0 is treated
      as "missing" for the sake of error calculation.

  Returns:
    A single image with shape (H, W, C) built by taking the pixel-wise argmin of
    the distance between each source image and the target image.
  """
  mask_union = jnp.max(source_masks, axis=0)

  sources = sources.transpose((1, 2, 0, 3))  # (H, W, S, C)
  source_masks = source_masks.transpose((1, 2, 0, 3))  # (H, W, S, C)

  fn = jax.vmap(jax.vmap(_pixel_argmin))  # map over (H, W)
  source_union = fn(sources, target, source_masks)

  return source_union, mask_union  # pytype: disable=bad-return-type  # jax-ndarray


def reproject_multisource(
    source_cameras: camera_type.Camera,  # (S,)
    source_imgs: basic_types.Image,  # (S, H, W, C)
    target_camera: camera_type.Camera,  # ()
    target_img: basic_types.Image,  # (H, W, C)
    target_depth: basic_types.DepthMap,  # (H, W, 1)
) -> basic_types.Image:  # (H, W, C)
  """Constructs a target image by reprojecting one or more sources.

  This function generalizes `reproject` to allow reconstructing an image
  from more than a single source.  The final image is obtained by taking
  a pixel-wise argmin of the error between each individual reprojection and a
  target image.

  This technique provides a simple non-geometric way to handle occlusions and
  dis-occlusions from perspective or motion.

  Args:
    source_cameras: (S,) A camera describing the acquisition of `source_imgs`.
    source_imgs: (S, H, W, C) tensor of S source images to warp.
    target_camera: (,) A camera describing the acquisition of `target_img`.
    target_img: The target image we're attempting to synthesize. Needed only to
      compute pixel-wise error for sampling from the reprojections.
    target_depth: (H, W, 1) depth map from the target view.

  Returns:
    (H, W, C) array containing the scene as viewed in the target frame
    using projections from all source images.
  """
  target_img = jax.lax.stop_gradient(target_img)  # Just to be safe.
  num_sources = source_imgs.shape[0]
  target_camera, target_depth = jax.tree.map(
      lambda x: jnp.stack([x] * num_sources, axis=0),
      (target_camera, target_depth),
  )

  projected_images, valid_masks = batched_reproject(source_cameras, source_imgs,
                                                    target_camera, target_depth)

  return pixelwise_argmin(projected_images, target_img, valid_masks)


def scale_camera(
    camera: camera_type.Camera,
    scale_factors: Tuple[float, float],
) -> camera_type.Camera:
  """Scale a camera in the x and y directons.

  Args:
    camera: The camera to scale.
    scale_factors: A tuple of floats `[scale_y, scale_x]` indicating the scaling
      of the camera image.

  Returns:
    A scaled camera.
  """
  scale_y, scale_x = scale_factors
  intrinsics_scale = jnp.diag(jnp.asarray([scale_x, scale_y, 1.]))
  scaled_intrinsics = _dot(intrinsics_scale, camera.intrinsics)
  return camera.replace(intrinsics=scaled_intrinsics)


def resize_camera(
    camera: camera_type.Camera,  # (<camera>)
    old_shape: Tuple[int, int],
    new_shape: Tuple[int, int],
) -> camera_type.Camera:  # (<camera>)
  """Resizes camera intrinsics corresponding an image resize of old -> new."""
  return scale_camera(
      camera, (new_shape[0] / old_shape[0], new_shape[1] / old_shape[1]))


def resize_image_and_camera(
    camera: camera_type.Camera,  # ()
    image: basic_types.Image,  # (H, W, C)
    new_shape: Tuple[int, int],
) -> Tuple[camera_type.Camera, basic_types.Image]:  # (), (H, W, C)
  """Resize an image and an asscociated camera.

  Args:
    camera: A camera that acquired the image.
    image: The image to be resized.
    new_shape: Target shape for the resulting image.

  Returns:
    scaled_camera: A camera that acquired the scaled image.
    scaled_image: The scaled image.
  """
  height, width, num_channels = image.shape
  new_height, new_width = new_shape
  scaled_camera = resize_camera(camera, (height, width), new_shape)

  # TODO(mdenil): use area interpolation for downscaling images?
  scaled_image = jax.image.resize(
      image, (new_height, new_width, num_channels), 'bilinear'
  )

  return scaled_camera, scaled_image
