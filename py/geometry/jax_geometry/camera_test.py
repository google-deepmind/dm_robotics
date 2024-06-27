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

"""Tests for geometry.jax.camera."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_robotics.geometry.jax_geometry import camera
import jax
import jax.numpy as jnp


DEFAULT_BATCH_SHAPES = [((),), ((5,),)]


def as_batch(array, batch_shape):
  return jnp.broadcast_to(array, batch_shape + array.shape)


class CameraTest(chex.TestCase, parameterized.TestCase):

  def _check_camera(self, cam, batch_shape):
    self.assertSequenceEqual(cam.batch_shape, batch_shape)
    chex.assert_shape(cam.extrinsics, batch_shape + (4, 4))
    chex.assert_shape(cam.intrinsics, batch_shape + (3, 3))
    chex.assert_trees_all_close(cam.extrinsics,
                                as_batch(jnp.eye(4), batch_shape))
    chex.assert_trees_all_close(cam.intrinsics,
                                as_batch(jnp.eye(3), batch_shape))

  @chex.all_variants
  @parameterized.parameters(DEFAULT_BATCH_SHAPES)
  def test_new(self, batch_shape):
    new_camera = self.variant(camera.Camera.new)
    extrinsics = jnp.eye(4)
    intrinsics = jnp.eye(3)

    # Check batched construction.
    cam = new_camera(
        extrinsics=as_batch(extrinsics, batch_shape),
        intrinsics=as_batch(intrinsics, batch_shape))
    self._check_camera(cam, batch_shape)

    # Check broadcasting against intrinsics.
    cam = new_camera(
        extrinsics=extrinsics, intrinsics=as_batch(intrinsics, batch_shape))
    self._check_camera(cam, batch_shape)

    # Check broadcasting against extrinsics
    cam = new_camera(
        extrinsics=as_batch(extrinsics, batch_shape), intrinsics=intrinsics)
    self._check_camera(cam, batch_shape)

    # Check bad extrinsics shape fails.
    with self.assertRaisesRegex(ValueError, 'extrinsics'):
      new_camera(
          extrinsics=as_batch(jnp.eye(3, 4), batch_shape),
          intrinsics=as_batch(intrinsics, batch_shape))

    # Check bad intrinics shape fails.
    with self.assertRaisesRegex(ValueError, 'intrinsics'):
      new_camera(
          extrinsics=as_batch(extrinsics, batch_shape),
          intrinsics=as_batch(jnp.eye(4), batch_shape))

  @chex.all_variants
  @parameterized.parameters(DEFAULT_BATCH_SHAPES)
  def test_at_origin(self, batch_shape):
    at_origin = self.variant(camera.Camera.at_origin)
    intrinsics = as_batch(jnp.eye(3), batch_shape)

    cam = at_origin(intrinsics=intrinsics)
    self._check_camera(cam, batch_shape)

  @chex.all_variants
  def test_invoke_vmap(self):
    batch_shape = (5,)
    extrinsics = jnp.eye(4)
    intrinsics = jnp.eye(3)

    cam = camera.Camera.new(
        extrinsics=as_batch(extrinsics, batch_shape),
        intrinsics=as_batch(intrinsics, batch_shape))

    def identity(my_cam):
      return my_cam

    # This is not a pointless test. There is a very surprising interaction
    # between vmap and custom dataclass __init__ (or __post_init__)
    # functions where jax will construct an instance of the data class with
    # `object()` as the values. If you try to do data verification in either
    # init function then vmap will fail unless you are super careful.
    self.variant(jax.vmap(identity))(cam)


if __name__ == '__main__':
  absltest.main()
