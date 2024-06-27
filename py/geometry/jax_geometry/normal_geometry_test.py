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

"""Tests for normal_geometry."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_robotics.geometry.jax_geometry import normal_geometry
import jax
from jax import numpy as jnp


class NormalGeometryTest(parameterized.TestCase):

  @chex.all_variants
  def test_points_to_normals(self):
    # This is a square plane that is a constant distance from the camera in the
    # x axis, and slopes away from the camera with increasing y.
    #
    # Normals should all be [0, 1/sqrt(2), -1/sqrt(2)]
    xs, ys = jnp.meshgrid(jnp.linspace(-1, 1, 32), jnp.linspace(-1, 1, 32))
    zs = 2.0 + ys
    points = jnp.dstack([xs, ys, zs])
    ref = jnp.ones_like(points)

    normals = self.variant(normal_geometry.points_to_normals)(
        points, ref=ref, alpha=1.0)

    expected_normal = jnp.asarray([[0., 1., -1.]]) / jnp.sqrt(2.)
    self.assertTrue(jnp.allclose(normals, expected_normal, atol=1e-6))

  @chex.all_variants
  def test_points_to_normals_stability(self):
    point_map = jnp.zeros((64, 64, 3))
    ref = jnp.zeros((64, 64, 3))
    normals = self.variant(normal_geometry.points_to_normals)(point_map, ref,
                                                              0.1)
    self.assertFalse(jnp.any(jnp.isnan(normals)))

  @chex.all_variants
  def test_points_to_normals_grad_not_nan(self):
    # This test exists because I (mdenil) had problems with the gradient
    # producing nans at some point.
    xs, ys = jnp.meshgrid(
        jnp.linspace(-1, 1, 32), jnp.linspace(-1, 1, 32))
    zs = 1.0 + xs ** 2 + ys ** 2

    points = jnp.dstack([xs, ys, zs])

    def loss(points):
      normals = normal_geometry.points_to_normals(
          points, ref=jnp.ones_like(points))
      value = normals @ jnp.asarray([0.0, 0.0, 1.0])
      return jnp.sum(value ** 2)

    value, grad = self.variant(jax.value_and_grad(loss))(points)

    self.assertFalse(jnp.isnan(value))
    self.assertFalse(jnp.any(jnp.isnan(grad)))


if __name__ == '__main__':
  absltest.main()
