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

"""Tests for frame_geometry."""

from absl.testing import absltest
import chex
from dm_robotics.geometry import geometry
from dm_robotics.geometry.jax_geometry import frame_geometry as fg
from dm_robotics.transformations import transformations as tr
import jax
import jax.numpy as jnp
import numpy as np
import tree


def _is_array_like(x):
  return isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)


class FrameGeometryTest(chex.TestCase):

  @chex.all_variants
  def test_identity(self):
    pose = fg.Pose.identity()
    self._assert_pose_types(pose)
    @self.variant
    def inv():
      return pose.inv()
    self.assertEqual(pose, inv())

  def test_pose_eq_operator(self):
    position1 = [1., 2., 3.]
    quaternion1 = tr.euler_to_quat([jnp.pi/2, 0., 0.], 'XYZ')
    position2 = [4., 5., 6.]
    quaternion2 = tr.euler_to_quat([0., jnp.pi/2, 0.], 'XYZ')

    pose1 = fg.Pose.new(position=position1, quaternion=quaternion1)
    pose2 = fg.Pose.new(position=position1, quaternion=quaternion1)
    pose3 = fg.Pose.new(position=position2, quaternion=quaternion2)
    self._assert_pose_types(pose1)
    self._assert_pose_types(pose2)
    self._assert_pose_types(pose3)

    self.assertEqual(pose1, pose2)
    self.assertNotEqual(pose1, pose3)

  def test_pose_is_immutable(self):
    # Tests that you cannot assign to position or quaternion fields.
    # Note: elements of position and quaternion are immutable by the nature of
    # jax, so we don't need to test setitem.
    position = [1., 2., 3.]
    quaternion = tr.euler_to_quat([jnp.pi/2, 0., 0.], 'XYZ')
    pose = fg.Pose.new(position=position, quaternion=quaternion)
    with self.assertRaises(AttributeError):
      pose.position = [4., 5., 6]
    with self.assertRaises(AttributeError):
      pose.quaternion = [1., 0., 0., 0]

  @chex.all_variants
  def test_relative_pose_vs_numpy(self):
    position1 = [1., 2., 3.]
    quaternion1 = tr.euler_to_quat([jnp.pi/2, 0., 0.], 'XYZ')
    position2 = [4., 5., 6.]
    quaternion2 = tr.euler_to_quat([jnp.pi/2, 0., 0.], 'XYZ')

    gpose1 = geometry.Pose(position=position1, quaternion=quaternion1)
    gpose2 = geometry.Pose(position=position2, quaternion=quaternion2)

    jpose1 = fg.Pose.new(position=position1, quaternion=quaternion1)
    jpose2 = fg.Pose.new(position=position2, quaternion=quaternion2)
    self._assert_pose_types(jpose1)
    self._assert_pose_types(jpose2)

    expected_pose = gpose1.inv().mul(gpose2)
    @self.variant
    def get_pose():
      return jpose1.inv().mul(jpose2)
    actual_pose = get_pose()

    np.testing.assert_allclose(
        expected_pose.position, actual_pose.position, rtol=1E-5, atol=1E-5)
    np.testing.assert_allclose(
        expected_pose.quaternion, actual_pose.quaternion, rtol=1E-5, atol=1E-5)

  @chex.all_variants
  def test_pose_from_and_to_posquat(self):
    posquat = jnp.concatenate(
        (jnp.array([1., 2., 3.]), tr.axisangle_to_quat([jnp.pi / 2, 0., 0.])))

    @self.variant
    def recover_posquat(posquat):
      return fg.Pose.from_posquat(posquat).to_posquat()

    np.testing.assert_allclose(
        posquat, recover_posquat(posquat), rtol=1E-5, atol=1E-5)

    # Test on (2, 7) array.
    batch_posquat = jnp.stack([posquat, posquat])
    np.testing.assert_allclose(
        batch_posquat, recover_posquat(batch_posquat), rtol=1E-5, atol=1E-5)

    # Test on (2, 2, 7) array.
    batch_posquat = jnp.stack([batch_posquat, batch_posquat])
    np.testing.assert_allclose(
        batch_posquat, recover_posquat(batch_posquat), rtol=1E-5, atol=1E-5)

  @chex.all_variants
  def test_pose_from_and_to_posaxisangle(self):
    posaxisangle = jnp.array([1., 2., 3., jnp.pi/2, 0., 0.])

    @self.variant
    def recover_posaxisangle(posaxisangle):
      return fg.Pose.from_posaxisangle(posaxisangle).to_posaxisangle()

    np.testing.assert_allclose(
        posaxisangle, recover_posaxisangle(posaxisangle), rtol=1E-5, atol=1E-5)

    # Test on (2, 6) array.
    batch_posaxisangle = jnp.stack([posaxisangle, posaxisangle])
    np.testing.assert_allclose(
        batch_posaxisangle,
        recover_posaxisangle(batch_posaxisangle),
        rtol=1E-5,
        atol=1E-5)

    # Test on (2, 2, 6) array.
    batch_posaxisangle = jnp.stack([batch_posaxisangle, batch_posaxisangle])
    np.testing.assert_allclose(
        batch_posaxisangle,
        recover_posaxisangle(batch_posaxisangle),
        rtol=1E-5,
        atol=1E-5)

  @chex.all_variants
  def test_pose_from_and_to_hmat(self):
    hmat = np.eye(4)
    hmat[:3, :3] = tr.euler_to_rmat([jnp.pi/2, jnp.pi/3, jnp.pi/4], 'XYZ')
    hmat[:3, 3] = np.array([1., 2., 3])

    @self.variant
    def recover_hmat(hmat):
      return fg.Pose.from_hmat(hmat).to_hmat()

    # Test non-batch case.
    np.testing.assert_allclose(hmat, recover_hmat(hmat), rtol=1E-5, atol=1E-5)

    # Test single leading batch-dimension (2, 4, 4).
    hmat = jnp.stack([hmat, hmat])
    np.testing.assert_allclose(hmat, recover_hmat(hmat), rtol=1E-5, atol=1E-5)

    # Test two leading batch-dimensions (2, 2, 4, 4).
    hmat = jnp.stack([hmat, hmat])
    np.testing.assert_allclose(hmat, recover_hmat(hmat), rtol=1E-5, atol=1E-5)

  @chex.all_variants
  def test_pose_from_and_to_posrot6(self):
    # Must start from a valid rotation because rot6->rmat is surjective.
    rot6 = tr.rmat_to_rot6(tr.euler_to_rmat([np.pi / 2, 0., 0.]))
    posrot6 = jnp.concatenate((jnp.array([0., 1., 2.]), rot6))

    @self.variant
    def recover_posrot6(posrot6):
      return fg.Pose.from_posrot6(posrot6).to_posrot6()

    # Test non-batch case.
    np.testing.assert_allclose(
        posrot6, recover_posrot6(posrot6), rtol=1E-5, atol=1E-5)

    # Test single leading batch-dimension (2, 9).
    posrot6 = jnp.stack([posrot6, posrot6])
    np.testing.assert_allclose(
        posrot6, recover_posrot6(posrot6), rtol=1E-5, atol=1E-5)

    # Test two leading batch-dimensions (2, 2, 9).
    posrot6 = jnp.stack([posrot6, posrot6])
    np.testing.assert_allclose(
        posrot6, recover_posrot6(posrot6), rtol=1E-5, atol=1E-5)

  @chex.all_variants
  def test_pose_batch_mul(self):
    # Tests pose-multiplication when `Pose` has batch-dimension.
    # Also tests batch `from_axisangle`, `to_hmat`, and `from_hmat`.
    axisangle1 = jnp.array([[1., 2., 3., jnp.pi / 2, 0., 0.],
                            [3., 2., 1., 0., jnp.pi / 3, 0.]])
    axisangle2 = jnp.array([[1., 2., 3., jnp.pi / 2, 0., 0.],
                            [3., 2., 1., 0., jnp.pi / 3, 0.]])

    pose1 = fg.Pose.from_posaxisangle(axisangle1)
    self._assert_pose_types(pose1)
    pose2 = fg.Pose.from_posaxisangle(axisangle2)
    self._assert_pose_types(pose2)

    @self.variant
    def get_pose():
      return pose1.mul(pose2)
    actual_pose = get_pose()

    hmat1 = pose1.to_hmat()
    hmat2 = pose2.to_hmat()
    hmat_prod = jnp.einsum('bij,bjk->bik', hmat1, hmat2)
    expected_pose = fg.Pose.from_hmat(hmat_prod)

    np.testing.assert_allclose(
        actual_pose.position, expected_pose.position, rtol=1E-5, atol=1E-5)
    np.testing.assert_allclose(
        actual_pose.quaternion, expected_pose.quaternion, rtol=1E-5, atol=1E-5)

  def test_maps(self):
    pose = fg.Pose.new([0., 0., 0.], [1., 0., 0., 0.])
    self._assert_pose_types(pose)

    # This map will fail if data validation is added to the default constructor.
    shape_pose = tree.map_structure(lambda x: x.shape, pose)
    def check_element(el):
      self.assertIsInstance(el, tuple)
      self.assertLen(el, 1)
    check_element(shape_pose.position)
    check_element(shape_pose.quaternion)

    # Verify that vmap works through Pose. Its easy to run afoul of the vmap
    # internal machinery if you do data validation in the default constructor.
    batch_pose = tree.map_structure(
        lambda x: jnp.stack([x] * 3, axis=0), pose
    )
    def transform_pose(p):
      self.assertSequenceEqual(p.position.shape, (3,))
      self.assertSequenceEqual(p.quaternion.shape, (4,))
      return tree.map_structure(lambda x: 2.0 * x, p)
    jax.vmap(transform_pose)(batch_pose)

  def _assert_pose_types(self, pose: fg.Pose):
    self.assertTrue(_is_array_like(pose.position))
    self.assertTrue(_is_array_like(pose.quaternion))

if __name__ == '__main__':
  absltest.main()
