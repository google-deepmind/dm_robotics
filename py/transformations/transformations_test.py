# Copyright 2020 DeepMind Technologies Limited.
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

"""Tests for transformations."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.transformations import transformations
import numpy as np

_NUM_RANDOM_SAMPLES = 1000


def _vel_transform_kdl(ht, t):
  # Sample implementation of velocity transform, from KDL source.
  r = ht[0:3, 0:3]
  p = ht[0:3, 3]

  tlin = t[0:3]
  tang = t[3:6]

  out_ang = r.dot(tang)
  out_lin = r.dot(tlin) + np.cross(p, out_ang)
  return np.concatenate([out_lin, out_ang])


def _force_transform_kdl(ht, w):
  # Sample implementation of force transform, from KDL source.
  r = ht[0:3, 0:3]
  p = ht[0:3, 3]

  f = w[0:3]
  t = w[3:6]
  out_lin = r.dot(f)

  out_ang = np.cross(p, out_lin) + r.dot(t)
  return np.concatenate([out_lin, out_ang])


class TransformationsTest(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._random_state = np.random.RandomState()

  @parameterized.parameters(
      {'cfg': (1, 2, 3, np.radians(45), np.radians(45), np.radians(45))},
      {'cfg': (0, 0, 0, np.radians(45), np.radians(45), np.radians(45))}
  )
  def test_homogeneous_matrix_construction(self, cfg):
    x, y, z, rx, ry, rz = cfg
    ht = transformations.poseuler_to_hmat(
        np.array([x, y, z, rx, ry, rz]), 'ZYZ')

    rotx = transformations.rotation_z_axis(rx, True)
    roty = transformations.rotation_y_axis(ry, True)
    rotz = transformations.rotation_z_axis(rz, True)
    ht_target = transformations.pos_to_hmat(
        np.array([x, y, z])).dot(rotx).dot(roty).dot(rotz)
    np.testing.assert_allclose(ht, ht_target)

  @parameterized.parameters(
      {'a': [1, 2, 3], 'b': [0, 1, 0]},
      {'a': [0, 1, 2], 'b': [-2, 1, 0]}
  )
  def test_cross_product(self, a, b):
    npver = np.cross(a, b)
    matver = transformations.cross_mat_from_vec3(a).dot(b)
    np.testing.assert_allclose(npver, matver)

  @parameterized.parameters(
      {
          'quat': [-0.41473841, 0.59483601, -0.45089078, 0.52044181],
          'truemat':
              np.array([[0.05167565, -0.10471773, 0.99315851],
                        [-0.96810656, -0.24937912, 0.02407785],
                        [0.24515162, -0.96272751, -0.11426475]])
      },
      {
          'quat': [0.08769298, 0.69897558, 0.02516888, 0.7093022],
          'truemat':
              np.array([[-0.00748615, -0.08921678, 0.9959841],
                        [0.15958651, -0.98335294, -0.08688582],
                        [0.98715556, 0.15829519, 0.02159933]])
      },
      {
          'quat': [0.58847272, 0.44682507, 0.51443343, -0.43520737],
          'truemat':
              np.array([[0.09190557, 0.97193884, 0.21653695],
                        [-0.05249182, 0.22188379, -0.97365918],
                        [-0.99438321, 0.07811829, 0.07141119]])
      },
  )
  def test_quat_to_mat(self, quat, truemat):
    """Tests hard-coded quat-mat pairs generated from mujoco if mj not avail."""
    mat = transformations.quat_to_mat(quat)
    np.testing.assert_allclose(mat[0:3, 0:3], truemat, atol=1e-7)

  @parameterized.parameters(
      {'twist': [1, 0, 0, 0, 0, 0],
       'cfg': (0, 0, 0, np.radians(0), np.radians(90), np.radians(0))},

      {'twist': [1, 2, 3, -3, 2, -1],
       'cfg': (-1, 2, 3, np.radians(30), np.radians(60), np.radians(90))}
  )
  def test_velocity_transform_special(self, twist, cfg):
    # Test for special values that often cause numerical issues.
    x, y, z, rx, ry, rz = cfg

    ht = transformations.poseuler_to_hmat(np.array([x, y, z, rx, ry, rz]),
                                          'ZYZ')
    tv = transformations.velocity_transform(ht)
    tt = tv.dot(twist)

    v2kdl = _vel_transform_kdl(ht, twist)
    np.testing.assert_allclose(tt, v2kdl)

  @parameterized.parameters(
      {'wrench': [1, 0, 0, 0, 0, 0],
       'cfg': (0, 0, 0, np.radians(0), np.radians(90), np.radians(0))},

      {'wrench': [1, 2, 3, -3, 2, -1],
       'cfg': (-1, 2, 3, np.radians(30), np.radians(60), np.radians(90))}
  )
  def test_force_transform_special(self, wrench, cfg):
    # Test for special values that often cause numerical issues.
    x, y, z, rx, ry, rz = cfg

    ht = transformations.poseuler_to_hmat(
        np.array([x, y, z, rx, ry, rz]), 'XYZ')
    tw = transformations.force_transform(ht)
    wt = tw.dot(wrench)

    w2kdl = _force_transform_kdl(ht, wrench)
    np.testing.assert_allclose(wt, w2kdl)

  @parameterized.parameters(
      {'state': [0, 0, 0]},
      {'state': [1.0, 2.0, np.radians(60)]}
  )
  def test_homogeneous_conversion_2d_special(self, state):
    # Test for special values that often cause numerical issues.
    x = np.array(state)
    ht = transformations.postheta_to_matrix_2d(x)
    x2 = transformations.matrix_to_postheta_2d(ht)
    np.testing.assert_allclose(x, x2)

  @parameterized.parameters(
      {'cfg': (0, 0, 0), 'vel': [0, 0, 0]},
      {'cfg': (1, -0.5, np.radians(-30)), 'vel': [-1, 1.5, 0.5]}
  )
  def test_velocity_transform_2d_special(self, cfg, vel):
    # Test for special values that often cause numerical issues.
    x, y, theta = cfg
    ht = transformations.postheta_to_matrix_2d(np.array([x, y, theta]))
    v_conv = transformations.velocity_transform_2d(ht, vel)
    v_oper = transformations.velocity_transform_2d(ht).dot(vel)
    np.testing.assert_allclose(v_conv, v_oper)

  @parameterized.parameters(
      {'cfg': (0, 0, 0), 'force': [0, 0, 0]},
      {'cfg': (1, -0.5, np.radians(-30)), 'force': [-1, 2, 3.22]}
  )
  def test_force_transform_2d_special(self, cfg, force):
    # Test for special values that often cause numerical issues.
    x, y, theta = cfg
    ht = transformations.postheta_to_matrix_2d(np.array([x, y, theta]))
    ft_conv = transformations.force_transform_2d(ht, force)
    ft_oper = transformations.force_transform_2d(ht).dot(force)
    np.testing.assert_allclose(ft_conv, ft_oper)

  @parameterized.parameters(
      {'angles': (0, 0, 0)},
      {'angles': (-0.1, 0.4, -1.3)}
  )
  def test_euler_to_rmat_special(self, angles):
    # Test for special values that often cause numerical issues.
    r1, r2, r3 = angles
    orderings = ('XYZ', 'XYX', 'XZY', 'ZYX', 'YZX', 'ZXY', 'YXZ', 'XZX', 'YXY',
                 'YZY', 'ZXZ', 'ZYZ')

    for ordering in orderings:
      r = transformations.euler_to_rmat(np.array([r1, r2, r3]), ordering)
      euler_angles = transformations.rmat_to_euler(r, ordering)
      np.testing.assert_allclose(euler_angles, [r1, r2, r3])

  @parameterized.parameters(
      {'rot': (np.pi, 0, 0)},
      {'rot': (0, 0, 0)},
      {'rot': (np.radians(10), np.radians(-30), np.radians(45))},
      {'rot': (np.radians(45), np.radians(45), np.radians(45))},
      {'rot': (0, np.pi, 0)},
      {'rot': (0, 0, np.pi)},
  )
  def test_rmat_axis_angle_conversion_special(self, rot):
    # Test for special values that often cause numerical issues.
    forward = transformations.euler_to_rmat(np.array(rot), ordering='ZYZ')
    w = transformations.rmat_to_axisangle(forward)
    backward = transformations.axisangle_to_rmat(w)
    np.testing.assert_allclose(forward, backward)

  @parameterized.parameters(
      {'rot6': np.zeros(6)},
      {'rot6': np.ones(6)},
      {'rot6': np.ones(6) * 1e-8},
      {'rot6': np.array([1., 2., 3., 0., 0., 0.])},
      {'rot6': np.array([0., 0., 0., 1., 2., 3.])},
      {'rot6': np.array([1., 2., 3., 4., 5., 6.])},
      {'rot6': np.array([1., 2., 3., 4., 5., 6.]) * -1},
  )
  def test_rot6_to_rmat(self, rot6):
    # Test that rot6 converts to valid rotations for arbitrary inputs.
    rmat = transformations.rot6_to_rmat(np.array(rot6))
    should_be_identity = rmat.T @ rmat
    np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-15)

  @parameterized.parameters(
      {'euler': (np.pi, 0, 0)},
      {'euler': (0, 0, 0)},
      {'euler': (np.radians(10), np.radians(-30), np.radians(45))},
      {'euler': (np.radians(45), np.radians(45), np.radians(45))},
      {'euler': (0, np.pi, 0)},
      {'euler': (0, 0, np.pi)},
  )
  def test_rmat_rot6_conversion_special(self, euler):
    # Test for special values that often cause numerical issues.
    rmat = transformations.euler_to_rmat(np.array(euler), ordering='ZYZ')
    rot6 = transformations.rmat_to_rot6(rmat)
    recovered_rmat = transformations.rot6_to_rmat(rot6)
    np.testing.assert_allclose(rmat, recovered_rmat)

  def test_rmat_rot6_conversion_random(self):
    # Tests cycle-consistency for a set of random valid orientations.
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat = self._random_quaternion()
      original_rmat = transformations.quat_to_mat(quat)[:3, :3]
      gs = transformations.rmat_to_rot6(original_rmat)
      recovered_rmat = transformations.rot6_to_rmat(gs)
      np.testing.assert_allclose(original_rmat, recovered_rmat)

  @parameterized.parameters(
      {'pos': (0, 0, 0), 'rot': (0, 0, 0)},
      {'pos': (1, 2, 3), 'rot': (0, 0, 0)},
      {'pos': (1, 2, 3), 'rot': (np.pi, 0., 0.)},
      {'pos': (1, 2, 3),
       'rot': (np.radians(30), np.radians(45), np.radians(60))}
  )
  def test_hmat_twist_conversion(self, pos, rot):
    x, y, z = pos
    r1, r2, r3 = rot
    ordering = 'XYZ'

    poseuler = np.array([x, y, z, r1, r2, r3])
    ht = transformations.poseuler_to_hmat(
        np.array([x, y, z, r1, r2, r3]), ordering)

    xi = transformations.hmat_to_twist(ht)
    ht2 = transformations.twist_to_hmat(xi)
    poseuler2 = transformations.hmat_to_poseuler(ht2, ordering)

    np.testing.assert_allclose(ht, ht2)
    np.testing.assert_allclose(poseuler, poseuler2)

  @parameterized.parameters(
      {'pos': (0.1, 0.2, 0.3), 'quat': (1., 0., 0., 0.)},
      {'pos': (0.1, 0.2, 0.3), 'quat': (0., 1., 0., 0.)},
      {'pos': (0.1, 0.2, 0.3), 'quat': (0., 0., 1., 0.)},
      {'pos': (0.1, 0.2, 0.3), 'quat': (0., 0., 0., 1.)},
      {'pos': (0.1, 0.2, 0.3), 'quat': (0.5, 0.5, 0.5, 0.5)},
  )
  def test_hmat_twist_conversion_from_quat(self, pos, quat):
    ht = transformations.quat_to_mat(quat)
    ht[0:3, 3] = pos

    xi = transformations.hmat_to_twist(ht)
    ht2 = transformations.twist_to_hmat(xi)
    quat2 = transformations.mat_to_quat(ht2)
    pos2 = ht2[:3, 3]

    np.testing.assert_allclose(ht, ht2, atol=1e-7)  # needed to drop prec.
    np.testing.assert_allclose(pos, pos2)
    self.assertTrue(np.allclose(quat, quat2) or np.allclose(quat, -quat2))

  @parameterized.parameters(
      {'pos': (0, 0, 0), 'rot': (0, 0, 0)},
      {'pos': (1, 2, 3), 'rot': (0, 0, 0)},
      {'pos': (-1, -2, -3),
       'rot': (np.radians(30), np.radians(-45), np.radians(60))}
  )
  def test_se3_integration(self, pos, rot):
    # Tests whether sucessive applications of a homogeneous transform
    # is equivalent to scaling the magnitude of the exponential coordinate
    # representation of that transform.
    # This is a useful result which illustrates that the twist parameterizes
    # the 6D manifold of the rotation H compactly, and can be used to
    # generalize or interpolate the effect of a transform over time.

    x, y, z = pos
    r1, r2, r3 = rot

    n = 3
    ordering = 'XYZ'

    ht = transformations.poseuler_to_hmat(
        np.array([x, y, z, r1, r2, r3]), ordering)
    xi = transformations.hmat_to_twist(ht)

    # verify that two applications of H is equivalent to doubling theta
    ht2_mult = np.linalg.matrix_power(ht, n)  # H.dot(H)
    ht2_exp = transformations.twist_to_hmat(xi * n)

    poseuler2_mult = transformations.hmat_to_poseuler(ht2_mult, ordering)
    poseuler2_exp = transformations.hmat_to_poseuler(ht2_exp, ordering)

    np.testing.assert_allclose(ht2_mult, ht2_exp)
    np.testing.assert_allclose(poseuler2_mult, poseuler2_exp)

  @parameterized.parameters(
      {
          'axisangle': np.array([0, 0, 0])
      },
      {'axisangle': np.array([np.pi / 6, -np.pi / 4, np.pi * 2. / 3])},
      {'axisangle': np.array([np.pi / 2, np.pi, np.pi / 2])},
      {'axisangle': np.array([np.pi, np.pi, np.pi])},
      {'axisangle': np.array([-np.pi, -np.pi, -np.pi])},
  )
  def test_axis_angle_to_quat_special(self, axisangle):
    # Test for special values that often cause numerical issues.
    rmat = transformations.axisangle_to_rmat(axisangle)
    quat_true = transformations.mat_to_quat(rmat)
    quat_test = transformations.axisangle_to_quat(axisangle)
    self.assertTrue(
        np.allclose(quat_true, quat_test) or np.allclose(quat_true, -quat_test))

  def test_axis_angle_to_quat_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      axisangle = self._random_state.rand(3)
      rmat = transformations.axisangle_to_rmat(axisangle)
      quat_true = transformations.mat_to_quat(rmat)
      quat_test = transformations.axisangle_to_quat(axisangle)
      self.assertTrue(
          np.allclose(quat_true, quat_test) or
          np.allclose(quat_true, -quat_test))

  @parameterized.parameters(
      {'euler_vec': np.array([0, 0, 0])},
      {'euler_vec': np.array([np.pi / 6, -np.pi / 4, np.pi * 2. / 3])},
      {'euler_vec': np.array([np.pi / 2, np.pi, np.pi / 2])},
      {'euler_vec': np.array([np.pi, np.pi, np.pi])},
      {'euler_vec': np.array([-np.pi, -np.pi, -np.pi])},
  )
  def test_quat_to_axis_angle_special(self, euler_vec):
    # Test for special values that often cause numerical issues.
    rmat = transformations.euler_to_rmat(euler_vec, ordering='XYZ')
    quat = transformations.euler_to_quat(euler_vec, ordering='XYZ')
    axisangle_true = transformations.rmat_to_axisangle(rmat)
    axisangle_test = transformations.quat_to_axisangle(quat)
    np.testing.assert_allclose(axisangle_true, axisangle_test)

  @parameterized.parameters(
      {'quat': np.array([0., 1., 2., 3.]),
       'expected_quat': np.array([0., 1., 2., 3.])},
      {'quat': np.array([1., 2., 3., 4.]),
       'expected_quat': np.array([1., 2., 3., 4.])},
      {'quat': np.array([-1., 2., 3., 4.]),
       'expected_quat': np.array([1., -2., -3., -4.])},
      {'quat': np.array([-1., -2., -3., -4.]),
       'expected_quat': np.array([1., 2., 3., 4.])},
      {'quat': np.array([
          [0., 1., 2., 3.],
          [1., 2., 3., 4.],
          [-1., 2., 3., 4.],
          [-1., -2., -3., -4.]]),
       'expected_quat': np.array([
           [0., 1., 2., 3.],
           [1., 2., 3., 4.],
           [1., -2., -3., -4.],
           [1., 2., 3., 4.]])},
  )
  def test_quat_leading_positive(self, quat, expected_quat):
    np.testing.assert_array_equal(
        transformations.positive_leading_quat(quat), expected_quat)

  def test_quat_to_axis_angle_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat = self._random_quaternion()
      mat = transformations.quat_to_mat(quat)
      axisangle_true = transformations.rmat_to_axisangle(mat[0:3, 0:3])
      axisangle_test = transformations.quat_to_axisangle(quat)
      np.testing.assert_allclose(axisangle_true, axisangle_test)

  def test_quat_mul_vs_mat_mul_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat1 = self._random_quaternion()
      quat2 = self._random_quaternion()
      rmat1 = transformations.quat_to_mat(quat1)[0:3, 0:3]
      rmat2 = transformations.quat_to_mat(quat2)[0:3, 0:3]
      quat_prod = transformations.quat_mul(quat1, quat2)
      rmat_prod_q = transformations.quat_to_mat(quat_prod)[0:3, 0:3]
      rmat_prod = rmat1.dot(rmat2)
      np.testing.assert_allclose(rmat_prod, rmat_prod_q)

  def test_quat_mul_vs_mat_mul_random_batched(self):
    quat1 = np.stack(
        [self._random_quaternion() for _ in range(_NUM_RANDOM_SAMPLES)], axis=0)
    quat2 = np.stack(
        [self._random_quaternion() for _ in range(_NUM_RANDOM_SAMPLES)], axis=0)
    quat_prod = transformations.quat_mul(quat1, quat2)
    for k in range(_NUM_RANDOM_SAMPLES):
      rmat1 = transformations.quat_to_mat(quat1[k])[0:3, 0:3]
      rmat2 = transformations.quat_to_mat(quat2[k])[0:3, 0:3]
      rmat_prod_q = transformations.quat_to_mat(quat_prod[k])[0:3, 0:3]
      rmat_prod = rmat1.dot(rmat2)
      np.testing.assert_allclose(rmat_prod, rmat_prod_q, atol=1e-5)

  def test_quat_slerp_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat0 = self._random_quaternion()
      quat1 = self._random_quaternion()

      # Test poles
      np.testing.assert_allclose(
          transformations.quat_slerp(quat0, quat1, 0.0), quat0)
      np.testing.assert_allclose(
          transformations.quat_slerp(quat0, quat1, 1.0), quat1)

      # Test slerp gives the same as rotating.
      full_angle = self._random_state.uniform(0, 90)
      frac = self._random_state.uniform(0, 1)

      # Full rotation and partial rotations
      full_quat_rot = transformations.euler_to_quat(
          [np.radians(full_angle), 0., 0.])
      partial_quat_rot = transformations.euler_to_quat(
          [np.radians(full_angle) * frac, 0., 0.])

      # Rotate the quaternion partially and check it is equivalent to slerp.
      full_rotated_quat = transformations.quat_mul(quat0, full_quat_rot)
      partial_rotated_quat = transformations.quat_mul(quat0, partial_quat_rot)
      slerp_quat = transformations.quat_slerp(quat0, full_rotated_quat, frac)
      np.testing.assert_allclose(partial_rotated_quat, slerp_quat, atol=1e-4)

      # Test that it takes the shortest path
      full_angle = self._random_state.uniform(0, 90)
      frac = self._random_state.uniform(0, 1)

      # Generate target quat by rotating fractional-angle around X from quat0.
      quat_fract_rot = transformations.euler_to_quat(
          [np.radians(-full_angle * frac), 0., 0.])
      quat_interp_true = transformations.quat_mul(quat0, quat_fract_rot)
      # Generate quat at target angle and interpolate using slerp.
      quat_rot = transformations.euler_to_quat(
          [np.radians(360. - full_angle), 0., 0.])
      quat2 = transformations.quat_mul(quat0, quat_rot)
      quat_interp_slerp = transformations.quat_slerp(quat0, quat2, frac)
      # Generate alternative interplated quat by scaling log along relative quat
      quat_interp_log = (
          transformations.quat_mul(
              quat0,
              transformations.quat_exp(
                  transformations.quat_log(
                      transformations.quat_diff_passive(quat0, quat2)) * frac)))
      self.assertTrue(
          np.allclose(quat_interp_slerp, quat_interp_true, atol=1e-4) or
          np.allclose(quat_interp_slerp, -1 * quat_interp_true, atol=1e-4))
      self.assertTrue(
          np.allclose(quat_interp_log, quat_interp_true, atol=1e-4) or
          np.allclose(quat_interp_log, -1 * quat_interp_true, atol=1e-4))

  def test_quat_diff_passive_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      # Get the source and target quaternions and their passive difference.
      source = self._random_quaternion()
      target = self._random_quaternion()
      diff = transformations.quat_diff_passive(source, target)

      # Take a vector expressed in the target frame and express it in the
      # source frame using the difference.
      vec_t = np.random.random(3)
      vec_s = transformations.quat_rotate(diff, vec_t)

      # Bring them both in the world frame and check they are the same.
      vec_w1 = transformations.quat_rotate(source, vec_s)
      vec_w2 = transformations.quat_rotate(target, vec_t)
      np.testing.assert_allclose(vec_w1, vec_w2)

  def test_quat_diff_passive_random_batched(self):
    # Get the source and target quaternions and their passive difference.
    source = np.stack(
        [self._random_quaternion() for _ in range(_NUM_RANDOM_SAMPLES)], axis=0)
    target = np.stack(
        [self._random_quaternion() for _ in range(_NUM_RANDOM_SAMPLES)], axis=0)
    diff = transformations.quat_diff_passive(source, target)

    for k in range(_NUM_RANDOM_SAMPLES):
      # Take a vector expressed in the target frame and express it in the
      # source frame using the difference.
      vec_t = np.random.random(3)
      vec_s = transformations.quat_rotate(diff[k], vec_t)

      # Bring them both in the world frame and check they are the same.
      vec_w1 = transformations.quat_rotate(source[k], vec_s)
      vec_w2 = transformations.quat_rotate(target[k], vec_t)
      np.testing.assert_allclose(vec_w1, vec_w2)

  def test_quat_diff_active_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      # Get the source and target quaternions and their active difference.
      source = self._random_quaternion()
      target = self._random_quaternion()
      diff = transformations.quat_diff_active(source, target)

      # Take a vector that has been rotated by source quaternion and rotate it
      # by target quaternion by applying the difference.
      vec_rotated_s = np.random.random(3)
      vec_rotated_t = transformations.quat_rotate(diff, vec_rotated_s)

      # Invert the rotations on both vectors and ensure the final vector is the
      # same.
      vec_1 = transformations.quat_rotate(
          transformations.quat_inv(source), vec_rotated_s)
      vec_2 = transformations.quat_rotate(
          transformations.quat_inv(target), vec_rotated_t)
      np.testing.assert_allclose(vec_1, vec_2)

  def test_quat_diff_active_random_batched(self):
    # Get the source and target quaternions and their passive difference.
    source = np.stack(
        [self._random_quaternion() for _ in range(_NUM_RANDOM_SAMPLES)], axis=0)
    target = np.stack(
        [self._random_quaternion() for _ in range(_NUM_RANDOM_SAMPLES)], axis=0)
    diff = transformations.quat_diff_active(source, target)

    for k in range(_NUM_RANDOM_SAMPLES):
      # Take a vector that has been rotated by source quaternion and rotate it
      # by target quaternion by applying the difference.
      vec_rotated_s = np.random.random(3)
      vec_rotated_t = transformations.quat_rotate(diff[k], vec_rotated_s)

      # Invert the rotations on both vectors and ensure the final vector is the
      # same.
      vec_1 = transformations.quat_rotate(
          transformations.quat_inv(source[k]), vec_rotated_s)
      vec_2 = transformations.quat_rotate(
          transformations.quat_inv(target[k]), vec_rotated_t)
      np.testing.assert_allclose(vec_1, vec_2)

  def test_quat_dist_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      # test with normalized quaternions for stability of test
      source = self._random_quaternion()
      target = self._random_quaternion()
      self.assertGreater(transformations.quat_dist(source, target), 0)
      np.testing.assert_allclose(
          transformations.quat_dist(source, source), 0, atol=1e-9)

  def test_quat_dist_random_batched(self):
    # Test batched quat dist
    source_quats = np.stack(
        [self._random_quaternion() for _ in range(_NUM_RANDOM_SAMPLES)], axis=0)
    target_quats = np.stack(
        [self._random_quaternion() for _ in range(_NUM_RANDOM_SAMPLES)], axis=0)
    np.testing.assert_allclose(
        transformations.quat_dist(source_quats, source_quats), 0, atol=1e-9)
    np.testing.assert_equal(
        transformations.quat_dist(source_quats, target_quats) > 0, 1)

  @parameterized.parameters(
      {'source': (1., 0., 0., 0.), 'target': (0., 1., 0., 0.), 'angle': np.pi},
      {'source': (1., 0., 0., 0.),
       'target': (0.86602540378, 0.5, 0., 0.),
       'angle': np.pi / 3
       },
      {'source': (1., 0., 0., 0.),
       'target': (1./ np.sqrt(2), 1./ np.sqrt(2), 0., 0.),
       'angle': np.pi / 2},
      {'source': np.array([
          [1., 0., 0., 0.],
          [1., 0., 0., 0.],
          [1., 0., 0., 0.]]),
       'target': np.array([
           [0., 1., 0., 0.],
           [0.86602540378, 0.5, 0., 0.],
           [1./ np.sqrt(2), 1./ np.sqrt(2), 0., 0.]]),
       'angle': np.array([np.pi, np.pi / 3, np.pi / 2])},
  )
  def test_quat_dist_deterministic(self, source, target, angle):
    predicted_angle = transformations.quat_dist(source, target)
    if np.asarray(source).ndim > 1:
      self.assertSequenceAlmostEqual(angle, predicted_angle)
    else:
      self.assertAlmostEqual(angle, predicted_angle)

  @parameterized.parameters(
      {'rot': (np.pi, 0, 0), 'angle': np.pi},
      {'rot': (0, 0, 0), 'angle': 0},
      {'rot': (np.radians(10), np.radians(-30), np.radians(45)),
       'angle': 0.9128419},
      {'rot': (np.radians(45), np.radians(45), np.radians(45)),
       'angle': 1.4975074},
      {'rot': (0, np.pi, 0), 'angle': np.pi},
      {'rot': (0, 0, np.pi), 'angle': np.pi},
      {'rot': np.array([
          [(0, np.pi, 0)],
          [(0, 0, np.pi)]]), 'angle': np.array([np.pi, np.pi])},
  )
  def test_quat_angle(self, rot, angle):
    # Test for special values that often cause numerical issues.
    if np.asarray(rot).ndim > 1:
      quat = np.stack([
          transformations.euler_to_quat(np.array(roti), ordering='XYZ')
          for roti in rot], axis=0)
      computed_angle = transformations.quat_angle(quat)
      self.assertSequenceAlmostEqual(angle, computed_angle)
    else:
      quat = transformations.euler_to_quat(np.array(rot), ordering='XYZ')
      computed_angle = transformations.quat_angle(quat)
      self.assertAlmostEqual(angle, computed_angle)

  def test_quat_between_vectors_random(self):
    # test quat_between_vectors with random vectors
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat = self._random_quaternion()
      source_vec = np.random.random(3)
      target_vec = transformations.quat_rotate(quat, source_vec)
      computed_quat = transformations.quat_between_vectors(
          source_vec, target_vec)
      computed_target = transformations.quat_rotate(computed_quat, source_vec)
      np.testing.assert_allclose(target_vec, computed_target, atol=0.005)

  def test_quat_between_vectors_inverse(self):
    # test quat_between_vectors with inverse vectors
    for _ in range(_NUM_RANDOM_SAMPLES):
      source_vec = np.random.random(3)
      target_vec = -source_vec
      computed_quat = transformations.quat_between_vectors(source_vec,
                                                           target_vec)
      computed_target = transformations.quat_rotate(computed_quat, source_vec)
      np.testing.assert_allclose(computed_target, target_vec)

  def test_quat_between_vectors_parallel(self):
    # test quat_between_vectors with parallel vectors
    for _ in range(_NUM_RANDOM_SAMPLES):
      source_vec = np.random.random(3)
      target_vec = source_vec
      computed_quat = transformations.quat_between_vectors(source_vec,
                                                           target_vec)
      computed_target = transformations.quat_rotate(computed_quat, source_vec)
      np.testing.assert_allclose(computed_target, target_vec)

  def test_quat_log_and_exp_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat = self._random_quaternion()
      log_quat = transformations.quat_log(quat)
      orig_quat = transformations.quat_exp(log_quat)
      np.testing.assert_allclose(quat, orig_quat, atol=1e-07)

  def test_quat_log_and_exp_random_batched(self):
    # Test batching of quats
    quat = np.stack(
        [self._random_quaternion() for k in range(_NUM_RANDOM_SAMPLES)], axis=0)
    log_quat = transformations.quat_log(quat)
    orig_quat = transformations.quat_exp(log_quat)
    np.testing.assert_allclose(quat, orig_quat, atol=1e-07)

  def test_quat_integration(self):

    for _ in range(_NUM_RANDOM_SAMPLES):
      integrated_quat = self._random_quaternion()
      target_quat = self._random_quaternion()

      # Because of the numerical approximations, we do 5 integrations steps to
      # reach the target and recompute the velocity to be applied at each step.
      for _ in range(5):

        # Get the quaternion difference between the current and target quat
        diff = transformations.quat_diff_active(integrated_quat, target_quat)

        # Get the angular velocity required to reach the target in one step
        angle = transformations.quat_angle(diff)
        axis = transformations.quat_axis(diff)

        # Scale the velocity for numerical stability
        vel = angle * axis * 0.999
        integrated_quat = transformations.integrate_quat(integrated_quat, vel)
        integrated_quat /= np.linalg.norm(integrated_quat)

      self.assertTrue(
          np.allclose(integrated_quat, target_quat, atol=1e-8) or
          np.allclose(integrated_quat, -1 * target_quat, atol=1e-8))

  def test_pos_rmat_to_hmat_batch(self):
    test_pos_nobatch = transformations.pos_to_hmat([1., 2., 3.])
    true_pos_nobatch = np.array([[1, 0, 0, 1.],
                                 [0, 1, 0, 2.],
                                 [0, 0, 1, 3.],
                                 [0, 0, 0, 1.]])
    np.testing.assert_allclose(test_pos_nobatch, true_pos_nobatch)

    test_pos_batch = transformations.pos_to_hmat([
        [1., 2., 3.], [4., 5., 6.]])
    true_pos_batch = np.array([
        [[1, 0, 0, 1.],
         [0, 1, 0, 2.],
         [0, 0, 1, 3.],
         [0, 0, 0, 1.]],
        [[1, 0, 0, 4.],
         [0, 1, 0, 5.],
         [0, 0, 1, 6.],
         [0, 0, 0, 1.]],
        ])
    np.testing.assert_allclose(test_pos_batch, true_pos_batch)

    test_rmat_nobatch = transformations.rmat_to_hmat(np.eye(3))
    true_rmat_nobatch = np.array([[1, 0, 0, 0.],
                                  [0, 1, 0, 0.],
                                  [0, 0, 1, 0.],
                                  [0, 0, 0, 1.]])
    np.testing.assert_allclose(test_rmat_nobatch, true_rmat_nobatch)

    test_rmat_batch = transformations.rmat_to_hmat(
        np.array([np.eye(3), np.eye(3)]))
    true_rmat_batch = np.array([
        [[1, 0, 0, 0.],
         [0, 1, 0, 0.],
         [0, 0, 1, 0.],
         [0, 0, 0, 1.]],
        [[1, 0, 0, 0.],
         [0, 1, 0, 0.],
         [0, 0, 1, 0.],
         [0, 0, 0, 1.]],
        ])
    np.testing.assert_allclose(test_rmat_batch, true_rmat_batch)

  def _random_quaternion(self):
    """Returns a normalized quaternion."""
    rand = self._random_state.rand(4)
    return rand / np.linalg.norm(rand)


def _normalize_and_make_positive_leading(quat):
  quat = quat.copy()
  quat /= np.linalg.norm(quat)
  if quat[0] < 0:
    quat = -1 * quat
  return quat

if __name__ == '__main__':
  absltest.main()
