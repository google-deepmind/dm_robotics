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

"""Tests for transformations.py.

Testing on TPU accelerators:
We constantly observe slightly different numerical output for matrix
calculations between JAX (TPU) and NumPy (CPU) libraries.

On TPU, 'transformations.py' is enforced to use 32bit precision, but NumPy's
default is np.float64. In multiplications and divisions operations, numerical
errors accumulate which can explain the numerical differences.
"""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.transformations import transformations as np_tr
from dm_robotics.transformations.jax import transformations as jax_tr

import jax
from jax import numpy as jnp
import numpy as np


def jax_device():
  return jax.devices()[0].platform


def _to_jax_array(*args):
  """Helper to cast list args to jax types."""
  res = [jnp.asarray(arr, dtype=jnp.float32) for arr in args]
  return res[0] if len(res) == 1 else res


class TransformationsTest(parameterized.TestCase):
  """Tests for jax transformations lib."""

  def assertAllClose(
      self,
      x: Any,
      y: Any,
      rtol=1e-5,
      atol=1e-5,
      **kwargs
  ):
    if isinstance(x, dict):
      self.assertEqual(x.keys(), y.keys())
      for k in x.keys():
        self.assertAllClose(x[k], y[k], rtol=rtol, atol=atol, **kwargs)
    elif np.isscalar(x) or isinstance(x, (jnp.ndarray, np.ndarray)):
      np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, **kwargs)
    else:
      self.assertLen(x, len(y))
      for xi, yi in zip(x, y):
        self.assertAllClose(xi, yi, rtol=rtol, atol=atol, **kwargs)

  @parameterized.parameters(
      {'a': [1, 2, 3], 'b': [0, 1, 0]},
      {'a': [0, 1, 2], 'b': [-2, 1, 0]})
  def test_cross_product(self, a, b):
    a, b = _to_jax_array(a, b)
    jnp_ver = jnp.cross(a, b)
    jtr_ver = jax_tr.cross_mat_from_vec3(a).dot(b)
    self.assertAllClose(jnp_ver, jtr_ver)

  @parameterized.parameters(
      {'quat': [1., 0., 0., 0.]},
      {'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'quat': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'quat': np_tr.euler_to_quat([0., 0., np.pi/3])},
      {'quat': np_tr.euler_to_quat([-1., 0.2, -0.3])},
  )
  def test_quat_conj_vs_numpy(self, quat):
    expected_output = np_tr.quat_conj(quat)
    actual_output = jax_tr.quat_conj(quat)
    self.assertAllClose(jnp.asarray(expected_output), actual_output)

  @parameterized.parameters(
      {'quat': [1., 0., 0., 0.]},
      {'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'quat': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'quat': np_tr.euler_to_quat([0., 0., np.pi/3])},
      {'quat': np_tr.euler_to_quat([-1., 0.2, -0.3])},
  )
  def test_quat_inv_vs_numpy(self, quat):
    expected_output = np_tr.quat_inv(quat)
    actual_output = jax_tr.quat_inv(quat)
    self.assertAllClose(jnp.asarray(expected_output), actual_output)

  @parameterized.parameters(
      {'a': [1., 0., 0., 0.],
       'b': [1./np.sqrt(2), 1./np.sqrt(2), 0., 0.]},
      {'a': np_tr.euler_to_quat([np.pi, 0., 0.]),
       'b': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'a': np_tr.euler_to_quat([-1., 0.2, -0.3]),
       'b': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'a': np_tr.euler_to_quat([0., 0., np.pi/3]),
       'b': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'a': np_tr.euler_to_quat([-1., 0.2, -0.3]),
       'b': np_tr.euler_to_quat([0., 0., np.pi/3])},
  )
  def test_quat_mul_vs_numpy(self, a, b):
    expected_output = np_tr.quat_mul(a, b)
    actual_output = jax_tr.quat_mul(a, b)
    self.assertAllClose(jnp.asarray(expected_output), actual_output)

  @parameterized.parameters(
      {'quat': [1., 0., 0., 0.],
       'vec': [1., 2., 3.,]},
      {'quat': np_tr.euler_to_quat([np.pi, 0., 0.]),
       'vec': [-1., 2., -3.,]},
      {'quat': np_tr.euler_to_quat([-1., 0.2, -0.3]),
       'vec': [0.2, -0.7, 0.9,]},
      {'quat': np_tr.euler_to_quat([0., 0., np.pi/3]),
       'vec': [-0.4, 0.1, 0.8,]},
      {'quat': np_tr.euler_to_quat([-1., 0.2, -0.3]),
       'vec': [1000., 200., 3000.,]},
  )
  def test_quat_rotate_vs_numpy(self, quat, vec):
    expected_output = np_tr.quat_rotate(quat, vec)
    actual_output = jax_tr.quat_rotate(quat, vec)
    self.assertAllClose(jnp.asarray(expected_output), actual_output)

  @parameterized.parameters(
      {'a': [1., 0., 0., 0.],
       'b': [1./np.sqrt(2), 1./np.sqrt(2), 0., 0.]},
      {'a': np_tr.euler_to_quat([np.pi, 0., 0.]),
       'b': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'a': np_tr.euler_to_quat([-1., 0.2, -0.3]),
       'b': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'a': np_tr.euler_to_quat([0., 0., np.pi/3]),
       'b': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'a': np_tr.euler_to_quat([-1., 0.2, -0.3]),
       'b': np_tr.euler_to_quat([0., 0., np.pi/3])},
  )
  def test_quat_diff_vs_numpy(self, a, b):
    expected_output = np_tr.quat_diff_passive(a, b)
    actual_output = jax_tr.quat_diff(a, b)
    self.assertAllClose(jnp.asarray(expected_output), actual_output)

  @parameterized.parameters(
      {'a': [1., 0., 0., 0.],
       'b': [1./np.sqrt(2), 1./np.sqrt(2), 0., 0.]},
      {'a': np_tr.euler_to_quat([np.pi, 0., 0.]),
       'b': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'a': np_tr.euler_to_quat([-1., 0.2, -0.3]),
       'b': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'a': np_tr.euler_to_quat([0., 0., np.pi/3]),
       'b': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'a': np_tr.euler_to_quat([-1., 0.2, -0.3]),
       'b': np_tr.euler_to_quat([0., 0., np.pi/3])},
  )
  def test_quat_dist_vs_numpy(self, a, b):
    expected_output = np_tr.quat_dist(a, b)
    actual_output = jax_tr.quat_dist(a, b)
    self.assertAllClose(jnp.asarray(expected_output), actual_output)

  @parameterized.parameters(
      {'quat': [1., 0., 0., 0.]},
      {'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'quat': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'quat': np_tr.euler_to_quat([0., 0., np.pi/3])},
      {'quat': np_tr.euler_to_quat([-1., 0.2, -0.3])},
  )
  def test_quat_to_axisangle_vs_numpy(self, quat):
    expected_output = np_tr.quat_to_axisangle(quat)
    actual_output = jax_tr.quat_to_axisangle(quat)
    self.assertAllClose(jnp.asarray(expected_output), actual_output)

  @parameterized.parameters(
      {'axisangle': [0., 0., 0.]},
      {'axisangle': [0.1, 0.2, 0.3]},
      {'axisangle': [-0.3, -0.2, -0.1]},
      {'axisangle': [np.pi, 0., 0.]},
      {'axisangle': [0., np.pi, 0.]},
      {'axisangle': [0., 0, np.pi]},
      {'axisangle': [np.pi/2, np.pi/2, np.pi/2]},
  )
  def test_axisangle_to_quat_vs_numpy(self, axisangle):
    expected_output = np_tr.axisangle_to_quat(axisangle)
    actual_output = jax_tr.axisangle_to_quat(axisangle)
    self.assertAllClose(jnp.asarray(expected_output), actual_output)

  @parameterized.parameters(
      {'quat': [1., 0., 0., 0.]},
      {'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'quat': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'quat': np_tr.euler_to_quat([0., 0., np.pi/3])},
      {'quat': np_tr.euler_to_quat([-1., 0.2, -0.3])},
  )
  def test_quat_to_mat_vs_numpy(self, quat):
    expected_output = np_tr.quat_to_mat(quat)
    actual_output = jax_tr.quat_to_mat(quat)
    self.assertAllClose(jnp.asarray(expected_output), actual_output, atol=2e-07)

  @parameterized.parameters(
      {'rmat': np_tr.euler_to_rmat([np.pi, 0., 0.], 'XYZ', full=True)},
      {'rmat': np_tr.euler_to_rmat([0., np.pi/2, 0.], 'XYZ', full=True)},
      {'rmat': np_tr.euler_to_rmat([0., 0., np.pi/3], 'XYZ', full=True)},
      {'rmat': np_tr.euler_to_rmat([-1., 0.2, -0.3], 'XYZ', full=True)},
  )
  def test_mat_to_quat_vs_numpy(self, rmat):
    expected_output = np_tr.mat_to_quat(rmat)
    actual_output = jax_tr.mat_to_quat(rmat)
    self.assertAlmostEqual(
        np_tr.quat_dist(expected_output, actual_output), 0, places=6)

  @parameterized.parameters(
      {'rmat': np_tr.euler_to_rmat([np.pi, 0., 0.], 'XYZ', full=True)},
      {'rmat': np_tr.euler_to_rmat([0., np.pi/2, 0.], 'XYZ', full=True)},
      {'rmat': np_tr.euler_to_rmat([0., 0., np.pi/3], 'XYZ', full=True)},
      {'rmat': np_tr.euler_to_rmat([-1., 0.2, -0.3], 'XYZ', full=True)},
  )
  def test_rmat_to_rot6_vs_numpy(self, rmat):
    expected_output = np_tr.rmat_to_rot6(rmat)
    actual_output = jax_tr.rmat_to_rot6(rmat)
    self.assertAllClose(jnp.asarray(expected_output), actual_output, atol=1e-07)

  @parameterized.parameters(
      {'rot6': np.zeros(6)},
      {'rot6': np.ones(6)},
      {'rot6': np.ones(6) * 1e-8},
      {'rot6': np.array([1., 2., 3., 0., 0., 0.])},
      {'rot6': np.array([1., 2., 3., 1., 2., 3.])},
      {'rot6': np.concatenate((np.array([1., 2., 3]),
                               np.array([1., 2., 3]) + 1e-8))},
      {'rot6': np.array([0., 0., 0., 1., 2., 3.])},
      {'rot6': np.array([1., 2., 3., 4., 5., 6.])},
      {'rot6': np.array([1., 2., 3., 4., 5., 6.]) * -1},
      {'rot6': np.array([1., 2., 3., 4., 5., 6.], dtype=np.float16)},
  )
  def test_rot6_to_rmat_vs_numpy(self, rot6):
    """Tests that rot6_to_rmat works with arbitrary inputs."""
    expected_output = np_tr.rot6_to_rmat(rot6)
    actual_output = jax_tr.rot6_to_rmat(rot6)
    atol = 1e-3 if rot6.dtype == np.float16 else 1e-7
    self.assertAllClose(jnp.asarray(expected_output), actual_output, atol=atol)

    # Verify it's a valid rotation too.
    should_be_identity = jnp.dot(
        actual_output.T, actual_output, precision=jax.lax.Precision.HIGHEST)
    self.assertAllClose(should_be_identity, jnp.eye(3), atol=atol)

  @parameterized.parameters(
      {'rmat': np.eye(3)},
      {'rmat': np_tr.euler_to_rmat([np.pi, 0., 0.])},
      {'rmat': np_tr.euler_to_rmat([0., np.pi/2, 0.])},
      {'rmat': np_tr.euler_to_rmat([0., 0., np.pi/3])},
      {'rmat': np_tr.euler_to_rmat([-1., 0.2, -0.3])},
  )
  def test_rmat_to_rot6_conversion(self, rmat):
    # Note: arbitrary values of `rot6` won't be cycle-consistent because
    # `rot6_to_rmat` is not a bijection. I.e. we can only test in the `rmat ->
    # rot6 -> rmat` order. See `test_rot6_to_rmat_vs_numpy` for arbitrary tests.
    rot6 = jax_tr.rmat_to_rot6(rmat)
    recovered_rmat = jax_tr.rot6_to_rmat(rot6)
    self.assertAllClose(rmat, recovered_rmat, atol=1e-07)

  @parameterized.parameters(
      {'pos': [1., 2., 3.],
       'quat': [1./np.sqrt(2), 1./np.sqrt(2), 0., 0.]},
      {'pos': [-3., -2., -1.],
       'quat': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'pos': [0., 0., 0.],
       'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'pos': [0.1, 0.2, 0.3],
       'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'pos': [1000., 2000., 3000.],
       'quat': np_tr.euler_to_quat([0., 0., np.pi/3])},
  )
  def test_pos_quat_to_hmat_conversion(self, pos, quat):
    hmat = jax_tr.pos_quat_to_hmat(pos, quat)
    recovered_pos, recovered_quat = jax_tr.hmat_to_pos_quat(hmat)
    self.assertAllClose(jnp.asarray(pos), recovered_pos)
    self.assertAllClose(jnp.asarray(quat), recovered_quat, atol=1e-07)

  @parameterized.parameters(
      {'pos': [1., 2., 3.],
       'quat': [1./np.sqrt(2), 1./np.sqrt(2), 0., 0.]},
      {'pos': [-3., -2., -1.],
       'quat': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'pos': [0., 0., 0.],
       'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'pos': [0.1, 0.2, 0.3],
       'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'pos': [1000., 2000., 3000.],
       'quat': np_tr.euler_to_quat([0., 0., np.pi/3])},
  )
  def test_pos_quat_inv_vs_numpy(self, pos, quat):
    pos, quat = _to_jax_array(pos, quat)
    actual_result = jax_tr.pos_quat_inv(pos, quat)

    inv_quat = np_tr.quat_inv(quat)
    expected_result = (np_tr.quat_rotate(inv_quat, -1 * pos), inv_quat)
    self.assertAllClose(_to_jax_array(*expected_result), actual_result)

  @parameterized.parameters(
      {'a': [[0., 0., 0.], [1., 0., 0., 0.]],
       'b': [[0., 0., 0.], [1./np.sqrt(2), 1./np.sqrt(2), 0., 0.]]},
      {'a': [[0.1, 0.2, 0.3], np_tr.euler_to_quat([np.pi, 0., 0.])],
       'b': [[-0.1, -0.2, -0.3], np_tr.euler_to_quat([0., np.pi/2, 0.])]},
      {'a': [[0.3, 0.2, 0.1], np_tr.euler_to_quat([-1., 0.2, -0.3])],
       'b': [[0.1, 0.2, 0.3], np_tr.euler_to_quat([np.pi, 0., 0.])]},
      {'a': [[10., 20., 30.], np_tr.euler_to_quat([0., 0., np.pi/3])],
       'b': [[-10., -30., 20.], np_tr.euler_to_quat([np.pi, 0., 0.])]},
      {'a': [[10000., 50000., -20000.], np_tr.euler_to_quat([-1., 0.2, -0.3])],
       'b': [[0.1, 0.2, 0.3], np_tr.euler_to_quat([0., 0., np.pi/3])]},
  )
  def test_pos_quat_mul_vs_numpy(self, a, b):
    pos1, quat1 = a
    pos2, quat2 = b
    expected_pos = pos1 + np_tr.quat_rotate(quat1, pos2)
    expected_quat = np_tr.quat_mul(quat1, quat2)
    expected_output = (expected_pos, expected_quat)
    actual_output = jax_tr.pos_quat_mul(*_to_jax_array(*a), *_to_jax_array(*b))
    self.assertAllClose(_to_jax_array(*expected_output), actual_output)

  @parameterized.parameters(
      {'axisangle': [0., 0., 0.]},
      {'axisangle': [0.1, 0.2, 0.3]},
      {'axisangle': [-0.3, -0.2, -0.1]},
      {'axisangle': [np.pi, 0., 0.]},
      {'axisangle': [0., np.pi, 0.]},
      {'axisangle': [0., 0, np.pi]},
      {'axisangle': [np.pi/2, np.pi/2, np.pi/2]},
  )
  def test_axisangle_rmat_conversion(self, axisangle):
    """Test internal consistency of axisangle to rmat conversion."""
    axisangle = _to_jax_array(axisangle)
    rmat = jax_tr.axisangle_to_rmat(axisangle)
    recovered_axisangle = jax_tr.rmat_to_axisangle(rmat)

    self.assertAllClose(recovered_axisangle, axisangle)

  @parameterized.parameters(
      {'axisangle': [0., 0., 0.]},
      {'axisangle': [0.1, 0.2, 0.3]},
      {'axisangle': [-0.3, -0.2, -0.1]},
      {'axisangle': [np.pi, 0., 0.]},
      {'axisangle': [0., np.pi, 0.]},
      {'axisangle': [0., 0, np.pi]},
      {'axisangle': [np.pi/2, np.pi/2, np.pi/2]},
  )
  def test_axisangle_to_rmat_vs_numpy(self, axisangle):
    """Test axisangle_to_rmat vs numpy implementation."""
    expected_output = np_tr.axisangle_to_rmat(axisangle)
    axisangle = _to_jax_array(axisangle)
    actual_output = jax_tr.axisangle_to_rmat(axisangle)

    atol = 1e-6 if jax_device() == 'tpu' else 1e-5
    self.assertAllClose(expected_output, actual_output, atol=atol)

  @parameterized.parameters(
      {'quat': np_tr.euler_to_quat([0., 0., 0.])},
      {'quat': np_tr.euler_to_quat([0.1, 0.2, 0.3])},
      {'quat': np_tr.euler_to_quat([0., np.pi/2, 0.])},
      {'quat': np_tr.euler_to_quat([np.pi, 0., 0.])},
      {'quat': np_tr.euler_to_quat([0., np.pi, 0.])},
      {'quat': np_tr.euler_to_quat([0., 0., np.pi])}
  )
  def test_rmat_to_axisangle_vs_numpy(self, quat):
    """Test rmat_to_axisangle vs numpy implementation."""
    rmat = jax_tr.quat_to_mat(_to_jax_array(quat))[:3, :3]
    expected_output = np_tr.rmat_to_axisangle(rmat)
    actual_output = jax_tr.rmat_to_axisangle(rmat)

    self.assertAllClose(expected_output, actual_output, atol=1e-3)

  @parameterized.parameters(
      {'pos': [0.1, 0.2, 0.3], 'quat': [1., 0., 0., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 1., 0., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 0., 1., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 0., 0., 1.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0.5, 0.5, 0.5, 0.5]},
  )
  def test_hmat_twist_conversion(self, pos, quat):
    """Test internal consistency of hmat to twist conversion."""
    pos, quat = _to_jax_array(pos, quat)
    hmat = jax_tr.pos_quat_to_hmat(pos, quat)
    xi = jax_tr.hmat_to_twist(hmat)
    hmat2 = jax_tr.twist_to_hmat(xi)
    quat2 = jax_tr.mat_to_quat(hmat2)
    pos2 = hmat2[:3, 3]

    atol = 5e-3 if jax_device() == 'tpu' else 1e-6
    self.assertAllClose(hmat, hmat2, atol=atol)

    atol = 5e-2 if jax_device() == 'tpu' else 1e-5
    self.assertAllClose(pos, pos2, atol=atol)

    atol = 5e-3 if jax_device() == 'tpu' else 1e-7
    self.assertTrue(
        jnp.allclose(quat, quat2, atol=atol) or
        jnp.allclose(quat, -quat2, atol=atol))

  def test_hmat_twist_no_nan_special_cases(self):
    """Small test to verify special cases to define as identity."""
    res = jax_tr.twist_to_hmat(jax_tr.hmat_to_twist(jnp.zeros((4, 4))))
    self.assertAllClose(res, jnp.eye(4), atol=1e-6)

    res = jax_tr.twist_to_hmat(jax_tr.hmat_to_twist(jnp.eye(4)))
    self.assertAllClose(res, jnp.eye(4), atol=1e-6)

  @parameterized.parameters(
      {'pos': [0.1, 0.2, 0.3], 'quat': [1., 0., 0., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 1., 0., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 0., 1., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 0., 0., 1.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0.5, 0.5, 0.5, 0.5]},
  )
  def test_hmat_posaxisangle_conversion(self, pos, quat):
    """Test internal consistency of hmat to twist conversion."""
    pos, quat = _to_jax_array(pos, quat)
    hmat = jax_tr.pos_quat_to_hmat(pos, quat)
    pa = jax_tr.hmat_to_posaxisangle(hmat)
    hmat2 = jax_tr.posaxisangle_to_hmat(pa)
    quat2 = jax_tr.mat_to_quat(hmat2)
    pos2 = hmat2[:3, 3]

    atol = 5e-3 if jax_device() == 'tpu' else 1e-6
    self.assertAllClose(hmat, hmat2, atol=atol)

    atol = 5e-2 if jax_device() == 'tpu' else 1e-5
    self.assertAllClose(pos, pos2, atol=atol)

    atol = 5e-3 if jax_device() == 'tpu' else 1e-7
    self.assertTrue(
        jnp.allclose(quat, quat2, atol=atol) or
        jnp.allclose(quat, -quat2, atol=atol))

  @parameterized.parameters(
      {'pos': [0.1, 0.2, 0.3], 'quat': [1., 0., 0., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 1., 0., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 0., 1., 0.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0., 0., 0., 1.]},
      {'pos': [0.1, 0.2, 0.3], 'quat': [0.5, 0.5, 0.5, 0.5]},
  )
  def test_hmat_to_twist_vs_numpy(self, pos, quat):
    """Test hmat_to_twist vs numpy implementation."""
    pos, quat = _to_jax_array(pos, quat)
    hmat = jax_tr.pos_quat_to_hmat(pos, quat)
    expected_output = np_tr.hmat_to_twist(hmat)
    actual_output = jax_tr.hmat_to_twist(hmat)

    atol = 5e-2 if jax_device() == 'tpu' else 1e-3
    self.assertAllClose(expected_output, actual_output, atol=atol)

  @parameterized.parameters(
      {'twist': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
      {'twist': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
      {'twist': [-0.1, 0.2, -0.3, -0.4, -0.5, 0.6]},
  )
  def test_twist_to_hmat_vs_numpy(self, twist):
    """Test twist_to_hmat vs numpy implementation."""
    expected_output = np_tr.twist_to_hmat(twist)
    twist = _to_jax_array(twist)
    actual_output = jax_tr.twist_to_hmat(twist)
    self.assertAllClose(expected_output, actual_output)

  @parameterized.parameters(
      {'twist': [1, 0, 0, 0, 0, 0],
       'poseuler': (0, 0, 0, np.radians(0), np.radians(90), np.radians(0))},
      {'twist': [1, 2, 3, -3, 2, -1],
       'poseuler': (-1, 2, 3, np.radians(30), np.radians(60), np.radians(90))}
  )
  def test_velocity_transform_vs_numpy(self, twist, poseuler):
    # Test for special values that often cause numerical issues.
    hmat = np_tr.poseuler_to_hmat(np.array(poseuler), 'ZYZ')
    expected_output = np_tr.velocity_transform(hmat, twist)
    twist = _to_jax_array(twist)
    actual_output = jax_tr.velocity_transform(hmat, twist)

    rtol = 5e-3 if jax_device() == 'tpu' else 1e-5
    self.assertAllClose(expected_output, actual_output, rtol=rtol)

  @parameterized.parameters(
      {'wrench': [1, 0, 0, 0, 0, 0],
       'poseuler': [0, 0, 0, np.radians(0), np.radians(90), np.radians(0)]},
      {'wrench': [1, 2, 3, -3, 2, -1],
       'poseuler': [-1, 2, 3, np.radians(30), np.radians(60), np.radians(90)]}
  )
  def test_force_transform_vs_numpy(self, wrench, poseuler):
    # Test for special values that often cause numerical issues.
    hmat = np_tr.poseuler_to_hmat(np.array(poseuler), 'ZYZ')
    expected_output = np_tr.force_transform(hmat, wrench)
    wrench = _to_jax_array(wrench)
    actual_output = jax_tr.force_transform(hmat, wrench)

    rtol = 5e-3 if jax_device() == 'tpu' else 1e-5
    self.assertAllClose(expected_output, actual_output, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
