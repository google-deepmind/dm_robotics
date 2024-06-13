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

"""JAX-based transformation utilities.

This is based on our in-house numpy transformations library:
https://github.com/google-deepmind/dm_robotics/blob/main/py/transformations/
"""

import functools
from typing import Optional, Tuple

import jax
from jax import numpy as jnp

# On TPU, JAX uses 16-bit values by default in 'matmul-like' operations, such as
# jax.numpy.dot. We enforce 32-bit based calculations.
# Internal dot product clamped to highest-precision.
_dot = functools.partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)


def cross_mat_from_vec3(
    vec: jnp.ndarray,   # (3,)
) -> jnp.ndarray:  # (3, 3)
  """Returns the skew-symmetric matrix cross-product operator.

  Args:
    vec: (3,) vector.

  Returns:
    A matrix cross-product operator P (3x3) for the vector vec = [x,y,z]^T,
    such that vec x b = Pb for any 3-vector b.
  """
  x, y, z = vec[0], vec[1], vec[2]
  return jnp.array([[0, -z, y],
                    [z, 0, -x],
                    [-y, x, 0]],
                   dtype=vec.dtype)


def axisangle_to_quat(
    axisangle: jnp.ndarray,  # (3,)
) -> jnp.ndarray:  # (4,)
  """Returns the quaternion corresponding to the provided axis-angle vector.

  Args:
    axisangle: (3,) array describing the axis of rotation, with angle encoded by
      its length

  Returns:
    quat: A quaternion [w, i, j, k]
  """
  axisangle = jnp.asarray(axisangle)
  theta = jnp.linalg.norm(axisangle)

  # Only called if theta > 0
  def _general() -> jnp.ndarray:
    wn = axisangle / theta
    return jnp.hstack([jnp.cos(theta / 2.), wn * jnp.sin(theta / 2.)])

  def _small_angle() -> jnp.ndarray:
    # Instead of simply returning identity quaternion we use this small angle
    # approximation to preserve gradients without a division by theta, which is
    # unstable near zero. Obtained by writing the imaginary part of `_general`
    # in terms of un-normalized axisangle and doing 2nd order taylor approx.
    # of sin (which cancels the theta normalization), and normalizing for the
    # real part.
    quat = jnp.array([
        jnp.sqrt(1 - (axisangle**2).sum() / 4), axisangle[0] / 2,
        axisangle[1] / 2, axisangle[2] / 2
    ])
    return quat

  quat = jax.lax.cond(
      jnp.allclose(theta, 0.),
      lambda _: _small_angle(),
      lambda _: _general(),
      operand=None)

  return quat


def axisangle_to_rmat(
    axisangle: jnp.ndarray,  # (3,)
) -> jnp.ndarray:  # (3, 3)
  """Returns rotation matrix corresponding to the exponential coordinates.

  See Murray1994: A Mathematical Introduction to Robotic Manipulation

  Args:
    axisangle: (3,) array describing the axis of rotation, with angle encoded by
      its length.

  Returns:
    Array of shape (3, 3) containing a rotation matrix.
  """
  theta = jnp.linalg.norm(axisangle)

  def _general(axisangle):
    wn = axisangle / theta
    s = cross_mat_from_vec3(wn)
    return jnp.eye(3) + s * jnp.sin(theta) + _dot(s, s) * (1 - jnp.cos(theta))

  def _small_angle(axisangle):
    # Instead of simply returning the identity rmat, we use this small angle
    # approximation to preserve gradients without a division by theta, which is
    # unstable near zero. Obtained by re-writing in terms of un-normalized
    # axisangle and doing 2nd order taylor approximation for sin and cos (which
    # cancels the divisions by theta).
    s_theta = cross_mat_from_vec3(axisangle)
    return jnp.eye(3) + s_theta + _dot(s_theta, s_theta) * 0.5

  rmat = jax.lax.cond(
      jnp.allclose(theta, 0, atol=1e-6), _small_angle, _general, axisangle)

  return rmat


def rmat_to_axisangle(
    rmat: jnp.ndarray,  # (3, 3)
) -> jnp.ndarray:  # (3,)
  """Returns exponential coordinates (w * theta) for the given rotation matrix.

  Lynch & Park 2017: Modern Robotics: Mechanics, Planning, and Control.

  Args:
    rmat: (3, 3) array containing a rotation matrix.

  Returns:
    Array of shape (3,) containing unit-vector describing the axis of rotation,
    scaled by the angle required to rotate about this axis to achieve `rmat`.
  """

  def _general(rmat):

    def _pole(rmat):
      # Note: original implementation in dm_robotics/transformations uses an
      # eigendecomposition to find the axis with eigenvalue=1.  However
      # `jnp.linalg.eig` is only supported on the CPU, so instead we borrow from
      # Lynch2017 and enumerate the 3 cases explicitly.
      def _a(rmat):
        axis = 1. / jnp.sqrt(2 * (1 + rmat[0, 0])) * jnp.array(
            [1 + rmat[0, 0], rmat[1, 0], rmat[2, 0]])
        return axis * jnp.pi

      def _b(rmat):
        axis = 1. / jnp.sqrt(2 * (1 + rmat[1, 1])) * jnp.array(
            [rmat[0, 1], 1 + rmat[1, 1], rmat[2, 1]])
        return axis * jnp.pi

      def _c(rmat):
        axis = 1. / jnp.sqrt(2 * (1 + rmat[2, 2])) * jnp.array(
            [rmat[0, 2], rmat[1, 2], 1. + rmat[2, 2]])
        return axis * jnp.pi

      # Select which branch to execute based on the first positive element on
      # the diagonal of `rmat`. Note this function is only invoked in the "pole"
      # case, meaning we have a 180-degree rotation around some axis resulting
      # in an `rmat` with a 1 and two "-1" along the diagonal, in some order.
      idxs = jnp.where(
          jnp.diag(rmat) > 0,  # Index of positive diagonal (will never be 3)
          jnp.array([0, 1, 2]),
          jnp.array([0, 0, 0]))

      # `idxs` will be zeros with at most a single non-zero entry.
      switch_idx = jnp.sum(idxs)  # like `jnp.nonzero`, but jittable
      axisangle = jax.lax.switch(switch_idx, [_a, _b, _c], rmat)

      return axisangle

    def _general(rmat):
      tr = (jnp.trace(rmat) - 1) / 2
      tr = jnp.clip(tr, -1, 1)
      angle = jnp.arccos(tr)
      axis = 1. / jnp.sin(angle) * jnp.array([
          rmat[2, 1] - rmat[1, 2],
          rmat[0, 2] - rmat[2, 0],
          rmat[1, 0] - rmat[0, 1]
      ])
      axis = axis / jnp.linalg.norm(axis)
      return axis * angle

    tr = jnp.trace(rmat)
    axisangle = jax.lax.cond(jnp.allclose(tr, -1.), _pole, _general, rmat)
    return axisangle

  def _small_angle(rmat):
    # Instead of simply returning zeros we use this small angle approximation to
    # preserve gradients without a division by theta, which is unstable near
    # zero. Obtained from `_general` by substituting the 1/sin term with the
    # 2nd order taylor approximation (which cancels theta).
    return jnp.array([
        rmat[2, 1] - rmat[1, 2],
        rmat[0, 2] - rmat[2, 0],
        rmat[1, 0] - rmat[0, 1]
    ]) * 0.5

  tr = (jnp.trace(rmat) - 1) / 2
  tr = jnp.clip(tr, -1, 1)
  theta = jnp.arccos(tr)
  axisangle = jax.lax.cond(
      jnp.allclose(theta, 0, atol=1e-6), _small_angle, _general, rmat)

  # The functionality above is best-effort, but doesn't check if the input rmat
  # was properly special-orthogonal (i.e. determinant=1 -> column normalized).
  # Jax & XLA don't have a runtime error mechanism, so we follow protocol of
  # signaling this case with NaNs.
  ortho_trace = jnp.trace(_dot(rmat.T, rmat))
  axisangle = jax.lax.cond(
      ortho_trace > (3. + 1e-5),  # Tuned for float32 precision on CPU & TPU.
      lambda x: jnp.nan * x,
      lambda x: x,
      axisangle)

  return axisangle


def quat_conj(
    quat: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # (4,)
  """Return conjugate of quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion [w, -i, -j, -k] representing the inverse of the rotation
    defined by `quat` (not assuming normalization).
  """
  quat = jnp.asarray(quat)
  return jnp.array((quat[0], -quat[1], -quat[2], -quat[3]))


def quat_inv(
    quat: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # (4,)
  """Return inverse of quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the inverse of the original rotation.
  """
  quat = jnp.asarray(quat)
  return quat_conj(quat) / _dot(quat, quat)


def quat_mul(
    quat1: jnp.ndarray,  # (4,)
    quat2: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # (4,)
  """Multiply quaternions.

  Args:
    quat1: A quaternion [w, i, j, k].
    quat2: A quaternion [w, i, j, k].

  Returns:
    The quaternion product, aka hamiltonian product.
  """
  quat2 = jnp.asarray(quat2)
  qmat = jnp.array([[quat1[0], -quat1[1], -quat1[2], -quat1[3]],
                    [quat1[1], quat1[0], -quat1[3], quat1[2]],
                    [quat1[2], quat1[3], quat1[0], -quat1[1]],
                    [quat1[3], -quat1[2], quat1[1], quat1[0]]])

  return _dot(qmat, quat2)


def quat_diff(
    source_quat: jnp.ndarray,  # (4,)
    target_quat: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # (4,)
  """Passive quaternion difference between source and target quaternions.

  This function gives the relative quaternion, i.e. the quaternion that brings a
  vector expressed in the target frame to the same vector expressed in the
  source frame.

  Args:
    source_quat: A unit quaternion [w, i, j, k] representing a passive rotation
      from the source frame to the world frame.
    target_quat: A unit quaternion [w, i, j, k] representing a passive rotation
      from the target frame to the world frame.

  Returns:
    A normalized quaternion representing the rotation that brings a vector
    expressed in the target frame to the same vector being expressed in the
    source frame.
  """
  quat = quat_mul(quat_conj(source_quat), target_quat)
  return quat / jnp.linalg.norm(quat)


def quat_dist(
    source: jnp.ndarray,  # (4,)
    target: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # ()
  """Computes angular distance between source and target quaternions.

  Args:
    source: A unit quaternion [w, i, j, k].
    target: A unit quaternion [w, i, j, k].

  Returns:
    The rotational distance from source to target in radians.
  """
  rel_quat = quat_diff(source, target)
  return quat_angle(rel_quat)


def quat_angle(
    quat: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # ()
  """Computes the angle of the rotation encoded by the unit quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k]. The norm of this vector should be 1.

  Returns:
    The angle in radians of the rotation encoded by the quaternion.
  """

  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = jnp.asarray(quat)

  # Ensure the quaternion is positive leading to get the shortest angle.
  quat = jax.lax.cond(quat[0] < 0, lambda q: q * -1, lambda q: q, quat)

  # We have w = cos(angle/2) with w the real part of the quaternion and
  # ||Im(q)|| = sin(angle/2) with Im(q) the imaginary part of the quaternion.
  # We choose the method that is less sensitive to a noisy evaluation of the
  # difference.
  def _arccos_angle(quat):
    """Extracts quat angle using arccos."""
    return 2 * jnp.arccos(quat[0])

  def _arcsin_angle(quat):
    """Extracts quat angle using arcsin."""
    # Note: the additive constant 1e-18 is tuned to be as small as possible
    # while still giving this function a gradient for identity quaternions on
    # a TPU at float32. This is important since optimization can be initialized
    # with identity quaternions and we want a gradient at those values.
    return 2 * jnp.arcsin(jnp.linalg.norm(quat[1:] + 1e-18))

  angle = jax.lax.cond(quat[0] < (1 / jnp.sqrt(2)), _arccos_angle,
                       _arcsin_angle, quat)

  return angle


def quat_axis(
    quat: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # (3,)
  """Returns the rotation axis of the corresponding quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k].

  Returns:
    axisangle: A 3x1 normalized numpy array describing the axis of rotation.
  """
  return quat[1:4] / jnp.linalg.norm(quat[1:4])


def quat_to_axisangle(
    quat: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # (3,)
  """Returns the axis-angle corresponding to the provided quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k].

  Returns:
    axisangle: (3,) array describing the axis of rotation, with angle encoded by
    its length.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = jnp.asarray(quat)

  # Ensure the quaternion is positive leading to get the shortest angle.
  quat = jax.lax.cond(quat[0] < 0, lambda q: q * -1, lambda q: q, quat)

  angle = quat_angle(quat)

  def _small_angle() -> jnp.ndarray:
    # This function is used as an alternative to `jnp.zeros(3)` which
    # evalutates to nearly zero on the angles in which it is invoked, but
    # provides a numerically-stable gradient at identity quaternions.
    # Obtained using identity `ax = qx / sin(angle/2)` and substituting the 1st-
    # order taylor expansion sin(x) ≈ x.  See
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/
    # quaternionToAngle/index.htm
    return 2 * quat[1:]

  def _general() -> jnp.ndarray:
    axis = quat_axis(quat)
    return axis * angle

  axisangle = jax.lax.cond(
      angle < 1e-6,
      lambda _: _small_angle(),
      lambda _: _general(),
      operand=None)

  return axisangle


def quat_rotate(
    quat: jnp.ndarray,  # (4,)
    vec: jnp.ndarray,  # (3,)
) -> jnp.ndarray:  # (3,)
  """Rotate a vector by a unit quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k]. The norm of this vector should be 1.
    vec: A 3-vector representing a position.

  Returns:
    (3,) The rotated vector.
  """
  qvec = jnp.hstack([[0], vec])
  return quat_mul(quat_mul(quat, qvec), quat_conj(quat))[1:]


def quat_to_mat(
    quat: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # (4, 4)
  """Return homogeneous rotation matrix from quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A 4x4 homogeneous matrix with the rotation corresponding to `quat`.
  """
  q = jnp.array(quat, copy=True)
  nq = _dot(q, q)

  def _quat_to_mat(q):
    q *= jnp.sqrt(2.0 / nq)
    q = jnp.outer(q, q)
    return jnp.array(
        ((1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0),
         (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0],
          0.0), (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2],
                 0.0), (0.0, 0.0, 0.0, 1.0)))

  return jax.lax.cond(nq < 1e-10, lambda _: jnp.identity(4), _quat_to_mat, q)


def mat_to_quat(
    mat: jnp.ndarray,  # (3, 3) or ((4, 4))
) -> jnp.ndarray:  # (4,)
  """Return quaternion from rotation matrix.

  Args:
    mat: A (3, 3) homogeneous transform.

  Returns:
    A quaternion [w, i, j, k].
  """
  m33 = jax.lax.cond(
      jnp.all(jnp.asarray(mat.shape) > 3), (mat), lambda mat: mat[3, 3], (mat),
      lambda mat: 1.)

  def _a(mat: jnp.ndarray, t: float):
    q = jnp.stack([
        t, mat[2, 1] - mat[1, 2], mat[0, 2] - mat[2, 0], mat[1, 0] - mat[0, 1]
    ])
    q *= 0.5 / jnp.sqrt(t * m33)
    return q

  def _b(mat: jnp.ndarray, unused_t: float):
    ijk = jnp.array([0, 1, 2])
    ijk = jax.lax.cond(mat[1, 1] > mat[0, 0], lambda _: jnp.array([1, 2, 0]),
                       lambda _: ijk, None)
    i = ijk[0]
    ijk = jax.lax.cond(mat[2, 2] > mat[i, i], lambda _: jnp.array([2, 0, 1]),
                       lambda _: ijk, None)
    i, j, k = ijk
    t = mat[i, i] - (mat[j, j] + mat[k, k]) + m33

    q = jnp.empty((4,))
    q = q.at[i + 1].set(t)
    q = q.at[j + 1].set(mat[i, j] + mat[j, i])
    q = q.at[k + 1].set(mat[k, i] + mat[i, k])
    q = q.at[0].set(mat[k, j] - mat[j, k])

    q *= 0.5 / jnp.sqrt(t * m33)
    return q

  mat = jnp.asarray(mat)
  t = jnp.trace(mat)
  q = jax.lax.cond(t > m33, lambda x: _a(*x), lambda x: _b(*x), (mat, t))
  return q


def rmat_to_rot6(
    rmat: jnp.ndarray,  # (3, 3)
) -> jnp.ndarray:  # (6,)
  """Projects rotation matrix to 6-dim "Gram-Schmidt-able" representation.

  The "rot6" representation is a 6-DOF representation of an orientation that is
  homeomorphic with SO(3). It is not minimal like an euler or axis-angle, but
  it is smooth over the full range of rotations, unlike eulers, quaternions, and
  axis-angle representations. See the original paper for details:
    "On the Continuity of Rotation Representations in Neural Networks"
    https://arxiv.org/pdf/1812.07035.pdf

  Args:
    rmat: A 3x3 rotation matrix.

  Returns:
    A 6-dimensional array containing the first two columns of `rmat`.  This
    representation can be mapped back to `rmat` using `rot6_to_rmat`.
  """
  return rmat[:3, :2].T.flatten()


def rot6_to_rmat(
    rot6: jnp.ndarray,  # (6,)
) -> jnp.ndarray:  # (3, 3)
  """Maps a 6-dim "Gram-Schmidt-able" representation back to a rotation matrix.

  The "rot6" representation is a 6-DOF representation of an orientation that is
  homeomorphic with SO(3). It is not minimal like an euler or axis-angle, but
  it is smooth over the full range of rotations, unlike eulers, quaternions, and
  axis-angle representations. See the original paper for details:
    "On the Continuity of Rotation Representations in Neural Networks"
    https://arxiv.org/pdf/1812.07035.pdf

  Args:
    rot6: An arbitrary 6-dimensional array representing a rotation. This
      representation can be obtained from an `rmat` using `rmat_to_rot6`.

  Returns:
    (3, 3) A rotation matrix obtained by normalizing and orthogonalizing the
    contents of `rot6`.
  """
  xu = rot6[0:3]
  yu = rot6[3:6]

  tol = 1e-6  # Tolerance below which the rot6 is replaced by a canonical basis.
  eps = 1e-5  # Safety factor to avoid zero case.

  def safe_interp(v: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Safe interpolation between input vector and a default basis."""
    # This function interpolates `v` and `b` a function of the norm of `v`.
    # The interpolation has a critical value in which the interpolation can be
    # zero if `v = -lambda * b`. We handle this by "jumping" over this value
    # for `alpha` with safety-factor `epsilon`. Achieved by defining a function
    # for alpha which grows from 0 to `crit - epsilon` over the range [0, tol].
    norm = jnp.linalg.norm(v)

    # Critical value for `v = -lambda * b` case, considering only projection
    # along `b` (if there are off-axis components there is no failure mode).
    crit = (1 - eps) / (1 - _dot(v, b))
    one = jnp.array(1.0, dtype=crit.dtype)

    alpha = jax.lax.cond(norm < tol, lambda _: crit * norm / tol, lambda _: one,
                         None)

    return alpha * v + (1 - alpha) * b

  # Interpolate `xu` and `yu` if they're close to zero.
  xu = safe_interp(xu, jnp.array([1, 0, 0], dtype=xu.dtype))
  yu = safe_interp(yu, jnp.array([0, 1, 0], dtype=yu.dtype))

  # If xu and yu are parallel, add arbitrary offset to allow orthogonalization.
  # Note: this introduces a discontinuity in the gradient when the
  # components of rot6 become parallel. Tried interpolating yu based on
  # `min(norm(yu), norm(jnp.cross(xu, yu))` to smooth this out, but the norm-
  # cross term generated nans in the jacobian. This also has a failure mode at
  # `xu = [0, 1, 0]`.
  yu = jax.lax.cond(
      jnp.allclose(jnp.cross(xu, yu), 0., atol=tol),
      lambda _: jnp.array([0, 1, 0], dtype=yu.dtype) + yu, lambda _: yu, None)

  # Rotation matrix obtained by orthogonalizing and normalizing.
  xn = xu / jnp.linalg.norm(xu)
  zu = jnp.cross(xn, yu)
  zn = zu / jnp.linalg.norm(zu)
  yn = jnp.cross(zn, xn)

  return jnp.stack([xn, yn, zn], axis=1)


def pos_quat_to_hmat(
    position: jnp.ndarray,  # (3,)
    quaternion: jnp.ndarray,  # (4,)
) -> jnp.ndarray:  # (4, 4)
  """Converts a position and quaternion to a homogeneous matrix.

  Args:
    position: (3,) array containing the position.
    quaternion: (4,) array containing the position.

  Returns:
    (4, 4) containing a homogeneous transform.
  """
  hmat = quat_to_mat(quaternion)
  hmat = hmat.at[:3, 3].set(position)
  return hmat


def hmat_to_pos_quat(
    hmat: jnp.ndarray,  # (4, 4)
) -> Tuple[jnp.ndarray, jnp.ndarray]:  # #((3,), (4,))
  """Converts a homogeneous matrix to a position and quaternion.

  Args:
    hmat: (4, 4) containing a homogeneous transform.

  Returns:
    ((3,), (4,)) 2-tuple containing a position (3,) and quaternion: (4,).
  """
  quaternion = mat_to_quat(hmat)
  position = hmat[:3, 3]
  return position, quaternion


def pos_quat_inv(
    pos: jnp.ndarray,  # (3,)
    quat: jnp.ndarray,  # (4,)
) -> Tuple[jnp.ndarray, jnp.ndarray]:  # ((3,), (4,))
  """Returns the inverse of the pose given by pos and quat."""
  inv_quat = quat_inv(quat)
  inv_pos = quat_rotate(inv_quat, pos * -1)

  return inv_pos, inv_quat


def pos_quat_mul(
    pos1: jnp.ndarray,  # (3,)
    quat1: jnp.ndarray,  # (4,)
    pos2: jnp.ndarray,  # (3,)
    quat2: jnp.ndarray,  # (4,)
) -> Tuple[jnp.ndarray, jnp.ndarray]:  # ((3,), (4,))
  """Returns the product of pose 1 and 2 given as pos and quat."""
  new_pos = pos1 + quat_rotate(quat1, pos2)
  new_quat = quat_mul(quat1, quat2)

  return new_pos, new_quat


def hmat_inv(
    hmat: jnp.ndarray,  # (4, 4)
) -> jnp.ndarray:  # (4, 4)
  """Numerically stable inverse of homogeneous transform."""
  rot = hmat[0:3, 0:3]
  pos = hmat[0:3, 3]
  hinv = jnp.eye(4)
  hinv = hinv.at[:3, :3].set(rot.T)
  hinv = hinv.at[:3, 3].set(_dot(rot.T, -pos))

  return hinv


def hmat_to_twist(
    hmat: jnp.ndarray,  # (4, 4)
) -> jnp.ndarray:  # (6,)
  """Returns the exponential coordinates for the homogeneous transform H.

  See Murray1994: A Mathematical Introduction to Robotic Manipulation
  Lynch & Park 2017: Modern Robotics: Mechanics, Planning, and Control

  Args:
    hmat: (4, 4) array containing a homogeneous transform.

  Returns:
    A vector with shape (6,) representing the instantaneous velocity and
    normalized axis of rotation, scaled by the magnitude of the twist.
    Intuitively, if this twist is integrated for unit time (by `twist_to_hmat`)
    it will recover `hmat`.
  """
  r = hmat[0:3, 0:3]
  p = hmat[0:3, 3]

  def _small_angle(r, p):
    del r
    wn = jnp.zeros(3)
    v = p
    theta = 1.
    return wn, v, theta

  def _general(r, p):
    w = rmat_to_axisangle(r)
    theta = jnp.linalg.norm(w)
    wn = w / theta
    s = cross_mat_from_vec3(wn)
    wn2d = jnp.atleast_2d(wn).T
    a = _dot(jnp.eye(3) - r, s) + _dot(wn2d, wn2d.T) * theta
    v = _dot(jnp.linalg.pinv(a), p)
    return wn, v, theta

  nearly_identity = jnp.allclose(r, jnp.eye(3), atol=1e-6)
  nearly_zero = jnp.allclose(r, jnp.zeros((3, 3)), atol=1e-6)
  wn, v, theta = jax.lax.cond(
      nearly_identity | nearly_zero,
      lambda x: _small_angle(*x),
      lambda x: _general(*x),
      (r, p),
  )
  xi = jnp.hstack([v, wn])
  return xi * theta


def twist_to_hmat(
    xi: jnp.ndarray,  # (6,)
) -> jnp.ndarray:  # (4, 4)
  """Returns homogeneous transform from exponential coordinates xi=[w, v],theta.

  The magnitude of the angle is encoded in the length of w if w is nonzero, else
  in the magnitude of v.
  See Murray 1994: A Mathematical Introduction to Robotic Manipulation or
  Lynch & Park 2017: Modern Robotics: Mechanics, Planning, and Control

  Args:
    xi: A 6-vector containing: v - 3-vector representing the instantaneous
      velocity. w - 3-vector representing the axis of rotation. Scaled by the
      magnitude of the rotation.

  Returns:
    H: A 4x4 numpy array containing a homogeneous transform.
  """

  def _small_angle(xi):
    v = xi[0:3]
    r = jnp.eye(3)
    p = v  # assume already scaled by theta
    return r, p

  def _general(xi):
    v = xi[0:3]
    w = xi[3:6]
    theta = jnp.linalg.norm(w)
    wn = w / theta
    vn = v / theta
    s = cross_mat_from_vec3(wn)
    r = jnp.eye(3) + s * jnp.sin(theta) + _dot(s, s) * (1 - jnp.cos(theta))
    p = _dot(_dot(jnp.eye(3) - r, s), vn) + wn * _dot(wn.T, vn) * theta
    return r, p

  r, p = jax.lax.cond(
      jnp.allclose(xi[3:6], 0, atol=1e-6), _small_angle, _general, (xi)
  )

  hmat = jnp.eye(4)
  hmat = hmat.at[:3, :3].set(r)
  hmat = hmat.at[:3, 3].set(p)
  return hmat


def posaxisangle_to_hmat(
    posaxisangle: jnp.ndarray,  # (6,)
) -> jnp.ndarray:  # (4, 4)
  """Converts position + axis-angle representation to homogeneous transform.

  Args:
    posaxisangle: (6) array containing [position, axisangle]

  Returns:
    A 4x4 numpy array containing a homogeneous transform.
  """
  pos, axisangle = posaxisangle[:3], posaxisangle[3:]
  rmat = axisangle_to_rmat(axisangle)
  hmat = jnp.eye(4)
  hmat = hmat.at[:3, 3].set(pos)
  hmat = hmat.at[:3, :3].set(rmat)
  return hmat


def hmat_to_posaxisangle(
    hmat: jnp.ndarray,  # (4, 4)
) -> jnp.ndarray:  # (6,)
  """Converts homogeneous transform to position + axis-angle representation.

  Args:
    hmat: (4, 4) array containing a homogeneous transform.

  Returns:
    posaxisangle: (6) array containing [position, axisangle]
  """
  pos, rmat = hmat[:3, 3], hmat[:3, :3]
  axisangle = rmat_to_axisangle(rmat)
  return jnp.concatenate((pos, axisangle))


def normal_to_rmat(
    normal: jnp.ndarray,  # (3,)
    xbasis: Optional[jnp.ndarray] = None,  # (3,)
) -> jnp.ndarray:  # (3, 3)
  """Converts a normal to a coordinate frame w/ normal on z-axis.

  Args:
    normal: (3,) array containing a vector normal to the plane of interest.
    xbasis: (3,) optional array containing a vector to cross with the normal to
      produce the x-basis of the rmat. This can be anything, but typically would
      be something that's constant w.r.t. `normal`, e.g. the x-basis of the
      canonical frame for the body. This allows the resulting rotation matrix to
      be consistent as the body changes orientation.

  Returns:
    A (3, 3) rotation matrix.
  """
  if xbasis is None:
    xbasis = jnp.array([1., 0., 0.])
  else:
    xbasis = xbasis / jnp.linalg.norm(xbasis)

  z = normal
  x = jnp.cross(z, xbasis)  # not guaranteed to be unit-len (inputs not ortho).
  y = jnp.cross(z, x)
  rmat = jnp.stack([x, y, z], axis=1)

  # Normalize to ensure valid roation.
  rmat = rmat / jnp.linalg.norm(rmat, axis=0)

  return rmat


def point_normal_to_hmat(
    pt: jnp.ndarray,  # (3,)
    normal: jnp.ndarray,  # (3,)
    xbasis: Optional[jnp.ndarray] = None,  # (3,)
) -> jnp.ndarray:  # (4, 4)
  """Converts a point and normal to homogeneous matrix w/ normal on z-axis.

  Args:
    pt: (3,) array containing the point of interest.
    normal: (3,) array containing a vector normal to the plane of interest.
    xbasis: (3,) optional array containing a vector to cross with the normal to
      produce the x-basis of the rmat. This can be anything, but typically would
      be something that's constant w.r.t. `normal`, e.g. the x-basis of the
      canonical frame for the body. This allows the resulting rotation matrix to
      be consistent as the body changes orientation.

  Returns:
    A (4, 4) homogeneous transformation matrix.
  """
  rmat = normal_to_rmat(normal, xbasis)
  hmat = jnp.eye(4)
  hmat = hmat.at[:3, :3].set(rmat)
  hmat = hmat.at[:3, 3].set(pt)

  return hmat


def velocity_transform(
    hmat: jnp.ndarray,  # (4, 4)
    vel: Optional[jnp.ndarray] = None,  # (6,)
) -> jnp.ndarray:  # (6, 6)
  """Returns a 6x6 matrix for mapping velocities to the defined frame.

  If R is the rotation part of hmat, and p the translation, and v the linear
  component of the twist and w the angular, this function computes the following
  matrix operator:
  [R,     (p+)R][v]
  [0,     R    ][w]
  Where "x" is cross-product, and "p+" is the 3x3 cross-product operator for
  3-vector p.

  Usage: recall that v is interpreted as the velocity of a point attached to the
  origin of some frame A. We can use velocity_transform to determine the
  equivalent velocity at a point in frame B relative to A using H_B_A, the
  transform from A to B (i.e. the pose of A in frame B). E.g. to compute the
  velocity v_orig of the origin at another point in the body frame, we use:
  v_pt = velocity_transform(H_point_origin, v_orig)
  Where H_point_origin defines the transform from the origin to target point.

  Args:
    hmat: (4, 4) A transform to the frame the target frame.
    vel: (6,) If provided, return the transformed velocity, else the full 6x6
      transform.

  Returns:
    A 6x6 matrix for mapping velocities, as 6d twists (vx,vy,vz,wx,wy,wz) to the
    frame defined in the homogeneous transform hmat.
  """

  r = hmat[0:3, 0:3]
  p = hmat[0:3, 3]
  pcross = cross_mat_from_vec3(p)
  tv = jnp.vstack(
      [jnp.hstack([r, _dot(pcross, r)]),
       jnp.hstack([jnp.zeros((3, 3)), r])])

  if vel is None:
    return tv
  else:
    return _dot(tv, vel)


def force_transform(
    ht: jnp.ndarray,  # (4, 4)
    wrench: Optional[jnp.ndarray] = None,  # (6,)
) -> jnp.ndarray:  # (6, 6)
  """Returns a 6x6 matrix for mapping forces as 6D wrenches.

  If R is the rotation part of H, and p the translation, and f the linear
  component of the wrench and t the angular, this function computes the
  following matrix operator:
  [R,     0][f]
  [(p+)R, R][t]
  Where x is cross-product, and p+ is the 3x3 cross-product operator for
  the 3-vector p.

  Args:
    ht: (4, 4) A transform from the source to target frame.
    wrench: (6,) If provided, return the transformed wrench, else the full 6x6
      transform.

  Returns:
    A 6x6 matrix for mapping forces, as 6d wrenches (fx,fy,fz,tx,ty,tz) to the
    frame defined in the homogeneous transform hmat.
  """
  r = ht[0:3, 0:3]
  p = ht[0:3, 3]
  pcross = cross_mat_from_vec3(p)

  tw = jnp.vstack(
      [jnp.hstack([r, jnp.zeros((3, 3))]),
       jnp.hstack([_dot(pcross, r), r])])

  if wrench is None:
    return tw
  else:
    return _dot(tw, wrench)
