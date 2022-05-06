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
"""Rigid-body transformations including velocities and static forces."""

from typing import Tuple, Union

from dm_robotics.transformations import _types as types
import numpy as np
import quaternion

_TOL = 1e-10


# Any functions in this file should always be drop-in replacements for functions
# in _transformations.py.

# LINT.IfChange


def axisangle_to_quat(axisangle: types.AxisAngleArray) -> types.QuatArray:
  """Returns the quaternion corresponding to the provided axis-angle vector.

  Args:
    axisangle: A [..,3] numpy array describing the axis of rotation, with angle
      encoded by its length

  Returns:
    quat: A quaternion [w, i, j, k]
  """
  quat = quaternion.from_rotation_vector(axisangle)
  return quaternion.as_float_array(quat)


def quat_conj(quat: types.QuatArray) -> types.QuatArray:
  """Return conjugate of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion [w, -i, -j, -k] representing the inverse of the rotation
    defined by `quat` (not assuming normalization).
  """
  quat = quaternion.from_float_array(quat)
  return quaternion.as_float_array(quat.conj())


def quat_inv(quat: types.QuatArray) -> types.QuatArray:
  """Return inverse of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the inverse of the original rotation.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = quaternion.from_float_array(quat)
  return quaternion.as_float_array(1.0 / quat)


def quat_mul(quat1: types.QuatArray, quat2: types.QuatArray) -> types.QuatArray:
  """Multiply quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat1: A quaternion [w, i, j, k].
    quat2: A quaternion [w, i, j, k].

  Returns:
    The quaternion product, aka hamiltonian product.
  """
  quat1 = quaternion.from_float_array(quat1)
  quat2 = quaternion.from_float_array(quat2)
  return quaternion.as_float_array(quat1 * quat2)


def quat_log(quat: types.QuatArray, tol: float = _TOL) -> types.QuatArray:
  """Log of a quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].
    tol: numerical tolerance to prevent nan.

  Returns:
    4D array representing the log of `quat`. This is analogous to
    `rmat_to_axisangle`.
  """
  if tol == 0:
    quat = quaternion.from_float_array(quat)
    return quaternion.as_float_array(np.log(quat))

  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  q_norm = np.linalg.norm(quat + tol, axis=-1, keepdims=True)
  a = quat[..., 0:1]
  v = np.stack([quat[..., 1], quat[..., 2], quat[..., 3]], axis=-1)
  # Clip to 2*tol because we subtract it here
  v_new = v / np.linalg.norm(
      v + tol, axis=-1, keepdims=True) * np.arccos(a / q_norm)
  return np.stack(
      [np.log(q_norm[..., 0]), v_new[..., 0], v_new[..., 1], v_new[..., 2]],
      axis=-1)


def quat_exp(quat: types.QuatArray, tol: float = _TOL) -> types.QuatArray:
  """Exp of a quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].
    tol: numerical tolerance to prevent nan.

  Returns:
    Exp of quaternion.
  """
  if tol == 0:
    quat = quaternion.from_float_array(quat)
    return quaternion.as_float_array(np.exp(quat))

  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  a = quat[..., 0:1]
  v = np.stack([quat[..., 1], quat[..., 2], quat[..., 3]], axis=-1)
  v_norm = np.linalg.norm(v + tol, axis=-1, keepdims=True)
  v_new = np.exp(a) * v / v_norm * np.sin(v_norm)
  a_new = np.exp(a) * np.cos(v_norm)
  return np.stack([a_new[..., 0], v_new[..., 0], v_new[..., 1], v_new[..., 2]],
                  axis=-1)


def quat_rotate(quat: types.QuatArray,
                vec: types.PositionArray) -> types.PositionArray:
  """Rotate a vector by a unit quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k]. The norm of this vector should be 1.
    vec: A 3-vector representing a position.

  Returns:
    The rotated vector.
  """

  quat = quaternion.from_float_array(quat)
  vec = quaternion.from_vector_part(vec)
  return quaternion.as_vector_part(quat * vec * quat.conj())


def quat_slerp(quat0: types.QuatArray, quat1: types.QuatArray,
               fraction: float) -> types.QuatArray:
  """Return spherical linear interpolation between two unit quaternions.

  Equivalent to:
  quat_mul(
    quat0, quat_exp(quat_log(quat_diff_passive(quat0, quat1)) * fraction)
  )

  Args:
    quat0: A unit quaternion [w, i, j, k].
    quat1: A unit quaternion [w, i, j, k].
    fraction: Scalar between 0.0 and 1.0.

  Returns:
    A unit quaternion `fraction` of the way from quat0 to quat1.

  Raises:
    ValueError: If invalid fraction passed.
  """
  quat0 = quaternion.from_float_array(quat0)
  quat1 = quaternion.from_float_array(quat1)
  quat = quaternion.slerp_evaluate(quat0, quat1, fraction)
  return quaternion.as_float_array(quat)


def quat_to_mat(quat: types.QuatArray) -> types.HomogeneousMatrix:
  """Return homogeneous rotation matrix from quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k].

  Returns:
    A 4x4 homogeneous matrix with the rotation corresponding to `quat`.
  """
  quat = quaternion.from_float_array(quat)
  mat = np.eye(4, dtype=np.float64)
  mat[:3, :3] = quaternion.as_rotation_matrix(quat)
  return mat


def pos_quat_to_hmat(pos: types.PositionArray,
                     quat: types.QuatArray) -> types.HomogeneousMatrix:
  """Returns a 4x4 Homogeneous transform for the given configuration.

  Args:
    pos: A cartesian position [x, y, z].
    quat: A unit quaternion [w, i, j, k].

  Returns:
    A 4x4 Homogenous transform as a numpy array.
  """
  hmat = quat_to_mat(quat)
  hmat[:3, 3] = pos
  return hmat


def integrate_quat(quat: types.QuatArray,
                   vel: types.AngVelArray) -> types.QuatArray:
  """Integrates the unit quaternion by the given angular velocity.

  For information on this operation see:
  https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/

  Args:
    quat: A unit quaternion [w, i, j, k] to integrate.
    vel: The 3D angular velocity used to integrate the orientation. It is
      assumed that the angular velocity is given in the same frame as the
      quaternion and it has been properly scaled by the timestep over which the
      integration is done. In particular the velocity should NOT be given in the
      frame of the rotating object.

  Returns:
    The normalized integrated quaternion.
  """
  vel = np.concatenate(([0], vel))
  quat = quat + 0.5 * quat_mul(vel, quat)
  return quat / np.linalg.norm(quat)


def mat_to_quat(
    mat: Union[types.RotationMatrix,
               types.HomogeneousMatrix]) -> types.QuatArray:
  """Return quaternion from homogeneous or rotation matrix.

  Args:
    mat: A homogeneous transform or rotation matrix

  Returns:
    A quaternion [w, i, j, k].
  """
  quat = quaternion.from_rotation_matrix(mat)
  return quaternion.as_float_array(quat)


def hmat_to_pos_quat(
    hmat: types.HomogeneousMatrix
) -> Tuple[types.PositionArray, types.QuatArray]:
  """Return a cartesian position and quaternion from a homogeneous matrix.

  Args:
    hmat: A homogeneous transform or rotation matrix

  Returns:
    A tuple containing:
    - A cartesian position [x, y, z].
    - A quaternion [w, i, j, k].
  """
  return hmat[:3, 3], mat_to_quat(hmat)


# LINT.ThenChange(_transformations.py)
