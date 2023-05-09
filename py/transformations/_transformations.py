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
from typing import Optional, Tuple, Union
from absl import logging
from dm_robotics.transformations import _types as types
import numpy as np

# Constants used to determine when a rotation is close to a pole.
_POLE_LIMIT = (1.0 - 1e-6)
_TOL = 1e-10
# Constant used to decide when to use the arcos over the arcsin
_TOL_ARCCOS = 1 / np.sqrt(2)
_IDENTITY_QUATERNION = np.array([1, 0, 0, 0], dtype=np.float64)


def _clip_within_precision(number: float, low: float,
                           high: float, precision: float = _TOL):
  """Clips input to the range [low, high], checking precision.

  Args:
    number: Number to be clipped.
    low: Lower bound (inclusive).
    high: Upper bound (inclusive).
    precision: Tolerance.

  Returns:
    Input clipped to given range.

  Raises:
    ValueError: If number is outside given range by more than given precision.
  """
  if number < low - precision or number > high + precision:
    raise ValueError(
        'Input {:.12f} not inside range [{:.12f}, {:.12f}] with precision {}'.
        format(number, low, high, precision))
  return np.clip(number, low, high)


def _batch_mm(m1, m2):
  """Batch matrix multiply.

  Args:
    m1: input lhs matrix with shape (batch, n, m).
    m2: input rhs matrix with shape (batch, m, o).

  Returns:
    product matrix with shape (batch, n, o).
  """
  return np.einsum('bij,bjk->bik', m1, m2)


def _rmat_to_euler_xyz(rmat: types.RotationMatrix) -> types.EulerArray['XYZ']:
  """Converts a 3x3 rotation matrix to XYZ euler angles."""
  # | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  # | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  # | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  if rmat[0, 2] > _POLE_LIMIT:
    logging.log_every_n_seconds(logging.WARNING, 'Angle at North Pole', 60)
    z = np.arctan2(rmat[1, 0], rmat[1, 1])
    y = np.pi/2
    x = 0.0
    return np.array([x, y, z])

  if rmat[0, 2] < -_POLE_LIMIT:
    logging.log_every_n_seconds(logging.WARNING, 'Angle at South Pole', 60)
    z = np.arctan2(rmat[1, 0], rmat[1, 1])
    y = -np.pi/2
    x = 0.0
    return np.array([x, y, z])

  z = -np.arctan2(rmat[0, 1], rmat[0, 0])
  y = np.arcsin(rmat[0, 2])
  x = -np.arctan2(rmat[1, 2], rmat[2, 2])

  # order of return is the order of input
  return np.array([x, y, z])


def _rmat_to_euler_xyx(rmat: types.RotationMatrix) -> types.EulerArray['XYX']:
  """Converts a 3x3 rotation matrix to XYX euler angles."""
  # | r00 r01 r02 |   |  cy      sy*sx1               sy*cx1             |
  # | r10 r11 r12 | = |  sy*sx0  cx0*cx1-cy*sx0*sx1  -cy*cx1*sx0-cx0*sx1 |
  # | r20 r21 r22 |   | -sy*cx0  cx1*sx0+cy*cx0*sx1   cy*cx0*cx1-sx0*sx1 |

  if rmat[0, 0] < 1.0:
    if rmat[0, 0] > -1.0:
      y = np.arccos(_clip_within_precision(rmat[0, 0], -1., 1.))
      x0 = np.arctan2(rmat[1, 0], -rmat[2, 0])
      x1 = np.arctan2(rmat[0, 1], rmat[0, 2])
      return np.array([x0, y, x1])
    else:
      # Not a unique solution:  x1_angle - x0_angle = atan2(-r12,r11)
      y = np.pi
      x0 = -np.arctan2(-rmat[1, 2], rmat[1, 1])
      x1 = 0.0
      return np.array([x0, y, x1])
  else:
    # Not a unique solution:  x1_angle + x0_angle = atan2(-r12,r11)
    y = 0.0
    x0 = -np.arctan2(-rmat[1, 2], rmat[1, 1])
    x1 = 0.0
    return np.array([x0, y, x1])


def _rmat_to_euler_zyx(rmat: types.RotationMatrix) -> types.EulerArray['ZYX']:
  """Converts a 3x3 rotation matrix to ZYX euler angles."""
  if rmat[2, 0] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    x = np.arctan2(rmat[0, 1], rmat[0, 2])
    y = -np.pi/2
    z = 0.0
    return np.array([z, y, x])

  if rmat[2, 0] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    x = np.arctan2(rmat[0, 1], rmat[0, 2])
    y = np.pi/2
    z = 0.0
    return np.array([z, y, x])

  x = np.arctan2(rmat[2, 1], rmat[2, 2])
  y = -np.arcsin(rmat[2, 0])
  z = np.arctan2(rmat[1, 0], rmat[0, 0])

  # order of return is the order of input
  return np.array([z, y, x])


def _rmat_to_euler_xzy(rmat: types.RotationMatrix) -> types.EulerArray['XZY']:
  """Converts a 3x3 rotation matrix to XZY euler angles."""
  if rmat[0, 1] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    y = np.arctan2(rmat[1, 2], rmat[1, 0])
    z = -np.pi/2
    x = 0.0
    return np.array([x, z, y])

  if rmat[0, 1] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    y = np.arctan2(rmat[1, 2], rmat[1, 0])
    z = np.pi/2
    x = 0.0
    return np.array([x, z, y])

  y = np.arctan2(rmat[0, 2], rmat[0, 0])
  z = -np.arcsin(rmat[0, 1])
  x = np.arctan2(rmat[2, 1], rmat[1, 1])

  # order of return is the order of input
  return np.array([x, z, y])


def _rmat_to_euler_yzx(rmat: types.RotationMatrix) -> types.EulerArray['YZX']:
  """Converts a 3x3 rotation matrix to YZX euler angles."""
  if rmat[1, 0] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    x = -np.arctan2(rmat[0, 2], rmat[0, 1])
    z = np.pi/2
    y = 0.0
    return np.array([y, z, x])

  if rmat[1, 0] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    x = -np.arctan2(rmat[0, 2], rmat[0, 1])
    z = -np.pi/2
    y = 0.0
    return np.array([y, z, x])

  x = -np.arctan2(rmat[1, 2], rmat[1, 1])
  z = np.arcsin(rmat[1, 0])
  y = -np.arctan2(rmat[2, 0], rmat[0, 0])

  # order of return is the order of input
  return np.array([y, z, x])


def _rmat_to_euler_zxy(rmat: types.RotationMatrix) -> types.EulerArray['ZXY']:
  """Converts a 3x3 rotation matrix to ZXY euler angles."""
  if rmat[2, 1] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    y = np.arctan2(rmat[0, 2], rmat[0, 0])
    x = np.pi/2
    z = 0.0
    return np.array([z, x, y])

  if rmat[2, 1] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    y = np.arctan2(rmat[0, 2], rmat[0, 0])
    x = -np.pi/2
    z = 0.0
    return np.array([z, x, y])

  y = -np.arctan2(rmat[2, 0], rmat[2, 2])
  x = np.arcsin(rmat[2, 1])
  z = -np.arctan2(rmat[0, 1], rmat[1, 1])

  # order of return is the order of input
  return np.array([z, x, y])


def _rmat_to_euler_yxz(rmat: types.RotationMatrix) -> types.EulerArray['YXZ']:
  """Converts a 3x3 rotation matrix to YXZ euler angles."""
  if rmat[1, 2] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    z = -np.arctan2(rmat[0, 1], rmat[0, 0])
    x = -np.pi/2
    y = 0.0
    return np.array([y, x, z])

  if rmat[1, 2] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    z = -np.arctan2(rmat[0, 1], rmat[0, 0])
    x = np.pi/2
    y = 0.0
    return np.array([y, x, z])

  z = np.arctan2(rmat[1, 0], rmat[1, 1])
  x = -np.arcsin(rmat[1, 2])
  y = np.arctan2(rmat[0, 2], rmat[2, 2])

  # order of return is the order of input
  return np.array([y, x, z])


def _rmat_to_euler_xzx(rmat: types.RotationMatrix) -> types.EulerArray['XZX']:
  """Converts a 3x3 rotation matrix to XZX euler angles."""
  # | r00 r01 r02 |   |  cz      -sz*cx1               sz*sx1              |
  # | r10 r11 r12 | = |  cx0*sz   cx0*cz*cx1-sx0*sx1  -sx0*cx1-cx0*cz*sx1  |
  # | r20 r21 r22 |   |  sx0*sz   sx0*cz*cx1+cx0*sx1   cx0*cx1-sx0*cz*sx1  |

  if rmat[0, 0] < 1.0:
    if rmat[0, 0] > -1.0:
      z = np.arccos(_clip_within_precision(rmat[0, 0], -1., 1.))
      x0 = np.arctan2(rmat[2, 0], rmat[1, 0])
      x1 = np.arctan2(rmat[0, 2], -rmat[0, 1])
      return np.array([x0, z, x1])
    else:
      # Not a unique solution:  x0_angle - x1_angle = atan2(r12,r11)
      z = np.pi
      x0 = np.arctan2(rmat[1, 2], rmat[1, 1])
      x1 = 0.0
      return np.array([x0, z, x1])
  else:
    # Not a unique solution:  x0_angle + x1_angle = atan2(-r12, r11)
    z = 0.0
    x0 = np.arctan2(-rmat[1, 2], rmat[1, 1])
    x1 = 0.0
    return np.array([x0, z, x1])


def _rmat_to_euler_yxy(rmat: types.RotationMatrix) -> types.EulerArray['YXY']:
  """Converts a 3x3 rotation matrix to YXY euler angles."""
  # | r00 r01 r02 | = | -sy0*sy1*cx+cy0*cy1   sx*sy0    sy0*cx*cy1+sy1*cy0  |
  # | r10 r11 r12 | = |  sx*sy1,              cx       -sx*cy1              |
  # | r20 r21 r22 | = | -sy0*cy1-sy1*cx*cy0   sx*cy0   -sy0*sy1+cx*cy0*cy1  |

  if rmat[1, 1] < 1.0:
    if rmat[1, 1] > -1.0:
      x = np.arccos(_clip_within_precision(rmat[1, 1], -1., 1.))
      y0 = np.arctan2(rmat[0, 1], rmat[2, 1])
      y1 = np.arctan2(rmat[1, 0], -rmat[1, 2])
      return np.array([y0, x, y1])
    else:
      # Not a unique solution:  y0_angle - y1_angle = atan2(r02, r22)
      x = np.pi
      y0 = np.arctan2(rmat[0, 2], rmat[2, 2])
      y1 = 0.0
      return np.array([y0, x, y1])
  else:
    # Not a unique solution:  y0_angle + y1_angle = atan2(r02, r22)
    x = 0.0
    y0 = np.arctan2(rmat[0, 2], rmat[2, 2])
    y1 = 0.0
    return np.array([y0, x, y1])


def _rmat_to_euler_yzy(rmat: types.RotationMatrix) -> types.EulerArray['YZY']:
  """Converts a 3x3 rotation matrix to YZY euler angles."""
  # | r00 r01 r02 | = | -sy0*sy1+cy0*cy1*cz  -sz*cy0    sy0*cy1+sy1*cy0*cz  |
  # | r10 r11 r12 | = |  sz*cy1               cz        sy1*sz              |
  # | r20 r21 r22 | = | -sy0*cy1*cz-sy1*cy0   sy0*sz   -sy0*sy1*cz+cy0*cy1  |

  if rmat[1, 1] < 1.0:
    if rmat[1, 1] > -1.0:
      z = np.arccos(_clip_within_precision(rmat[1, 1], -1., 1.))
      y0 = np.arctan2(rmat[2, 1], -rmat[0, 1])
      y1 = np.arctan2(rmat[1, 2], rmat[1, 0])
      return np.array([y0, z, y1])
    else:
      # Not a unique solution:  y0_angle - y1_angle = atan2(r02, r22)
      z = np.pi
      y0 = np.arctan2(rmat[0, 2], rmat[2, 2])
      y1 = 0.0
      return np.array([y0, z, y1])
  else:
    # Not a unique solution:  y0_angle + y1_angle = atan2(r02, r22)
    z = 0.0
    y0 = np.arctan2(rmat[0, 2], rmat[2, 2])
    y1 = 0.0
    return np.array([y0, z, y1])


def _rmat_to_euler_zxz(rmat: types.RotationMatrix) -> types.EulerArray['ZXZ']:
  """Converts a 3x3 rotation matrix to ZXZ euler angles."""
  # | r00 r01 r02 | = | -sz0*sz1*cx+cz0*cz1   -sz0*cx*cz1-sz1*cz0   sx*sz0  |
  # | r10 r11 r12 | = |  sz0*cz1+sz1*cx*cz0   -sz0*sz1+cx*cz0*cz1  -sx*cz0  |
  # | r20 r21 r22 | = |  sx*sz1                sx*cz1               cx      |

  if rmat[2, 2] < 1.0:
    if rmat[2, 2] > -1.0:
      x = np.arccos(_clip_within_precision(rmat[2, 2], -1., 1.))
      z0 = np.arctan2(rmat[0, 2], -rmat[1, 2])
      z1 = np.arctan2(rmat[2, 0], rmat[2, 1])
      return np.array([z0, x, z1])
    else:
      # Not a unique solution:  z0_angle - z1_angle = atan2(r10, r00)
      x = np.pi
      z0 = np.arctan2(rmat[1, 0], rmat[0, 0])
      z1 = 0.0
      return np.array([z0, x, z1])
  else:
    # Not a unique solution:  z0_angle + z1_angle = atan2(r10, r00)
    x = 0.0
    z0 = np.arctan2(rmat[1, 0], rmat[0, 0])
    z1 = 0.0
    return np.array([z0, x, z1])


def _rmat_to_euler_zyz(rmat: types.RotationMatrix) -> types.EulerArray['ZYZ']:
  """Converts a 3x3 rotation matrix to ZYZ euler angles."""
  # | r00 r01 r02 | = | -sz0*sz1+cy*cz0*cz1  -sz0*cz1-sz1*cy*cz0  sy*cz0  |
  # | r10 r11 r12 | = |  sz0*cy*cz1+sz1*cz0  -sz0*sz1*cy+cz0*cz1  sy*sz0  |
  # | r20 r21 r22 | = | -sy*cz1               sy*sz1              cy      |

  if rmat[2, 2] < 1.0:
    if rmat[2, 2] > -1.0:
      y = np.arccos(_clip_within_precision(rmat[2, 2], -1., 1.))
      z0 = np.arctan2(rmat[1, 2], rmat[0, 2])
      z1 = np.arctan2(rmat[2, 1], -rmat[2, 0])
      return np.array([z0, y, z1])
    else:
      # Not a unique solution:  z0_angle - z1_angle = atan2(r10, r00)
      y = np.pi
      z0 = np.arctan2(rmat[1, 0], rmat[0, 0])
      z1 = 0.0
      return np.array([z0, y, z1])
  else:
    # Not a unique solution:  z0_angle + z1_angle = atan2(r10, r00)
    y = 0.0
    z0 = np.arctan2(rmat[1, 0], rmat[0, 0])
    z1 = 0.0
    return np.array([z0, y, z1])


def _axis_rotation(theta, full: bool):
  """Returns the theta dim, cos and sin, and blank matrix for axis rotation."""
  n = 1 if np.isscalar(theta) else len(theta)
  ct = np.cos(theta)
  st = np.sin(theta)

  if full:
    rmat = np.zeros((n, 4, 4))
    rmat[:, 3, 3] = 1.
  else:
    rmat = np.zeros((n, 3, 3))

  return n, ct, st, rmat

# map from full rotation orderings to euler conversion functions
_eulermap = {
    'XYZ': _rmat_to_euler_xyz,
    'XYX': _rmat_to_euler_xyx,
    'XZY': _rmat_to_euler_xzy,
    'ZYX': _rmat_to_euler_zyx,
    'YZX': _rmat_to_euler_yzx,
    'ZXY': _rmat_to_euler_zxy,
    'YXZ': _rmat_to_euler_yxz,
    'XZX': _rmat_to_euler_xzx,
    'YXY': _rmat_to_euler_yxy,
    'YZY': _rmat_to_euler_yzy,
    'ZXZ': _rmat_to_euler_zxz,
    'ZYZ': _rmat_to_euler_zyz,
}


def cross_mat_from_vec3(v):
  """Returns the skew-symmetric matrix cross-product operator.

  Args:
      v: A 3x1 vector.

  Returns:
      A matrix cross-product operator P (3x3) for the vector v = [x,y,z]^T,
      such that  v x b = Pb for any 3-vector b
  """
  x, y, z = v[0], v[1], v[2]
  return np.array([[0, -z, y],
                   [z, 0, -x],
                   [-y, x, 0]])


def axisangle_to_euler(axisangle: types.AxisAngleArray,
                       ordering: str = 'XYZ') -> types.SomeEulerArray:
  """Returns euler angles corresponding to the exponential coordinates.

  Args:
      axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length.
      ordering: Desired euler angle ordering.

  Returns: A euler triple
  """
  rmat = axisangle_to_rmat(axisangle)
  return rmat_to_euler(rmat, ordering)


def axisangle_to_rmat(axisangle: types.AxisAngleArray) -> types.RotationMatrix:
  """Returns rotation matrix corresponding to the exponential coordinates.

  See Murray1994: A Mathematical Introduction to Robotic Manipulation

  Args:
      axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length.

  Returns: A tuple (w, theta)
      R: a 3x3 numpy array describing the rotation
  """
  theta = np.linalg.norm(axisangle)
  if np.allclose(theta, 0):
    s_theta = cross_mat_from_vec3(axisangle)
    return np.eye(3) + s_theta + s_theta.dot(s_theta) * 0.5
  else:
    wn = axisangle / theta
    s = cross_mat_from_vec3(wn)
    return np.eye(3) + s * np.sin(theta) + s.dot(s) * (1-np.cos(theta))


# LINT.IfChange
def axisangle_to_quat(axisangle: types.AxisAngleArray) -> types.QuatArray:
  """Returns the quaternion corresponding to the provided axis-angle vector.

  Args:
    axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length

  Returns:
    quat: A quaternion [w, i, j, k]
  """
  theta = np.linalg.norm(axisangle)
  if np.allclose(theta, 0):
    return _IDENTITY_QUATERNION
  else:
    wn = axisangle/theta
    return np.hstack([np.cos(theta/2), wn * np.sin(theta/2)])
# LINT.ThenChange(_transformations_quat.py)


def euler_to_axisangle(euler_vec: types.SomeEulerArray,
                       ordering: str = 'XYZ') -> types.AxisAngleArray:
  """Returns the euler angles corresponding to the provided axis-angle vector.

  Args:
    euler_vec: The euler angle rotations.
    ordering: Desired euler angle ordering.

  Returns:
    axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length
  """
  rmat = euler_to_rmat(euler_vec, ordering=ordering)
  return rmat_to_axisangle(rmat)


def euler_to_quat(euler_vec: types.SomeEulerArray,
                  ordering: str = 'XYZ') -> types.QuatArray:
  """Returns the quaternion corresponding to the provided euler angles.

  Args:
    euler_vec: The euler angle rotations.
    ordering: Desired euler angle ordering.

  Returns:
    quat: A quaternion [w, i, j, k]
  """
  mat = euler_to_rmat(euler_vec, ordering=ordering)
  return mat_to_quat(mat)


def euler_to_rmat(
    euler_vec: types.SomeEulerArray,
    ordering: str = 'ZXZ',
    full: bool = False,
    extrinsic: bool = False
) -> Union[types.HomogeneousMatrix, types.RotationMatrix]:
  """Returns rotation matrix (or transform) for the given Euler rotations.

  Euler*** methods compose a Rotation matrix corresponding to the given
  rotations r1, r2, r3 following the given rotation ordering.
  This operation follows the INTRINSIC rotation convention, i.e. defined w.r.t
  the axes of the rotating system.  Intrinsic rotations are evaluated in the
  order provided.  E.g. for XYZ we return rotX(r1) * rotY(r2) * rotZ(r3).
  This is equivalent to ZYX extrinsic, because rotZ is evaluated first in the
  fixed frame, which is then transformed by rotY and rotX.

  From Wikipedia: http://en.wikipedia.org/wiki/Euler_angles
  Any extrinsic rotation is equivalent to an extrinsic rotation by the same
  angles but with inverted order of elemental rotations, and vice-versa. For
  instance, the extrinsic rotations x-y'-z" by angles alpha, beta, gamma are
  equivalent to the extrinsic rotations z-y-x by angles gamma, beta, alpha.

  Args:
    euler_vec: The euler angle rotations.
    ordering: euler angle ordering string (see _euler_orderings).
    full: If true, returns a full 4x4 transform.
    extrinsic: Whether to use the extrinsic or intrinsic rotation convention.

  Returns:
    The rotation matrix or homogeneous transform corresponding to the given
    Euler rotation.
  """

  # map from partial rotation orderings to rotation functions
  rotmap = {'X': rotation_x_axis, 'Y': rotation_y_axis, 'Z': rotation_z_axis}
  rotations = [rotmap[c] for c in ordering]

  if extrinsic:
    rotations.reverse()

  euler_vec = np.atleast_2d(euler_vec)

  rots = []
  for i in range(len(rotations)):
    rots.append(rotations[i](euler_vec[:, i], full))

  if rots[0].ndim == 3:
    result = _batch_mm(_batch_mm(rots[0], rots[1]), rots[2])
    return result.squeeze()
  else:
    return (rots[0].dot(rots[1])).dot(rots[2])


def positive_leading_quat(quat: types.QuatArray) -> types.QuatArray:
  """Returns the positive leading version of the quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    The equivalent quaternion [w, i, j, k] with w > 0.
  """

  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  quat = np.where(np.tile(quat[..., 0:1] < 0, quat.shape[-1]), -quat, quat)

  return quat


# LINT.IfChange
def quat_conj(quat: types.QuatArray) -> types.QuatArray:
  """Return conjugate of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion [w, -i, -j, -k] representing the inverse of the rotation
    defined by `quat` (not assuming normalization).
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  return np.stack(
      [quat[..., 0], -quat[..., 1],
       -quat[..., 2], -quat[..., 3]], axis=-1).astype(np.float64)
# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
def quat_inv(quat: types.QuatArray) -> types.QuatArray:
  """Return inverse of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the inverse of the original rotation.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  return quat_conj(quat) / np.sum(quat * quat, axis=-1, keepdims=True)
# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
def quat_mul(quat1: types.QuatArray, quat2: types.QuatArray) -> types.QuatArray:
  """Multiply quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat1: A quaternion [w, i, j, k].
    quat2: A quaternion [w, i, j, k].

  Returns:
    The quaternion product, aka hamiltonian product.
  """
  # Ensure quats are np.arrays in case a tuple or a list is passed
  quat1, quat2 = np.asarray(quat1), np.asarray(quat2)

  # Construct a 4x4 matrix representation of quat1 for use with matmul
  w1, x1, y1, z1 = [quat1[..., i] for i in range(4)]
  qmat = np.stack(
      [np.stack([w1, -x1, -y1, -z1], axis=-1),
       np.stack([x1, w1, -z1, y1], axis=-1),
       np.stack([y1, z1, w1, -x1], axis=-1),
       np.stack([z1, -y1, x1, w1], axis=-1)],
      axis=-2)

  # Compute (batched) hamiltonian product
  qdot = qmat @ np.expand_dims(quat2, axis=-1)
  return np.squeeze(qdot, axis=-1)
# LINT.ThenChange(_transformations_quat.py)


def quat_diff_passive(source_quat: types.QuatArray,
                      target_quat: types.QuatArray) -> types.QuatArray:
  """Passive quaternion difference between source and target quaternions.

  This quaternion difference is used when one is trying to find the quaternion
  that brings a vector expressed in the target frame to the same vector
  expressed in the source frame.

  Note: `source_quat` and `target_quat` should represent world-frame
  orientations, i.e. both should rotate a vector expressed in their respective
  frames to world.

  This is the passive quaternion difference as the vector is not moving, only
  the frame in which it is expressed.

  For more information on active/passive rotations please refer to:
  https://www.tu-chemnitz.de/informatik/KI/edu/robotik/ws2016/lecture-02_p2.pdf

  This function supports inputs with or without leading batch dimensions.

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
  # Ensure quats are np.arrays in case a tuple or a list is passed
  source_quat, target_quat = np.asarray(source_quat), np.asarray(target_quat)
  quat = quat_mul(quat_conj(source_quat), target_quat)
  return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def quat_diff_active(source_quat: types.QuatArray,
                     target_quat: types.QuatArray) -> types.QuatArray:
  """Active quaternion difference between source and target quaternions.

  Given the unit vectors of the source frame (expressed in the world frame),
  this function gives the quaternion that rotates these vectors into the unit
  vectors of the target frame (expressed in the world frame).

  Note: `source_quat` and `target_quat` should represent active rotations of
  vectors, i.e. both should rotate the unit vectors of the world frame
  into the unit vectors of their respective frame (expressed in the world
  frame).

  This is the active quaternion difference as the vectors are being rotated
  while the reference frame they are expressed in stays the same.

  For more information on active/passive rotations please refer to:
  https://www.tu-chemnitz.de/informatik/KI/edu/robotik/ws2016/lecture-02_p2.pdf

  This function supports inputs with or without leading batch dimensions.

  Args:
    source_quat: A unit quaternion [w, i, j, k], or multi-dimensional array of
      unit quaternions.
    target_quat: A unit quaternion [w, i, j, k], or multi-dimensional array of
      unit quaternions.

  Returns:
    A normalized quaternion representing the rotation that brings the source
    frame into the target frame.
  """
  # Ensure quats are np.arrays in case a tuple or a list is passed
  source_quat, target_quat = np.asarray(source_quat), np.asarray(target_quat)
  quat = quat_mul(target_quat, quat_conj(source_quat))
  return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


# LINT.IfChange
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
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  q_norm = np.linalg.norm(quat + tol, axis=-1, keepdims=True)
  a = quat[..., 0:1]
  v = np.stack([quat[..., 1], quat[..., 2], quat[..., 3]], axis=-1)
  # Clip to 2*tol because we subtract it here
  v_new = v / np.linalg.norm(v + tol, axis=-1, keepdims=True) * np.arccos(
      a / q_norm)
  return np.stack(
      [np.log(q_norm[..., 0]), v_new[..., 0], v_new[..., 1], v_new[..., 2]],
      axis=-1)
# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
def quat_exp(quat: types.QuatArray, tol: float = _TOL) -> types.QuatArray:
  """Exp of a quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].
    tol: numerical tolerance to prevent nan.

  Returns:
    Exp of quaternion.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  a = quat[..., 0:1]
  v = np.stack([quat[..., 1], quat[..., 2], quat[..., 3]], axis=-1)
  v_norm = np.linalg.norm(v + tol, axis=-1, keepdims=True)
  v_new = np.exp(a) * v/v_norm * np.sin(v_norm)
  a_new = np.exp(a) * np.cos(v_norm)
  return np.stack([a_new[..., 0], v_new[..., 0], v_new[..., 1], v_new[..., 2]],
                  axis=-1)
# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
def quat_dist(source: types.QuatArray, target: types.QuatArray) -> np.ndarray:
  """Computes distance between source and target quaternions.

  This function supports inputs with or without leading batch dimensions.

  Note: operates on unit quaternions

  Args:
    source: A unit quaternion [w, i, j, k].
    target: A unit quaternion [w, i, j, k].

  Returns:
    The rotational distance from source to target in radians.
  """
  # Ensure quats are np.arrays in case a tuple or a list is passed
  source, target = np.asarray(source), np.asarray(target)
  quat_err = quat_mul(source, quat_inv(target))
  quat_err /= np.linalg.norm(quat_err, axis=-1, keepdims=True)
  return quat_angle(quat_err)
# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
def quat_angle(quat: types.QuatArray) -> np.ndarray:
  """Computes the angle of the rotation encoded by the unit quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A unit quaternion [w, i, j, k]. The norm of this vector should be 1.

  Returns:
    The angle in radians of the rotation encoded by the quaternion.
  """

  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)

  # Ensure the quaternion is positive leading to get the shortest angle.
  quat = positive_leading_quat(quat)

  # We have w = cos(angle/2) with w the real part of the quaternion and
  # ||Im(q)|| = sin(angle/2) with Im(q) the imaginary part of the quaternion.
  # We choose the method that is less sensitive to a noisy evaluation of the
  # difference.
  condition = quat[..., 0] < _TOL_ARCCOS
  angle = np.where(
      condition,
      np.arccos(quat[..., 0], where=condition),
      np.arcsin(np.linalg.norm(quat[..., 1:], axis=-1), where=~condition),
  )
  return 2 * angle

# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
def quat_rotate(quat: types.QuatArray,
                vec: types.PositionArray) -> types.PositionArray:
  """Rotate a vector by a unit quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k]. The norm of this vector should be 1.
    vec: A 3-vector representing a position.

  Returns:
    The rotated vector.
  """
  vec = np.atleast_2d(vec)
  qvec = np.hstack([np.zeros(vec.shape[0:-1] + (1,)), vec])
  return quat_mul(quat_mul(quat, qvec), quat_conj(quat))[:, 1:].squeeze()
# LINT.ThenChange(_transformations_quat.py)


def quat_between_vectors(source_vec: types.PositionArray,
                         target_vec: types.PositionArray) -> types.QuatArray:
  """Returns the minimal quaternion that rotates `source_vec` to `target_vec`.

  An explanation for the math can be found here (under Quaternion Result):
  http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
  The input vectors can be any non-zero vectors. The returned unit quaternion is
  the shortest arc rotation from source_vec to target_vec.

  Args:
    source_vec: A 3-vector representing the source vector.
    target_vec: A 3-vector representing the target vector.

  Returns:
    A quaternion rotation between source and target vectors, such that
    quat_rotate(quat, source_vec) == target_vec.
  """
  if np.linalg.norm(source_vec) == 0 or np.linalg.norm(target_vec) == 0:
    raise ValueError('Source or target vector is a 0 vector; cannot compute '
                     'rotation for a vector with no direction.')

  dot_product_normalized = np.dot(source_vec / np.linalg.norm(source_vec),
                                  target_vec / np.linalg.norm(target_vec))
  # check if source and target vectors are parallel with same direction
  if dot_product_normalized > _POLE_LIMIT:
    # return identity rotation
    return _IDENTITY_QUATERNION

  # check if source and target vectors are parallel with opposite direction
  elif dot_product_normalized < -_POLE_LIMIT:
    # In this case we need to return a 180 degree rotation around any vector
    # that is orthogonal to source_vec.
    # To compute the orthogonal vector, we can take the cross product of the
    # source vector and any other vector that is nonparallel to source_vec.
    # To find a nonparallel vector, we can take these 3 unit vectors and find
    # which one has the smallest dot product.
    unit_vectors = np.eye(3)
    min_dotproduct = np.argmin(np.dot(unit_vectors, source_vec))
    nonparallel_vector = unit_vectors[min_dotproduct]
    # Compute the orthogonal vector.
    orthogonal_vector = np.cross(source_vec, nonparallel_vector)
    # Return the 180 degree rotation around this orthogonal vector.
    return np.concatenate(
        ([0], orthogonal_vector / np.linalg.norm(orthogonal_vector)))

  # compute the i, j, k terms of the quaternion
  ijk = np.cross(source_vec, target_vec)
  real = np.linalg.norm(source_vec) * np.linalg.norm(target_vec) + np.dot(
      source_vec, target_vec)
  q_rotation = np.concatenate(([real], ijk))
  return q_rotation / np.linalg.norm(q_rotation)


# LINT.IfChange
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
  q0 = np.array(quat0)
  q1 = np.array(quat1)
  if fraction < 0.0 or fraction > 1.0:
    raise ValueError('fraction must be between 0 and 1 (inclusive)')
  if fraction < _TOL:
    return q0
  elif fraction > 1.0 - _TOL:
    return q1
  d = np.dot(q0, q1)
  if abs(abs(d) - 1.0) < _TOL:
    return q0
  if d < 0.0:
    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all
    # four components. Fix by reversing one quaternion.
    d = -d
    q1 *= -1.0
  angle = np.arccos(_clip_within_precision(d, -1., 1.))
  if abs(angle) < _TOL:
    return q0
  isin = 1.0 / np.sin(angle)
  s0 = np.sin((1.0 - fraction) * angle) * isin
  s1 = np.sin(fraction * angle) * isin
  interp_quat = q0 * s0 + q1 * s1
  return interp_quat / np.linalg.norm(interp_quat)
# LINT.ThenChange(_transformations_quat.py)


def quat_axis(quat: types.QuatArray) -> types.AxisAngleArray:
  """Returns the rotation axis of the corresponding quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A unit quaternion [w, i, j, k].

  Returns:
    axisangle: A 3x1 normalized numpy array describing the axis of rotation.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  return quat[..., 1:4] / np.linalg.norm(quat[..., 1:4], axis=-1, keepdims=True)


def quat_to_axisangle(quat: types.QuatArray) -> types.AxisAngleArray:
  """Returns the axis-angle corresponding to the provided quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k].

  Returns:
    axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)

  # Ensure the quaternion is positive leading to get the shortest angle.
  quat = positive_leading_quat(quat)
  angle = quat_angle(quat)

  if angle < _TOL:
    return np.zeros(3)
  else:
    axis = quat_axis(quat)
    return axis * angle


def quat_to_euler(quat: types.QuatArray,
                  ordering: str = 'XYZ') -> types.SomeEulerArray:
  """Returns the euler angles corresponding to the provided quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k].
    ordering: Desired euler angle ordering.

  Returns:
    euler_vec: The euler angle rotations.
  """
  mat = quat_to_mat(quat)
  return rmat_to_euler(mat[0:3, 0:3], ordering=ordering)


# LINT.IfChange
def quat_to_mat(quat: types.QuatArray) -> types.HomogeneousMatrix:
  """Return homogeneous rotation matrix from quaternion.

  Args:
    quat: A unit quaternion [w, i, j, k].

  Returns:
    A 4x4 homogeneous matrix with the rotation corresponding to `quat`.
  """
  q = np.array(quat, dtype=np.float64, copy=True)
  nq = np.dot(q, q)
  if nq < _TOL:
    return np.identity(4)
  q *= np.sqrt(2.0 / nq)
  q = np.outer(q, q)
  return np.array(
      ((1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0),
       (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0),
       (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0),
       (0.0, 0.0, 0.0, 1.0)),
      dtype=np.float64)
# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
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
# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
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
# LINT.ThenChange(_transformations_quat.py)


def rotation_x_axis(
    theta: np.ndarray,
    full: bool = False) -> Union[types.RotationMatrix, types.HomogeneousMatrix]:
  """Returns a rotation matrix of a rotation about the X-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: If true, returns a full 4x4 transfom.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.array([[1, 0, 0]])
  rmat[:, 1, 0:3] = np.vstack([np.zeros(n), ct, -st]).T
  rmat[:, 2, 0:3] = np.vstack([np.zeros(n), st, ct]).T

  return rmat.squeeze()


def rotation_y_axis(
    theta: np.ndarray,
    full: bool = False) -> Union[types.RotationMatrix, types.HomogeneousMatrix]:
  """Returns a rotation matrix of a rotation about the Y-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: If true, returns a full 4x4 transfom.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.vstack([ct, np.zeros(n), st]).T
  rmat[:, 1, 0:3] = np.array([[0, 1, 0]])
  rmat[:, 2, 0:3] = np.vstack([-st, np.zeros(n), ct]).T

  return rmat.squeeze()


def rotation_z_axis(
    theta: np.ndarray,
    full: bool = False) -> Union[types.RotationMatrix, types.HomogeneousMatrix]:
  """Returns a rotation matrix of a rotation about the z-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: If true, returns a full 4x4 transfom.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.vstack([ct, -st, np.zeros(n)]).T
  rmat[:, 1, 0:3] = np.vstack([st, ct, np.zeros(n)]).T
  rmat[:, 2, 0:3] = np.array([[0, 0, 1]])

  return rmat.squeeze()


def rmat_to_axisangle(rmat: types.RotationMatrix) -> types.AxisAngleArray:
  """Returns exponential coordinates (w * theta) for the given rotation matrix.

  See Murray1994: A Mathematical Introduction to Robotic Manipulation

  Args:
      rmat: a 3x3 numpy array describing the rotation.

  Returns:
    A 3D numpy unit-vector describing the axis of rotation, scaled by the angle
    required to rotate about this axis to achieve `rmat`.
  """
  theta = np.arccos(
      _clip_within_precision((np.trace(rmat) - 1) / 2, -1., 1.))

  if np.allclose(theta, 0):
    return np.zeros(3)

  w = 1./np.sin(theta) * np.array([
      rmat[2, 1] - rmat[1, 2],
      rmat[0, 2] - rmat[2, 0],
      rmat[1, 0] - rmat[0, 1]])

  wnorm = np.linalg.norm(w)
  if np.allclose(wnorm, 0.):
    # rotation matrix is symmetric, fall back to eigen-decomposition
    w, v = np.linalg.eig(rmat)
    i = np.where(abs(np.real(w) - 1.0) < _TOL)[0][0]  # index of eigenvalue=1
    return np.real(v[:, i]) * theta

  else:
    wnormed = w / np.linalg.norm(w)
    return wnormed * theta


def rmat_to_euler(rmat: types.RotationMatrix,
                  ordering: str = 'ZXZ') -> types.SomeEulerArray:
  """Returns the euler angles corresponding to the provided rotation matrix.

  Args:
    rmat: The rotation matrix.
    ordering: (str) Desired euler angle ordering.

  Returns:
    Euler angles corresponding to the provided rotation matrix.
  """
  return _eulermap[ordering](rmat)


def rmat_to_rot6(rmat: types.RotationMatrix) -> np.ndarray:
  """Projects rotation matrix to 6-dim "Gram-Schmidt-able" representation.

  The "rot6" representation is a 6-DOF representation of an orientation that is
  homeomorphic with SO(3). It is not minimal like an euler or axis-angle, but
  it is smooth over the full range of rotations, unlike eulers, quaternions, and
  axis-angle representations. See the original paper for details:
    "On the Continuity of Rotation Representations in Neural Networks"
    https://arxiv.org/pdf/1812.07035.pdf

  Args:
    rmat: A 3x3 rotation matrix, or larger rank-2 matrix containing a 3x3
      rotation matrix in the leading 3-dimensions (e.g. a homogeneous 4x4).

  Returns:
    A 6-dimensional array containing the first two columns of `rmat`.  This
    representation can be mapped back to `rmat` using `rot6_to_rmat`.
  """
  return rmat[:3, :2].T.flatten()  # concatenate the first 2 columns of `rmat`.


# LINT.IfChange
def mat_to_quat(
    mat: Union[types.RotationMatrix,
               types.HomogeneousMatrix]) -> types.QuatArray:
  """Return quaternion from homogeneous or rotation matrix.

  Args:
    mat: A homogeneous transform or rotation matrix

  Returns:
    A quaternion [w, i, j, k].
  """
  if mat.shape == (3, 3):
    tmp = np.eye(4)
    tmp[0:3, 0:3] = mat
    mat = tmp

  q = np.empty((4,), dtype=np.float64)
  t = np.trace(mat)
  if t > mat[3, 3]:
    q[0] = t
    q[3] = mat[1, 0] - mat[0, 1]
    q[2] = mat[0, 2] - mat[2, 0]
    q[1] = mat[2, 1] - mat[1, 2]
  else:
    i, j, k = 0, 1, 2
    if mat[1, 1] > mat[0, 0]:
      i, j, k = 1, 2, 0
    if mat[2, 2] > mat[i, i]:
      i, j, k = 2, 0, 1
    t = mat[i, i] - (mat[j, j] + mat[k, k]) + mat[3, 3]
    q[i + 1] = t
    q[j + 1] = mat[i, j] + mat[j, i]
    q[k + 1] = mat[k, i] + mat[i, k]
    q[0] = mat[k, j] - mat[j, k]
  q *= 0.5 / np.sqrt(t * mat[3, 3])
  return q
# LINT.ThenChange(_transformations_quat.py)


# LINT.IfChange
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
# LINT.ThenChange(_transformations_quat.py)


def rot6_to_rmat(rot6: np.ndarray) -> np.ndarray:
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
    A rotation matrix obtained by normalizing and orthogonalizing the contents
    of `rot6`.
  """
  xu = rot6[0:3]
  yu = rot6[3:6]

  tol = 1e-6  # Tolerace below which the rot6 is replaced by a canonical basis.
  eps = 1e-5  # Safety factor to avoid zero case.

  def safe_interp(v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe interpolation between input vector and a default basis."""
    # This function interpolates `v` and `b` a function of the norm of `v`.
    # The interpolation has a critical value in which the interpolation can be
    # zero if `v = -lambda * b`. We handle this by "jumping" over this value
    # for `alpha` with safety-factor `epsilon`. Achieved by defining a function
    # for alpha which grows from 0 to `crit - epsilon` over the range [0, tol].
    norm = np.linalg.norm(v)

    # Critical value for `v = -lambda * b` case, considering only projection
    # along `b` (if there are off-axis components then the is no failure mode).
    crit = (1 - eps) / (1 - v @ b)

    if norm < tol:
      alpha = crit * norm / tol
    else:
      alpha = 1.

    return alpha * v + (1 - alpha) * b

  # Interpolate `xu` and `yu` if they're close to zero.
  xu = safe_interp(xu, np.array([1, 0, 0], dtype=xu.dtype))
  yu = safe_interp(yu, np.array([0, 1, 0], dtype=yu.dtype))

  # If xu and yu are parallel, add arbitrary offset to allow orthogonalization.
  if np.allclose(np.cross(xu, yu), 0., atol=tol):
    yu = np.array([0, 1, 0], dtype=yu.dtype) + yu

  # Rotation matrix obtained by orthogonalizing and normalizing.
  xn = xu / np.linalg.norm(xu)
  zu = np.cross(xn, yu)
  zn = zu / np.linalg.norm(zu)
  yn = np.cross(zn, xn)

  return np.stack([xn, yn, zn], axis=1)


def hmat_inv(hmat: types.HomogeneousMatrix) -> types.HomogeneousMatrix:
  """Numerically stable inverse of homogeneous transform."""
  rot = hmat[0:3, 0:3]
  pos = hmat[0:3, 3]
  hinv = np.eye(4)
  hinv[0:3, 3] = rot.T.dot(-pos)
  hinv[0:3, 0:3] = rot.T
  return hinv


def hmat_to_poseuler(ht: types.HomogeneousMatrix, ordering: str) -> np.ndarray:
  """Returns a configuration vector for the given homogeneous transform.

  Args:
    ht: The homogeneous transform.
    ordering: Desired euler angle ordering.

  Returns:
    A 6x1 configuration vector containing the x,y,z position and r1, r2, r3
    euler-angles from the provided homogeneous transform ht, following the given
    rotation ordering.
  """
  return np.hstack([ht[0:3, 3], rmat_to_euler(ht[0:3, 0:3], ordering)])


def hmat_to_twist(ht: types.HomogeneousMatrix) -> np.ndarray:
  """Returns the exponential coordinates for the homogeneous transform H.

  See Murray1994: A Mathematical Introduction to Robotic Manipulation
  Lynch & Park 2017: Modern Robotics: Mechanics, Planning, and Control

  Args:
      ht: A 4x4 numpy array containing a homogeneous transform.

  Returns:
    A 6-vector representing the instantaneous velocity and normalized axis of
    rotation, scaled by the magnitude of the twist.  Intuitively, if this twist
    is integrated for unit time (by `twist_to_hmat`) it will recover `ht`.
  """
  r = ht[0:3, 0:3]
  p = ht[0:3, 3]

  if np.allclose(r, np.eye(3), atol=1e-6):
    wn = np.zeros(3)
    v = p
    theta = 1.
  else:
    w = rmat_to_axisangle(r)
    theta = np.linalg.norm(w)
    wn = w/theta
    s = cross_mat_from_vec3(wn)
    wn2d = np.atleast_2d(wn).T
    a = (np.eye(3) - r).dot(s) + wn2d.dot(wn2d.T) * theta
    v = np.linalg.pinv(a).dot(p)

  xi = np.hstack([v, wn])
  return xi * theta


def pos_to_hmat(pos: types.PositionArray) -> types.HomogeneousMatrix:
  """Returns homogeneous translation matrix for the given position.

  Args:
    pos: 1-dim position vector, or 2-dim tensor of positions with batch in
      leading dimension.
  """
  pos = np.atleast_2d(pos)
  hmat = np.zeros((pos.shape[0], 4, 4))
  hmat[:, np.arange(4), np.arange(4)] = 1
  hmat[:, 0:3, 3] = pos
  return hmat.squeeze()


def rmat_to_hmat(rmat: types.RotationMatrix) -> types.HomogeneousMatrix:
  """Returns homogeneous translation matrix for the given rotation matrix.

  Args:
    rmat: 2-dim rotation matrix, or 3-dim tensor of matrices with batch in
      leading dimension.
  """
  if rmat.ndim == 2:
    rmat = np.expand_dims(rmat, 0)
  hmat = np.zeros((rmat.shape[0], 4, 4))
  hmat[:, :3, :3] = rmat
  hmat[:, 3, 3] = 1.
  return hmat.squeeze()


def poseuler_to_hmat(pe: np.ndarray, ordering: str) -> types.HomogeneousMatrix:
  """Returns a 4x4 Homogeneous transform for the given configuration.

  Args:
    pe: position (x, y, z) and euler angles (r1, r2, r3) following the
      order specified in ordering (e.g. "XYZ").
    ordering: The ordering of euler angles in the configuration array.

  Returns:
    A 4x4 Homogenous transform as a numpy array.
  """

  pos = pe[0:3]
  euler_vec = pe[3:]
  hmat = euler_to_rmat(euler_vec, ordering, full=True)
  hmat[0:3, 3] = pos
  return hmat


def velocity_transform(ht: types.HomogeneousMatrix,
                       vel: Optional[types.ArrayLike] = None) -> np.ndarray:
  """Returns a 6x6 matrix for mapping velocities to the defined frame.

  If R is the rotation part of ht, and p the translation, and v the linear
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
    ht: A transform to the frame the target frame.
    vel: If provided, return the transformed velocity, else the full 6x6
      transform.

  Returns:
    A 6x6 matrix for mapping velocities, as 6d twists (vx,vy,vz,wx,wy,wz) to the
    frame defined in the homogeneous transform ht.
  """

  r = ht[0:3, 0:3]
  p = ht[0:3, 3]
  pcross = cross_mat_from_vec3(p)

  tv = np.vstack([np.hstack([r, pcross.dot(r)]),
                  np.hstack([np.zeros((3, 3)), r])])

  if vel is None:
    return tv
  else:
    return tv.dot(vel)


def twist_to_hmat(xi: np.ndarray) -> types.HomogeneousMatrix:
  """Returns homogeneous transform from exponential coordinates xi=[w, v],theta.

  The magnitude of the angle is encoded in the length of w if w is nonzero, else
  in the magnitude of v.
  See Murray 1994: A Mathematical Introduction to Robotic Manipulation or
  Lynch & Park 2017: Modern Robotics: Mechanics, Planning, and Control

  Args:
      xi: A 6-vector containing:
          v - 3-vector representing the instantaneous velocity.
          w - 3-vector representing the axis of rotation.
          Scaled by the magnitude of the rotation.

  Returns:
      H: A 4x4 numpy array containing a homogeneous transform.
  """
  v = xi[0:3]
  w = xi[3:6]
  ht = np.eye(4)

  if np.allclose(w, 0):
    r = np.eye(3)
    p = v  # assume already scaled by theta
  else:
    w = xi[3:6]
    theta = np.linalg.norm(w)
    wn = w/theta
    vn = v/theta
    s = cross_mat_from_vec3(wn)
    r = np.eye(3) + s * np.sin(theta) + s.dot(s) * (1-np.cos(theta))
    p = (np.eye(3) - r).dot(s.dot(vn)) + wn * (wn.T.dot(vn)) * theta

  ht[0:3, 0:3] = r
  ht[0:3, 3] = p
  return ht


#########################
# Control-Support Utils #
#########################


def force_transform(ht: types.HomogeneousMatrix,
                    wrench: Optional[types.ArrayLike] = None) -> np.ndarray:
  """Returns a 6x6 matrix for mapping forces as 6D wrenches.

  If R is the rotation part of H, and p the translation, and f the linear
  component of the wrench and t the angular, this function computes the
  following matrix operator:
  [R,     0][f]
  [(p+)R, R][t]
  Where x is cross-product, and p+ is the 3x3 cross-product operator for
  the 3-vector p.

  Args:
    ht: A transform to the frame the target frame.
    wrench: If provided, return the transformed wrench, else the full 6x6
        transform.

  Returns:
    A 6x6 transform matrix.
  """
  r = ht[0:3, 0:3]
  p = ht[0:3, 3]
  pcross = cross_mat_from_vec3(p)

  tw = np.vstack([np.hstack([r, np.zeros((3, 3))]),
                  np.hstack([pcross.dot(r), r])])

  if wrench is None:
    return tw
  else:
    return tw.dot(wrench)


def rotate_vec6(mat: Union[types.RotationMatrix, types.HomogeneousMatrix],
                vec6: np.ndarray) -> np.ndarray:
  """Returns a rotated 6-vector based on rotation component of mat.

  Args:
    mat: A homogeneous transform or rotation matrix.
    vec6: A 6-vector to rotate, e.g. twist, wrench, accel, etc.
  """
  rmat = mat[0:3, 0:3]
  rvec = np.zeros(6)
  rvec[0:3] = rmat.dot(vec6[0:3])
  rvec[3:6] = rmat.dot(vec6[3:6])
  return rvec


def integrate_hmat(
    hmat: types.HomogeneousMatrix,
    twist: types.Twist,
    rotate_twist_to_hmat: bool = True) -> types.HomogeneousMatrix:
  """Integrates hmat by the given twist.

  This function is useful for driving a position reference around using a
  velocity signal, e.g. a spacenav or joystick.

  Args:
    hmat: The homogeneous transform to integrate.
    twist: A 6-dof twist containing the linear velocity and angular velocity
      axis.  If the angular velocity is nonzero, the norm of this axis is
      interpreted as the angle to integrate both components over.  Otherwise
      the magnitude of the linear component is used.  See `twist_to_hmat`.
    rotate_twist_to_hmat: If True, rotate twist into the hmat frame (assumes
      twist is defined in the same frame as hmat, e.g. world).  Else interpret
      twist as local to hmat.

  Returns:
    hmat_new: The resulting transform.
  """
  if rotate_twist_to_hmat:
    twist_local = rotate_vec6(hmat.T, twist)
  else:
    twist_local = twist
  hmat_delta = twist_to_hmat(twist_local)
  return hmat.dot(hmat_delta)


################
# 2D Functions #
################


def postheta_to_matrix_2d(pose: np.ndarray) -> types.HomogeneousMatrix2d:
  """Converts 2D pose vector (x, y, theta) to 2D homogeneous transform matrix.

  Args:
    pose: (np.array) Pose vector with x,y,theta elements.

  Returns:
    A 3x3 transform matrix.
  """
  ct = np.cos(pose[2])
  st = np.sin(pose[2])
  return np.array([
      [ct, -st, pose[0]],
      [st, ct, pose[1]],
      [0., 0., 1.]
  ])


def matrix_to_postheta_2d(mat: types.HomogeneousMatrix2d) -> np.ndarray:
  """Converts 2D homogeneous transform matrix to a 2D pose vector (x, y, theta).

  Args:
    mat: (np.array) 3x3 transform matrix.

  Returns:
    An x,y,theta 2D pose.
  """
  return np.array([mat[0, 2], mat[1, 2], np.arctan2(mat[1, 0], mat[0, 0])])


def rotation_matrix_2d(theta: float) -> types.RotationMatrix2d:
  ct = np.cos(theta)
  st = np.sin(theta)
  return np.array([
      [ct, -st],
      [st, ct]
  ])


def cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Performs a 2D cross product, returning a scalar or vector as appropriate.

  Two vectors -> scalar
  Vector and scalar -> vector
  Scalar and vector -> vector

  Args:
    a: first argument (scalar or vector).
    b: second argument (scalar or vector).

  Returns:
    A 2D cross product of the two given vectors.

  Raises:
    Exception: if the vector lengths are incompatible.
  """
  va, vb = np.atleast_1d(a), np.atleast_1d(b)
  l1, l2 = len(va), len(vb)

  if l1 == 2 and l2 == 2:
    # Perform the cross product on two vectors.
    # In 2D this produces a scalar.
    return va[0] * vb[1] - va[1] * vb[0]
  elif l1 == 2 and l2 == 1:
    # Perform the cross product on a vector and a scalar.
    # In 2D this produces a vector.
    return np.array([vb[0] * va[1], -vb[0] * va[0]])
  elif l1 == 1 and l2 == 2:
    # Perform the cross product on a scalar and a vector.
    # In 2D this produces a vector.
    return np.array([-va[0] * vb[1], va[0] * vb[0]])
  else:
    raise Exception('Unsupported argument vector lengths')  # pylint: disable=broad-exception-raised


def velocity_transform_2d(ht: types.HomogeneousMatrix2d,
                          vel: Optional[types.ArrayLike] = None) -> np.ndarray:
  """Returns a matrix for mapping 2D velocities.

  This is a 2-dimensional version of velocity_transform which expects a numpy
  homogeneous transform and a numpy velocity array

  Args:
    ht: A 3x3 numpy homogeneous transform to the target frame.
    vel: A numpy velocity vector (3x1 mini-twist).  If provided, return the
      transformed velocity.  Else the full 3x3 transform operator which can be
      used to transform velocities.
  """
  r = ht[0:2, 0:2]
  p = ht[0:2, 2]

  # linear part is two columns of rotation and a column of cross product
  tv = np.hstack([r, np.array([[p[1], -p[0]]]).T])

  # angular part is identity b/c angular vel not affect by in-plane transform
  tv = np.vstack([tv, np.array([0., 0., 1.])])

  if vel is None:
    return tv
  else:
    return tv.dot(vel)


def force_transform_2d(ht: types.HomogeneousMatrix2d,
                       force_torque: Optional[types.ArrayLike] = None):
  """Returns a matrix for mapping 2D forces.

  This is a 2-dimensional version of force_transform which expects a numpy
  homogeneous transform and a numpy force-torque array

  Args:
    ht: A 3x3 numpy homogeneous transform to the target frame.
    force_torque: A numpy force-torque vector (3x1).

  Returns:
    A 3x3 transform matrix.
  """
  # extract position and cos(theta) and sin(theta) from transform
  x, y, ct, st = ht[0, 2], ht[1, 2], ht[0, 0], ht[1, 0]

  # linear part is two columns of rotation and nothing ()
  tv = np.hstack([ht[0:2, 0:2], np.zeros((2, 1))])

  # angular part needs to compute: [-y, x].T * R * ft_xy + ft_theta
  # i.e. angular part is two columns of cross product and one of identity
  tv = np.vstack([tv, np.array([[x*st - y*ct, x*ct + y*st, 1]])])

  if force_torque is None:
    return tv
  else:
    return tv.dot(force_torque)
