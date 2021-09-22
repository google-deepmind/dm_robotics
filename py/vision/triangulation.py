# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module performing triangulation of 2D points from multiple views."""

from typing import Optional, Sequence, Tuple

import cv2
from dmr_vision import types
import numpy as np
from tf import transformations as tf


def _plane_basis_from_normal(normal: np.ndarray):
  """Returns a 3x2 matrix with orthonormal columns tangent to normal."""
  norm = np.linalg.norm(normal)
  if norm < 1e-6:
    raise ValueError("The norm of the normal vector is less than 1e-6. "
                     "Consider rescaling it or passing a unit vector.")
  normal = normal / norm
  # Choose some vector which is guaranteed not to be collinear with the normal.
  if normal[0] > 0.8:
    tangent_0 = np.array([0., 1., 0.])
  else:
    tangent_0 = np.array([1., 0., 0.])
  # Make it orthonormal to normal.
  tangent_0 = tangent_0 - normal * np.dot(tangent_0, normal)
  tangent_1 = np.cross(normal, tangent_0)
  return np.stack([tangent_0, tangent_1], axis=1)


class Triangulation:
  """Triangulates points in 3D space from multiple 2D image observations.

  We assume that at least two observations from two different view points
  corresponding to the same 3D point are available. This triangulation will
  work both with fixed or moving cameras, as long as the camera pose at
  measurement time is known.

  In its current implementation, triangulation will not work for points at
  or near infinity.
  """

  def __init__(self, camera_matrices: Sequence[np.ndarray],
               distortions: Optional[Sequence[np.ndarray]],
               extrinsics: Sequence[np.ndarray],
               planar_constraint: Optional[types.Plane] = None) -> None:
    """Initializes the class given the scene configuration.

    Args:
      camera_matrices: List of 3x3 camera (projection) matrices. Length of this
        list corresponds to number of measurements. If multiple measurements are
        taken from the same camera, the camera matrix needs to be provided
        multiple times.
      distortions: None or List of distortion parameters corresponding to each
        measurement. If None, measurements are assumed to be taken from
        rectified images. Distortion parameters can be of any type supported by
        OpenCV.
      extrinsics: List of camera extrinsics (i.e. poses) expressed in the
        'world' frame in which we want to estimate the 3D point in. Each pose
        must be provided as a NumPy array of size (7, 1) where the first three
        entries are position and the last four are the quaternion [x, y, z, w].
      planar_constraint: An optional plane (in global frame) that the
        triangulated point must lie on.

    Raises:
      ValueError: Will be raised if dimensions between parameters mismatch.
    """
    self._camera_matrices = camera_matrices
    self._distortions = distortions
    self._extrinsics = extrinsics
    # Optional plane constraint parameterization.
    self._has_planar_constraint = planar_constraint is not None
    self._offset = None
    self._basis = None
    if self._has_planar_constraint:
      self._offset = planar_constraint.point
      self._basis = _plane_basis_from_normal(planar_constraint.normal)

    if len(self._camera_matrices) != len(self._extrinsics):
      raise ValueError("Number of camera matrices and extrinsics should match.")

    if distortions is not None:
      if len(self._camera_matrices) != len(self._extrinsics):
        raise ValueError(
            "Number of camera matrices and distortion parameters should match.")

  def triangulate(
      self,
      pixel_measurements: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Triangulates a 3D point from multiple 2D observations.

    Args:
      pixel_measurements: List of 2D pixel measurements (u, v) corresponding to
        the observation. If distortion parameters were set in the `__init__`
        method, these pixel measurements should be distored as well.

    Returns:
      A NumPy array of size 3 with [x, y, z] coordinates of the triangulated
      point and a NumPy array of size 1 with the residual of the triangulated
      point.

    Raises:
      ValueError: Will be raised if number of configured cameras and
        observations do not match.
    """
    if len(self._camera_matrices) != len(pixel_measurements):
      raise ValueError(
          "Number of camera matrices and measurements should match.")

    undistorted_points = []
    for i in range(len(pixel_measurements)):
      distortion = self._distortions[i] if self._distortions else np.zeros(4)
      # OpenCV expects the following convoluted format.
      pixel_list = np.array([np.array([
          np.array(pixel_measurements[i]),
      ],)])
      undistorted_points.append(
          cv2.undistortPoints(pixel_list, self._camera_matrices[i], distortion))
    return self._linear_triangulate(undistorted_points)

  def _linear_triangulate(
      self,
      undistorted_points: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Implements linear triangulation using least-squares.

    Args:
      undistorted_points: List of 2D image points in undistorted, normalized
        camera coordinates (not pixel coordinates).

    Returns:
      3D position of triangulation as NumPy array of size (3,).
      Residual of the 3D position of triangulation of size (1,) or an empty
      array if the 3D position can be reconstructed exactly from the
      measurements and constraints.

    Raises:
      ValueError: Will be raised if observations do not allow triangulation.
      np.linalg.LinAlgError: Raised if numeric errors occur.

    Implementation according to 5.1 in

    Hartley, Richard I., and Peter Sturm. "Triangulation."
    Computer vision and image understanding 68.2 (1997): 146-157.

    https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf

    While the article above talks about two cameras only, the multi-camera
    equaivalent can be understood very intuitively: The 3D point to triangulate
    needs to lie on the line constructed by the cameras focal point (its
    position) and the detected point on the image plane. This line is given as:
      l = x_c1 + s_1 * v_c1 (1)
    where x_c1 is the position of camera 1 and v_c1 the detected point. Given
    that the 3D point is somewhat along this line at an unknown distance s_1,
    the intersection equation is given as
      p - l = 0 (2)
    where p is the position of the 3D point, we want to triangulate. To carry
    out this computation, we need a coherent reference frame for all quantities.
    Since we want to triangulate our 3D point in a global frame (called W),
    equation (2) becomes:
      W_p - W_l (3)
    The camera position in equation (1) is already expressed in the global
    frame. Hence we only need to rotate v_c1, which is usually expressed in the
    camera frame, into the global frame
      W_v_c1 = R_W_C * C_v_c1 (4)
    where R_W_C is the orientation of the camera.
    Substituting (4) into (3), we get:
      W_p - (W_x_c1 + s_1 * W_v_c1) (5)

    We can then replicate equation (5) for all cameras and form it as a
    linear system of equations in the form of Ax = b where
    A = [I, W_v_c1
         I, W_v_c2
         ...       ]
    x = [W_p_x, W_p_y, W_p_z, s_1, s_2, ...]
    b = [-W_x_c1, -W_x_c2, ...]

    According to the article above, we then solve this system using the SVD
    and discarding the scale.

    If a planar constraint is provided, then we reparameterize
    [W_p_x, W_p_y, W_p_z] using 2d coordinates in this plane, solve
    for these coordinates, and then transform back to 3d.

    This can be improved in two ways:
    - implement the iterative method outlined in 5.2 in the article above
    - implement a Levenberg-Marquardt implementation, also outlined in 5.2

    LM is available, so this is relatively easy to do.
    """

    num_measurements = len(undistorted_points)

    if num_measurements < 2 and not self._has_planar_constraint:
      raise ValueError("We need at least two measurements to triangulate.")

    # For linear algebra as well as expressing reference frames, it really
    # improves readability, if we allow variable names to start with capital
    # letters.
    # pylint: disable=invalid-name
    rows = 3 * num_measurements
    cols = 3 + num_measurements
    A = np.zeros([rows, cols])
    b = np.zeros([rows, 1])

    for i in range(num_measurements):
      # Normalize/convert to 3D.
      C_v = np.append(undistorted_points[i], 1.)

      # Transform measured point to global frame.
      R_W_C = tf.quaternion_matrix(self._extrinsics[i][3:7])[0:3, 0:3]
      W_v = R_W_C.dot(C_v)

      # Fill A and b for the given observation, using the formulas above.
      A[3 * i:3 * (i + 1), 0:3] = np.eye(3)
      A[3 * i:3 * (i + 1), 3 + i] = -W_v
      b[3 * i:3 * (i + 1), 0] = np.array(self._extrinsics[i][0:3]).T

    # Maybe add a planar constraint.
    if self._has_planar_constraint:
      # Reparametrize x = Zx' + y -> AZx' = b - Ay.
      Z = np.zeros([cols, cols - 1])
      y = np.zeros([cols, 1])
      Z[3:, 2:] = np.eye(num_measurements)
      Z[:3, :2] = self._basis
      y[:3, 0] = self._offset
      orig_A = A
      A = np.matmul(orig_A, Z)  # [3 * num_measurments, 2 + num_measurments]
      b = b - np.matmul(orig_A, y)

    # Solve for the 3D point and the scale.
    try:
      x, residual, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as err:
      err.message = """Triangulation failed, possibly due to invalid
        data provided. Numeric error: """ + err.message
      raise
    if self._has_planar_constraint:
      # Map from 2d plane to 3d world coordinates.
      scales = x[2:]
      pose = (np.matmul(self._basis, x[:2]) +
              np.expand_dims(self._offset, axis=-1))
      num_free_variables = 3 + num_measurements - 1
    else:
      scales = x[3:]
      pose = x[:3]
      num_free_variables = 3 + num_measurements

    # Verify the rank to ensure visibility.
    if rank < num_free_variables:
      raise ValueError("Insufficient observations to triangulate.")

    # Verify that scaling factors are positive.
    if (scales < 0).any():
      raise ValueError("3D point lies behind at least one camera")

    return pose, np.sqrt(residual)
    # pylint: enable=invalid-name
