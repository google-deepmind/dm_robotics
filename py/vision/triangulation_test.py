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

"""Tests for `triangulation.py`."""

from absl.testing import absltest
import cv2
from dmr_vision import triangulation
import numpy as np
from tf import transformations as tf


class TriangulationTest(absltest.TestCase):

  def setUp(self):
    super(TriangulationTest, self).setUp()

    # Test parameters are taken from random real world camera calibrations.
    self.camera_matrices = [
        np.array([[1418.1731532081515, 0.0, 951.1884329781567],
                  [0.0, 1418.7685215216047, 582.078606694917],
                  [0.0, 0.0, 1.0]]),
        np.array([[1408.1781154186797, 0.0, 969.3781055884897],
                  [0.0, 1409.3923562102611, 600.0994341386963],
                  [0.0, 0.0, 1.0]]),
        np.array([[1401.351249806934, 0.0, 979.9264199776977],
                  [0.0, 1402.777176503527, 615.5376637166237],
                  [0.0, 0.0, 1.0]]),
    ]

    # Test parameters are taken from one real robot Lego cell (cell 1).
    self.extrinsics = [
        (0.978, 0.363, 0.225, 0.332, 0.750, -0.523, -0.231),
        (0.975, -0.362, 0.222, -0.754, -0.305, 0.179, 0.554),
        (0.123, -0.203, 0.324, 0.717, -0.520, 0.230, -0.403),
    ]

    # Taken from random cameras, not corresponding to projection matrices.
    self.distortions = [
        (-0.1716, 0.1010, -0.000725, -0.000551, 0.0),
        (-0.1645, 0.0945, 0.000205, 0.000527, 0.0),
        (-0.1676, 0.0935, -0.000168, -0.000842, 0.0),
    ]

    # Roughly the Lego basket center, expressed in the robot base frame.
    self.point_3d = np.array([0.605, 0.0, 0.05])

    # Theoretic, distorted pixel measurements corresponding to the point and
    # the camera settings above.
    self.pixels = [
        (905.87075049, 541.33127149),
        (837.30620605, 582.11931161),
        (993.52079234, 440.15403317),
    ]

  def test_undistorted(self):
    point_3d_triangulated, residual = self._run_triangulation(
        self.camera_matrices, None, self.extrinsics, self.point_3d)
    self.assertSequenceAlmostEqual(point_3d_triangulated.flatten(),
                                   self.point_3d)
    self.assertAlmostEqual(residual.item(), 0)

  def test_distorted(self):
    point_3d_triangulated, residual = self._run_triangulation(
        self.camera_matrices, self.distortions, self.extrinsics, self.point_3d)
    self.assertSequenceAlmostEqual(point_3d_triangulated.flatten(),
                                   self.point_3d)
    self.assertAlmostEqual(residual.item(), 0)

  def test_from_two_viewpoints(self):
    point_3d_triangulated, residual = self._run_triangulation(
        self.camera_matrices[0:2], self.distortions[0:2], self.extrinsics[0:2],
        self.point_3d)
    self.assertSequenceAlmostEqual(point_3d_triangulated.flatten(),
                                   self.point_3d)
    self.assertAlmostEqual(residual.item(), 0)

  def test_from_same_viewpoints(self):
    camera_matrices = [self.camera_matrices[0], self.camera_matrices[0]]
    distortions = [self.distortions[0], self.distortions[0]]
    extrinsics = [self.extrinsics[0], self.extrinsics[0]]

    with self.assertRaises(ValueError):
      self._run_triangulation(camera_matrices, distortions, extrinsics,
                              self.point_3d)

  def test_single_observation(self):
    with self.assertRaises(ValueError):
      self._run_triangulation(
          [self.camera_matrices[0],], [self.distortions[0],],
          [self.extrinsics[0],], self.point_3d)

  def test_behind_camera(self):
    with self.assertRaises(ValueError):
      self._run_triangulation(
          self.camera_matrices, None, self.extrinsics, np.array([10., 0., 0.]))

  def test_distorted_pixel_measurements(self):
    point_3d_triangulated, residual = self._run_triangulation_from_pixels(
        self.camera_matrices, self.distortions, self.extrinsics, self.pixels)
    self.assertSequenceAlmostEqual(point_3d_triangulated.flatten(),
                                   self.point_3d)
    self.assertAlmostEqual(residual.item(), 0)

  def _run_triangulation(self, camera_matrices, distortions, extrinsics, point):
    triangulator = triangulation.Triangulation(
        camera_matrices, distortions, extrinsics)

    # pylint: disable=invalid-name
    pixel_measurements = []
    for i in range(len(camera_matrices)):
      extrinsics_mat = tf.quaternion_matrix(extrinsics[i][3:7])[0:3, 0:3]
      extrinsics_pos = np.array(extrinsics[i][0:3])

      point_3d_list = np.array([point,])
      # OpenCV expects the camera position expressed in the camera frame.
      W_x_C = extrinsics_mat.T.dot(-extrinsics_pos)
      distortion = distortions[i] if distortions else np.zeros(4)
      point_projected, _ = cv2.projectPoints(
          point_3d_list, extrinsics_mat.T, W_x_C,
          camera_matrices[i], distortion)

      pixel_measurements.append(point_projected[0][0])

    return triangulator.triangulate(pixel_measurements)
    # pylint: enable=invalid-name

  def _run_triangulation_from_pixels(self, camera_matrices, distortions,
                                     extrinsics, pixel_measurements):
    triangulator = triangulation.Triangulation(
        camera_matrices, distortions, extrinsics)

    return triangulator.triangulate(pixel_measurements)


if __name__ == "__main__":
  absltest.main()
