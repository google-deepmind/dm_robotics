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
"""Tests for PoseDistribution implementations."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.geometry import pose_distribution
from dm_robotics.transformations import transformations as tr
import numpy as np


class PoseDistributionTest(parameterized.TestCase):

  def testLookAtPoseDistribution(self):

    target_dist = mock.MagicMock(spec=pose_distribution.UniformDistribution)
    source_dist = mock.MagicMock(spec=pose_distribution.UniformDistribution)
    target_dist.sample.return_value = np.array([0.1, 0.5, 0.3])
    source_dist.sample.return_value = np.array([0.4, 0.2, 0.6])
    target_dist.mean.return_value = np.array([0.1, 0.2, 0.3]) * -1
    source_dist.mean.return_value = np.array([0.3, 0.2, 0.1]) * -1

    xnormal = np.array([0, 1, 0])

    look_at_dist = pose_distribution.LookAtPoseDistribution(
        target_dist, source_dist, xnormal)

    # Test `sample_pose`
    actual_pos, actual_quat = look_at_dist.sample_pose(None)
    expected_pos, expected_quat = pose_distribution._points_to_pose(
        target_dist.sample(), source_dist.sample(), xnormal)

    np.testing.assert_allclose(actual_pos, expected_pos)
    self.assertTrue(
        np.allclose(actual_quat, expected_quat) or
        np.allclose(actual_quat, expected_quat * -1))

    # Test `mean_pose`
    actual_pos, actual_quat = look_at_dist.mean_pose(None)
    expected_pos, expected_quat = pose_distribution._points_to_pose(
        target_dist.mean(), source_dist.mean(), xnormal)

    np.testing.assert_allclose(actual_pos, expected_pos)
    self.assertTrue(
        np.allclose(actual_quat, expected_quat) or
        np.allclose(actual_quat, expected_quat * -1))

  def testTruncatedNormalPoseDistribution(self):
    """Test normal pose with limits."""
    random_state = np.random.RandomState(1)

    def _check_limits(mean_poseuler, pos_sd, rot_sd, pos_clip_sd, rot_clip_sd):
      pose_dist = pose_distribution.truncated_normal_pose_distribution(
          mean_poseuler, pos_sd, rot_sd, pos_clip_sd, rot_clip_sd)
      pos, quat = pose_dist.sample_pose(random_state)

      # Check that position and axis-angle don't exceed clip_sd
      # Obtain the orientation relative to the mean
      mean_quat = tr.euler_to_quat(mean_poseuler[3:])
      quat_mean_inv = tr.quat_conj(mean_quat)
      quat_samp = tr.quat_mul(quat_mean_inv, quat)
      # Convert to axisangle and compare to threshold.
      axisangle_samp = tr.quat_to_axisangle(quat_samp)

      self.assertTrue(
          np.all(np.logical_or(pos > -pos_clip_sd, pos < pos_clip_sd)))
      self.assertTrue(
          np.all(
              np.logical_or(axisangle_samp > -rot_clip_sd,
                            axisangle_samp < rot_clip_sd)))

    # Check special cases
    _check_limits(
        mean_poseuler=np.array([0.1, 0.2, 0.3, 0, 0, 0]),
        pos_sd=np.array([0.3, 0.2, 0.1]),
        rot_sd=np.array([0.3, 0.0, 0.0]),
        pos_clip_sd=2.,
        rot_clip_sd=1.)

    # Check a bunch of random inputs
    for _ in range(100):
      mean_poseuler = random_state.uniform([-1, -2, -3, -np.pi, -np.pi, -np.pi],
                                           [1, 2, 3, np.pi, np.pi, np.pi])
      pos_sd = random_state.uniform([0, 0, 0], [1, 2, 3])
      rot_sd = random_state.uniform([0, 0, 0], [1, 2, 3])
      pos_clip_sd = random_state.uniform(0, 10)
      rot_clip_sd = random_state.uniform(0, 10)
      _check_limits(mean_poseuler, pos_sd, rot_sd, pos_clip_sd, rot_clip_sd)

      # Check that pos and axis only vary along non-zero sd dims
      pos_sd = np.array([0.0, 0.2, 0.0])
      rot_sd = np.array([0.1, 0.0, 0.3])
      pose_dist = pose_distribution.truncated_normal_pose_distribution(
          mean_pose=np.array([0., 0., 0., 0., 0., 0.]),
          pos_sd=pos_sd,
          rot_sd=rot_sd,
          pos_clip_sd=2.,
          rot_clip_sd=1.)
      pos, quat = pose_dist.sample_pose(random_state)
      axisangle_samp = tr.quat_to_axisangle(quat)

      self.assertTrue(np.all(np.nonzero(pos)[0] == np.nonzero(pos_sd)[0]))
      self.assertTrue(
          np.all(np.nonzero(axisangle_samp)[0] == np.nonzero(rot_sd)[0]))

  def testWeightedDiscretePoseDistribution(self):
    """Test weighted samples of a discrete set of poses."""
    rs = np.random.RandomState(0)
    base_dist = pose_distribution.UniformPoseDistribution(
        np.ones(6) * -1, np.ones(6) * 1
    )
    poses = [base_dist.sample_pose(rs) for _ in range(3)]

    dist = pose_distribution.WeightedDiscretePoseDistribution(poses, [1, 10, 0])
    sampled_poses = [dist.sample_pose(rs) for _ in range(100)]

    is_close = lambda a, b: np.allclose(np.concatenate(a), np.concatenate(b))
    num_times_sampled = [
        np.sum([is_close(o, p) for p in sampled_poses]) for o in poses
    ]

    # Poses 0 and 1 should be sampled with nonzero probability.
    self.assertGreater(num_times_sampled[0], 0)
    self.assertGreater(num_times_sampled[1], 0)

    # Poses 2 should never be sampled.
    self.assertEqual(num_times_sampled[2], 0)

    # The second pose should be higher probability than the first.
    self.assertGreater(num_times_sampled[1], num_times_sampled[0])


if __name__ == '__main__':
  absltest.main()
