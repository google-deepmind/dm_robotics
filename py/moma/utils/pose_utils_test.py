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

"""Tests for pose_utils."""


from absl.testing import absltest
from dm_robotics.moma.utils import pose_utils
import numpy as np


class PoseUtilsTest(absltest.TestCase):

  def test_positive_leading_quat(self):
    # Should not change a quaternion with a positive leading scalar.
    input_quat = [1., 2., 3., 4.]  # unnormalized, but doesn't matter.
    expected_quat = input_quat
    np.testing.assert_almost_equal(
        pose_utils.positive_leading_quat(np.array(input_quat)),
        expected_quat, decimal=3)
    # But it should change a quaternion with a negative leading scalar.
    input_quat = [-1., 2., 3., 4.]  # unnormalized, but doesn't matter.
    expected_quat = [1., -2., -3., -4.]
    np.testing.assert_almost_equal(
        pose_utils.positive_leading_quat(np.array(input_quat)),
        expected_quat, decimal=3)


if __name__ == '__main__':
  absltest.main()
