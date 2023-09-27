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

"""Tests for the sawyer_constants files."""
from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.moma.models.robots.robot_arms import sawyer_constants
import numpy as np


@parameterized.named_parameters(
    {
        'testcase_name': 'test_function_from_pinch_pose_to_tcp_pose',
        'sawyer_pinch_pose': np.array([[
            5.8391762e-01,
            7.3800012e-02,
            1.4042723e-01,
            2.6832174e-03,
            4.0328479e-01,
            9.1507059e-01,
            1.8798358e-04,
        ]]),
        'sawyer_tcp_pose': np.array([[
            5.8289909e-01,
            7.4166231e-02,
            3.4162432e-01,
            2.6832174e-03,
            4.0328479e-01,
            9.1507059e-01,
            1.8798358e-04,
        ]]),
    },
)
class SawyerTest(parameterized.TestCase):

  def test_function_from_pinch_pose_to_tcp_pose(
      self, sawyer_pinch_pose, sawyer_tcp_pose
  ):
    """Check function tcp_pose_from_pinch_pose."""
    expected_tcp_pose = sawyer_constants.tcp_pose_from_pinch_pose(
        sawyer_pinch_pose
    )
    self.assertTrue(np.allclose(sawyer_tcp_pose, expected_tcp_pose))


if __name__ == '__main__':
  absltest.main()
