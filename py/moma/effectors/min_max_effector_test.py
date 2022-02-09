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

"""Tests for min_max_effector."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.moma.effectors import min_max_effector
from dm_robotics.moma.effectors import test_utils
import numpy as np


class MinMaxEffectorTest(parameterized.TestCase):

  def test_min_max_effector_sends_correct_command(self):
    spy_effector = test_utils.SpyEffector(dofs=3)
    min_action = np.array([-0.9, -0.5, -0.2])
    max_action = np.array([0.2, 0.5, 0.8])
    test_effector = min_max_effector.MinMaxEffector(
        base_effector=spy_effector,
        min_action=min_action,
        max_action=max_action)

    # Ensure that the effector correctly transforms the input command.
    sent_command = np.array([-0.8, 0., 0.3])
    expected_command = np.array([-0.9, -0.5, 0.8])
    test_effector.set_control(None, sent_command)
    np.testing.assert_allclose(expected_command, spy_effector.previous_action)

  def test_default_spec_min_max_effector_sends_correct_command(self):
    spy_effector = test_utils.SpyEffector(dofs=3)
    test_effector = min_max_effector.MinMaxEffector(base_effector=spy_effector)

    # Ensure that the effector correctly transforms the input command.
    sent_command = np.array([-0.3, 0., 0.6])
    expected_command = np.array([-1., -1., 1.])
    test_effector.set_control(None, sent_command)
    np.testing.assert_allclose(expected_command, spy_effector.previous_action)

  @parameterized.named_parameters(
      ('min_action_wrong', np.array([1., 2.]), np.array([4., 5., 6.])),
      ('max_action_wrong', np.array([1., 2., 3.]), np.array([4., 5.])),)
  def test_raises_if_wrong_shaped_action_is_passed(
      self, min_action, max_action):
    spy_effector = test_utils.SpyEffector(dofs=3)
    test_effector = min_max_effector.MinMaxEffector(
        base_effector=spy_effector,
        min_action=min_action,
        max_action=max_action)

    # Assert the effector raises an error.
    with self.assertRaises(ValueError):
      test_effector.action_spec(None)

if __name__ == '__main__':
  absltest.main()
