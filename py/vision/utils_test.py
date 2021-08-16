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
"""Tests for `utils.py`."""

from absl.testing import absltest
from dmr_vision import robot_config
from dmr_vision import types
from dmr_vision import utils
import numpy as np


class PoseValidatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    r_config = robot_config.get_robot_config("STANDARD_SAWYER")
    center = np.append(r_config.basket_center, r_config.basket_height)
    self.limits = types.PositionLimit(
        upper=center + np.array([0.45, 0.45, 0.20]),
        lower=center + np.array([-0.45, -0.45, -0.02]),
    )
    self.pose_validator = utils.PoseValidator(self.limits)

  def testIsValid(self):
    eps = np.array([1e-4, 0., 0.])
    pos_slightly_above = self.limits.upper + eps
    self.assertFalse(self.pose_validator.is_valid(pos_slightly_above))
    pos_slightly_below = self.limits.lower - eps
    self.assertFalse(self.pose_validator.is_valid(pos_slightly_below))
    pos_in_limits = self.limits.upper - eps
    self.assertTrue(self.pose_validator.is_valid(pos_in_limits))


if __name__ == "__main__":
  absltest.main()
