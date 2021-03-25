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

# python3
"""Tests for dm_robotics.agentflow.logging.utils."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.agentflow.loggers import utils
import numpy as np


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple', [0., 1., 2.], [1., 1., 0.], 0.9),
      ('random', np.random.rand(3), np.random.rand(3), np.random.rand(1)),
  )
  def test_compute_return(self, rewards, discounts, additional_discount):
    actual_return = utils.compute_return(
        rewards,
        np.asarray(discounts) * additional_discount)
    expected_return = (
        rewards[0] + rewards[1] * discounts[0] * additional_discount +
        rewards[2] * discounts[0] * discounts[1] * additional_discount**2)
    np.testing.assert_almost_equal(actual_return, expected_return)


if __name__ == '__main__':
  absltest.main()
