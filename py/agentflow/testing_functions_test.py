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
"""Tests SpyEnvironment."""

from absl.testing import absltest
import dm_env
from dm_robotics.agentflow import testing_functions
import numpy as np


def action(spec, value):
  return np.full(shape=spec.shape, fill_value=value, dtype=spec.dtype)


class SpyEnvironmentTest(absltest.TestCase):

  def testSomeSteps(self):
    env = testing_functions.SpyEnvironment()
    ts1 = env.reset()
    action1 = action(env.action_spec(), 1.0)
    ts2 = env.step(action1)
    action2 = action(env.action_spec(), 2.0)
    ts3 = env.step(action2)

    # Check step_count.
    self.assertEqual(env.get_step_count(ts1), 0)
    self.assertEqual(env.get_step_count(ts2), 1)
    self.assertEqual(env.get_step_count(ts3), 2)

    # Check last_action.
    np.testing.assert_array_almost_equal(env.get_last_action(ts2), action1)
    np.testing.assert_array_almost_equal(env.get_last_action(ts3), action2)

    # Check the step types.
    self.assertIs(ts1.step_type, dm_env.StepType.FIRST)
    self.assertIs(ts2.step_type, dm_env.StepType.MID)
    self.assertIs(ts3.step_type, dm_env.StepType.MID)

  def testReset(self):
    env = testing_functions.SpyEnvironment(episode_length=3)

    action1 = action(env.action_spec(), 1)
    action2 = action(env.action_spec(), 2)
    action3 = action(env.action_spec(), 3)
    action4 = action(env.action_spec(), 4)
    action5 = action(env.action_spec(), 5)

    ts1 = env.step(action1)  # Should reset, ignoring action.
    ts2 = env.step(action2)  # Step 1 of 3
    ts3 = env.step(action3)  # Step 2 of 3
    ts4 = env.step(action4)  # Step 3 of 3
    ts5 = env.reset()
    ts6 = env.step(action5)

    # Check step types.
    self.assertIs(ts1.step_type, dm_env.StepType.FIRST)
    self.assertIs(ts2.step_type, dm_env.StepType.MID)
    self.assertIs(ts3.step_type, dm_env.StepType.MID)
    self.assertIs(ts4.step_type, dm_env.StepType.LAST)
    self.assertIs(ts5.step_type, dm_env.StepType.FIRST)
    self.assertIs(ts6.step_type, dm_env.StepType.MID)

  def testImplicitResets(self):
    env = testing_functions.SpyEnvironment(episode_length=2)

    action1 = action(env.action_spec(), 1)
    action2 = action(env.action_spec(), 2)
    action3 = action(env.action_spec(), 3)
    action4 = action(env.action_spec(), 4)
    action5 = action(env.action_spec(), 5)
    action6 = action(env.action_spec(), 6)

    ts1 = env.step(action1)  # Should reset, ignoring action.
    ts2 = env.step(action2)  # Step 1 of 2
    ts3 = env.step(action3)  # Step 2 of 2
    ts4 = env.step(action4)  # Should reset.
    ts5 = env.step(action5)  # Step 1 of 2
    ts6 = env.step(action6)  # Step 2 of 2

    # Check step types.
    self.assertIs(ts1.step_type, dm_env.StepType.FIRST)
    self.assertIs(ts2.step_type, dm_env.StepType.MID)
    self.assertIs(ts3.step_type, dm_env.StepType.LAST)
    self.assertIs(ts4.step_type, dm_env.StepType.FIRST)
    self.assertIs(ts5.step_type, dm_env.StepType.MID)
    self.assertIs(ts6.step_type, dm_env.StepType.LAST)

    # Check in-episode step count.
    self.assertEqual(env.get_step_count(ts1), 0)
    self.assertEqual(env.get_step_count(ts2), 1)
    self.assertEqual(env.get_step_count(ts3), 2)
    self.assertEqual(env.get_step_count(ts4), 0)
    self.assertEqual(env.get_step_count(ts5), 1)
    self.assertEqual(env.get_step_count(ts6), 2)

    # Check global step count.
    self.assertEqual(env.get_global_step_count(ts1), 0)
    self.assertEqual(env.get_global_step_count(ts2), 1)
    self.assertEqual(env.get_global_step_count(ts3), 2)
    self.assertEqual(env.get_global_step_count(ts4), 3)
    self.assertEqual(env.get_global_step_count(ts5), 4)
    self.assertEqual(env.get_global_step_count(ts6), 5)


if __name__ == '__main__':
  absltest.main()
