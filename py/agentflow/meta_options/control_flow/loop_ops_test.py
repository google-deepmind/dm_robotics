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
"""Tests for dm_robotics.agentflow.meta_options.control_flow.cond."""

from unittest import mock

from absl.testing import absltest
import dm_env
from dm_robotics.agentflow import core
from dm_robotics.agentflow.meta_options.control_flow import loop_ops
import numpy as np


class WhileOptionTest(absltest.TestCase):

  def _timestep(self, first=False, last=False, observation=None):
    step_type = dm_env.StepType.MID
    if first:
      step_type = dm_env.StepType.FIRST
    if last:
      step_type = dm_env.StepType.LAST
    if first and last:
      raise ValueError('FIRST and LAST not allowed')

    reward = 0
    discount = np.random.random()
    observation = observation or {}
    return dm_env.TimeStep(step_type, reward, discount, observation)

  def assert_mock_step_type(self, mock_step, step_type):
    self.assertEqual(mock_step.call_args[0][0].step_type, step_type)

  def assert_delegate_just_restarted(self, option):
    self.assertEqual(option.step.call_args_list[-2][0][0].step_type,
                     dm_env.StepType.LAST)
    self.assertEqual(option.step.call_args_list[-1][0][0].step_type,
                     dm_env.StepType.FIRST)

  def test_while_terminates_if_false(self):
    # Test whether While terminates if cond is False.

    option = mock.MagicMock(spec=core.Option)
    option.result.return_value = core.OptionResult(core.TerminationType.SUCCESS)
    first_timestep = self._timestep(first=True)
    mid_timestep = self._timestep()
    last_timestep = self._timestep(last=True)
    result_success = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS,
        data='discarded result')
    cond = mock.MagicMock()

    cond.return_value = True
    option.pterm.return_value = 0.0

    # If cond is true first step should go through.
    while_option = loop_ops.While(cond, option, eval_every_step=True)
    while_option.on_selected(first_timestep, result_success)
    while_option.step(first_timestep)
    option.step.assert_called_with(first_timestep)
    self.assertEqual(while_option.pterm(first_timestep), 0.0)

    # Regular step should go through.
    while_option.step(mid_timestep)
    option.step.assert_called_with(mid_timestep)

    # If cond goes false should terminate and push last step to delegate.
    cond.return_value = False
    while_option.step(mid_timestep)
    self.assertEqual(while_option.pterm(mid_timestep), 1.0)
    self.assert_mock_step_type(option.step, dm_env.StepType.LAST)

    cond.return_value = True
    while_option.on_selected(first_timestep, result_success)
    while_option.step(last_timestep)
    option.step.assert_called_with(last_timestep)

  def test_while_calls_cond_selectively(self):
    # Test whether While terminates if cond is False.

    option = mock.MagicMock(spec=core.Option)
    first_timestep = self._timestep(first=True)
    mid_timestep = self._timestep()
    result_success = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS,
        data='discarded result')
    cond = mock.MagicMock()

    cond.return_value = True
    option.pterm.return_value = 0.0

    # on_selected shouldn't call the cond.
    while_option_default = loop_ops.While(cond, option, eval_every_step=True)
    while_option_lazy = loop_ops.While(cond, option, eval_every_step=False)
    while_option_default.on_selected(first_timestep, result_success)
    while_option_lazy.on_selected(first_timestep, result_success)
    cond.assert_not_called()

    # Should only call on step if eval_every_step=True
    while_option_lazy.step(first_timestep)
    cond.assert_not_called()
    while_option_default.step(first_timestep)
    cond.assert_called_with(first_timestep)

    # step should always call the cond if delegate terminates.
    cond.reset_mock()
    cond.return_value = True
    option.pterm.return_value = 1.0
    while_option_lazy.step(mid_timestep)
    cond.assert_called_with(mid_timestep)
    cond.reset_mock()
    cond.return_value = True
    while_option_default.step(mid_timestep)
    cond.assert_called_with(mid_timestep)

  def test_while_restarts_delegate(self):
    # Test that While restarts option if cond is True and delegate requests term

    option = mock.MagicMock(spec=core.Option)
    first_timestep = self._timestep(first=True)
    mid_timestep = self._timestep()
    result_success = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS,
        data='discarded result')
    cond = mock.MagicMock()

    cond.return_value = True
    option.pterm.return_value = 0.0
    option.result.return_value = core.OptionResult(core.TerminationType.SUCCESS)

    # If cond is true first step should go through.
    while_option = loop_ops.While(cond, option, eval_every_step=True)
    while_option.on_selected(first_timestep, result_success)
    while_option.step(first_timestep)
    for _ in range(10):
      while_option.step(mid_timestep)
    option.step.assert_called_with(mid_timestep)
    self.assertEqual(option.step.call_count, 11)
    self.assertEqual(option.on_selected.call_count, 1)

    # Terminate delegate and verify it's restarted properly.
    option.pterm.return_value = 1.0
    while_option.step(mid_timestep)
    self.assertEqual(option.on_selected.call_count, 2)
    self.assert_delegate_just_restarted(option)

    # Verify that delegate sees last step if cond flips to false.
    option.reset_mock()
    option.pterm.return_value = 0.0
    option.result.return_value = core.OptionResult(core.TerminationType.SUCCESS)
    cond.return_value = False
    while_option.step(mid_timestep)
    self.assert_mock_step_type(option.step, dm_env.StepType.LAST)

  def test_for(self):
    option = mock.MagicMock(spec=core.Option)
    first_timestep = self._timestep(first=True)
    mid_timestep = self._timestep()
    result_success = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS,
        data='discarded result')
    cond = mock.MagicMock()

    cond.return_value = True
    option.pterm.return_value = 0.0

    # Run first iteration of delegate.
    for_option = loop_ops.Repeat(3, option)
    for_option.on_selected(first_timestep, result_success)
    for_option.step(first_timestep)
    for _ in range(10):
      for_option.step(mid_timestep)
    self.assertEqual(for_option.delegate_episode_ctr, 0)

    # Verify it resets and increments if delegate terminates.
    option.pterm.return_value = 1.0
    for_option.step(mid_timestep)
    self.assert_delegate_just_restarted(option)
    self.assertEqual(for_option.delegate_episode_ctr, 1)

    # Verify it doesn't increment while stepping delegate on 2nd iteration.
    option.pterm.return_value = 0.0
    for _ in range(10):
      for_option.step(mid_timestep)
      self.assertEqual(for_option.delegate_episode_ctr, 1)

    # Verify it resets and increments if delegate terminates.
    option.pterm.return_value = 1.0
    for_option.step(mid_timestep)
    self.assert_delegate_just_restarted(option)
    self.assertEqual(for_option.delegate_episode_ctr, 2)

    # Verify we can run 3rd iteration.
    option.pterm.return_value = 0.0
    for _ in range(10):
      for_option.step(mid_timestep)
      self.assertEqual(for_option.delegate_episode_ctr, 2)
      self.assertEqual(for_option.pterm(mid_timestep), 0.0)

    # Verify that loop terminates when this iteration ends.
    option.pterm.return_value = 1.0  # delegate asks for termination
    for_option.step(mid_timestep)  # delegate gets terminated

    # Assert delegate is terminated without restarting.
    self.assert_mock_step_type(option.step, dm_env.StepType.LAST)

    # Assert loop requests termination.
    self.assertEqual(for_option.pterm(mid_timestep), 1.0)


if __name__ == '__main__':
  absltest.main()
