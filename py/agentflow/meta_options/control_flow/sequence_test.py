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

"""Tests for dm_robotics.agentflow.meta_options.bt.sequence."""

from typing import List, Tuple
from unittest import mock

from absl.testing import absltest
import dm_env
from dm_robotics.agentflow import core
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.meta_options.control_flow import cond
from dm_robotics.agentflow.meta_options.control_flow import sequence
from dm_robotics.agentflow.options import basic_options
import numpy as np

_SUCCESS_RESULT_TEXT = 'All good.'
_SUCCESS_RESULT = core.OptionResult(
    core.TerminationType.SUCCESS,
    termination_text=_SUCCESS_RESULT_TEXT)
_FAILURE_RESULT = core.OptionResult(core.TerminationType.FAILURE)
_FIXED_ACTION = [0.1, 0.2, -0.3, 0.05]


def _make_simple_option(result=_SUCCESS_RESULT):
  fixed_action = _FIXED_ACTION
  option = mock.MagicMock(spec=basic_options.FixedOp)
  option.step.return_value = fixed_action
  option.pterm.return_value = 1.0
  option.result.return_value = result
  return option


class SequenceTest(absltest.TestCase):
  """Test case for Sequence Option."""

  def _make_agent(
      self, *, num_options, terminate_on_option_failure
  ) -> Tuple[sequence.Sequence, List[core.Option]]:

    # create options
    option_list = []
    for _ in range(num_options):
      option_list.append(_make_simple_option())

    agent = sequence.Sequence(
        option_list=option_list,
        terminate_on_option_failure=terminate_on_option_failure,
        name='TestSequence')
    return agent, option_list

  def test_basic(self):
    """Test that sets up a basic sequence and runs a few steps."""

    agent, option_list = self._make_agent(
        num_options=3, terminate_on_option_failure=False)

    # Ensure we can directly step an option before it is selected.
    first_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    mid_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.MID)
    last_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.LAST)
    agent.step(first_timestep)  # marks first option for termination.
    self.assertTrue(agent._terminate_option)

    # Select option and verify no option has been touched yet.
    agent.on_selected(first_timestep)
    self.assertIsNone(agent._current_option)
    self.assertIsNone(agent._previous_option)

    # Step first option.
    agent.step(first_timestep)
    self.assertTrue(agent._terminate_option)  # marked for termination.
    self.assertIsNone(agent._previous_option)  # haven't advanced yet.
    # Assert sequence isn't terminal yet.
    self.assertEqual(agent.pterm(mid_timestep), 0.0)

    # Step through second option.
    option_list[1].pterm.return_value = 0.0  # make non-terminal
    agent.step(mid_timestep)  # now switches to option1.
    # Assert we haven't advanced yet.
    self.assertIs(option_list[0], agent._previous_option)
    # Step again and assert we still haven't advanced
    agent.step(mid_timestep)
    self.assertIs(option_list[0], agent._previous_option)
    # Make option terminal and assert we advance.
    option_list[1].pterm.return_value = 1.0  # make terminal
    agent.step(mid_timestep)  # marks option1 for termination
    self.assertTrue(agent._terminate_option)
    # Assert sequence isn't terminal yet.
    self.assertEqual(agent.pterm(mid_timestep), 0.0)

    # Step through third option.
    agent.step(mid_timestep)  # transitions & steps option2.
    self.assertEqual(agent.pterm(mid_timestep), 1.0)  # immediately terminal.
    self.assertTrue(agent._terminate_option)  # wants to terminate option2.
    # Assert we haven't transitioned yet.
    self.assertIs(option_list[1], agent._previous_option)
    self.assertIs(option_list[2], agent._current_option)

    agent.step(last_timestep)  # transitions to terminal state.
    self.assertIs(option_list[2], agent._previous_option)
    # Assert sequence is terminal.
    self.assertEqual(agent.pterm(mid_timestep), 1.0)

  def test_sequence_failure_on_option_failure(self):
    """Test that sets up a basic sequence and runs until failure."""

    agent, option_list = self._make_agent(
        num_options=3, terminate_on_option_failure=True)
    # Select option and verify no option has been touched yet.
    first_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    mid_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.MID)
    last_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.LAST)
    agent.on_selected(first_timestep)
    self.assertIsNone(agent._current_option)
    self.assertIsNone(agent._previous_option)

    # Step first option (won't switch until next step).
    agent.step(first_timestep)
    # Assert sequence isn't terminal yet.
    self.assertEqual(agent.pterm(first_timestep), 0.0)

    # Step through second option.
    option_list[1].pterm.return_value = 0.0  # make non-terminal
    agent.step(mid_timestep)
    agent.step(mid_timestep)
    self.assertIs(option_list[1], agent._current_option)

    # Make option terminal and FAIL and assert we advance.
    option_list[1].pterm.return_value = 1.0  # make terminal
    option_list[1].result.return_value = _FAILURE_RESULT  # make option fail.
    agent.step(mid_timestep)  # option1 marked for termination.
    self.assertTrue(agent._terminate_option)  # option1 terminal
    # Assert we haven't advanced yet.
    self.assertIs(option_list[0], agent._previous_option)
    self.assertIs(option_list[1], agent._current_option)
    # At this point option_list[1] has not yet received a timestep
    # with StepType.LAST. Because of that, the result of
    # option_list[1] cannot be guaranteed to be valid. We'll only find
    # out about that after the next timestep. Because of that, despite
    # terminate_on_option_failure=True, pterm is still 0.
    self.assertEqual(agent.pterm(mid_timestep), 0.0)
    # Pass another MID timestep and assert we haven't advanced yet.
    agent.step(mid_timestep)  # stays stuck on option1 b/c sequence_terminal
    self.assertIs(option_list[1], agent._previous_option)
    self.assertIs(None, agent._current_option)
    # Now the sequence knows the result of option_list[1], so we can
    # assert that the sequence is terminal.
    self.assertEqual(agent.pterm(mid_timestep), 1.0)
    # Pass a LAST timestep and assert still haven't advanced (sequence_terminal)
    agent.step(last_timestep)
    # The current/previous options have not changed.
    self.assertIs(option_list[1], agent._previous_option)
    self.assertIs(None, agent._current_option)
    self.assertEqual(
        agent.result(last_timestep).termination_reason,
        core.TerminationType.FAILURE)

  def test_sequence_failure_on_immediate_option_failure(self):
    """Test that sets up a basic sequence and runs a few steps."""

    agent, option_list = self._make_agent(
        num_options=3, terminate_on_option_failure=True)
    # Select option and verify no option has been touched yet.
    first_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    last_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.LAST)
    agent.on_selected(first_timestep)
    self.assertIsNone(agent._current_option)
    self.assertIsNone(agent._previous_option)

    option_list[0].pterm.return_value = 1.0
    option_list[0].result.return_value = _FAILURE_RESULT  # make option fail.

    agent.on_selected(first_timestep)
    agent.step(first_timestep)
    agent.step(last_timestep)

    self.assertIs(option_list[0], agent._previous_option)
    # Assert sequence is terminal and failure because the option failed.
    self.assertEqual(agent.pterm(last_timestep), 1.0)
    self.assertEqual(
        agent.result(last_timestep).termination_reason,
        core.TerminationType.FAILURE)

  def test_wrapped_sequence_failure_on_immediate_option_failure(self):
    """Test that sets up a basic sequence and runs a few steps."""

    agent, option_list = self._make_agent(
        num_options=3, terminate_on_option_failure=True)
    # Select option and verify no option has been touched yet.
    wrapped_agent = sequence.Sequence([agent])
    first_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    mid_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.MID)
    last_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.LAST)
    wrapped_agent.on_selected(first_timestep)

    self.assertIsNone(agent._current_option)
    self.assertIsNone(agent._previous_option)

    option_list[0].pterm.return_value = 1.0
    option_list[0].result.return_value = _FAILURE_RESULT  # make option fail.

    wrapped_agent.on_selected(first_timestep)
    wrapped_agent.step(first_timestep)
    wrapped_agent.step(mid_timestep)
    wrapped_agent.step(last_timestep)

    # Check inner agent
    self.assertIs(option_list[0], agent._previous_option)
    # Assert sequence is terminal and failure because the option failed.
    self.assertEqual(agent.pterm(last_timestep), 1.0)
    self.assertEqual(agent.result(last_timestep).termination_reason,
                     core.TerminationType.FAILURE)

    self.assertEqual(
        wrapped_agent.result(last_timestep).termination_reason,
        core.TerminationType.FAILURE)

    self.assertEqual(
        wrapped_agent._previous_option_result.termination_reason,
        core.TerminationType.FAILURE)

  def test_nested_with_cond(self):
    """Test that composes a nested set of Sequence and Cond options."""

    # State used to track which options are executed.
    counter = 0
    true_counter = 0
    truth_test_counter = 0

    def increment_counter(unused_timestep, unused_result):
      nonlocal counter
      counter += 1

    def increment_true_counter(unused_timestep, unused_result):
      nonlocal true_counter
      true_counter += 1

    def counter_is_even(unused_timestep, unused_result):
      nonlocal counter
      nonlocal truth_test_counter
      truth_test_counter += 1
      return counter % 2 == 0

    # options that update the tracking state - these are the instrumented
    # options.
    inc_counter_op = basic_options.LambdaOption(
        delegate=_make_simple_option(),
        on_selected_func=increment_counter)

    inc_true_counter_op = basic_options.LambdaOption(
        delegate=_make_simple_option(),
        on_selected_func=increment_true_counter)

    # inner increments counter unconditionally, and
    # increments true_counter on the condition's True branch.
    option_list = [inc_counter_op,
                   cond.Cond(counter_is_even,
                             inc_true_counter_op,
                             _make_simple_option())]
    inner = sequence.Sequence(option_list=option_list, name='inner')

    # outer executes inner a few times.
    # We could use a loop here to test all meta_options.
    outer = sequence.Sequence(option_list=[inner] * 10, name='outer')

    timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    outer.on_selected(timestep)
    outer.step(timestep)
    # outer.step -> inner.step -> inc_counter_op.step
    # We should have incrementd the counter, but not checked the condition.
    self.assertEqual(counter, 1)
    self.assertEqual(truth_test_counter, 0)
    self.assertEqual(true_counter, 0)

    outer.step(testing_functions.random_timestep(step_type=dm_env.StepType.MID))
    # outer.step -> inner.step -> cond.step ->
    #   counter_is_even (False) and the simple option
    self.assertEqual(counter, 1)  # on_selected, step(FIRST)
    self.assertGreaterEqual(truth_test_counter, 1)  # on_selected, step(FIRST)
    truth_test_counter = 0  # Reset for future tests.
    self.assertEqual(true_counter, 0)

    outer.step(testing_functions.random_timestep(step_type=dm_env.StepType.MID))
    # outer.step -> inner.step -> inc_counter_op.step
    self.assertEqual(counter, 2)
    self.assertGreaterEqual(truth_test_counter, 0)
    self.assertEqual(true_counter, 0)

    outer.step(testing_functions.random_timestep(step_type=dm_env.StepType.MID))
    # outer.step -> inner.step -> cond.step ->
    #   counter_is_even (True) and the simple option
    self.assertEqual(counter, 2)
    self.assertGreaterEqual(truth_test_counter, 1)  # on_selected, step(FIRST)
    truth_test_counter = 0  # Reset for future tests.
    self.assertEqual(true_counter, 1)

    outer.step(testing_functions.random_timestep(step_type=dm_env.StepType.MID))
    # outer.step -> inner.step -> inc_counter_op.step
    self.assertEqual(counter, 3)
    self.assertGreaterEqual(truth_test_counter, 0)
    self.assertEqual(true_counter, 1)

  def test_with_single_child(self):
    """Test that Sequence wrapping a single child can step and pterm."""
    child_op = _make_simple_option()
    sequence_op = sequence.Sequence([child_op])

    # Verify pterm can be called before option is stepped.
    # Not guaranteed to match pterm of child bc Sequence needs to be FIRST
    # stepped before the child is activated.
    timestep = testing_functions.random_timestep()
    pterm = sequence_op.pterm(timestep)
    self.assertEqual(pterm, 0.)  # Should be zero before sequence is stepped.

    # FIRST-step the sequence_op and verify the underlying op pterm gets through
    first_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    sequence_op.on_selected(first_timestep)
    sequence_op.step(first_timestep)
    expected_pterm = child_op.pterm(first_timestep)
    actual_pterm = sequence_op.pterm(first_timestep)
    self.assertEqual(expected_pterm, actual_pterm)

  def test_success_result_single_child(self):
    """Test that Sequence wrapping a single child can return results."""
    child_op = _make_simple_option()
    sequence_op = sequence.Sequence([child_op])

    timestep = testing_functions.random_timestep()
    pterm = sequence_op.pterm(timestep)
    self.assertEqual(pterm, 0.)  # Should be zero before sequence is stepped.

    first_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    last_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.LAST)
    sequence_op.on_selected(first_timestep)
    sequence_op.step(first_timestep)
    sequence_op.step(last_timestep)

    result = sequence_op.result(last_timestep)
    self.assertEqual(result.termination_reason,
                     core.TerminationType.SUCCESS)
    self.assertEqual(result.termination_text, _SUCCESS_RESULT_TEXT)

  def test_failure_result_single_child(self):
    """Test that Sequence wrapping a single child can return results."""
    child_op = _make_simple_option(_FAILURE_RESULT)
    sequence_op = sequence.Sequence([child_op])

    timestep = testing_functions.random_timestep()
    pterm = sequence_op.pterm(timestep)
    self.assertEqual(pterm, 0.)  # Should be zero before sequence is stepped.

    first_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    last_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.LAST)
    sequence_op.on_selected(first_timestep)
    sequence_op.step(first_timestep)
    sequence_op.step(last_timestep)

    result = sequence_op.result(last_timestep)
    self.assertEqual(result, _FAILURE_RESULT)

  def test_first_op_gets_last_step_before_second_gets_first_step(
      self):
    first_op = testing_functions.SpyOp(np.ndarray(0),
                                       pterm=1,
                                       result=_FAILURE_RESULT)
    second_op = testing_functions.SpyOp(np.ndarray(0))
    sequence_op = sequence.Sequence([first_op, second_op])

    first_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.FIRST)
    mid_timestep = testing_functions.random_timestep(
        step_type=dm_env.StepType.MID)

    sequence_op.on_selected(first_timestep)
    sequence_op.step(first_timestep)
    sequence_op.step(mid_timestep)

    def get_timesteps_passed_to_step_function(spyop):
      return [spyopcall.step for spyopcall in spyop.timesteps
              if spyopcall.step]

    self.assertEqual(
        get_timesteps_passed_to_step_function(first_op),
        [first_timestep, mid_timestep._replace(
            step_type=dm_env.StepType.LAST)])
    self.assertEqual(
        get_timesteps_passed_to_step_function(second_op),
        [mid_timestep._replace(step_type=dm_env.StepType.FIRST)])
    self.assertEqual(second_op.previous_option_result,
                     _FAILURE_RESULT)


if __name__ == '__main__':
  absltest.main()
