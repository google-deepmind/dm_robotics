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

from typing import Any, List, Text
from unittest import mock

from absl.testing import absltest
import dm_env
from dm_robotics.agentflow import core
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.meta_options.control_flow import cond
import numpy as np


class CondOptionTest(absltest.TestCase):

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

  def _test_called_correct_delegate_method(self, cond_option, on_delegate,
                                           off_delegate, method_name, *args,
                                           **kwargs):
    initial_call_count = getattr(off_delegate, method_name).call_count
    getattr(cond_option, method_name)(*args, **kwargs)
    getattr(on_delegate, method_name).assert_called_with(*args, **kwargs)
    self.assertEqual(
        getattr(off_delegate, method_name).call_count, initial_call_count)

  def _test_delegation(self, option: cond.Cond, on_delegate: core.Option,
                       off_delegate: core.Option, methods_to_test: List[Text],
                       method_args: List[Any]):
    """Helper for testing that option calls the appropriate delegate.

    Args:
      option: The cond-option.
      on_delegate: The option that should be called.
      off_delegate: The option that shouldn't be called.
      methods_to_test: A list of strings naming methods to test.
      method_args: A list of  args to pass, one list per entry in
        `methods_to_test`
    """

    for i, method_name in enumerate(methods_to_test):
      self._test_called_correct_delegate_method(option, on_delegate,
                                                off_delegate, method_name,
                                                *method_args[i])

    # Result is called on the delegate option to ensure proper forwarding to
    # the delegate.
    on_delegate_result = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS,
        data='random data')
    on_delegate.result.return_value = on_delegate_result
    timestep = mock.MagicMock()
    self.assertEqual(option.result(timestep), on_delegate_result)
    on_delegate.result.assert_called_with(timestep)
    self.assertEqual(off_delegate.result.call_count, 0)

  def test_cond_static(self):
    # Test whether Cond selects and properly delegates to proper branches for
    # one-time cond
    option1 = mock.MagicMock(spec=core.Option)
    option2 = mock.MagicMock(spec=core.Option)
    timestep = mock.MagicMock()
    previous_result = mock.MagicMock()
    methods_to_test = ['on_selected', 'pterm', 'step']
    method_args = [[timestep, previous_result], [timestep], [timestep]]

    true_cond_option = cond.Cond(lambda timestep, _: True, option1, option2)
    self._test_delegation(true_cond_option, option1, option2, methods_to_test,
                          method_args)

    option1.reset_mock()
    option2.reset_mock()
    false_cond_option = cond.Cond(lambda timestep, _: False, option1, option2)

    self._test_delegation(false_cond_option, option2, option1, methods_to_test,
                          method_args)

  def test_cond_dynamic(self):
    # Test whether Cond selects and properly delegates to proper branches for
    # a dynamic cond which is evaluated every step, and cleans up properly.

    def assert_mock_step_type(mock_step, step_type):
      self.assertEqual(mock_step.call_args[0][0].step_type, step_type)

    class FlipFlopCond(object):
      """Simple callable that returns true once every `freq` calls."""

      def __init__(self, freq: int):
        self._call_ctr = 0
        self._freq = freq

      def __call__(self, timestep, result):
        res = self._call_ctr % self._freq == 0
        if timestep.mid():  # don't increment during on_selected.
          self._call_ctr += 1
        return res

    random_action_spec = testing_functions.random_array_spec(shape=(4,))
    random_observation_spec = {
        'doubles': testing_functions.random_array_spec(shape=(10,))
    }

    # create options
    fixed_action1 = testing_functions.valid_value(random_action_spec)
    fixed_action2 = testing_functions.valid_value(random_action_spec)

    option1 = mock.MagicMock(spec=core.Option)
    option2 = mock.MagicMock(spec=core.Option)
    option1.step.return_value = fixed_action1
    option2.step.return_value = fixed_action2

    flip_flop_cond = FlipFlopCond(freq=3)
    cond_option = cond.Cond(
        flip_flop_cond, option1, option2, eval_every_step=True)

    first_timestep = self._timestep(
        first=True,
        observation=testing_functions.valid_value(random_observation_spec))
    mid_timestep = self._timestep(
        observation=testing_functions.valid_value(random_observation_spec))
    last_timestep = self._timestep(
        last=True,
        observation=testing_functions.valid_value(random_observation_spec))

    # Select option1 (call_ctr 0->0).
    cond_option.on_selected(first_timestep)
    self.assertEqual(option1.on_selected.call_count, 1)
    self.assertEqual(option2.on_selected.call_count, 0)
    self.assertSetEqual(set([option1]), cond_option.options_selected)

    # Step 1; No switch (call_ctr 0->0).
    # Should first-step option1.
    cond_option.step(first_timestep)
    self.assertEqual(flip_flop_cond._call_ctr, 0)
    self.assertEqual(option1.on_selected.call_count, 1)
    self.assertEqual(option2.on_selected.call_count, 0)
    self.assertEqual(option1.step.call_count, 1)
    self.assertEqual(option2.step.call_count, 0)
    assert_mock_step_type(option1.step, dm_env.StepType.FIRST)
    self.assertSetEqual(set([option1]), cond_option.options_selected)

    # Step 2; No switch (call_ctr 0->1).
    # Should step option1
    cond_option.step(mid_timestep)
    self.assertEqual(flip_flop_cond._call_ctr, 1)
    self.assertEqual(option1.on_selected.call_count, 1)
    self.assertEqual(option2.on_selected.call_count, 0)
    self.assertEqual(option1.step.call_count, 2)
    self.assertEqual(option2.step.call_count, 0)
    assert_mock_step_type(option1.step, dm_env.StepType.MID)
    self.assertSetEqual(set([option1]), cond_option.options_selected)

    # Step 3; Switch 1->2; (call_ctr 1->2)
    # Should term-step option1 and select + first-step option2.
    cond_option.step(mid_timestep)
    self.assertEqual(flip_flop_cond._call_ctr, 2)
    self.assertEqual(option1.on_selected.call_count, 1)
    self.assertEqual(option2.on_selected.call_count, 1)
    self.assertEqual(option1.step.call_count, 3)
    self.assertEqual(option2.step.call_count, 1)
    assert_mock_step_type(option1.step, dm_env.StepType.LAST)
    assert_mock_step_type(option2.step, dm_env.StepType.FIRST)
    self.assertSetEqual(set([option1, option2]), cond_option.options_selected)

    # Step 4; No switch; (call_ctr 2->3)
    # Should step option2.
    cond_option.step(mid_timestep)
    self.assertEqual(flip_flop_cond._call_ctr, 3)
    self.assertEqual(option1.on_selected.call_count, 1)
    self.assertEqual(option2.on_selected.call_count, 1)
    self.assertEqual(option1.step.call_count, 3)
    self.assertEqual(option2.step.call_count, 2)
    assert_mock_step_type(option2.step, dm_env.StepType.MID)
    self.assertSetEqual(set([option1, option2]), cond_option.options_selected)

    # Step 5; Switch 2->1; (call_ctr 3->4)
    # Should term-step option2 and select + first-step option1.
    cond_option.step(mid_timestep)
    self.assertEqual(flip_flop_cond._call_ctr, 4)
    self.assertEqual(option1.on_selected.call_count, 2)
    self.assertEqual(option2.on_selected.call_count, 1)
    self.assertEqual(option1.step.call_count, 4)
    self.assertEqual(option2.step.call_count, 3)
    assert_mock_step_type(option1.step, dm_env.StepType.FIRST)
    assert_mock_step_type(option2.step, dm_env.StepType.LAST)
    self.assertSetEqual(set([option1, option2]), cond_option.options_selected)

    # Step 6; Switch 1->2; (call_ctr 4->5)
    # Should term-step option1 and select + first-step option2.
    cond_option.step(mid_timestep)
    self.assertEqual(flip_flop_cond._call_ctr, 5)
    self.assertEqual(option1.on_selected.call_count, 2)
    self.assertEqual(option2.on_selected.call_count, 2)
    self.assertEqual(option1.step.call_count, 5)
    self.assertEqual(option2.step.call_count, 4)
    assert_mock_step_type(option1.step, dm_env.StepType.LAST)
    assert_mock_step_type(option2.step, dm_env.StepType.FIRST)
    self.assertSetEqual(set([option1, option2]), cond_option.options_selected)

    # Step 7; No switch; (call_ctr 5->5)
    # Send terminal step and verify both options see it.
    # Should term-step option1 and option2
    # The set of selected options should be reinitialized.
    cond_option.step(last_timestep)
    self.assertEqual(flip_flop_cond._call_ctr, 5)
    self.assertEqual(option1.step.call_count, 6)
    self.assertEqual(option2.step.call_count, 5)
    cond_option._true_branch.step.assert_called_with(last_timestep)
    cond_option._false_branch.step.assert_called_with(last_timestep)
    self.assertSetEqual(set(), cond_option.options_selected)


if __name__ == '__main__':
  absltest.main()
