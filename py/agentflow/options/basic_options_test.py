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

"""Tests for basic_options."""

from typing import Callable, Text
from unittest import mock

from absl.testing import absltest
import dm_env
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.options import basic_options
import numpy as np


class FixedOpTest(absltest.TestCase):

  def test_action_returned(self):
    expected_action = testing_functions.random_action()
    timestep = mock.MagicMock()
    num_steps = None
    option = basic_options.FixedOp(expected_action, num_steps, 'test_fixed_op')
    actual_action = option.step(timestep)
    np.testing.assert_almost_equal(actual_action, expected_action)
    self.assertEqual(option.pterm(timestep), 0.)

  def test_termination(self):
    random_action = testing_functions.random_action()
    timestep = mock.MagicMock()

    # If num_steps is None should never terminate.
    num_steps = None
    option = basic_options.FixedOp(random_action, num_steps, 'test_fixed_op')
    option.step(timestep)
    self.assertEqual(option.pterm(timestep), 0.)

    # If num_steps is 0 should request termination immediately, even before step
    num_steps = 0
    option = basic_options.FixedOp(random_action, num_steps, 'test_fixed_op')
    self.assertEqual(option.pterm(timestep), 1.)

    # If num_steps = n should terminate after nth step.
    num_steps = 5
    option = basic_options.FixedOp(random_action, num_steps, 'test_fixed_op')
    for i in range(num_steps):
      option.step(timestep)
      expected_pterm = 1. if i == num_steps else 0.
      self.assertEqual(option.pterm(timestep), expected_pterm)

  def test_set_action(self):
    expected_action1 = testing_functions.random_action()
    timestep = mock.MagicMock()
    num_steps = None
    option = basic_options.FixedOp(expected_action1, num_steps, 'test_fixed_op')

    actual_action1 = option.step(timestep)
    np.testing.assert_almost_equal(actual_action1, expected_action1)

    expected_action2 = testing_functions.random_action()
    option.set_action(expected_action2)
    actual_action2 = option.step(timestep)
    np.testing.assert_almost_equal(actual_action2, expected_action2)


class RandomOptionTest(absltest.TestCase):

  def test_action_returned(self):
    action_spec = testing_functions.random_array_spec()
    timestep = mock.MagicMock()
    option = basic_options.RandomOption(action_spec)
    for _ in range(5):
      output_action = option.step(timestep)
      spec_utils.validate(action_spec, output_action)
      self.assertEqual(option.pterm(timestep), 0.)


class LambdaOptionsTest(absltest.TestCase):

  def test_delegation(self):
    delegate = mock.MagicMock(spec=core.Option)
    option = basic_options.LambdaOption(
        on_selected_func=lambda timestep, prev_result: True,
        func_as_result=True,
        delegate=delegate)

    timestep = mock.MagicMock()
    previous_result = mock.MagicMock()

    option.on_selected(timestep, previous_result)
    delegate.on_selected.assert_called_with(timestep, previous_result)

    delegate.pterm.return_value = 0.4
    self.assertEqual(option.pterm(timestep), 0.4)
    delegate.pterm.assert_called_with(timestep)

    # For a LambdaOption if `func_as_result=True` we discard the delegate's
    # result and replace it with the result of the function.
    discarded_result = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS, data='random data')
    delegate.result.return_value = discarded_result
    expected_result = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS, data=True)
    self.assertNotEqual(option.result(timestep), discarded_result)
    self.assertEqual(option.result(timestep), expected_result)
    delegate.result.assert_called_with(timestep)

  def test_result(self):
    delegate = mock.MagicMock(spec=core.Option)
    lambda_result = 'lambda result'
    option = basic_options.LambdaOption(
        on_selected_func=lambda timestep, prev_result: lambda_result,
        func_as_result=True,
        delegate=delegate)

    timestep = mock.MagicMock()
    previous_result = mock.MagicMock()
    delegate.result.return_value = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS,
        data='Delegate result should be ignored')
    delegate.pterm.return_value = 1.0

    option.on_selected(timestep, previous_result)
    option.step(timestep)
    pterm = option.pterm(timestep)
    self.assertEqual(pterm, 1.0)
    result = option.result(timestep)
    self.assertEqual(result.data, lambda_result)

  def test_callables_invoked(self):
    delegate = mock.MagicMock(spec=core.Option)
    on_selected_func = mock.MagicMock()
    on_step_func = mock.MagicMock()
    pterm_func = mock.MagicMock()
    option = basic_options.LambdaOption(
        on_selected_func=on_selected_func,
        on_step_func=on_step_func,
        pterm_func=pterm_func,
        delegate=delegate)

    timestep = mock.MagicMock()
    previous_result = mock.MagicMock()

    option.on_selected(timestep, previous_result)
    on_selected_func.assert_called_with(timestep, previous_result)

    option.step(timestep)
    on_step_func.assert_called_with(timestep)

    option.pterm(timestep)
    pterm_func.assert_called_with(timestep)


class ConcurrentOptionTest(absltest.TestCase):

  def assert_timestep(self, expected: dm_env.TimeStep, actual: dm_env.TimeStep):
    self.assertIs(expected.step_type, actual.step_type)
    np.testing.assert_almost_equal(expected.discount, actual.discount)
    np.testing.assert_almost_equal(expected.reward, actual.reward)
    testing_functions.assert_value(expected.observation, actual.observation)

  def test_action_merging(self):
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=[0, 0],
        maximum=[1, 1],
        name='spec')
    value_a = np.asarray([np.nan, 0.2], dtype=np.float32)
    value_b = np.asarray([0.1, np.nan], dtype=np.float32)

    option = basic_options.ConcurrentOption(
        options_list=[(basic_options.FixedOp(action=value_a)),
                      (basic_options.FixedOp(action=value_b))],
        action_spec=spec)

    expected_action = np.asarray([0.1, 0.2], dtype=np.float32)
    merged_action = option.step(_timestep_with_no_values())
    testing_functions.assert_value(merged_action, expected_action)

  def test_action_merging_with_empty_actions(self):
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=[0, 0],
        maximum=[1, 1],
        name='spec')
    value_a = np.asarray([0.1, 0.2], dtype=np.float32)
    value_b = np.asarray([np.nan, np.nan], dtype=np.float32)

    option = basic_options.ConcurrentOption(
        options_list=[(basic_options.FixedOp(action=value_a)),
                      (basic_options.FixedOp(action=value_b))],
        action_spec=spec)

    expected_action = np.asarray([0.1, 0.2], dtype=np.float32)
    merged_action = option.step(_timestep_with_no_values())
    testing_functions.assert_value(merged_action, expected_action)

  def test_action_emitting_nans(self):
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=[0, 0],
        maximum=[1, 1],
        name='spec')
    value_a = np.asarray([np.nan, 0.2], dtype=np.float32)
    value_b = np.asarray([np.nan, np.nan], dtype=np.float32)

    option = basic_options.ConcurrentOption(
        options_list=[(basic_options.FixedOp(action=value_a)),
                      (basic_options.FixedOp(action=value_b))],
        action_spec=spec,
        allow_nan_actions=True)

    expected_action = np.asarray([np.nan, 0.2], dtype=np.float32)
    merged_action = option.step(_timestep_with_no_values())
    testing_functions.assert_value(merged_action, expected_action)

  def test_pterm(self):
    spec_a, value_a = _rand_spec_and_value(shape=(1,), dtype=np.float32)
    spec_b, value_b = _rand_spec_and_value(shape=(2,), dtype=np.float32)
    overall_spec = testing_functions.composite_spec(spec_a, spec_b)

    option = basic_options.ConcurrentOption(
        options_list=[(testing_functions.SpyOp(action=value_a,
                                               pterm=0.2)),
                      (testing_functions.SpyOp(action=value_b,
                                               pterm=0.5))],
        action_spec=overall_spec)

    pterm = option.pterm(_timestep_with_no_values())
    np.testing.assert_almost_equal(0.6, pterm)

  def test_pterm_function_invocation(self):
    spec_a, value_a = _rand_spec_and_value(shape=(1,), dtype=np.float32)
    spec_b, value_b = _rand_spec_and_value(shape=(2,), dtype=np.float32)
    overall_spec = testing_functions.composite_spec(spec_a, spec_b)

    option1 = testing_functions.SpyOp(action=value_a, pterm=0.2)
    option2 = testing_functions.SpyOp(action=value_b, pterm=0.5)

    actual_pterms = []

    def custom_termination_function(
        pterms: basic_options.OptionPterms) -> float:
      actual_pterms.extend(pterms)
      return 0.9

    option = basic_options.ConcurrentOption(
        options_list=[option1, option2],
        action_spec=overall_spec,
        termination=custom_termination_function)

    pterm = option.pterm(_timestep_with_no_values())

    np.testing.assert_almost_equal(0.9, pterm)

    expected_pterms = [(option1, 0.2), (option2, 0.5)]
    actual_pterms_sorted = sorted(actual_pterms, key=lambda opt: opt[0].uid)
    expected_pterms_sorted = sorted(expected_pterms, key=lambda opt: opt[0].uid)
    self.assertEqual(actual_pterms_sorted, expected_pterms_sorted)

  def test_any_terminates(self):
    option = basic_options.FixedOp(np.random.random(size=(2,)))
    self.assertEqual(1.0, basic_options.any_terminates([(option, 1.0)]))
    self.assertEqual(
        1.0, basic_options.any_terminates([(option, 0.0), (option, 1.0)]))
    self.assertEqual(
        0.0, basic_options.any_terminates([(option, 0.0), (option, 0.0)]))
    np.testing.assert_almost_equal(
        0.3, basic_options.any_terminates([(option, 0.0), (option, 0.3)]))
    np.testing.assert_almost_equal(
        0.64, basic_options.any_terminates([(option, 0.4), (option, 0.4)]))

  def test_all_terminate(self):
    option = basic_options.FixedOp(np.random.random(size=(2,)))
    self.assertEqual(1.0, basic_options.all_terminate([(option, 1.0)]))
    self.assertEqual(0.0, basic_options.all_terminate([(option, 0.0)]))
    self.assertEqual(
        0.25, basic_options.all_terminate([(option, 0.5), (option, 0.5)]))

  def test_options_terminate(self):
    option1 = basic_options.FixedOp(np.random.random(size=(2,)))
    option2 = basic_options.FixedOp(np.random.random(size=(2,)))
    option3 = basic_options.FixedOp(np.random.random(size=(2,)))

    o1_terminates = basic_options.options_terminate(option1)
    o1_o2_terminates = basic_options.options_terminate(option1, option2)

    # o1_terminates should ignore other options
    self.assertEqual(1.0, o1_terminates([(option1, 1.0)]))
    self.assertEqual(0.3, o1_terminates([(option1, 0.3), (option2, 1.0)]))

    # o1_o2_terminates should return the product of option1 and option2 pterms.
    self.assertEqual(0.6, o1_o2_terminates([(option1, 0.6)]))
    self.assertEqual(0.36, o1_o2_terminates([(option1, 0.6), (option2, 0.6)]))

    # o1_o2_terminates should return the product of option1 and option2 pterms.
    self.assertEqual(0.6, o1_o2_terminates([(option1, 0.6)]))
    self.assertEqual(0.36, o1_o2_terminates([
        (option1, 0.6),
        (option2, 0.6),
    ]))
    self.assertEqual(
        0.36, o1_o2_terminates([
            (option1, 0.6),
            (option2, 0.6),
            (option3, 0.6),
        ]))

  def test_result_overall_failure_if_one_fails(self):
    # The result of a ConcurrentOption is a list of result values,
    # There should be a single termination reason - which ever is the 'worst'
    # termination reason.  I.e. if one option failed the whole thing failed.
    spec_1, value_1 = _rand_spec_and_value(shape=(1,), dtype=np.float32)
    spec_2, value_2 = _rand_spec_and_value(shape=(2,), dtype=np.float32)
    overall_spec = testing_functions.composite_spec(spec_1, spec_2)

    result_1 = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS, data='data_1')
    result_2 = core.OptionResult(
        termination_reason=core.TerminationType.FAILURE, data='data_2')

    option_1 = testing_functions.SpyOp(action=value_1, result=result_1)
    option_2 = testing_functions.SpyOp(action=value_2, result=result_2)
    option = basic_options.ConcurrentOption(
        options_list=[option_1, option_2], action_spec=overall_spec)

    result = option.result(_timestep_with_no_values())
    self.assertIs(result.termination_reason, core.TerminationType.FAILURE)
    self.assertIsInstance(result.data, list)
    self.assertListEqual(result.data, ['data_1', 'data_2'])

  def test_result_successful(self):
    spec_1, value_1 = _rand_spec_and_value(shape=(1,), dtype=np.float32)
    spec_2, value_2 = _rand_spec_and_value(shape=(2,), dtype=np.float32)
    overall_spec = testing_functions.composite_spec(spec_1, spec_2)

    result_1 = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS, data='data_1')
    result_2 = core.OptionResult(
        termination_reason=core.TerminationType.SUCCESS, data='data_2')

    option_1 = testing_functions.SpyOp(action=value_1, result=result_1)
    option_2 = testing_functions.SpyOp(action=value_2, result=result_2)
    option = basic_options.ConcurrentOption(
        options_list=[option_1, option_2], action_spec=overall_spec)

    result = option.result(_timestep_with_no_values())
    self.assertIs(result.termination_reason, core.TerminationType.SUCCESS)
    self.assertIsInstance(result.data, list)
    self.assertListEqual(result.data, ['data_1', 'data_2'])

  def test_child_timesteps(self):
    # Child options should have the observation in their timestep altered
    # to include their arg, if it is present in the input observation.

    # This test runs through the methods that are supposed to do this.
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=[0, 0],
        maximum=[1, 1],
        name='spec')
    value_a = np.asarray([np.nan, 0.2], dtype=np.float32)
    value_b = np.asarray([0.1, np.nan], dtype=np.float32)

    arg_spec_1, arg_1 = _rand_spec_and_value(shape=(3,))
    arg_spec_2, arg_2 = _rand_spec_and_value(shape=(4,))

    option_1 = testing_functions.SpyOp(action=value_a,
                                       arg_spec=arg_spec_1)
    option_2 = testing_functions.SpyOp(action=value_b,
                                       arg_spec=arg_spec_2)
    option = basic_options.ConcurrentOption(
        options_list=[option_1, option_2], action_spec=spec)

    observation_with_args = {option.arg_key: np.concatenate([arg_1, arg_2])}
    parent_timestep = dm_env.TimeStep(
        step_type=np.random.choice(list(dm_env.StepType)),
        reward=np.random.random(),
        discount=np.random.random(),
        observation=observation_with_args)
    expected_timestep_1 = dm_env.TimeStep(
        step_type=parent_timestep.step_type,
        reward=parent_timestep.reward,
        discount=parent_timestep.discount,
        observation={
            option.arg_key: np.concatenate([arg_1, arg_2]),
            option_1.arg_key: arg_1
        },
    )
    expected_timestep_2 = dm_env.TimeStep(
        step_type=parent_timestep.step_type,
        reward=parent_timestep.reward,
        discount=parent_timestep.discount,
        observation={
            option.arg_key: np.concatenate([arg_1, arg_2]),
            option_2.arg_key: arg_2
        },
    )

    option.on_selected(parent_timestep)
    self.assert_timestep(expected_timestep_1, option_1.timestep)
    self.assert_timestep(expected_timestep_2, option_2.timestep)
    option_1.clear_timesteps()

    option.step(parent_timestep)
    self.assert_timestep(expected_timestep_1, option_1.timestep)
    self.assert_timestep(expected_timestep_2, option_2.timestep)
    option_1.clear_timesteps()

    option.pterm(parent_timestep)
    self.assert_timestep(expected_timestep_1, option_1.timestep)
    self.assert_timestep(expected_timestep_2, option_2.timestep)
    option_1.clear_timesteps()

    option.result(parent_timestep)
    self.assert_timestep(expected_timestep_1, option_1.timestep)
    self.assert_timestep(expected_timestep_2, option_2.timestep)
    option_1.clear_timesteps()

  def test_arg_spec(self):
    spec_a, action_a = _rand_spec_and_value(shape=(1,),
                                            dtype=np.float32)
    spec_b, action_b = _rand_spec_and_value(shape=(2,),
                                            dtype=np.float32)
    overall_spec = testing_functions.composite_spec(spec_a, spec_b)

    arg_spec_a = testing_functions.random_array_spec(
        shape=(3,), dtype=np.float32)
    arg_spec_b = testing_functions.random_array_spec(
        shape=(4,), dtype=np.float32)

    option_1 = testing_functions.SpyOp(action=action_a,
                                       arg_spec=arg_spec_a)
    option_2 = testing_functions.SpyOp(action=action_b,
                                       arg_spec=arg_spec_b)
    option = basic_options.ConcurrentOption(
        options_list=[option_1, option_2], action_spec=overall_spec)

    testing_functions.assert_spec(
        option.arg_spec,
        spec_utils.merge_specs([option_1.arg_spec(), option_2.arg_spec()]))


class DelegateOptionTest(absltest.TestCase):

  def testDelegateOption(self):
    base = basic_options.FixedOp(np.arange(2))
    delegate1 = TrivialDelegateOption(base)
    delegate2 = TrivialDelegateOption(base)

    # uid is delegated, BUT the delegate is not the thing it delegates to
    # and therefore it is not (and shouldn't be) considered equal.
    self.assertNotEqual(delegate1.uid, base.uid)
    self.assertNotEqual(delegate2.uid, base.uid)

    # Check __eq__
    self.assertIsNot(delegate1, delegate2)
    self.assertNotEqual(delegate1, delegate2)

    # Check __hash__
    self.assertNotEqual(hash(delegate1), hash(delegate2))


class TrivialDelegateOption(basic_options.DelegateOption):

  def step(self, timestep):
    return np.arange(2)


class FakeActionSpace(core.ActionSpace[core.Spec]):

  def __init__(self, func: Callable[[np.ndarray], np.ndarray]):
    super().__init__()
    self._func = func

  @property
  def name(self) -> Text:
    return 'FakeActionSpace'

  def spec(self) -> core.Spec:
    raise NotImplementedError()

  def project(self, action: np.ndarray) -> np.ndarray:
    return self._func(action)


class PadOptionTest(absltest.TestCase):

  def testRestructureOutput(self):
    value_from_base = np.arange(2)
    value_from_padded = np.arange(3)

    def adjuster(value: np.ndarray) -> np.ndarray:
      np.testing.assert_almost_equal(value, value_from_base)
      return value_from_padded

    base = basic_options.FixedOp(value_from_base)
    padded = basic_options.PadOption(
        base, action_space=FakeActionSpace(adjuster))

    action = padded.step(timestep=_timestep_with_no_values())
    np.testing.assert_almost_equal(action, value_from_padded)

  def testArgSpec(self):
    expected_arg_spec = specs.Array(
        shape=(2, 2), dtype=np.float32, name='expected_arg_spec')
    action_spec = specs.Array(shape=(3,), dtype=np.float32, name='action_spec')

    class HasSpec(basic_options.FixedOp):

      def arg_spec(self):
        return expected_arg_spec

    base = HasSpec(testing_functions.valid_value(action_spec))
    padded = basic_options.PadOption(
        base, action_space=core.IdentityActionSpace(action_spec))

    self.assertEqual(padded.arg_spec(), expected_arg_spec)


class ArgAdaptorTest(absltest.TestCase):

  def testArgToNone(self):
    fixed_action = np.arange(2)
    random_arg_spec, random_arg = _rand_spec_and_value((4,))
    op_without_arg = mock.MagicMock(spec=basic_options.FixedOp)
    op_without_arg.arg_spec.return_value = None
    type(op_without_arg).arg_key = mock.PropertyMock(
        return_value='op_without_arg_key')
    op_without_arg.step.return_value = fixed_action

    adaptor_func = mock.Mock()
    adaptor_func.return_value = None
    adapted_op_without_arg = basic_options.ArgAdaptor(
        op_without_arg, random_arg_spec, adaptor_func=adaptor_func)

    observation = {'op_without_arg_key': random_arg}
    timestep = testing_functions.random_timestep(observation=observation)
    adapted_op_without_arg.step(timestep)
    timestep_without_arg = timestep._replace(observation={})
    op_without_arg.step.assert_called_with(timestep_without_arg)
    self.assertEqual(adaptor_func.call_count, 0)

  def testArgToOtherArg(self):
    fixed_action = np.arange(2)
    parent_arg_spec, parent_arg = _rand_spec_and_value((4,))
    adapted_arg_spec, adapted_arg = _rand_spec_and_value((2,))

    op_with_arg = mock.MagicMock(spec=basic_options.FixedOp)
    op_with_arg.arg_spec.return_value = adapted_arg_spec
    type(op_with_arg).arg_key = mock.PropertyMock(
        return_value='op_with_arg_key')
    op_with_arg.step.return_value = fixed_action

    adaptor_func = mock.Mock()
    adaptor_func.return_value = adapted_arg
    adapted_op_with_arg = basic_options.ArgAdaptor(
        op_with_arg, parent_arg_spec, adaptor_func=adaptor_func)

    parent_observation = {'op_with_arg_key': parent_arg}
    timestep = testing_functions.random_timestep(observation=parent_observation)
    adapted_op_with_arg.step(timestep)
    timestep_with_replaced_arg = timestep._replace(
        observation={'op_with_arg_key': adapted_arg})
    op_with_arg.step.assert_called_with(timestep_with_replaced_arg)
    adaptor_func.assert_called_with(parent_arg)


class IgnoreErrorDelegateOptionTest(absltest.TestCase):

  def testResultIsSuccess(self):
    _, value = _rand_spec_and_value(shape=(1,), dtype=np.float32)

    delegate_result = core.OptionResult(
        termination_reason=core.TerminationType.FAILURE, data='failure')

    delegate = testing_functions.SpyOp(action=value, result=delegate_result)
    option = basic_options.IgnoreErrorDelegateOption(delegate, 'name')

    result = option.result(_timestep_with_no_values())
    self.assertIs(result.termination_reason, core.TerminationType.SUCCESS)


def _rand_spec_and_value(shape, dtype=None):
  spec = testing_functions.random_array_spec(shape=shape, dtype=dtype)
  return spec, testing_functions.valid_value(spec)


def _timestep_with_no_values():
  return dm_env.TimeStep(
      step_type=(np.random.choice(list(dm_env.StepType))),
      reward=np.random.random(),
      discount=np.random.random(),
      observation={})


if __name__ == '__main__':
  absltest.main()
