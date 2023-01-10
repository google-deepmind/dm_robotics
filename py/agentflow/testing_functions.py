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
"""Utility functions that are helpful in testing."""

import random
import string
import sys
import typing
from typing import Any, Dict, List, Mapping, NewType, Optional, Text, Tuple, Type, Union

from absl import logging
from dm_control import composer
from dm_control.composer import arena
import dm_env
from dm_env import specs
import dm_robotics.agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import overrides
import numpy as np

Arg = NewType('Arg', Any)  # pylint: disable=invalid-name


# Container for caching the timestep argument to Option methods.
class SpyOpCall(typing.NamedTuple):
  pterm: Optional[dm_env.TimeStep]
  on_selected: Optional[dm_env.TimeStep]
  step: Optional[dm_env.TimeStep]
  result: Optional[dm_env.TimeStep]


def _equal_or_close(x: Arg, y: Arg) -> bool:
  """An equals function that can take floats, dicts, or == compatible inputs."""
  if isinstance(x, float) or isinstance(x, np.ndarray):
    return np.allclose(x, y, equal_nan=True)
  elif isinstance(x, dict) and (set(map(type, x)) == {str}):
    # If args are themselves dicts with string keys, recurse on the values.
    return all(
        [_equal_or_close(vx, vy) for vx, vy in zip(x.values(), y.values())])
  else:
    return x == y


class SpyOp(af.FixedOp):
  """FixedOp that records the timestep it's given.

  It also records the previous option result for on_selected.
  """

  def __init__(self,
               action,
               pterm=None,
               result=None,
               arg_spec=None,
               num_steps: Optional[int] = 0,
               name='SpyOp'):
    super().__init__(action, num_steps=num_steps, name=name)
    self._pterm = pterm
    self._result = result
    self._timesteps = []  # type: List[SpyOpCall]
    self._previous_option_result = None
    self._arg_spec = arg_spec
    self._default_call = SpyOpCall(
        pterm=None, on_selected=None, step=None, result=None)

  def pterm(self, timestep):
    self._timesteps.append(self._default_call._replace(pterm=timestep))
    if self._pterm is None:
      return super().pterm(timestep)
    else:
      return self._pterm

  def result(self, timestep):
    self._timesteps.append(self._default_call._replace(result=timestep))
    if self._result is None:
      return super().result(timestep)
    else:
      return self._result

  def step(self, timestep: dm_env.TimeStep):
    self._timesteps.append(self._default_call._replace(step=timestep))
    return super().step(timestep)

  def on_selected(self, timestep, prev_option_result=None):
    self._timesteps.append(self._default_call._replace(on_selected=timestep))
    self._previous_option_result = prev_option_result
    super().on_selected(timestep, prev_option_result)

  def arg_spec(self):
    if self._arg_spec is None:
      return super().arg_spec()
    else:
      return self._arg_spec

  @property
  def timestep(self):
    if not self._timesteps:
      return None
    timesteps = self._timesteps[-1]._asdict().values()
    return next((t for t in timesteps if t is not None), None)

  @property
  def timesteps(self):
    return self._timesteps

  @property
  def previous_option_result(self):
    return self._previous_option_result

  def clear_timesteps(self):
    del self._timesteps[:]


class FixedOpWithArg(af.FixedOp):
  """A FixedOp which expects a runtime-argument via the timestep."""

  def __init__(self,
               action: np.ndarray,
               arg_spec: af.ArgSpec,
               **kwargs):
    super().__init__(action, **kwargs)
    self._arg_spec = arg_spec

  def arg_spec(self) -> af.ArgSpec:
    return self._arg_spec


class ActionSequenceOp(af.Option):
  """An option that returns a fixed sequence of actions."""

  def __init__(self, actions, name='ActionSequenceOp'):
    super().__init__(name=name)
    self._actions = actions
    self._next_action_index = 0

  @overrides(af.Option)
  def step(self, timestep: dm_env.TimeStep):
    logging.debug('ActionSequenceOp.step(%s) _next_action_index: %d', timestep,
                  self._next_action_index)
    if timestep.first():
      self._next_action_index = 0

    index = self._next_action_index
    self._next_action_index += 1

    if timestep.last():
      # This is not ok, it /is/ ok for index == len(self._actions)
      # but not to go over that.
      assert index <= len(self._actions), 'Too many steps without a LAST step.'
      # self._next_action_index = 0
      if index == len(self._actions):
        return None

    return self._actions[index]

  def pterm(self, timestep) -> float:
    return 0.0 if self._next_action_index < len(self._actions) else 1.0


class IdentitySubtask(af.SubTask):
  """Trivial subtask that does not change the task."""

  def __init__(self,
               observation_spec: specs.Array,
               action_spec: specs.Array,
               steps: int,
               name: Optional[Text] = None) -> None:
    super().__init__(name)
    self._max_steps = steps
    self._steps_taken = 0
    self._observation_spec = observation_spec
    self._action_spec = action_spec

  @overrides(af.SubTask)
  def observation_spec(self):
    return self._observation_spec

  @overrides(af.SubTask)
  def action_spec(self):
    return self._action_spec

  @overrides(af.SubTask)
  def arg_spec(self):
    return None

  @overrides(af.SubTask)
  def agent_to_parent_action(self, agent_action: np.ndarray) -> np.ndarray:
    return agent_action

  @overrides(af.SubTask)
  def reset(self, parent_timestep: dm_env.TimeStep):
    self._steps_taken = 0
    return parent_timestep

  @overrides(af.SubTask)
  def parent_to_agent_timestep(
      self,
      parent_timestep: dm_env.TimeStep,
      arg_key: Optional[Text] = None) -> Tuple[dm_env.TimeStep, float]:
    self._steps_taken += 1
    return parent_timestep, self.pterm(parent_timestep, arg_key)

  def pterm(self, parent_timestep: dm_env.TimeStep, arg_key: Text) -> float:
    """Option termination probability.

    This implementation assumes sparse reward; returns 1.0 if reward() > 0.

    Args:
      parent_timestep: A timestep from the parent.
      arg_key: The part of the observation that stores the arg.

    Returns:
      Option termination probability.
    """
    return 1.0 if self._steps_taken >= self._max_steps else 0.0


class SpyEnvironment(dm_env.Environment):
  """An environment for testing."""

  def __init__(self, episode_length: int = sys.maxsize):
    self.actions_received = []  # type: List[np.ndarray]
    self._episode_length = episode_length
    self._episode_step = -1
    self._global_step = -1
    trivial_arena = arena.Arena()
    self._task = composer.NullTask(trivial_arena)

  def reset(self) -> dm_env.TimeStep:
    self._episode_step = 0
    self.actions_received = []
    return self._timestep()

  def step(self, agent_action: np.ndarray) -> dm_env.TimeStep:
    self.actions_received.append(agent_action)
    self._episode_step += 1
    self._global_step += 1

    if self._episode_step > self._episode_length or self._episode_step == 0:
      return self.reset()
    else:
      return self._timestep()

  def _timestep(self) -> dm_env.TimeStep:
    step_type = self._step_type()
    observation = self._observation()

    return dm_env.TimeStep(
        step_type=step_type,
        reward=np.full(
            shape=self.reward_spec().shape,
            fill_value=0,
            dtype=self.reward_spec().dtype),
        discount=np.full(
            self.discount_spec().shape,
            fill_value=0,
            dtype=self.discount_spec().dtype),
        observation=observation,
    )

  def _step_type(self):
    if self._episode_step == 0:
      return dm_env.StepType.FIRST
    elif self._episode_step >= self._episode_length:
      return dm_env.StepType.LAST
    else:
      return dm_env.StepType.MID

  def _observation(self):
    if self.actions_received:
      last_action = self.actions_received[-1]
    else:
      last_action = np.full(
          shape=self.action_spec().shape,
          fill_value=-1,
          dtype=self.action_spec().dtype)

    return {
        'step_count':
            np.asarray([self._episode_step], dtype=np.float32),
        'global_step_count':
            np.asarray([self._global_step], dtype=np.float32),
        'last_action':
            np.copy(last_action)
    }

  def observation_spec(self):
    return {
        'step_count':
            specs.BoundedArray(
                shape=(1,),
                dtype=np.float32,
                minimum=[0],
                maximum=[sys.maxsize]),
        'global_step_count':
            specs.BoundedArray(
                shape=(1,),
                dtype=np.float32,
                minimum=[0],
                maximum=[sys.maxsize]),
        'last_action':
            self.action_spec()
    }

  def action_spec(self):
    return specs.BoundedArray(
        shape=(0,), dtype=np.float32, minimum=[], maximum=[])

  def get_step_count(self, timestep: dm_env.TimeStep):
    assert 'step_count' in timestep.observation
    step_count_array = timestep.observation['step_count']
    assert step_count_array.shape == (1,)
    return int(step_count_array[0])

  def get_global_step_count(self, timestep: dm_env.TimeStep):
    assert 'global_step_count' in timestep.observation
    step_count_array = timestep.observation['global_step_count']
    assert step_count_array.shape == (1,)
    return int(step_count_array[0])

  def get_last_action(self, timestep: dm_env.TimeStep):
    assert 'last_action' in timestep.observation
    return timestep.observation['last_action']

  @property
  def physics(self):
    return None

  def task(self):
    return self._task


def atomic_option_with_name(name, action_size=2):
  return af.FixedOp(
      action=random_action(action_size), name=name)


def random_action(action_size=2):
  return np.random.random(size=(action_size,))


def random_string(length=None) -> Text:
  length = length or random.randint(5, 10)
  return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def random_shape(ndims=None) -> Tuple[int, ...]:
  ndims = ndims or random.randint(1, 3)
  return tuple([random.randint(1, 10) for _ in range(ndims)])


def random_dtype() -> Type[np.floating]:
  return random.choice([float, np.float32, np.float64])


def unit_array_spec(shape: Optional[Tuple[int, ...]] = None,
                    name=None) -> specs.BoundedArray:
  shape = shape or random_shape()
  dtype = random.choice([np.float32, np.float64])
  minimum = np.zeros(shape=shape, dtype=dtype)
  maximum = np.ones(shape=shape, dtype=dtype)
  name = name or random_string()
  return specs.BoundedArray(shape, dtype, minimum, maximum, name)


def random_array_spec(
    shape: Optional[Tuple[int, ...]] = None,
    name: Optional[Text] = None,
    dtype: Optional[Type[np.floating]] = None,
    minimum: Optional[np.ndarray] = None,
    maximum: Optional[np.ndarray] = None) -> specs.BoundedArray:
  """Create BoundedArray spec with unspecified parts randomized."""

  shape = shape or random_shape()
  name = name or random_string()
  dtype = dtype or random.choice([np.float32, np.float64])
  if minimum is None:
    minimum = np.random.random(size=shape) * random.randint(0, 10)
  minimum = minimum.astype(dtype)
  if maximum is None:
    maximum = np.random.random(size=shape) * random.randint(0, 10) + minimum
  maximum = maximum.astype(dtype)
  return specs.BoundedArray(shape, dtype, minimum, maximum, name)


def random_observation_spec(
    size: Optional[int] = None,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[Type[np.floating]] = None) -> Dict[Text, specs.Array]:
  size = random.randint(3, 6) if size is None else size
  obs_spec = {}
  for _ in range(size):
    name = random_string(3)
    obs_spec[name] = random_array_spec(shape, name, dtype)
  return obs_spec


def random_step_type() -> dm_env.StepType:
  return random.choice(list(dm_env.StepType))


def valid_value(spec: Union[specs.Array, spec_utils.ObservationSpec,
                            spec_utils.TimeStepSpec]):
  """Returns a valid value from the primitive, dict, or timestep spec."""

  def valid_primitive(prim_spec):
    value = np.random.random(size=prim_spec.shape).astype(prim_spec.dtype)
    if isinstance(prim_spec, specs.BoundedArray):
      # Clip specs to handle +/- np.inf in the specs.
      maximum = np.clip(prim_spec.maximum, -1e10, 1e10)
      minimum = np.clip(prim_spec.minimum, -1e10, 1e10)
      value *= (maximum - minimum)
      value += minimum
    else:
      value *= 1e10  # Make range / magnitude assumptions unlikely to hold.
    return value.astype(prim_spec.dtype)

  if isinstance(spec, dict):
    return {k: valid_primitive(v) for k, v in spec.items()}
  elif isinstance(spec, specs.Array):
    return valid_primitive(spec)
  elif isinstance(spec, spec_utils.TimeStepSpec):
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=valid_value(spec.reward_spec),
        discount=valid_value(spec.discount_spec),
        observation=valid_value(spec.observation_spec))
  else:
    raise ValueError('bad spec, type: {}'.format(type(spec)))


def assert_value(expected, actual, path=None):
  """Fails if the expected and actual values are different."""
  if expected is actual:
    return
  path = path or ''
  msg = '\nexpected: {}\ngot: {}'.format(expected, actual)

  if isinstance(expected, np.ndarray):
    if not isinstance(actual, np.ndarray):
      raise AssertionError('array vs not at {}: {}'.format(path, msg))
    np.testing.assert_almost_equal(expected, actual)
  elif isinstance(expected, dict):
    if not isinstance(actual, dict):
      raise AssertionError('dict vs not at {}: {}'.format(path, msg))
    if sorted(expected.keys()) != sorted(actual.keys()):
      raise AssertionError('wrong keys at {}: {}'.format(path, msg))
    for key in expected.keys():
      assert_value(expected[key], actual[key], path + '/{}'.format(key))
  else:
    raise AssertionError('Bad type given: {}: {}'.format(type(expected), msg))


def assert_spec(expected: spec_utils.TimeStepSpec,
                actual: spec_utils.TimeStepSpec):
  return expected == actual


def assert_timestep(lhs: dm_env.TimeStep, rhs: dm_env.TimeStep):
  assert_value(lhs.observation, rhs.observation)
  np.testing.assert_almost_equal(lhs.reward, rhs.reward)
  np.testing.assert_almost_equal(lhs.discount, rhs.discount)
  if lhs.step_type != rhs.step_type:
    raise AssertionError('step types differ left: {}, right {}'.format(
        lhs, rhs))


def _call_string(call):
  """Converts the provided call to string."""
  positional_args = call[0]
  keyword_args = call[1]

  arg_strings = []
  if positional_args:
    arg_strings.append(', '.join([str(arg) for arg in positional_args]))
  if keyword_args:
    arg_strings.append(', '.join(
        [f'{k}={v}' for (k, v) in keyword_args.items()]))
  arg_string = ', '.join(arg_strings)
  return f'({arg_string})'


def _call_strings(calls):
  return [_call_string(call) for call in calls]


def _args_match(actual_arg, expected_arg, equals_fn):
  """Return True if actual_arg matched expected_arg."""
  if actual_arg is expected_arg:
    return True
  if actual_arg is None or expected_arg is None:
    return False  # They're not /both/ None.
  return equals_fn(actual_arg, expected_arg)


def call_matches(mock_call_obj,
                 expected_args: Tuple,  # pylint: disable=g-bare-generic
                 expected_kwargs: Dict[Text, Any],
                 equals_fn=None):
  """Return True if the args and kwargs in mock_call_obj are as expected."""
  if equals_fn is None:
    equals_fn = _equal_or_close

  mock_positional_args = mock_call_obj[0]
  mock_keyword_args = mock_call_obj[1]

  args_matches = [
      _args_match(*args, equals_fn=equals_fn)
      for args in zip(mock_positional_args, expected_args)
  ]

  if mock_keyword_args.keys() != expected_kwargs.keys():
    return False

  aligned_kwargs = [[mock_keyword_args[k], expected_kwargs[k]]
                    for k in mock_keyword_args.keys()]
  kwargs_matches = [
      _args_match(*args, equals_fn=equals_fn)
      for args in aligned_kwargs
  ]

  return all(args_matches) and all(kwargs_matches)


def assert_calls(mock_obj,
                 expected_args: Optional[List[Tuple]] = None,  # pylint: disable=g-bare-generic
                 expected_kwargs: Optional[List[Dict[Text, Any]]] = None,
                 equals_fn=None):
  """Checks that the calls made to the given match the args given.

  This function takes a mock function on which function calls may have occurred,
  and corresponding lists of args and kwargs for what those calls should have
  been.

  It then checks that the args and kwargs match the mock call list exactly,
  using np.testing.assert_equal for numpy arguments.

  It does not check the call order, but does check that there are no extra or
  missing calls.

  Args:
    mock_obj: The mock (function) to check calls on.
    expected_args: A list (one per call) of positional argument tuples.
    expected_kwargs: A list (one per call) of keyword argument dicts.
    equals_fn: Custom comparison function which will be called with potential
      argument pairs. If the default (None) is used than we will compare the
      objects using a default function which can handle numpy arrays and
      string-key specially, and deferrs everything else to ==.

  Returns:
    None

  Raises:
    AssertionError: if the calls do not match.
  """
  if expected_args is None and expected_kwargs is not None:
    expected_args = [()] * len(expected_kwargs)
  elif expected_args is not None and expected_kwargs is None:
    expected_kwargs = [{}] * len(expected_args)
  elif expected_args is not None and expected_kwargs is not None:
    assert len(expected_args) == len(expected_kwargs)
  expected_calls = list(zip(expected_args, expected_kwargs))
  actual_calls = mock_obj.call_args_list
  if len(actual_calls) != len(expected_args):
    raise AssertionError('Expected {} calls, but got {}'.format(
        len(expected_args), len(actual_calls)))

  found = [False] * len(actual_calls)
  matched = [False] * len(actual_calls)
  for expected_index, expected_call in enumerate(expected_calls):
    for actual_index, actual_call in enumerate(actual_calls):
      if not matched[actual_index] and call_matches(
          actual_call,
          expected_args=expected_call[0],
          expected_kwargs=expected_call[1],
          equals_fn=equals_fn):
        matched[actual_index] = True
        found[expected_index] = True
        break

  if not all(found):
    expected_not_found = [
        call for found, call in zip(found, expected_calls) if not found
    ]
    unmatched_actual = [
        call for matched, call in zip(matched, actual_calls) if not matched
    ]
    raise AssertionError(('Did not find all expected calls in actual calls.\n'
                          'Expected but not found:\n{}\n'
                          'Unmatched, actual calls:\n{}').format(
                              _call_strings(expected_not_found),
                              _call_strings(unmatched_actual)))


def random_reward_spec(shape: Optional[Tuple[int, ...]] = None,
                       dtype: Optional[Type[np.floating]] = None,
                       name: Text = 'reward') -> specs.Array:
  if not shape:
    shape = ()
  if not dtype:
    dtype = random_dtype()

  return specs.Array(shape=shape, dtype=dtype, name=name)


def random_discount_spec(shape: Optional[Tuple[int, ...]] = None,
                         dtype: Optional[Type[np.floating]] = None,
                         minimum: float = 0.,
                         maximum: float = 1.,
                         name: Text = 'discount') -> specs.BoundedArray:
  """Generate a discount spec."""
  if not shape:
    shape = ()
  if not dtype:
    dtype = random_dtype()
  return specs.BoundedArray(
      shape=shape, dtype=dtype, minimum=minimum, maximum=maximum, name=name)


def random_timestep_spec(
    observation_spec: Optional[Mapping[Text, specs.Array]] = None,
    reward_spec: Optional[specs.Array] = None,
    discount_spec: Optional[specs.BoundedArray] = None
    ) -> spec_utils.TimeStepSpec:
  """Generate a timestep spec."""
  if not observation_spec:
    observation_spec = random_observation_spec()
  if not reward_spec:
    reward_spec = random_reward_spec()
  if not discount_spec:
    discount_spec = random_discount_spec()

  return spec_utils.TimeStepSpec(
      observation_spec=observation_spec,
      reward_spec=reward_spec,
      discount_spec=discount_spec)


def random_timestep(spec: Optional[spec_utils.TimeStepSpec] = None,
                    step_type: Optional[dm_env.StepType] = None,
                    reward: Optional[np.floating] = None,
                    discount: Optional[np.floating] = None,
                    observation: Optional[spec_utils.ObservationValue] = None):
  """Create a timestep."""

  if spec and not observation:
    # Create a valid observation:
    observation = valid_value(spec.observation_spec)
  else:
    observation = observation or {}  # no spec => no observation.

  if step_type is None:
    step_type = random_step_type()

  if reward is None:
    if spec:
      reward = valid_value(spec.reward_spec)
    else:
      reward = random_dtype()(np.random.random())

  if discount is None:
    if spec:
      discount = valid_value(spec.discount_spec)
    else:
      discount = random_dtype()(np.random.random())

  timestep = dm_env.TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observation)

  # We should not return and invalid timestep.
  if spec:
    spec_utils.validate_timestep(spec, timestep)

  return timestep


class EnvironmentSpec(object):
  """Convenience class for creating valid timesteps and actions."""

  @classmethod
  def random(cls):
    """Create an EnvironmentSpec with randomly created specs."""

    ts_spec = random_timestep_spec()
    action_spec = random_array_spec()
    return EnvironmentSpec(timestep_spec=ts_spec, action_spec=action_spec)

  @classmethod
  def for_subtask(cls, subtask: af.SubTask):
    timestep_spec = spec_utils.TimeStepSpec(
        observation_spec=subtask.observation_spec(),
        reward_spec=subtask.reward_spec(),
        discount_spec=subtask.discount_spec())
    action_spec = subtask.action_spec()
    return EnvironmentSpec(timestep_spec=timestep_spec, action_spec=action_spec)

  def __init__(self, timestep_spec: spec_utils.TimeStepSpec,
               action_spec: specs.Array):
    self.action_spec = action_spec
    self.spec = timestep_spec

  def create_timestep(
      self,
      step_type: Optional[dm_env.StepType] = None,
      reward: Optional[np.floating] = None,
      discount: Optional[np.floating] = None,
      observation: Optional[spec_utils.ObservationValue] = None
  ) -> dm_env.TimeStep:
    return random_timestep(self.spec, step_type, reward, discount, observation)

  def create_action(self) -> np.ndarray:
    val = valid_value(self.action_spec)  # type: np.ndarray  # pytype: disable=annotation-type-mismatch
    return val


def composite_spec(*components: specs.BoundedArray) -> specs.BoundedArray:
  """Create a spec by composing / concatenating the given specs."""

  if not components:
    raise ValueError('No specs to compose')

  for spec in components:
    if len(spec.shape) != 1:
      raise ValueError('Not creating composite spec: not all shapes are 1-D')

  if not all(spec.dtype == components[0].dtype for spec in components):
    raise ValueError('not all dtypes match')

  shape = (sum(spec.shape[0] for spec in components),)
  dtype = components[0].dtype
  minimum = np.hstack([spec.minimum for spec in components])
  maximum = np.hstack([spec.maximum for spec in components])
  name = '\t'.join(spec.name for spec in components)
  return specs.BoundedArray(shape, dtype, minimum, maximum, name)
