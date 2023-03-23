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
"""Core option implementations.  See class comments for details."""

import abc
import copy
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from absl import logging
import dm_env
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import overrides
import numpy as np

OptionPterms = Iterable[Tuple[core.Option, float]]


def any_terminates(pterms: OptionPterms) -> float:
  """Get the probability of any option terminating from options P(terminate).

  P(continue) for each option is calculated as `1.0 - pterm`.
  The P(continue) values are multiplied together and then a P(terminate) is
  returned as the remaining probability: `1.0 - P(continue)`

  Args:
    pterms: Iterable of (option, P(terminate)).
  Returns:
    Probability of any option terminating.
  """

  # Calculate pcont from pterm.
  pconts = [(opt, 1.0 - pterm) for opt, pterm in pterms]

  # Log which options may cause termination (assumes binary pterms).
  if logging.level_debug():
    for option, pcont in pconts:
      if pcont < 0.5:
        logging.debug('Option %s P(cont): %f', option.name, pcont)

  pcont_product = np.prod([pcont for _, pcont in pconts])
  return 1 - pcont_product


def all_terminate(pterms: OptionPterms) -> float:
  """Returns the probability of all options terminating.

  Args:
    pterms: Iterable of (option, P(terminate))
  Returns:
    The product of termination probabilities.
  """

  return np.prod([pterm for _, pterm in pterms])


def options_terminate(*options: core.Option) -> Callable[[OptionPterms], float]:
  """Returns a callable that gives a termination probability.

  The returned callable calculates the probability of all the given options
  terminating.

  This can be useful when creating a `ConcurrentOption` when you want more
  control over the termination of that option. `any_terminates` and
  `all_terminate` operate on *all* the options in the ConcurrentOption, this
  function can be used to terminate when *specific* options terminate, E.g.

  ```
  af.ConcurrentOption(
      option_list=[move_left, move_right, hold_others],
      termination=options_terminate(move_left, move_right))
  ```

  In this case, the option will terminate when both move_left and move_right
  want to terminate (their P(termination) values are combined), regardless of
  `hold_others.pterm()`.

  Args:
    *options: The options to combine the P(termination) values of.
  """

  def pterm(pterms: OptionPterms) -> float:
    """Returns product of input P(terminate) values."""
    return np.prod([pterm for option, pterm in pterms if option in options])

  return pterm


class FixedOp(core.Option):
  """A fixed action, that always succeeds and is valid everywhere.

  The action it returns is fixed, whatever the input timestep is.  Used when
  building tasks that require an action placeholder (not in the tf-sense).
  """

  def __init__(self,
               action: np.ndarray,
               num_steps: Optional[int] = 0,
               name: Optional[str] = None):
    """Initialized FixedOp.

    Args:
      action: The action to return.
      num_steps: The number of steps to run before requesting termination. If
        None, pterm is always zero.
      name: A name for this Option.
    """
    super().__init__(name=name)
    self._action = action
    self._num_steps = num_steps
    self._steps_remaining = num_steps or 0

  @overrides(core.Option)
  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    if timestep.first():
      self._steps_remaining = self._num_steps or 0

    self._steps_remaining -= 1
    return self._action

  def pterm(self, timestep) -> float:
    return 0.0 if (self._num_steps is None) else (self._steps_remaining <= 0)

  def set_action(self, action: np.ndarray):
    self._action = action


class RandomOption(core.Option):
  """An option that generates uniform random actions for a provided spec.

  The pterm of this option is always zero.
  """

  def __init__(self,
               action_spec: specs.BoundedArray,
               random_state: Optional[np.random.RandomState] = None,
               name: Optional[str] = None):
    """Initializer.

    Args:
      action_spec: Expected output action specification.
      random_state: Pseudo RNG state - optional.
      name: A name for this Option.
    """
    super().__init__(name=name)
    self._action_spec = action_spec
    self._random_state = random_state or np.random.RandomState()

  @overrides(core.Option)
  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    del timestep
    return self._random_state.uniform(
        low=self._action_spec.minimum,
        high=self._action_spec.maximum,
        size=self._action_spec.shape).astype(self._action_spec.dtype)

  def pterm(self, timestep) -> float:
    del timestep
    return 0.


class ConcurrentOption(core.Option):
  """Container-option which steps multiple options in order on each step.

  Use to implement concurrent behaviors or sequential processing pipelines,
  e.g. injecting perceptual features and acting on them.

  On each step() of the ConcurrentOption, all the options within the
  ConcurrentOption are stepped once. Those options are stepped in the order
  that they are passed to `__init__` so one can depend on the output of another.

  Merging behaviour:
  * Actions: The actions emitted by this option come from merging the output of
    its constituent options. see `spec_utils.merge_primitives`
  * Option arguments: This option creates an argument spec from the argument
    specs of the options it's created from. On each step the arguments (if any)
    for each sub-option are separated and passed to that option.
  * Result: Termination reasons are 'escalated' so if any option returns a
    termination type of FAILURE, the whole option returns this termination type.
    The option's result values are combined in a list.
  """

  def __init__(self,
               options_list: Sequence[core.Option],
               action_spec: specs.Array,
               name: Optional[str] = None,
               termination: Optional[Callable[[OptionPterms], float]] = None,
               allow_nan_actions: bool = False):
    """ConcurrentOption constructor.

    Args:
      options_list: The options to run. They're stepped in this order.
      action_spec: Expected output action specification.
      name: Name of the option.
      termination: Configures how the option terminates, `any_terminates` is the
        default, which means that if any sub-option terminates this option will
        terminate.
      allow_nan_actions: Whether this option can emit actions with NaNs or not.
        If this is False, the options within this option can still emit NaN
        values, as long as there are no NaNs after those actions are combined
        with `spec_utils.merge_primitives`.
    Raises:
      ValueError: If no options are given.
    """
    if not options_list:
      raise ValueError('options_list should have non-zero length')

    super().__init__(name=name)
    self._options_list = options_list
    self._action_spec = action_spec
    self._termination = termination or any_terminates
    self._ignore_nans = allow_nan_actions

    self._arg_spec = None  # type: core.ArgSpec
    self._child_args = {}  # type: Dict[int, core.Arg]

  @property
  def options_list(self):
    return self._options_list

  def child_policies(self):
    return self.options_list

  @overrides(core.Option)
  def arg_spec(self) -> core.ArgSpec:
    """Returns an argument specification for the option.

    Returns:
      The arg specs for each child/sub option, merged into one spec.
    """
    if self._arg_spec:
      return self._arg_spec

    child_specs = []
    for child in self._options_list:
      child_spec = child.arg_spec()
      if child_spec is not None:
        if not isinstance(child_spec, specs.BoundedArray):
          child_spec = specs.BoundedArray(
              shape=child_spec.shape,
              dtype=child_spec.dtype,
              minimum=np.ones_like(child_spec.generate_value()) * -np.inf,
              maximum=np.ones_like(child_spec.generate_value()) * np.inf)
        child_specs.append(child_spec)

    self._arg_spec = spec_utils.merge_specs(child_specs)
    return self._arg_spec

  def _get_child_arg(self, timestep: dm_env.TimeStep,
                     child_index: int) -> Optional[core.Arg]:
    if child_index in self._child_args:
      return self._child_args[child_index]

    child_args = timestep.observation.get(self.arg_key, None)
    if child_args is None:
      return None

    target_spec = self._options_list[child_index].arg_spec()
    if target_spec is None:
      return None

    start_idx = 0
    for i in range(child_index):
      child_spec = self._options_list[i].arg_spec()
      if child_spec is not None:
        start_idx += child_spec.shape[0]

    end_idx = start_idx + target_spec.shape[0]

    self._child_args[child_index] = child_args[start_idx:end_idx]
    return self._child_args[child_index]

  def _get_child_timestep(self, timestep: dm_env.TimeStep,
                          child_index: int) -> dm_env.TimeStep:
    child_arg = self._get_child_arg(timestep, child_index)
    if child_arg is None:
      return timestep

    child = self._options_list[child_index]
    child_observation = copy.copy(timestep.observation)
    child_observation[child.arg_key] = child_arg
    return timestep._replace(observation=child_observation)

  @overrides(core.Policy)
  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    actions = self._get_child_actions(timestep)
    output_action = spec_utils.merge_primitives(actions)
    spec_utils.validate(
        self._action_spec, output_action, ignore_nan=self._ignore_nans)
    return output_action

  def _get_child_actions(self, timestep: dm_env.TimeStep) -> List[np.ndarray]:
    actions = []  # type: List[np.ndarray]
    for index, opt in enumerate(self._options_list):
      actions.append(opt.step(self._get_child_timestep(timestep, index)))
    return actions

  @overrides(core.Option)
  def on_selected(
      self,
      timestep: dm_env.TimeStep,
      prev_option_result: Optional[core.OptionResult] = None) -> None:
    for index, opt in enumerate(self._options_list):
      opt_timestep = self._get_child_timestep(timestep, index)
      opt.on_selected(opt_timestep, prev_option_result)

  @overrides(core.Option)
  def pterm(self, timestep: dm_env.TimeStep) -> float:
    """Calculate pterm from the termination condition."""
    pterms = {}  # type: Dict[core.Option, float]
    for index, opt in enumerate(self._options_list):
      opt_timestep = self._get_child_timestep(timestep, index)
      pterms[opt] = opt.pterm(opt_timestep)

    return self._termination(pterms.items())

  @overrides(core.Option)
  def result(self, timestep: dm_env.TimeStep) -> core.OptionResult:
    """Returns result and termination reason.

    The returned termination_reason is the max of the termination_reason of the
      options list. This model assumes that termination_reason is coded in order
      of increasing priority, and takes advantage that `None` (the default when
      not terminating) evaluates to min.

    Args:
      timestep: input timestep.
    """

    termination_reason = None
    result_data = []

    for index, opt in enumerate(self._options_list):
      opt_timestep = self._get_child_timestep(timestep, index)
      result = opt.result(opt_timestep)
      result_data.append(result.data)
      if termination_reason is None:
        termination_reason = result.termination_reason
      else:
        termination_reason = max(termination_reason, result.termination_reason)

    return core.OptionResult(termination_reason, result_data)

  @overrides(core.Option)
  def render_frame(self, canvas) -> None:
    for opt in self._options_list:
      opt.render_frame(canvas)


class PolicyAdapter(core.Policy):
  """A policy that delegates `step` to a given object.

  Used to up-class an arbitrary agent-like object to be usable as a `Policy`.
  """

  def __init__(self, delegate: Any):
    super().__init__()
    self._delegate = delegate

  def child_policies(self) -> Iterable[core.Policy]:
    return [self._delegate]

  def step(self, timestep: dm_env.TimeStep):
    return self._delegate.step(timestep)

  def render_frame(self, canvas) -> None:
    # Pass-through `render_frame` call if available
    if callable(getattr(self._delegate, 'render_frame', None)):
      self._delegate.render_frame(canvas)


class OptionAdapter(core.Option):
  """An Option that delegates `step` to a given object.

  Used to up-class an arbitrary agent-like object to be usable as an `Option`.
  Note that this Option will never terminate.
  """

  def __init__(self, delegate: Any):
    super().__init__()
    self._delegate = delegate

  def child_policies(self) -> Iterable[core.Policy]:
    return [self._delegate]

  def step(self, timestep: dm_env.TimeStep):
    return self._delegate.step(timestep)

  def render_frame(self, canvas) -> None:
    # Pass-through `render_frame` call if available
    if callable(getattr(self._delegate, 'render_frame', None)):
      self._delegate.render_frame(canvas)


class DelegateOption(core.Option, abc.ABC):
  """An Option that delegates all methods to a given option."""

  def __init__(self, delegate: core.Option, name: Optional[str] = None):
    super().__init__(name=name)
    self._delegate = delegate  # subclasses may overwrite, e.g. Cond.
    self._name = name  # Overwrite; delegates if None.

  @property
  def name(self):
    return self._name or self._delegate.name

  def child_policies(self) -> Iterable[core.Policy]:
    return [self._delegate]

  @property
  def key_prefix(self) -> str:
    return self._delegate.key_prefix

  @property
  def arg_key(self) -> str:
    return self._delegate.arg_key

  def arg_spec(self) -> Optional[core.ArgSpec]:
    return self._delegate.arg_spec()

  def on_selected(self, timestep, prev_option_result=None):
    self._delegate.on_selected(timestep, prev_option_result)

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    return self._delegate.step(timestep)

  def pterm(self, timestep: dm_env.TimeStep) -> float:
    return self._delegate.pterm(timestep)

  def result(self, timestep: dm_env.TimeStep) -> core.OptionResult:
    return self._delegate.result(timestep)

  def render_frame(self, canvas):
    self._delegate.render_frame(canvas)

  @property
  def delegate(self) -> core.Option:
    return self._delegate

  @delegate.setter
  def delegate(self, delegate: core.Option):
    self._delegate = delegate

  def __str__(self):
    return f'DelegateOption({str(self._delegate)})'

  def __repr__(self):
    return f'DelegateOption({repr(self._delegate)})'


class LambdaOption(DelegateOption):
  """An option which can wrap another option and invoke various callables.

  The user can specify callables to be invoked at any of the following times:
  1) When `on_selected` is called.
  2) On every step()
  3) On `pterm`, to override the termination signal from the delegate.

  The value returned by `on_selected_func` can be inserted as the data in the
  `OptionResult` returned from the delegate by setting `func_as_result`.
  If the returned value is None, a warning is emitted.

  The action itself is delegated to the wrapped option, along with `pterm` and
  `result` if not explicitly overridden by the appropriate callables.
  """

  def __init__(self,
               delegate: core.Option,
               func_as_result: bool = False,
               on_selected_func: Optional[Callable[
                   [dm_env.TimeStep, Optional[core.OptionResult]], Any]] = None,
               on_step_func: Optional[Callable[[dm_env.TimeStep], Any]] = None,
               pterm_func: Optional[Callable[[dm_env.TimeStep], float]] = None,
               name: Optional[str] = None,
               **kwargs) -> None:
    """Construct LambdaOption.

    Args:
      delegate: An option to delegate option behavior to.
      func_as_result: If True, pack the output of `on_selected_func` in the
        OptionResult. If the on_selected_func returns a
        core.OptionResult, that will be used directly and the result.
      on_selected_func: A callable to invoke when the option is selected.
      on_step_func: A callable to invoke when the option is stepped.
      pterm_func: Optional function which overrides the pterm of the delegate.
      name: Name of the option.
      **kwargs: Unused keyword arguments.
    """
    super().__init__(delegate=delegate, name=name)
    if on_selected_func is not None:
      assert callable(on_selected_func)
    if on_step_func is not None:
      assert callable(on_step_func)
    if pterm_func is not None:
      assert callable(pterm_func)
    self._on_selected_func = on_selected_func
    self._on_step_func = on_step_func
    self._pterm_func = pterm_func
    self._func_as_result = func_as_result
    self._func_output = None  # type: Any

  @overrides(core.Option)
  def on_selected(
      self,
      timestep: dm_env.TimeStep,
      prev_option_result: Optional[core.OptionResult] = None) -> None:
    if self._on_selected_func is not None:
      # Store result to return from `result`.
      self._func_output = self._on_selected_func(timestep, prev_option_result)
    return self._delegate.on_selected(timestep, prev_option_result)

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    if self._on_step_func is not None:
      self._on_step_func(timestep)

    return super().step(timestep)

  @overrides(core.Option)
  def result(self, timestep: dm_env.TimeStep) -> Optional[core.OptionResult]:
    delegate_result = super().result(timestep)
    if self._func_as_result:
      if delegate_result is not None and delegate_result.data is not None:
        logging.warning('Discarding delegate option result: %s',
                        delegate_result)

      if isinstance(self._func_output, core.OptionResult):
        return self._func_output
      else:
        # Pack result into an OptionResult.
        return core.OptionResult(
            termination_reason=core.TerminationType.SUCCESS,
            data=self._func_output)
    else:
      return delegate_result

  @overrides(core.Option)
  def pterm(self, timestep: dm_env.TimeStep) -> float:
    if self._pterm_func is not None:
      return self._pterm_func(timestep)
    else:
      return super().pterm(timestep)


class PadOption(DelegateOption):
  """An Option that applies an `ActionSpace` to another option.

  This can be used to convert an action for part of the environment (e.g. the
  gripper) into an action for the whole environment (e.g. arm and gripper).
  """

  def __init__(self, delegate: core.Option, action_space: core.ActionSpace,
               **kwargs):
    super().__init__(delegate, **kwargs)
    self._action_space = action_space

  def step(self, timestep) -> np.ndarray:
    action = self._delegate.step(timestep)
    return self._action_space.project(action)


class ArgAdaptor(DelegateOption):
  """An option that adapts the argument for a wrapped option.

  This is helpful when composing options with different arg_specs in a parent
  that requires a consistent arg_spec for all children (see TensorflowMdpPolicy)

  E.g.:
  >>> base_spec = specs.Array(shape=(4,), dtype=np.float32)
  >>> sub_spec = specs.Array(shape=(2,), dtype=np.float32)
  >>> op_with_spec = SomeOption() # Expects sub_spec
  >>> adapted_op = ArgAdaptor(    # Expects base_spec
  >>>         op_with_spec, base_spec, lambda: arg: arg[:2])
  >>> adapted_op.step(timestep)  # op_with_spec will see sliced arg.
  """

  def __init__(self,
               delegate: core.Option,
               arg_spec: core.ArgSpec,
               adaptor_func: Callable[[core.Arg], Optional[core.Arg]],
               name: Optional[str] = None):
    """Initialize ArgAdaptor.

    Args:
      delegate: An option to delegate option behavior to.
      arg_spec: An arg_spec for the context in which this option will run.
      adaptor_func: A callable that takes an arg matching `arg_spec` and returns
        an arg matching `delegate.arg_spec`.
      name: Name of the option.
    """
    super().__init__(delegate=delegate, name=name)
    self._arg_spec = arg_spec
    self._adaptor_func = adaptor_func

  def arg_spec(self) -> Optional[core.ArgSpec]:
    return self._arg_spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Adapts the argument and steps the delegate with the modified argument."""
    adapted_observation = copy.copy(timestep.observation)
    child_spec = self.delegate.arg_spec()
    if child_spec is None:
      # Remove the arg.
      if self.arg_key in timestep.observation:
        adapted_observation.pop(self.arg_key)
    else:
      # Adapt to spec of delegate.
      initial_arg = timestep.observation.get(self.arg_key, None)
      adapted_arg = self._adaptor_func(initial_arg)
      if adapted_arg is None:
        raise ValueError(f'Delegate expects arg matching {child_spec} but '
                         'adaptor_func generated `None`.')
      else:
        spec_utils.validate(child_spec, adapted_arg)
      adapted_observation[self.arg_key] = adapted_arg

    timestep = timestep._replace(observation=adapted_observation)

    return super().step(timestep)


class IgnoreErrorDelegateOption(DelegateOption):
  """A DelegateOption that ignores the result of the delegate."""

  def result(self, timestep: dm_env.TimeStep) -> core.OptionResult:
    del timestep
    return core.OptionResult(termination_reason=core.TerminationType.SUCCESS)
