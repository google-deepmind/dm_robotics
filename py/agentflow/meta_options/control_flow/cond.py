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
"""Module for defining a Cond Option for basic control flow."""

from typing import Optional, Text, Callable, Iterable, Set

import dm_env

from dm_robotics.agentflow import core
from dm_robotics.agentflow.decorators import overrides
from dm_robotics.agentflow.options import basic_options


class Cond(basic_options.DelegateOption):
  """An option which branches on a condition to one of two delegates.

  The condition can be evaluated on option-selection or every timestep.
  """

  def __init__(self,
               cond: Callable[[dm_env.TimeStep, Optional[core.OptionResult]],
                              bool],
               true_branch: core.Option,
               false_branch: core.Option,
               eval_every_step: bool = False,
               name: Optional[Text] = None) -> None:
    """Initialize Cond.

    Args:
      cond: A callable accepting the current timestep, and the result of the
        last option - if one exists.
      true_branch: The option to run if `cond` is True.
      false_branch: The option to run if `cond` is False.
      eval_every_step: If True evaluate `cond` and re-select every step.  If
        False (default) `cond` is evaluated once when the option is selected.
      name: A name for this option.
    """
    super(Cond, self).__init__(delegate=true_branch, name=name)
    assert callable(cond)
    self._cond = cond
    self._true_branch = true_branch
    self._false_branch = false_branch
    self._eval_every_step = eval_every_step

    self._options_selected: set(core.Option) = set()
    self._just_switched = False

  def _maybe_other_branch(self) -> Optional[core.Option]:
    """Returns the non-active branch only if it has been selected previously."""
    cur, tb, fb = self.delegate, self._true_branch, self._false_branch
    other_option = fb if cur is tb else tb
    if other_option in self._options_selected:
      return other_option
    else:
      return None

  def _select_option(
      self,
      timestep: dm_env.TimeStep,
      prev_option_result: Optional[core.OptionResult] = None) -> None:
    """Evaluates condition and activates the appropriate option."""
    prev_option = self.delegate
    if self._cond(timestep, prev_option_result):
      self.delegate = self._true_branch
    else:
      self.delegate = self._false_branch

    # Keep track of which options have been selected for later termination.
    self._options_selected.add(self.delegate)
    self._just_switched = prev_option is not self.delegate

  def _switch_step_options(self, on_option: core.Option,
                           off_option: core.Option,
                           timestep: dm_env.TimeStep) -> None:
    """Last-step off_option and first-step on_option."""
    # End old option.
    last_timestep = timestep._replace(step_type=dm_env.StepType.LAST)
    off_option.step(last_timestep)
    result = off_option.result(timestep)

    # Start new option.
    on_option.on_selected(timestep, result)

  def child_policies(self) -> Iterable[core.Policy]:
    return [self._true_branch, self._false_branch]

  @property
  def true_branch(self):
    return self._true_branch

  @property
  def false_branch(self):
    return self._false_branch

  @property
  def options_selected(self) -> Set[core.Option]:
    return self._options_selected

  @overrides(core.Option)
  def on_selected(
      self,
      timestep: dm_env.TimeStep,
      prev_option_result: Optional[core.OptionResult] = None) -> None:
    self._select_option(timestep, prev_option_result)
    return super(Cond, self).on_selected(timestep, prev_option_result)

  def step(self, timestep: dm_env.TimeStep):
    """Evaluates the condition if requested, and runs the selected option."""

    if (
        ((timestep.first() and not self._options_selected)
         or self._eval_every_step or not self._options_selected)
        and not timestep.last()  # <-- Never switch options on a LAST step.
    ):
      # TODO(b/186731743): Test selecting an option for 1st time on a LAST step.
      self._select_option(timestep, None)

    if timestep.last():
      # Clean up other branch if appropriate.
      other_branch = self._maybe_other_branch()
      if other_branch is not None:
        other_branch.step(timestep)
      # Re-initialized the set of option selected.
      self._options_selected: set(core.Option) = set()

    if timestep.mid() and self._eval_every_step and self._just_switched:
      # Step through option lifecycle appropriately.
      other_branch = self._maybe_other_branch()
      if other_branch is not None:
        self._switch_step_options(self.delegate, other_branch, timestep)
        # Convert timestep to FIRST for stepping the newly-selected option.
        timestep = timestep._replace(step_type=dm_env.StepType.FIRST)

    # Step the active option.
    return super(Cond, self).step(timestep)

  @overrides(core.Option)
  def result(self, timestep: dm_env.TimeStep) -> core.OptionResult:
    return super(Cond, self).result(timestep)
