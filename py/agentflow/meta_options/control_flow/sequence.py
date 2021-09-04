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
"""Module for an agent to run options in a sequence."""

import typing
from typing import Any, List, Optional, Text, Tuple

import dm_env
from dm_robotics.agentflow import core
from dm_robotics.agentflow import util
from dm_robotics.agentflow.decorators import overrides
import numpy as np

OptionTimestep = typing.NewType("OptionTimestep", Any)


class Step(typing.NamedTuple):
  option: core.Option
  action: np.ndarray
  result: core.OptionResult


class Sequence(core.MetaOption):
  """An Option that executes options in a sequence.

  Sequence runs each Option from a list in order until one fails, or the final
  one succeeds.
  """

  def __init__(self,
               option_list: List[core.Option],
               terminate_on_option_failure: bool = False,
               allow_stepping_after_terminal: bool = True,
               name: Optional[Text] = None):
    """Initialize Sequence.

    Args:
      option_list: A list of Options to run.
      terminate_on_option_failure: If True exits with a `FAILURE` code when any
        option fails.  If False it proceeds to the next option.
      allow_stepping_after_terminal: If True, allows this option to be stepped
        even after it has requested termination and received a LAST timestep.
        This is required when the option is driven by a run-loop that doesn't
        respect the option life-cycle (e.g. a standard run loop).
      name: A name for this Option.
    """
    super().__init__(name=name)
    self._option_list = option_list
    self._terminate_on_option_failure = terminate_on_option_failure
    self._allow_stepping_after_terminal = allow_stepping_after_terminal
    self._initialize()

  def _initialize(self):
    self._option_idx = 0
    self._current_option = None  # type: core.Option
    self._current_option_result = core.OptionResult.success_result()
    self._current_option_first_step = None  # type: bool

    self._previous_option = None  # type: core.Option
    self._previous_option_result = core.OptionResult.success_result()
    self._terminate_option = False
    self._terminate_sequence = False
    self._logged_termination = False

  def _make_current_option_previous(self):
    self._previous_option = self._current_option
    self._previous_option_result = self._current_option_result

    self._current_option = None
    self._current_option_result = core.OptionResult.success_result()
    self._option_idx += 1

  def _select_option(self, timestep: dm_env.TimeStep):
    if self._option_idx >= len(self._option_list):
      # If no options available just return; step() will handle last timestep.
      return

    self._current_option = self._option_list[self._option_idx]
    self._current_option_result = core.OptionResult.success_result()
    self._current_option_first_step = True

  def _step_current_option(
      self, timestep: dm_env.TimeStep) -> Tuple[np.ndarray, bool, bool]:
    if self._current_option is None:
      raise ValueError("No current option")

    option_timestep = OptionTimestep(timestep)

    if self._current_option_first_step:
      option_timestep = option_timestep._replace(
          step_type=dm_env.StepType.FIRST)
      self._current_option.on_selected(option_timestep,
                                       self._previous_option_result)
      self._current_option_first_step = False

    action = self._current_option.step(option_timestep)
    pterm = self._current_option.pterm(option_timestep)

    terminate_option = pterm > np.random.uniform()
    terminate_sequence = False
    if terminate_option:
      self._current_option_result = self._current_option.result(option_timestep)
      terminate_early = (
          self._terminate_on_option_failure and
          self._current_option_result.termination_reason ==
          core.TerminationType.FAILURE)
      if terminate_early:
        # If the option failed and the user asked, terminate the whole sequence.
        terminate_sequence = True

    return action, terminate_option, terminate_sequence

  def _step_completed(self) -> None:
    if not self._terminate_option and self._current_option_first_step:
      self._current_option_first_step = False

  def _last_step_previous_option(
      self, timestep: dm_env.TimeStep) -> np.ndarray:
    prev_option = self._previous_option
    if prev_option is None:
      raise ValueError("Expected previous option is None.")

    timestep = OptionTimestep(timestep)
    termination_timestep = timestep._replace(step_type=dm_env.StepType.LAST)
    action = prev_option.step(termination_timestep)

    return action

  def arg_spec(self):
    return  # Sequence cannot take runtime arguments.

  def on_selected(self, timestep, prev_option_result=None):
    self._initialize()

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    if timestep.first():
      self._initialize()  # in case `on_selected` not called (top-level agent).
      self._select_option(timestep)

    on_last_option = self._option_idx == len(self._option_list) - 1
    block_termination = ((on_last_option or self._terminate_sequence) and
                         not timestep.last())
    if self._terminate_option and not block_termination:
      # Advance to next option iff parent respected pterm and sent LAST timestep
      util.log_termination_reason(self._current_option,
                                  self._current_option_result)
      self._make_current_option_previous()
      self._select_option(timestep)

      action = self._last_step_previous_option(timestep)
      self._terminate_option = False
      if timestep.last():  # Done.
        return action

    if self._current_option is None:
      if self._allow_stepping_after_terminal:
        # Agentflow options will nominally pass only a single LAST step when
        # options become terminal, but the top-level run loop may continue to
        # drive with MID timesteps (which become LAST if the option is >1 level
        # deep in the hierarchy).  We can allow this by stepping the last child,
        # if the user wishes not to treat as an error.
        return self._last_step_previous_option(timestep)
      raise ValueError(f"{str(self)} is terminal but was stepped. Is "
                       f"this agent in a non-agentflow run_loop?")

    action, self._terminate_option, self._terminate_sequence = (
        self._step_current_option(timestep))

    self._step_completed()
    return action

  def pterm(self, timestep) -> float:
    terminate_early = self._terminate_sequence
    out_of_options = self._option_idx >= len(self._option_list)
    on_last_option = self._option_idx == len(self._option_list) - 1
    last_option_finished = False
    if on_last_option:
      if self._current_option is None:
        return 0.  # If not FIRST stepped should still be able to query pterm.
      pterm = self._current_option.pterm(OptionTimestep(timestep))
      last_option_finished = pterm > np.random.uniform()
    if terminate_early or out_of_options or last_option_finished:
      return 1.
    return 0.

  def result(self, unused_timestep) -> core.OptionResult:
    return self._previous_option_result

  def child_policies(self):
    return self._option_list

  @property
  def terminate_on_option_failure(self) -> bool:
    return self._terminate_on_option_failure

  @overrides(core.Option)
  def render_frame(self, canvas) -> None:
    if self._current_option:
      self._current_option.render_frame(canvas)
