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
"""Module for defining a While Option for basic control flow."""

from typing import Callable, Optional, Text

import dm_env
from dm_robotics.agentflow import core
from dm_robotics.agentflow import util
from dm_robotics.agentflow.decorators import overrides
from dm_robotics.agentflow.options import basic_options
import numpy as np


class While(basic_options.DelegateOption):
  """An option which runs until a condition is False."""

  def __init__(self,
               cond: Callable[[dm_env.TimeStep], bool],
               delegate: core.Option,
               eval_every_step: bool = True,
               name: Optional[Text] = None) -> None:
    """Construct While Option.

    Per step `cond` will be called:
    * 0 times: if passed a LAST timestep (since the option is exiting anyway).
    * 0 times: if eval_every_step is False and the delegate doesn't terminate.
    * 1 time: if eval_every_step is False and the delegate terminates.
    * 1 time: if eval_every_step is True and the delegate doesn't terminate.
    * 2 times: if eval_every_step is True and the delegate terminates.

    Args:
      cond: A callable accepting the current timestep.
      delegate: The option to run while `cond` is True.
      eval_every_step: If True re-evaluate `cond` every step.  If False `cond`
        is evaluated only upon each termination.  See breakdown above. Note that
        the cond is not evaluated by pterm(), as this result is cached during
        step().
      name: A name for this option.
    """
    super().__init__(delegate=delegate, name=name)
    assert callable(cond)
    self._cond = cond
    self._eval_every_step = eval_every_step
    self._delegate_episode_ctr = 0
    self._continue = None

  @property
  def delegate_episode_ctr(self):
    return self._delegate_episode_ctr

  @overrides(core.Option)
  def pterm(self, timestep: dm_env.TimeStep):
    """Terminates if `cond` is False."""
    if self._continue is None:
      raise ValueError('pterm called before step')
    return 0.0 if self._continue else 1.0

  @overrides(core.Option)
  def step(self, timestep: dm_env.TimeStep):
    """Evaluates the condition if requested, and runs the selected option."""
    self._continue = True  # Flag used in pterm (T, F, None (error state))

    terminate_delegate = super().delegate.pterm(timestep) > np.random.rand()
    if terminate_delegate or self._eval_every_step:
      # Evaluate condition if delegate is terminal or asked to by user.
      self._continue = self._cond(timestep)
    terminate_loop = timestep.last() or not self._continue

    if terminate_delegate and not terminate_loop:
      # Terminate & restart delegate.
      last_timestep = timestep._replace(step_type=dm_env.StepType.LAST)
      last_action = self._delegate.step(last_timestep)
      self._delegate_episode_ctr += 1
      self._continue = self._cond(timestep)
      if self._continue:
        # Only restart if cond is still True.  This calls cond twice in a given
        # step, but is required in order to allow stateful conditions (e.g.
        # Repeat) to see the state after the delegate terminates.
        result = self._delegate.result(last_timestep)
        timestep = timestep._replace(step_type=dm_env.StepType.FIRST)
        self._delegate.on_selected(timestep, result)
      else:
        return last_action

    elif terminate_loop and not timestep.last():
      timestep = timestep._replace(step_type=dm_env.StepType.LAST)
      util.log_termination_reason(self, self.result(timestep))

    return super().step(timestep)


class Repeat(While):
  """An option which runs for a given number of iterations."""

  def __init__(self,
               num_iter: int,
               delegate: core.Option,
               name: Optional[Text] = None) -> None:
    """Construct Repeat Option.

    Args:
      num_iter: Number of times to repeat the option.
      delegate: The option to run for `num_iter` iterations.
      name: A name for this option.
    """
    self._num_iter = num_iter
    cond = lambda _: self.delegate_episode_ctr < num_iter
    super().__init__(
        cond=cond, delegate=delegate, eval_every_step=False, name=name)

  @property
  def num_iters(self) -> int:
    return self._num_iter
