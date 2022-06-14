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

"""An AgentFlow Subtask to Environment adaptor."""

import re
import traceback
from typing import Callable, List, Sequence

from absl import logging
from dm_control import composer
from dm_control import mjcf
import dm_env
from dm_robotics import agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.moma import base_task
from dm_robotics.moma import effector
from dm_robotics.moma import moma_option
import numpy as np

# Internal profiling


def _fixed_timestep(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  if timestep.reward is None:
    timestep = timestep._replace(reward=0.0)

  if timestep.discount is None:
    timestep = timestep._replace(discount=1.0)

  return timestep


class SubTaskEnvironment(dm_env.Environment):
  """SubTask to `dm_env.Environment` adapter.

  One important note is that users need to either call close() on the
  environment when they are done using it or they should use them within
  a context manager.

  Example:
  with subtask_env() as env:
    ...
  """

  def __init__(self,
               env: composer.Environment,
               effectors: Sequence[effector.Effector],
               subtask: af.SubTask,
               reset_option: moma_option.MomaOption):
    if env is None:
      raise ValueError('env is None')
    if subtask is None:
      raise ValueError('subtask is None')
    if reset_option is None:
      raise ValueError('reset_option is None')
    if not effectors:
      raise ValueError('no effectors specified')

    self._env = env
    self._effectors = effectors
    self._subtask = subtask
    self._reset_option = reset_option
    self._reset_required = True
    self._last_internal_timestep = None  # type: dm_env.TimeStep
    self._last_external_timestep = None  # type: dm_env.TimeStep

    # Stub action for the moma task step. Actual actuation is done directly
    # through effectors.
    self._stub_env_action = np.zeros_like(
        self._env.action_spec().generate_value())

    self._effectors_action_spec = None

    self._observers = []  # type: List [af.SubTaskObserver]
    self._teardown_callables = []  # type: List  [Callable[[], None]]
    self._is_closed = False  # type: bool
    # If the user does not close the environment we issue an error and
    # print out where the environment was created
    self._traceback = traceback.format_stack()

  def __del__(self):
    if not self._is_closed:
      logging.error(
          'You must call .close() on the environment created at:\n %s',
          ''.join(self._traceback))

  # Profiling for .wrap()
  def reset(self) -> dm_env.TimeStep:
    if self._is_closed:
      raise RuntimeError(
          'The environment has been closed, it can no longer be used.')

    env_timestep = _fixed_timestep(self._env.reset())
    self._reset_option.on_selected(timestep=env_timestep)

    # Run the reset option to completion.
    pterm = 0  # An option is stepped before we ask for pterm.
    while pterm < np.random.random():
      # The reset_option is a MomaOption which handles actuating the effectors
      # internally.
      self._reset_option.step(env_timestep)
      env_timestep = self._env.step(self._stub_env_action)
      pterm = self._reset_option.pterm(env_timestep)

    # Send LAST timestep to reset option's delegate. This means we want to
    # step the option without actuating the effectors. Ignore the action it
    # returns.
    self._reset_option.step_delegate(
        env_timestep._replace(step_type=dm_env.StepType.LAST))

    # send env_timestep through the subtask.
    self._last_internal_timestep = env_timestep._replace(
        step_type=dm_env.StepType.FIRST)
    self._subtask.reset(env_timestep)
    timestep = self._subtask.parent_to_agent_timestep(
        self._last_internal_timestep,
        own_arg_key=self._subtask.get_arg_key(None))

    self._reset_required = False
    if not timestep.first():
      raise ValueError(f'SubTask returned non FIRST timestep: {timestep}')

    timestep = timestep._replace(reward=None)
    timestep = timestep._replace(discount=None)

    self._last_external_timestep = timestep
    return timestep

  # Profiling for .wrap()
  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    if self._is_closed:
      raise RuntimeError(
          'The environment has been closed, it can no longer be used.')

    if self._reset_required:
      return self.reset()  # `action` is deliberately ignored.

    # subtask_env does not use Option-argument mechanism.
    dummy_arg_key = self._subtask.get_arg_key(None)

    pterm = self._subtask.pterm(self._last_internal_timestep, dummy_arg_key)

    action = action.astype(self._subtask.action_spec().dtype)

    external_action = np.clip(action,
                              self._subtask.action_spec().minimum,
                              self._subtask.action_spec().maximum)
    internal_action = self._subtask.agent_to_parent_action(external_action)

    for obs in self._observers:
      obs.step(
          parent_timestep=self._last_internal_timestep,
          parent_action=internal_action,
          agent_timestep=self._last_external_timestep,
          agent_action=external_action)

    self._actuate_effectors(internal_action)
    internal_timestep = self._env.step(self._stub_env_action)

    # If subtask wanted to stop, this is the last timestep.
    if pterm > np.random.random():
      internal_timestep = internal_timestep._replace(
          step_type=dm_env.StepType.LAST)

    self._last_internal_timestep = internal_timestep
    external_timestep = self._subtask.parent_to_agent_timestep(
        internal_timestep, dummy_arg_key)

    # If subtask or base env emit a LAST timestep, we need to reset next.
    if external_timestep.last():
      self._reset_required = True

      # For a LAST timestep, step the observers with a None action. This ensures
      # the observers will see every timestep of the task.
      for obs in self._observers:
        obs.step(
            parent_timestep=internal_timestep,
            parent_action=None,
            agent_timestep=external_timestep,
            agent_action=None)

    # This shouldn't happen, but just in case.
    if external_timestep.first():
      external_timestep = external_timestep._replace(reward=None)
      external_timestep = external_timestep._replace(discount=None)

    self._last_external_timestep = external_timestep
    return external_timestep

  def _actuate_effectors(self, action):
    if self._effectors_action_spec is None:
      aspecs = [a.action_spec(self.physics) for a in self._effectors]
      self._effectors_action_spec = spec_utils.merge_specs(aspecs)

    spec_utils.validate(self._effectors_action_spec, action)
    for ef in self._effectors:
      e_cmd = action[self._find_effector_indices(ef)]
      ef.set_control(self.physics, e_cmd)

  def _find_effector_indices(self, ef: effector.Effector) -> List[bool]:
    actuator_names = self._effectors_action_spec.name.split('\t')
    prefix_expr = re.compile(ef.prefix)
    return [re.match(prefix_expr, name) is not None for name in actuator_names]

  def observation_spec(self):
    return self._subtask.observation_spec()

  def action_spec(self):
    return self._subtask.action_spec()

  def reward_spec(self):
    return self._subtask.reward_spec()

  def discount_spec(self):
    return self._subtask.discount_spec()

  @property
  def base_env(self) -> composer.Environment:
    return self._env

  @property
  def physics(self) -> mjcf.Physics:
    return self._env.physics

  @property
  def task(self) -> composer.Task:
    """Returns the underlying composer.Task, defining the world."""
    return self._env.task

  @property
  def subtask(self) -> af.SubTask:
    """Returns the underlying agentflow.SubTask, defining the task."""
    return self._subtask

  @property
  def reset_option(self) -> moma_option.MomaOption:
    return self._reset_option

  @reset_option.setter
  def reset_option(self, reset_option: moma_option.MomaOption) -> None:
    """Changes the reset option.

    Sometimes the correct reset option is not constructible when the
    SubTaskEnvironment is initialized, so this property allows the caller to
    overwrite the original reset option.

    Args:
      reset_option: New reset option for this environment.
    """
    self._reset_option = reset_option

  def add_observer(self, observer: af.SubTaskObserver) -> None:
    """Adds a subtask observer to the environment."""
    self._observers.append(observer)

  def add_teardown_callable(self, teardown_fn: Callable[[], None]):
    """Adds a function to be called when the environment is closed.

    When running our environment we might need to start processes that need
    to be closed when we are done using the environment.

    Args:
      teardown_fn: Function to run when we close the environment.
    """
    self._teardown_callables.append(teardown_fn)

  def close(self) -> None:
    """Cleanup when we are done using the environment."""
    if self._is_closed:
      logging.warning('The environment has already been closed.')
      return

    # A MoMa base tasks holds all the effectors and sensors. When running a
    # real environment we need to make sure that we close all the sensors and
    # effectors used with the real robot. The `close` method of the task ensures
    # this.
    if isinstance(self.task, base_task.BaseTask):
      self.task.close()

    # Call all the provided teardowns when closing the environment.
    for teardown_callable in self._teardown_callables:
      teardown_callable()

    # Close the base class
    super().close()

    self._is_closed = True
