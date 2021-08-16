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

"""Builder for a SubTaskEnvironment."""

from typing import Optional, Sequence

from dm_control import composer
from dm_robotics import agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
from dm_robotics.agentflow.subtasks import parameterized_subtask
from dm_robotics.moma import base_task
from dm_robotics.moma import effector
from dm_robotics.moma import moma_option
from dm_robotics.moma import subtask_env

# Internal profiling


class SubtaskEnvBuilder(object):
  """Builder for a SubTaskEnvironment."""

  def __init__(self):
    self._task = None  # type: Optional[base_task.BaseTask]
    self._base_env = None  # type: Optional[composer.Environment]
    self._action_space = None  # type: Optional[af.ActionSpace]
    self._preprocessors = []  # timestep_preprocessor.TimestepPreprocessor
    self._reset_option = None  # type: Optional[moma_option.MomaOption]
    self._effectors = None  # type: Optional[Sequence[effector.Effector]]

  def set_task(self, task: base_task.BaseTask) -> 'SubtaskEnvBuilder':
    self._task = task
    return self

  def set_action_space(self, action_space) -> 'SubtaskEnvBuilder':
    self._action_space = action_space
    return self

  def set_reset_option(self, reset_option: moma_option.MomaOption
                       ) -> 'SubtaskEnvBuilder':
    self._reset_option = reset_option
    return self

  def set_effectors(
      self, effectors: Sequence[effector.Effector]) -> 'SubtaskEnvBuilder':
    """Allow the effectors to be changed, post-construction."""

    self._effectors = effectors
    return self

  def add_preprocessor(
      self, preprocessor: timestep_preprocessor.TimestepPreprocessor
  ) -> 'SubtaskEnvBuilder':
    self._preprocessors.append(preprocessor)
    return self

  def build_base_env(self) -> composer.Environment:
    """Builds the base composer.Environment.

    Factored out as a separate call to allow users to build the base composer
    env before calling the top-level `build` method.  This can be necessary when
    preprocessors require parameters from the env, e.g. action specs.

    Returns:
      The base composer.Environment
    """
    if self._task is None:
      raise ValueError('Cannot build the base_env until the task is built')

    if self._base_env is None:
      self._base_env = composer.Environment(self._task,
                                            strip_singleton_obs_buffer_dim=True)

    return self._base_env

  # Profiling for .wrap()
  def build(self) -> subtask_env.SubTaskEnvironment:
    """Builds the SubTaskEnvironment from the set components."""
    # Ensure base_env has been built.
    base_env = self.build_base_env()

    parent_spec = spec_utils.TimeStepSpec(base_env.observation_spec(),
                                          base_env.reward_spec(),
                                          base_env.discount_spec())

    preprocessor = timestep_preprocessor.CompositeTimestepPreprocessor(
        *self._preprocessors)

    if self._action_space is None:
      raise ValueError(
          'Cannot build the subtask envrionment until the action space is set')

    subtask = parameterized_subtask.ParameterizedSubTask(
        parent_spec=parent_spec,
        action_space=self._action_space,
        timestep_preprocessor=preprocessor,
        name=self._task.name())

    # Use the specified effectors if they exist, otherwise default to using all
    # of the effectors in the base task.
    effectors = self._effectors or list(self._task.effectors)

    # Check if there are effectors.
    if not effectors:
      raise ValueError(
          'Cannot build the subtask envrionment if there are no effectors.')

    reset_option = (self._reset_option or
                    self._build_noop_reset_option(base_env, effectors))

    return subtask_env.SubTaskEnvironment(
        env=base_env,
        effectors=effectors,
        subtask=subtask,
        reset_option=reset_option)

  def _build_noop_reset_option(
      self, env: composer.Environment, effectors: Sequence[effector.Effector]
      ) -> moma_option.MomaOption:
    """Builds a no-op MoMa option."""

    parent_action_spec = self._task.effectors_action_spec(
        physics=env.physics, effectors=effectors)
    noop_action = spec_utils.zeros(parent_action_spec)
    delegate = af.FixedOp(noop_action, name='NoOp')

    return moma_option.MomaOption(
        physics_getter=lambda: env.physics,
        effectors=effectors,
        delegate=delegate)
