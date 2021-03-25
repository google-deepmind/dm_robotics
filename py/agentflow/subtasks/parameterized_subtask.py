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
"""A subtask with customizable timestep preprocessors and termination functions."""

from typing import Any, Callable, Optional, Text

import dm_env
from dm_env import specs
from dm_robotics import agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import overrides
from dm_robotics.agentflow.preprocessors import timestep_preprocessor as processing
import numpy as np


class ParameterizedSubTask(af.SubTask):
  """A subtask with injectable components for actions, obs, and termination."""

  def __init__(
      self,
      parent_spec: spec_utils.TimeStepSpec,
      action_space: af.ActionSpace[specs.BoundedArray],
      timestep_preprocessor: Optional[processing.TimestepPreprocessor] = None,
      default_termination_type: af.TerminationType = af.TerminationType.FAILURE,
      render_frame_cb: Optional[Callable[[Any], None]] = None,
      name: Optional[Text] = None):
    """Constructor.

    Args:
      parent_spec: The spec that is exposed by the environment.
      action_space: The action space that projects the action received by the
        agent to one that the environment expects.
      timestep_preprocessor: The preprocessor that transforms the observation
        of the environment to the one received by the agent.
      default_termination_type: Termination type used by default if the
        `timestep_preprocessor does not return one.
      render_frame_cb: Callable that is used to render a frame on a canvas.
      name: Name of the subtask.
    """
    super(ParameterizedSubTask, self).__init__(name=name)
    self._action_space = action_space
    self._parent_spec = parent_spec
    self._timestep_preprocessor = timestep_preprocessor

    if timestep_preprocessor is None:
      self._spec = parent_spec
    else:
      self._spec = timestep_preprocessor.setup_io_spec(self._parent_spec)

    self._default_termination_type = default_termination_type
    self._pterm = 0.
    self._render_frame_cb = render_frame_cb

  @overrides(af.SubTask)
  def observation_spec(self):
    return self._spec.observation_spec

  @overrides(af.SubTask)
  def arg_spec(self) -> None:
    return  # ParameterizedSubTask cannot take runtime arguments.

  @overrides(af.SubTask)
  def reward_spec(self):
    return self._spec.reward_spec

  @overrides(af.SubTask)
  def discount_spec(self):
    return self._spec.discount_spec

  @overrides(af.SubTask)
  def action_spec(self) -> specs.BoundedArray:
    return self._action_space.spec()

  @overrides(af.SubTask)
  def agent_to_parent_action(self, agent_action: np.ndarray) -> np.ndarray:
    return self._action_space.project(agent_action)

  @overrides(af.SubTask)
  def parent_to_agent_timestep(
      self,
      parent_timestep: dm_env.TimeStep,
      own_arg_key: Optional[Text] = None) -> dm_env.TimeStep:
    """Applies timestep preprocessors to generate the agent-side timestep."""
    del own_arg_key  # ParameterizedSubTask doesn't handle signals from parent.
    preprocessor_timestep = (
        processing.PreprocessorTimestep.from_environment_timestep(
            parent_timestep, pterm=0., result=None))
    if self._timestep_preprocessor is not None:
      preprocessor_timestep = self._timestep_preprocessor.process(
          preprocessor_timestep)

    # Cache preprocessed timestep pterm and result since these fields are lost
    # when converting back to dm_env.TimeStep.
    self._pterm = preprocessor_timestep.pterm
    self._result = preprocessor_timestep.result

    return preprocessor_timestep.to_environment_timestep()

  @overrides(af.SubTask)
  def pterm(self, parent_timestep: dm_env.TimeStep,
            own_arg_key: Text) -> float:
    del parent_timestep
    del own_arg_key
    return self._pterm

  @overrides(af.SubTask)
  def subtask_result(self, parent_timestep: dm_env.TimeStep,
                     own_arg_key: Text) -> af.OptionResult:
    result = self._result or af.OptionResult(
        self._default_termination_type, data=None)
    return result

  @overrides(af.SubTask)
  def render_frame(self, canvas) -> None:
    if self._render_frame_cb is not None:
      self._render_frame_cb(canvas)

  def set_timestep_preprocessor(
      self, preprocessor: processing.TimestepPreprocessor):
    self._spec = preprocessor.setup_io_spec(self._parent_spec)
    self._timestep_preprocessor = preprocessor
