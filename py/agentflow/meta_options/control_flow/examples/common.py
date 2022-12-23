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
"""Common boilerplate for examples."""

from typing import Mapping, Optional, Text, Tuple

import dm_env
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import subtask
import numpy as np


class DummyEnv(dm_env.Environment):
  """A dummy environment with some spec."""

  def observation_spec(self) -> Mapping[Text, specs.Array]:
    return {
        'dummy_obs':
            specs.BoundedArray(
                shape=(3,),
                dtype=np.float32,
                minimum=0.,
                maximum=1.,
                name='dummy_obs')
    }

  def action_spec(self) -> specs.Array:
    return specs.BoundedArray(
        shape=(4,), dtype=np.float32, minimum=0., maximum=1., name='dummy_act')

  def step(self, action) -> dm_env.TimeStep:
    return dm_env.TimeStep(
        reward=0.,
        discount=1.,
        observation={'dummy_obs': np.random.rand(3)},
        step_type=dm_env.StepType.MID)

  def reset(self) -> dm_env.TimeStep:
    return dm_env.TimeStep(
        reward=0.,
        discount=1.,
        observation={'dummy_obs': np.random.rand(3)},
        step_type=dm_env.StepType.FIRST)


class DummySubTask(subtask.SubTask):
  """A dummy subtask."""

  def __init__(self,
               base_obs_spec: Mapping[Text, specs.Array],
               name: Optional[Text] = None):
    super().__init__(name)
    self._base_obs_spec = base_obs_spec

  def observation_spec(self) -> Mapping[Text, specs.Array]:
    return self._base_obs_spec

  def arg_spec(self) -> Optional[specs.Array]:
    return

  def action_spec(self) -> specs.Array:  # pytype: disable=signature-mismatch  # overriding-return-type-checks
    return specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0., maximum=1., name='dummy_act')

  def agent_to_parent_action(self, agent_action: np.ndarray) -> np.ndarray:
    return np.hstack((agent_action, np.zeros(2)))  # Return full action.

  def parent_to_agent_timestep(self, parent_timestep: dm_env.TimeStep,  # pytype: disable=signature-mismatch  # overriding-return-type-checks
                               arg_key: Text) -> Tuple[dm_env.TimeStep, float]:
    return (parent_timestep, 1.0)

  def pterm(self, parent_timestep: dm_env.TimeStep,
            own_arg_key: Text) -> float:
    return 0.


class DummyPolicy(core.Policy):
  """A dummy policy."""

  def __init__(self, action_spec: specs.Array,
               unused_observation_spec: Mapping[Text, specs.Array],
               name: Optional[Text] = None):
    super().__init__(name)
    self._action_spec = action_spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    return np.random.rand(self._action_spec.shape[0])


class DummyOption(core.Option):
  """A dummy option."""

  def __init__(self, action_spec: specs.Array,
               unused_observation_spec: Mapping[Text, specs.Array],
               name: Optional[Text] = None):
    super().__init__(name)
    self._action_spec = action_spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    option_action = np.random.rand(self._action_spec.shape[0])
    return np.hstack((option_action, np.zeros(2)))
