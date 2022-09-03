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
"""Module for logging subtasks.
"""

import abc
from typing import Any, List, Mapping, Optional

import dm_env
from dm_robotics.agentflow import subtask
from dm_robotics.agentflow.loggers import types
from dm_robotics.agentflow.loggers import utils
import numpy as np


class Aggregator(abc.ABC):
  """Base class for data-aggregators for SubTaskLogger.

  An `Aggregator` handles the job of accumulating data to log from parent and
  child timesteps & actions within a subtask.
  """

  @abc.abstractmethod
  def accumulate(self, parent_timestep: dm_env.TimeStep,
                 parent_action: np.ndarray, agent_timestep: dm_env.TimeStep,
                 agent_action: np.ndarray) -> Optional[Mapping[str, Any]]:
    """Step aggregator and optionally return a dict of information to log.

    Args:
      parent_timestep: The timestep passed to the SubTask by its parent.
      parent_action: The action being returned to the parent. Typically an
        exteneded or modified version of `agent_action`.
      agent_timestep: The timestep this subtask passed to its agent. Typically a
        reduced or modified version of `parent_timestep`.
      agent_action: The action returned by the agent this step.

    Returns:
      A dictionary of information that can be passed to an acme logger. Can also
      return None, which skips logging this step.
    """
    pass


class EpisodeReturnAggregator(Aggregator):
  """An Aggregator that computes episode return and length when subtask ends."""

  def __init__(self,
               additional_discount: float = 1.,
               return_name: str = 'episode_return',
               length_name: str = 'episode_length'):
    self._additional_discount = additional_discount
    self._return_name = return_name
    self._length_name = length_name
    self._episode_rewards = []  # type: List[float]
    self._episode_discounts = []  # type: List[float]

  def accumulate(self, parent_timestep: dm_env.TimeStep,
                 parent_action: np.ndarray, agent_timestep: dm_env.TimeStep,
                 agent_action: np.ndarray) -> Optional[Mapping[str, Any]]:
    if agent_timestep.first():
      self._episode_rewards.clear()
      self._episode_discounts.clear()

    if agent_timestep.reward is None or agent_timestep.discount is None:
      return  # Some environments omit reward and discount on first step.

    self._episode_rewards.append(agent_timestep.reward)
    self._episode_discounts.append(agent_timestep.discount)

    if agent_timestep.last():
      return {
          self._return_name: utils.compute_return(
              self._episode_rewards,
              np.array(self._episode_discounts) * self._additional_discount),
          self._length_name: len(self._episode_rewards)
      }
    return


class SubTaskLogger(subtask.SubTaskObserver):
  """A subtask observer that logs agent performance to an Acme logger."""

  def __init__(self, logger: types.Logger, aggregator: Aggregator):
    """Initialize SubTaskLogger."""
    self._logger = logger
    self._aggregator = aggregator

  def step(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, parent_timestep: dm_env.TimeStep, parent_action: np.ndarray,
      agent_timestep: dm_env.TimeStep, agent_action: np.ndarray) -> None:
    # Fetch current data to log.
    data = self._aggregator.accumulate(parent_timestep, agent_action,
                                       agent_timestep, agent_action)

    # Log the given results.
    if data is not None:
      self._logger.write(data)
