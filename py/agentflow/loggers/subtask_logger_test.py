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
"""Tests for dm_robotics.agentflow.logging.subtask_logger."""

import json
from typing import Any, List, Mapping

from absl.testing import absltest
import dm_env
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.loggers import subtask_logger
from dm_robotics.agentflow.loggers import utils
import numpy as np


class SubtaskLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._parent_timestep_spec = testing_functions.random_timestep_spec(
        reward_spec=testing_functions.random_reward_spec(dtype=np.float64),
        discount_spec=testing_functions.random_discount_spec(dtype=np.float64))
    self._agent_timestep_spec = testing_functions.random_timestep_spec(
        reward_spec=testing_functions.random_reward_spec(dtype=np.float64),
        discount_spec=testing_functions.random_discount_spec(dtype=np.float64))

  def _step_through_sequence(self, observer: subtask_logger.SubTaskLogger,
                             rewards: List[float], discounts: List[float],
                             step_type: dm_env.StepType):
    for reward, discount in zip(rewards, discounts):
      dummy_parent_timestep = testing_functions.random_timestep(
          self._parent_timestep_spec)
      agent_timestep = testing_functions.random_timestep(
          self._agent_timestep_spec,
          step_type=step_type,
          reward=reward,
          discount=discount)
      dummy_parent_action = testing_functions.random_action()
      dummy_agent_action = testing_functions.random_action()

      observer.step(dummy_parent_timestep, dummy_parent_action,
                    agent_timestep, dummy_agent_action)

  def test_episode_return_logger(self):
    additional_discount = 0.8
    episode_len = 3

    aggregator = subtask_logger.EpisodeReturnAggregator(additional_discount)
    logger = FakeLogger()
    observer = subtask_logger.SubTaskLogger(logger, aggregator)

    rewards = np.hstack(([0], np.random.rand(episode_len - 1)))
    discounts = np.hstack(([1], np.random.rand(episode_len - 1)))

    # Initialize; Shouldn't call logger.write until a LAST step is received.
    # First timestep has no reward or discount by convention.
    self._step_through_sequence(observer, rewards[:1], discounts[:1],
                                dm_env.StepType.FIRST)
    self.assertEmpty(logger.logs())

    # Run episode up to last step.
    self._step_through_sequence(observer, rewards[1:-1], discounts[1:-1],
                                dm_env.StepType.MID)

    # Shouldn't call logger.write until a LAST step is received.
    self.assertEmpty(logger.logs())

    # Last-step observer, should call logger.
    self._step_through_sequence(observer, rewards[-1:], discounts[-1:],
                                dm_env.StepType.LAST)

    expected_return = utils.compute_return(rewards,
                                           discounts * additional_discount)
    self.assertTrue(logger.is_logged(episode_return=expected_return,
                                     episode_length=episode_len))


class FakeLogger:

  def __init__(self):
    super().__init__()
    self._log = []

  def write(self, values: Mapping[str, Any]):
    self._log.append(json.dumps(values, sort_keys=True))

  def logs(self):
    return list(self._log)

  def is_logged(self, **kwargs):
    expected_entry = json.dumps(kwargs, sort_keys=True)
    return expected_entry in self._log


if __name__ == '__main__':
  absltest.main()
