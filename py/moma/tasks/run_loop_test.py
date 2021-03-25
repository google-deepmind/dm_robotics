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

"""Tests for run_loop."""

import copy
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from dm_robotics.moma.tasks import run_loop
import numpy as np


@parameterized.parameters([True, False])
class RunLoopTest(parameterized.TestCase):

  def test_actions_given_to_environment(self, use_scalar_rewards: bool):
    env = SpyEnv(use_scalar_rewards)
    action_1 = np.asarray([0.1])
    action_2 = np.asarray([0.2])
    agent = CycleStepsAgent([action_1, action_2])
    run_loop.run(env, agent, [], max_steps=5)

    # env method,  agent timestep, agent action produced
    # env.reset,   FIRST,          0.1
    # env.step,    MID,            0.2
    # env.step,    MID,            0.1
    # env.step,    LAST,           0.2  (Discarded)
    # env.step,    FIRST,          0.1

    expected_actions_sent_to_environment = [
        SpyEnv.RESET_CALLED,  # reset comes without an action.
        action_1,  # agent step 1
        action_2,  # agent step 2
        action_1,  # agent step 3
        SpyEnv.RESET_CALLED,  # agent step 4, action discarded
        action_1,  # agent step 5, the last step the agent is asked for.
    ]
    self.assertEqual(env.actions_received, expected_actions_sent_to_environment)

  def test_timesteps_given_to_agent(self, use_scalar_rewards: bool):
    env = SpyEnv(use_scalar_rewards)
    agent = CycleStepsAgent([(np.asarray([0.1]))])
    run_loop.run(env, agent, [], max_steps=5)

    expected_timestep_types = [
        dm_env.StepType.FIRST,
        dm_env.StepType.MID,
        dm_env.StepType.MID,
        dm_env.StepType.LAST,
        dm_env.StepType.FIRST,
    ]
    actual_timestep_types = [
        timestep.step_type for timestep in agent.timesteps_received
    ]
    self.assertEqual(actual_timestep_types, expected_timestep_types)

  def test_observer_calls(self, use_scalar_rewards: bool):
    env = SpyEnv(use_scalar_rewards)
    action_1 = np.asarray([0.1])
    action_2 = np.asarray([0.2])
    agent = CycleStepsAgent([action_1, action_2])
    observer = SpyObserver()
    run_loop.run(env, agent, [observer], max_steps=5)

    expected_observations = [
        (SpyObserver.BEGIN_EPISODE, None, None),
        (SpyObserver.STEP, dm_env.StepType.FIRST, action_1),
        (SpyObserver.STEP, dm_env.StepType.MID, action_2),
        (SpyObserver.STEP, dm_env.StepType.MID, action_1),
        (SpyObserver.END_EPISODE, dm_env.StepType.LAST, None),  # a2
        (SpyObserver.BEGIN_EPISODE, None, None),  # no agent interaction
        (SpyObserver.STEP, dm_env.StepType.FIRST, action_1),
    ]

    # "act" = actual, "ex" = expected.
    # unzip the call, timestep and actions from the SpyObserver.
    act_calls, act_timesteps, act_actions = zip(*observer.notifications)
    ex_calls, ex_step_types, ex_actions = zip(*expected_observations)
    act_step_types = [ts.step_type if ts else None for ts in act_timesteps]

    self.assertEqual(act_calls, ex_calls)
    self.assertEqual(act_step_types, list(ex_step_types))
    self.assertEqual(act_actions, ex_actions)


class SpyObserver:
  BEGIN_EPISODE = 'begin_ep'
  STEP = 'step'
  END_EPISODE = 'end_ep'

  def __init__(self):
    self.notifications = []

  def begin_episode(self, agent_id):
    del agent_id
    self.notifications.append((SpyObserver.BEGIN_EPISODE, None, None))

  def step(self, agent_id, timestep, action):
    del agent_id
    self.notifications.append((SpyObserver.STEP, timestep, action))

  def end_episode(self, agent_id, term_reason, timestep):
    del agent_id, term_reason
    self.notifications.append((SpyObserver.END_EPISODE, timestep, None))


class SpyEnv(dm_env.Environment):

  RESET_CALLED = 'reset'

  def __init__(self, use_scalar_rewards: bool):
    self._step_types = self._initialize_step_type_sequence()
    self.actions_received = []
    self.steps_emitted = []
    self._use_scalar_rewards = use_scalar_rewards

  def _initialize_step_type_sequence(self):
    return iter(
        itertools.cycle([
            dm_env.StepType.FIRST,
            dm_env.StepType.MID,
            dm_env.StepType.MID,
            dm_env.StepType.LAST,
        ]))

  def reset(self) -> dm_env.TimeStep:
    self._step_types = self._initialize_step_type_sequence()
    step = self._create_step(next(self._step_types))
    self.actions_received.append(SpyEnv.RESET_CALLED)
    self.steps_emitted.append(step)
    return copy.deepcopy(step)

  def _create_step(self, step_type):
    return dm_env.TimeStep(
        step_type=step_type,
        reward=0.0 if self._use_scalar_rewards else np.zeros(3,),
        discount=0.9,
        observation={'state': np.random.random(size=(1,))})

  def step(self, action) -> dm_env.TimeStep:
    step = self._create_step(next(self._step_types))
    self.actions_received.append(np.copy(action))
    self.steps_emitted.append(step)
    return copy.deepcopy(step)

  def reward_spec(self):
    shape = () if self._use_scalar_rewards else (3,)
    return specs.Array(shape=shape, dtype=float, name='reward')

  def discount_spec(self):
    return specs.BoundedArray(
        shape=(), dtype=float, minimum=0., maximum=1., name='discount')

  def observation_spec(self):
    return {
        'state':
            specs.BoundedArray(
                shape=(1,), dtype=np.float32, minimum=[0], maximum=[1])
    }

  def action_spec(self):
    return specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=[0], maximum=[1])

  def close(self):
    pass


class CycleStepsAgent:

  def __init__(self, steps):
    self._steps = itertools.cycle(steps)
    self.timesteps_received = []

  def step(self, timestep) -> np.ndarray:
    self.timesteps_received.append(timestep)
    return next(self._steps)


if __name__ == '__main__':
  absltest.main()
