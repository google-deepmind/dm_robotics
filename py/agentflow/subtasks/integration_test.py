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
"""Integration test for parameterized_subtask and termination components."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import subtask
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.preprocessors import rewards
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
from dm_robotics.agentflow.subtasks import parameterized_subtask
from dm_robotics.agentflow.subtasks import subtask_termination
import numpy as np


class IntegrationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Parameters for MaxStepsTermination.
    self._max_steps = 10
    self._max_steps_terminal_discount = 0.123

    # Parameters for RewardThresholdTermination.
    self._reward_threshold = 0.5
    self._reward_threshold_terminal_discount = 0.

    # Parameters for LeavingWorkspaceTermination.
    self._tcp_pos_obs = 'tcp_pos'
    self._workspace_center = np.array([0.2, -0.4, 0.6])
    self._workspace_radius = 1.
    self._leaving_workspace_terminal_discount = 0.456
    self._tcp_observation_spec = {
        self._tcp_pos_obs:
            specs.BoundedArray(
                shape=(3,),
                name=self._tcp_pos_obs,
                minimum=np.ones(3) * -10,
                maximum=np.ones(3) * 10,
                dtype=np.float64)
    }

    # Parameters for ThresholdedL2Reward.
    self._reward_l2_threshold = 0.1
    self._success_reward_value = 1.
    self._world_center_obs = 'world_center'
    self._world_center_observation_spec = {
        self._world_center_obs:
            specs.BoundedArray(
                shape=(3,),
                name=self._world_center_obs,
                minimum=self._workspace_center,
                maximum=self._workspace_center,
                dtype=np.float64)
    }

  def _build(self, sparse_mode: bool):
    # Configure all timestep preprocessors.
    max_steps_tsp = subtask_termination.MaxStepsTermination(
        self._max_steps, self._max_steps_terminal_discount)
    leaving_workspace_tsp = subtask_termination.LeavingWorkspaceTermination(
        self._tcp_pos_obs, self._workspace_center, self._workspace_radius,
        self._leaving_workspace_terminal_discount)
    reward_setting_tsp = rewards.ThresholdedL2Reward(
        obs0=self._tcp_pos_obs, obs1=self._world_center_obs,
        threshold=self._reward_l2_threshold, reward=self._success_reward_value)
    reward_threshold_tsp = subtask_termination.RewardThresholdTermination(
        self._reward_threshold,
        self._reward_threshold_terminal_discount,
        sparse_mode=sparse_mode)

    # Compose to a single CompositeTimestepPreprocessor that runs in order.
    self._preprocesors = [
        max_steps_tsp, leaving_workspace_tsp, reward_setting_tsp,
        reward_threshold_tsp
    ]
    ts_preprocessor = timestep_preprocessor.CompositeTimestepPreprocessor(
        *self._preprocesors)

    # Create task spec and instantiate subtask.
    self._action_spec = testing_functions.random_array_spec(shape=(4,))
    random_action_space = core.IdentityActionSpace(self._action_spec)
    random_observation_spec = testing_functions.random_observation_spec(
        dtype=np.float64)
    random_observation_spec.update(self._tcp_observation_spec)
    random_observation_spec.update(self._world_center_observation_spec)
    self._parent_spec = testing_functions.random_timestep_spec(
        observation_spec=random_observation_spec)

    self._subtask = parameterized_subtask.ParameterizedSubTask(
        parent_spec=self._parent_spec,
        action_space=random_action_space,
        timestep_preprocessor=ts_preprocessor,
        name='TestParameterizedSubTask')

    # Update types on all reward and discount members for tests.
    discount_type = self._parent_spec.discount_spec.dtype.type
    reward_type = self._parent_spec.reward_spec.dtype.type
    self._default_discount = discount_type(1.)
    self._default_reward = reward_type(0.)
    self._success_reward_value = reward_type(self._success_reward_value)
    self._reward_threshold_terminal_discount = discount_type(
        self._reward_threshold_terminal_discount)
    self._max_steps_terminal_discount = discount_type(
        self._max_steps_terminal_discount)
    self._leaving_workspace_terminal_discount = discount_type(
        self._leaving_workspace_terminal_discount)

  def _run(self, default_tcp_obs, event_tcp_obs, event_step_idx,
           expected_rewards, expected_discounts):
    # Steps through an episode, setting `event_tcp_obs` on step `event_step_idx`
    # and checks that the policy saw the correct rewards and discounts.
    mock_policy = mock.MagicMock(spec=core.Policy)
    valid_action = testing_functions.valid_value(self._action_spec)
    mock_policy.step.return_value = valid_action
    agent = subtask.SubTaskOption(sub_task=self._subtask,
                                  agent=mock_policy, name='Subtask Option')

    random_first_timestep = testing_functions.random_timestep(
        self._parent_spec, step_type=dm_env.StepType.FIRST,
        discount=self._default_discount)

    random_first_timestep.observation[self._tcp_pos_obs] = default_tcp_obs

    _ = agent.step(random_first_timestep)

    for i in range(1, self._max_steps * 2):
      random_mid_timestep = testing_functions.random_timestep(
          self._parent_spec, step_type=dm_env.StepType.MID,
          discount=self._default_discount)
      if i == event_step_idx:
        random_mid_timestep.observation[self._tcp_pos_obs] = event_tcp_obs
      else:
        random_mid_timestep.observation[self._tcp_pos_obs] = default_tcp_obs
      _ = agent.step(random_mid_timestep)

      if agent.pterm(random_mid_timestep) > np.random.rand():
        random_last_timestep = testing_functions.random_timestep(
            self._parent_spec, step_type=dm_env.StepType.LAST,
            discount=self._default_discount)
        # TCP doesn't have to be in-bounds for subtask to provide terminal
        # reward, so just to verify we set back to out-of-bounds.
        random_last_timestep.observation[self._tcp_pos_obs] = event_tcp_obs
        _ = agent.step(random_last_timestep)
        break

    actual_rewards = [
        call[0][0].reward for call in mock_policy.step.call_args_list
    ]
    actual_discounts = [
        call[0][0].discount for call in mock_policy.step.call_args_list
    ]

    self.assertEqual(expected_rewards, actual_rewards)
    self.assertEqual(expected_discounts, actual_discounts)

  @parameterized.parameters([False, True])
  def test_terminating_with_sparse_reward(self, sparse_mode: bool):
    # Tests that agent sees proper reward and discount when solving the task.

    self._build(sparse_mode)

    # Make sure tcp is not within radius of target.
    random_unit_offset = np.random.rand(3)
    random_unit_offset /= np.linalg.norm(random_unit_offset)
    within_bounds_obs = (
        self._workspace_center + random_unit_offset *
        (self._workspace_radius / 2.))
    success_obs = (
        self._workspace_center + random_unit_offset *
        (self._reward_l2_threshold / 2.))

    event_step_idx = 5
    event_tcp_obs = success_obs
    default_tcp_obs = within_bounds_obs
    expected_event_reward = self._success_reward_value
    expected_event_discount = self._reward_threshold_terminal_discount

    # Expected results:
    if sparse_mode:
      # If `sparse_mode` then agent should see reward=0 and discount=1 for first
      # step and following `event_step_idx - 1` steps, and then reward=1 and
      # discount=0 for step `event_step_idx + 1`. The episode should terminate
      # on that step.
      expected_rewards = ([self._default_reward] +  # first step.
                          [self._default_reward] * event_step_idx +  # steps 1:n
                          [expected_event_reward])  # last step.
      expected_discounts = ([self._default_discount] +
                            [self._default_discount] * event_step_idx +
                            [expected_event_discount])
    else:
      # In default mode agent should see `event_step_idx` regular steps with
      # reward=0 and discount=1, and TWO "success" steps with reward=1 and
      # discount=0.
      expected_rewards = ([self._default_reward] * event_step_idx +  # 0:(n-1).
                          [expected_event_reward] +  # first success step.
                          [expected_event_reward])  # last step.
      expected_discounts = ([self._default_discount] * event_step_idx +
                            [expected_event_discount] +  # first success step.
                            [expected_event_discount])  # last step.
    self._run(default_tcp_obs, event_tcp_obs, event_step_idx, expected_rewards,
              expected_discounts)

  def test_terminating_out_of_bounds(self):
    # Tests that agent sees proper reward and discount when violating workspace
    # limits.

    self._build(False)

    # Make sure tcp is not within radius of target.
    random_unit_offset = np.random.rand(3)
    random_unit_offset /= np.linalg.norm(random_unit_offset)
    out_of_bounds_obs = (
        self._workspace_center + random_unit_offset *
        (self._workspace_radius * 2.))
    within_bounds_obs = (
        self._workspace_center + random_unit_offset *
        (self._workspace_radius / 2.))

    event_step_idx = 5
    event_tcp_obs = out_of_bounds_obs
    default_tcp_obs = within_bounds_obs
    expected_event_reward = self._default_reward
    expected_event_discount = self._leaving_workspace_terminal_discount

    # Expected results:
    # Agent should see `event_step_idx` regular steps with  reward=0 and
    # discount=1, and TWO "failure" steps with reward=0 and discount=0.
    expected_rewards = ([self._default_reward] * event_step_idx +  # 0:(n-1).
                        [expected_event_reward] +      # first success step.
                        [expected_event_reward])       # last step.
    expected_discounts = ([self._default_discount] * event_step_idx +
                          [expected_event_discount] +  # first success step.
                          [expected_event_discount])   # last step.

    self._run(default_tcp_obs, event_tcp_obs, event_step_idx, expected_rewards,
              expected_discounts)

  def test_terminating_max_steps(self):
    # Tests that agent sees proper reward and discount when running out of time.

    self._build(False)

    # Make sure tcp is not within radius of target.
    random_unit_offset = np.random.rand(3)
    random_unit_offset /= np.linalg.norm(random_unit_offset)
    within_bounds_obs = (
        self._workspace_center + random_unit_offset *
        (self._workspace_radius / 2.))

    event_step_idx = self._max_steps - 1
    event_tcp_obs = within_bounds_obs
    default_tcp_obs = within_bounds_obs
    expected_event_reward = self._default_reward
    expected_event_discount = self._max_steps_terminal_discount

    # Expected results:
    # Agent should see `event_step_idx` regular steps with reward=0 and
    # discount=1, and TWO "failure" steps with reward=0 and discount=0.
    expected_rewards = ([self._default_reward] * event_step_idx +  # 0:(n-1).
                        [expected_event_reward] +      # first success step.
                        [expected_event_reward])       # last step.
    expected_discounts = ([self._default_discount] * event_step_idx +
                          [expected_event_discount] +  # first success step.
                          [expected_event_discount])   # last step.

    self._run(default_tcp_obs, event_tcp_obs, event_step_idx, expected_rewards,
              expected_discounts)


if __name__ == '__main__':
  absltest.main()
