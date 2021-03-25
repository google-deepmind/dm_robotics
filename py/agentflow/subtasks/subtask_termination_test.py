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
"""Tests for dm_robotics.agentflow.subtasks.parameterized_subtask."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
from dm_robotics.agentflow.subtasks import subtask_termination
from dm_robotics.geometry import geometry
import numpy as np

_NUM_RUNS = 10

to_preprocessor_timestep = (
    timestep_preprocessor.PreprocessorTimestep.from_environment_timestep)


class SubtaskTerminationTest(parameterized.TestCase):

  def test_max_steps_termination(self):
    max_steps = 3
    terminal_discount = 0.

    for _ in range(_NUM_RUNS):
      random_timestep_spec = testing_functions.random_timestep_spec()
      random_first_timestep = testing_functions.random_timestep(
          random_timestep_spec, step_type=dm_env.StepType.FIRST)
      tsp = subtask_termination.MaxStepsTermination(
          max_steps, terminal_discount)
      tsp.setup_io_spec(random_timestep_spec)

      # First-step, increments counter but no change to timestep.
      preprocessor_timestep = to_preprocessor_timestep(
          random_first_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 0.)
      self.assertEqual(result.discount, random_first_timestep.discount)
      self.assertIsNone(result.result)

      # Second-step, increments counter but no change to timestep.
      random_mid_timestep = testing_functions.random_timestep(
          random_timestep_spec, step_type=dm_env.StepType.MID)
      preprocessor_timestep = to_preprocessor_timestep(
          random_mid_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 0.)
      self.assertEqual(result.discount, random_mid_timestep.discount)
      self.assertIsNone(result.result)

      # Third-step, signal termination.
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 1.)
      self.assertEqual(result.discount, terminal_discount)
      self.assertEqual(result.result, core.OptionResult.failure_result())

  @parameterized.parameters([False, True])
  def test_reward_threshold_termination(self, sparse_mode: bool):
    reward_threshold = 0.5
    terminal_discount = 0.2

    for _ in range(_NUM_RUNS):
      reward_spec = specs.BoundedArray(
          shape=(), dtype=np.float64, minimum=0., maximum=1., name='reward')
      discount_spec = specs.BoundedArray(
          shape=(), dtype=np.float64, minimum=0., maximum=1., name='discount')
      random_timestep_spec = testing_functions.random_timestep_spec(
          reward_spec=reward_spec, discount_spec=discount_spec)
      random_mid_timestep = testing_functions.random_timestep(
          random_timestep_spec, step_type=dm_env.StepType.MID)
      tsp = subtask_termination.RewardThresholdTermination(
          reward_threshold, terminal_discount, sparse_mode)
      tsp.setup_io_spec(random_timestep_spec)

      # If reward is sub-threshold timestep should be unchanged.
      sub_thresh_reward = 0.3
      random_mid_timestep = random_mid_timestep._replace(
          reward=sub_thresh_reward)
      preprocessor_timestep = to_preprocessor_timestep(
          random_mid_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 0.)
      self.assertEqual(result.discount, random_mid_timestep.discount)
      self.assertEqual(result.reward, sub_thresh_reward)
      self.assertIsNone(result.result)

      # If reward is super-threshold timestep should be terminal.
      super_thresh_reward = 0.7
      random_mid_timestep = random_mid_timestep._replace(
          reward=super_thresh_reward)
      preprocessor_timestep = to_preprocessor_timestep(
          random_mid_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      if not sparse_mode:
        self.assertEqual(result.pterm, 1.)
        self.assertEqual(result.discount, terminal_discount)
        self.assertEqual(result.reward, super_thresh_reward)
        self.assertEqual(result.result, core.OptionResult.success_result())
      else:
        # First step signals termination, but sets cached reward and doesn't set
        # terminal discount
        self.assertEqual(result.pterm, 1.)
        self.assertEqual(result.discount, random_mid_timestep.discount)
        self.assertEqual(result.reward, sub_thresh_reward)
        self.assertEqual(result.result, core.OptionResult.success_result())

        # Stepping again with a MID timestep still does NOT terminate.
        result = tsp.process(preprocessor_timestep)
        self.assertEqual(result.pterm, 1.)
        self.assertEqual(result.discount, random_mid_timestep.discount)
        self.assertEqual(result.reward, sub_thresh_reward)
        self.assertEqual(result.result, core.OptionResult.success_result())

        # Finally providing a LAST timestep yields proper termination.
        preprocessor_timestep = preprocessor_timestep._replace(
            step_type=dm_env.StepType.LAST)
        result = tsp.process(preprocessor_timestep)
        self.assertEqual(result.pterm, 1.)
        self.assertEqual(result.discount, terminal_discount)
        self.assertEqual(result.reward, super_thresh_reward)
        self.assertEqual(result.result, core.OptionResult.success_result())

  @parameterized.parameters([False, True])
  def test_reward_array_threshold_termination(self, use_threshold_array):
    reward_threshold = 0.5 * np.ones(2) if use_threshold_array else 0.5
    terminal_discount = 0.2
    reward_spec = specs.BoundedArray(
        shape=(2,), dtype=np.float64, minimum=0., maximum=1., name='reward')
    discount_spec = specs.BoundedArray(
        shape=(), dtype=np.float64, minimum=0., maximum=1., name='discount')
    random_timestep_spec = testing_functions.random_timestep_spec(
        reward_spec=reward_spec, discount_spec=discount_spec)
    random_mid_timestep = testing_functions.random_timestep(
        random_timestep_spec, step_type=dm_env.StepType.MID)

    tsp = subtask_termination.RewardThresholdTermination(
        reward_threshold, terminal_discount)
    tsp.setup_io_spec(random_timestep_spec)

    # If reward is sub-threshold timestep should be unchanged.
    sub_thresh_reward = 0.3 * np.ones(reward_spec.shape, reward_spec.dtype)
    random_mid_timestep = random_mid_timestep._replace(
        reward=sub_thresh_reward)
    preprocessor_timestep = to_preprocessor_timestep(
        random_mid_timestep, pterm=0., result=None)
    result = tsp.process(preprocessor_timestep)
    self.assertEqual(result.pterm, 0.)
    self.assertEqual(result.discount, random_mid_timestep.discount)
    self.assertIsNone(result.result)

    # If reward is super-threshold timestep should be terminal.
    super_thresh_reward = 0.7 * np.ones(reward_spec.shape, reward_spec.dtype)
    random_mid_timestep = random_mid_timestep._replace(
        reward=super_thresh_reward)
    preprocessor_timestep = to_preprocessor_timestep(
        random_mid_timestep, pterm=0., result=None)
    result = tsp.process(preprocessor_timestep)
    self.assertEqual(result.pterm, 1.)
    self.assertEqual(result.discount, terminal_discount)
    self.assertEqual(result.result, core.OptionResult.success_result())

  def test_leaving_workspace_termination(self):
    tcp_pos_obs = 'tcp_pos_obs'
    workspace_center = np.array([0.1, 0.2, 0.3])
    workspace_radius = 0.5
    terminal_discount = 0.1
    observation_spec = {
        tcp_pos_obs:
            specs.BoundedArray(
                shape=(3,),
                name=tcp_pos_obs,
                minimum=np.ones(3) * -10,
                maximum=np.ones(3) * 10,
                dtype=np.float64)
    }
    discount_spec = specs.BoundedArray(
        shape=(), dtype=np.float64, minimum=0., maximum=1., name='discount')

    for _ in range(_NUM_RUNS):
      random_timestep_spec = testing_functions.random_timestep_spec(
          observation_spec=observation_spec,
          discount_spec=discount_spec)
      tsp = subtask_termination.LeavingWorkspaceTermination(
          tcp_pos_obs, workspace_center, workspace_radius, terminal_discount)
      tsp.setup_io_spec(random_timestep_spec)

      random_unit_offset = np.random.rand(3)
      random_unit_offset /= np.linalg.norm(random_unit_offset)
      within_bounds_obs = workspace_center + random_unit_offset * 0.4
      out_of_bounds_obs = workspace_center + random_unit_offset * 0.6

      # If observation is within bounds the timestep should be unchanged.
      random_in_bounds_timestep = testing_functions.random_timestep(
          random_timestep_spec, observation={tcp_pos_obs: within_bounds_obs})
      preprocessor_timestep = to_preprocessor_timestep(
          random_in_bounds_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 0.)
      self.assertEqual(result.discount, random_in_bounds_timestep.discount)
      self.assertIsNone(result.result)

      # If observation is out of bounds the timestep should be terminal.
      random_out_of_bounds_timestep = testing_functions.random_timestep(
          random_timestep_spec, observation={tcp_pos_obs: out_of_bounds_obs})
      preprocessor_timestep = to_preprocessor_timestep(
          random_out_of_bounds_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 1.)
      self.assertEqual(result.discount, terminal_discount)
      self.assertEqual(result.result, core.OptionResult.failure_result())

  def test_leaving_workspace_box_termination(self):
    tcp_pos_obs = 'tcp_pos_obs'
    tcp_quat_obs = 'tcp_quat_obs'

    workspace_center_poseuler = np.array([0.1, 0.2, 0.3, -0.1, 0.2, -0.3])
    workspace_center = geometry.PoseStamped(
        geometry.Pose.from_poseuler(workspace_center_poseuler), None)

    workspace_limits = np.array([0.3, 0.2, 0.1, 0.5, 0.5, 0.5])

    terminal_discount = 0.1
    observation_spec = {
        tcp_pos_obs:
            specs.BoundedArray(
                shape=(3,),
                name=tcp_pos_obs,
                minimum=np.ones(3) * -10,
                maximum=np.ones(3) * 10,
                dtype=np.float64),
        tcp_quat_obs:
            specs.BoundedArray(
                shape=(4,),
                name=tcp_quat_obs,
                minimum=np.ones(4) * -1.0,
                maximum=np.ones(4) * 1.0,
                dtype=np.float64)
    }
    discount_spec = specs.BoundedArray(
        shape=(), dtype=np.float64, minimum=0., maximum=1., name='discount')

    for _ in range(_NUM_RUNS):
      random_timestep_spec = testing_functions.random_timestep_spec(
          observation_spec=observation_spec, discount_spec=discount_spec)
      tsp = subtask_termination.LeavingWorkspaceBoxTermination(
          tcp_pos_obs=tcp_pos_obs,
          tcp_quat_obs=tcp_quat_obs,
          workspace_centre=workspace_center,
          workspace_limits=workspace_limits,
          terminal_discount=terminal_discount)
      tsp.setup_io_spec(random_timestep_spec)

      random_unit_offset = np.random.rand(6)
      random_unit_offset /= np.linalg.norm(random_unit_offset)

      # pos-euler offsets.
      within_bounds_offset = random_unit_offset * 0.99 * workspace_limits
      out_of_bounds_offset = (random_unit_offset + 1.01) * workspace_limits

      within_bounds_pose = geometry.PoseStamped(
          geometry.Pose.from_poseuler(within_bounds_offset),
          frame=workspace_center)
      within_bounds_pos = within_bounds_pose.get_world_pose().position
      within_bounds_quat = within_bounds_pose.get_world_pose().quaternion

      out_of_bounds_pose = geometry.PoseStamped(
          geometry.Pose.from_poseuler(out_of_bounds_offset),
          frame=workspace_center)
      out_of_bounds_pos = out_of_bounds_pose.get_world_pose().position
      out_of_bounds_quat = out_of_bounds_pose.get_world_pose().quaternion

      # If observation is within bounds the timestep should be unchanged.
      random_in_bounds_timestep = testing_functions.random_timestep(
          random_timestep_spec,
          observation={
              tcp_pos_obs: within_bounds_pos,
              tcp_quat_obs: within_bounds_quat
          })
      preprocessor_timestep = to_preprocessor_timestep(
          random_in_bounds_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 0.)
      self.assertEqual(result.discount, random_in_bounds_timestep.discount)
      self.assertIsNone(result.result)

      # If observation is out of bounds the timestep should be terminal.
      random_out_of_bounds_timestep = testing_functions.random_timestep(
          random_timestep_spec,
          observation={
              tcp_pos_obs: out_of_bounds_pos,
              tcp_quat_obs: out_of_bounds_quat
          })
      preprocessor_timestep = to_preprocessor_timestep(
          random_out_of_bounds_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 1.)
      self.assertEqual(result.discount, terminal_discount)
      self.assertEqual(result.result, core.OptionResult.failure_result())

  def test_observation_threshold_termination(self):
    obs = 'joint_configuration'
    desired_obs = np.array([0.1, 0.2, 0.3, 0.4])
    error_norm = 0.5
    terminal_discount = 0.1
    observation_spec = {
        obs:
            specs.BoundedArray(
                shape=(4,),
                name=obs,
                minimum=np.ones(4) * -10,
                maximum=np.ones(4) * 10,
                dtype=np.float64)
    }
    discount_spec = specs.BoundedArray(
        shape=(), dtype=np.float64, minimum=0., maximum=1., name='discount')

    for _ in range(_NUM_RUNS):
      random_timestep_spec = testing_functions.random_timestep_spec(
          observation_spec=observation_spec, discount_spec=discount_spec)
      tsp = subtask_termination.ObservationThresholdTermination(
          obs, desired_obs, error_norm, terminal_discount)
      tsp.setup_io_spec(random_timestep_spec)

      random_unit_offset = np.random.rand(4)
      random_unit_offset /= np.linalg.norm(random_unit_offset)
      within_threshold_obs = desired_obs + random_unit_offset * 0.4
      outside_threshold_obs = desired_obs + random_unit_offset * 0.6

      # If observation is within the threshold, the timestep should be terminal.
      random_within_threshold_timestep = testing_functions.random_timestep(
          random_timestep_spec, observation={obs: within_threshold_obs})
      preprocessor_timestep = to_preprocessor_timestep(
          random_within_threshold_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 1.)
      self.assertEqual(result.discount, terminal_discount)
      self.assertEqual(result.result, core.OptionResult.failure_result())

      # If observation is outside the threshold, the timestep should be
      # unchanged.
      random_outside_threshold_timestep = testing_functions.random_timestep(
          random_timestep_spec, observation={obs: outside_threshold_obs})
      preprocessor_timestep = to_preprocessor_timestep(
          random_outside_threshold_timestep, pterm=0., result=None)
      result = tsp.process(preprocessor_timestep)
      self.assertEqual(result.pterm, 0.)
      self.assertEqual(result.discount,
                       random_outside_threshold_timestep.discount)
      self.assertIsNone(result.result)


if __name__ == '__main__':
  absltest.main()
