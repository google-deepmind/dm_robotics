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
"""Tests for dm_robotics.agentflow.preprocessors.rewards."""

from typing import Sequence, Text, Union

from absl.testing import absltest
from absl.testing import parameterized

from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.decorators import overrides
from dm_robotics.agentflow.preprocessors import rewards
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
import numpy as np


def random_scalar_spec(name):
  return specs.Array(
      shape=(), dtype=np.random.choice([np.float32, np.float64]), name=name)


def create_timestep_spec(observation_spec):
  return spec_utils.TimeStepSpec(
      observation_spec,
      reward_spec=random_scalar_spec('reward'),
      discount_spec=random_scalar_spec('discount'))


def create_timestep(
    input_spec: spec_utils.TimeStepSpec,
    name: Text,
    value: Sequence[float],
) -> timestep_preprocessor.PreprocessorTimestep:
  dtype = input_spec.observation_spec[name].dtype
  observation = testing_functions.valid_value(input_spec.observation_spec)
  observation[name] = np.asarray(value, dtype=dtype)
  timestep = testing_functions.random_timestep(
      spec=input_spec, observation=observation)
  return timestep_preprocessor.PreprocessorTimestep.from_environment_timestep(
      timestep, pterm=0.0)


class RewardsTest(absltest.TestCase):

  def test_threshold_reward(self):
    input_spec = create_timestep_spec({})

    threshold = 0.5
    hi = 1.
    lo = 0.
    reward_preprocessor = rewards.ThresholdReward(
        threshold=threshold, hi=hi, lo=lo)

    output_spec = reward_preprocessor.setup_io_spec(input_spec)

    # Assert discount specs are unchanged.
    self.assertEqual(input_spec.reward_spec,
                     output_spec.reward_spec)
    self.assertEqual(input_spec.discount_spec,
                     output_spec.discount_spec)

    # Test logic.
    to_target_dtype = input_spec.reward_spec.dtype.type
    random_input = testing_functions.random_timestep(spec=input_spec)

    # Greater than threshold -> hi.
    random_input = random_input._replace(reward=to_target_dtype(0.8))
    output = reward_preprocessor.process(random_input)
    np.testing.assert_allclose(output.reward, to_target_dtype(hi))

    # Equal to threshold -> hi.
    random_input = random_input._replace(reward=to_target_dtype(0.5))
    output = reward_preprocessor.process(random_input)
    np.testing.assert_allclose(output.reward, to_target_dtype(hi))

    # Less than threshold -> lo.
    random_input = random_input._replace(reward=to_target_dtype(0.4))
    output = reward_preprocessor.process(random_input)
    np.testing.assert_allclose(output.reward, to_target_dtype(lo))

  def test_l2_spec_updated_properly(self):
    observation_spec = {
        'obs0': testing_functions.random_array_spec(),
        'obs1': testing_functions.random_array_spec()
    }
    input_spec = create_timestep_spec(observation_spec)

    reward_preprocessor = rewards.L2Reward(
        obs0='obs0', obs1='obs1', reward_scale=1.0, reward_offset=1.0)

    output_spec = reward_preprocessor.setup_io_spec(input_spec)

    # Assert discount specs are unchanged.
    self.assertEqual(input_spec.discount_spec,
                     output_spec.discount_spec)

    # Assert observation specs are unchanged.
    self.assertEqual(input_spec.observation_spec,
                     output_spec.observation_spec)

    # Assert reward specs match observation spec dtypes.
    type0 = input_spec.observation_spec['obs0']
    type1 = input_spec.observation_spec['obs1']
    targ_type = np.promote_types(type0, type1)
    self.assertEqual(output_spec.reward_spec.dtype, targ_type)

  def test_thresholded_l2_spec_unchanged(self):
    observation_spec = {
        'obs0': testing_functions.random_array_spec(),
        'obs1': testing_functions.random_array_spec()
    }
    input_spec = create_timestep_spec(observation_spec)

    reward_preprocessor = rewards.ThresholdedL2Reward(
        obs0='obs0', obs1='obs1', threshold=0.5, reward=1.0)

    output_spec = reward_preprocessor.setup_io_spec(input_spec)
    self.assertEqual(input_spec, output_spec)

  def test_spec_validation_missing_observation(self):
    observation_spec = {
        'wrong_name':
            specs.Array(shape=(2,), dtype=np.int32, name='bool as two'),
    }
    input_spec = create_timestep_spec(observation_spec)

    reward_preprocessors = [
        rewards.L2Reward(
            obs0='obs0', obs1='obs1', reward_scale=1.0, reward_offset=1.0),
        rewards.ThresholdedL2Reward(
            obs0='obs0', obs1='obs1', threshold=0.5, reward=1.0)
    ]

    for rp in reward_preprocessors:
      try:
        rp.setup_io_spec(input_spec)
        self.fail('Exception expected due to missing observation')
      except KeyError:
        pass  # expected

  def test_l2_spec_numerics(self):
    random_arr_spec = testing_functions.random_array_spec()
    observation_spec = {
        'obs0': random_arr_spec,
        'obs1': random_arr_spec
    }
    input_spec = create_timestep_spec(observation_spec)

    reward_preprocessor = rewards.L2Reward(
        obs0='obs0', obs1='obs1', reward_scale=1.5, reward_offset=2.0)

    output_spec = reward_preprocessor.setup_io_spec(input_spec)
    timestep = testing_functions.valid_value(input_spec)
    processed_timestep = reward_preprocessor.process(timestep)

    dist = np.linalg.norm(timestep.observation['obs0'] -
                          timestep.observation['obs1'])
    expected_reward = output_spec.reward_spec.dtype.type(-1 * dist * 1.5 + 2.0)

    self.assertEqual(expected_reward,
                     processed_timestep.reward)
    self.assertEqual(expected_reward.dtype,
                     processed_timestep.reward.dtype)

  def test_thresholded_l2_spec_numerics(self):
    random_arr_spec = testing_functions.random_array_spec()
    observation_spec = {
        'obs0': random_arr_spec,
        'obs1': random_arr_spec
    }
    input_spec = create_timestep_spec(observation_spec)

    reward_preprocessor = rewards.ThresholdedL2Reward(
        obs0='obs0', obs1='obs1', threshold=0.5, reward=1.0)

    output_spec = reward_preprocessor.setup_io_spec(input_spec)
    timestep = testing_functions.valid_value(input_spec)
    processed_timestep = reward_preprocessor.process(timestep)

    self.assertEqual(output_spec.reward_spec.dtype,
                     processed_timestep.reward.dtype)


class ComputeRewardTest(absltest.TestCase):

  def test_scalar_reward_computed_based_on_observation(self):
    reward_fn = lambda obs: obs['obs'][0]
    observation_spec = {
        'obs': specs.Array(shape=(2,), dtype=np.float32)
    }
    input_spec = create_timestep_spec(observation_spec)
    input_timestep = create_timestep(input_spec, 'obs', [2.0, 3.0])
    reward_preprocessor = rewards.ComputeReward(reward_function=reward_fn)
    reward_preprocessor.setup_io_spec(input_spec)
    output_timestep = reward_preprocessor.process(input_timestep)
    self.assertEqual(output_timestep.reward, 2.0)

  def test_array_rewards_fail_without_correct_shape(self):
    reward_fn = lambda obs: obs['obs']
    observation_spec = {
        'obs': specs.Array(shape=(2,), dtype=np.float32)
    }
    input_spec = create_timestep_spec(observation_spec)
    input_timestep = create_timestep(input_spec, 'obs', [2.0, 3.0])
    reward_preprocessor = rewards.ComputeReward(reward_function=reward_fn)
    with self.assertRaises(ValueError):
      reward_preprocessor.setup_io_spec(input_spec)
      reward_preprocessor.process(input_timestep)

  def test_array_rewards_succeeds_with_correct_shape(self):
    reward_fn = lambda obs: obs['obs']
    observation_spec = {
        'obs': specs.Array(shape=(2,), dtype=np.float32)
    }
    input_spec = create_timestep_spec(observation_spec)
    input_timestep = create_timestep(input_spec, 'obs', [2.0, 3.0])
    reward_preprocessor = rewards.ComputeReward(
        reward_function=reward_fn, output_spec_shape=(2,))
    reward_preprocessor.setup_io_spec(input_spec)
    output_timestep = reward_preprocessor.process(input_timestep)
    np.testing.assert_allclose(output_timestep.reward, [2.0, 3.0])


class CombineRewardsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._reward_1 = _TestReward(reward_value=1.)
    self._reward_10 = _TestReward(reward_value=10.)

    observation_spec = {
        'unused_obs': testing_functions.random_array_spec(
            shape=(2,),
            minimum=np.asarray([0, 0]),
            maximum=np.asarray([10, 10]))
    }
    self._input_spec = create_timestep_spec(observation_spec)

    self._input_timestep = create_timestep(
        self._input_spec, 'unused_obs', [2.0, 3.0])

  def test_default_combination(self):
    # Default combination is maximum of all rewards.
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=[self._reward_1, self._reward_10])

    combined_reward.setup_io_spec(self._input_spec)

    output_timestep = combined_reward.process(self._input_timestep)

    self.assertEqual(output_timestep.reward, 10.0)

  def test_max_combination(self):
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=[self._reward_1, self._reward_10],
        combination_strategy=np.max)

    combined_reward.setup_io_spec(self._input_spec)

    output_timestep = combined_reward.process(self._input_timestep)

    self.assertEqual(output_timestep.reward, 10.0)

  def test_min_combination(self):
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=[self._reward_1, self._reward_10],
        combination_strategy=np.min)

    combined_reward.setup_io_spec(self._input_spec)

    output_timestep = combined_reward.process(self._input_timestep)

    self.assertEqual(output_timestep.reward, 1.0)

  def test_sum_combination(self):
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=[self._reward_1, self._reward_10],
        combination_strategy=np.sum)

    combined_reward.setup_io_spec(self._input_spec)

    output_timestep = combined_reward.process(self._input_timestep)

    self.assertEqual(output_timestep.reward, 11.0)

  def test_sum_combination_with_list_input(self):
    reward_array = _TestReward(reward_value=np.ones(3))
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=[self._reward_1, reward_array],
        combination_strategy=np.sum)

    combined_reward.setup_io_spec(self._input_spec)

    output_timestep = combined_reward.process(self._input_timestep)

    self.assertEqual(output_timestep.reward, 4.0)

  def test_output_list_of_rewards_fails_without_correct_shape(self):
    # Must update the output shape when returning an array of rewards.
    with self.assertRaises(ValueError):
      combined_reward = rewards.CombineRewards(
          reward_preprocessors=[self._reward_1, self._reward_10],
          combination_strategy=np.stack)
      combined_reward.setup_io_spec(self._input_spec)
      combined_reward.process(self._input_timestep)

  def test_output_list_of_rewards_succeeds_with_correct_shape(self):
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=[self._reward_1, self._reward_10],
        combination_strategy=np.stack, output_spec_shape=(2,))
    combined_reward.setup_io_spec(self._input_spec)
    output_timestep = combined_reward.process(self._input_timestep)
    np.testing.assert_allclose(output_timestep.reward, [1., 10.])

  def test_processing_unflattened_rewards(self):
    zero_rewards = _TestReward(np.zeros(3))
    one_rewards = _TestReward(np.ones(3))
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=[zero_rewards, one_rewards],
        combination_strategy=lambda rewards: np.mean(rewards, axis=0),
        output_spec_shape=(3,), flatten_rewards=False)
    combined_reward.setup_io_spec(self._input_spec)
    output_timestep = combined_reward.process(self._input_timestep)
    np.testing.assert_allclose(output_timestep.reward, [0.5, 0.5, 0.5])

    # Check to make sure the flattened version gives a different result.
    # Reset things to help set up the specs.
    input_spec = self._input_spec.replace()  # makes a copy of the spec.
    zero_rewards = _TestReward(np.zeros(3))
    one_rewards = _TestReward(np.ones(3))
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=[zero_rewards, one_rewards],
        combination_strategy=lambda rewards: np.mean(rewards, axis=0),
        output_spec_shape=(), flatten_rewards=True)
    combined_reward.setup_io_spec(input_spec)
    output_timestep = combined_reward.process(self._input_timestep)
    self.assertEqual(output_timestep.reward, 0.5)

  def test_staged_active_monotonous(self):
    """More stages above threshold mean more reward, no matter the values."""
    reward_100 = _TestReward(reward_value=100.)
    reward_0 = _TestReward(reward_value=0.)
    staged_combination = rewards.StagedWithActiveThreshold(threshold=1e-6)
    combined_reward_1_above_thresh = rewards.CombineRewards(
        reward_preprocessors=[reward_100, reward_0],
        combination_strategy=staged_combination)

    reward_100 = _TestReward(reward_value=100.)
    reward_001 = _TestReward(reward_value=0.001)
    staged_combination = rewards.StagedWithActiveThreshold(threshold=1e-6)
    combined_reward_2_above_thresh = rewards.CombineRewards(
        reward_preprocessors=[reward_100, reward_001],
        combination_strategy=staged_combination)

    combined_reward_1_above_thresh.setup_io_spec(self._input_spec)
    combined_reward_2_above_thresh.setup_io_spec(self._input_spec)

    output_timestep_1_above_thresh = combined_reward_1_above_thresh.process(
        self._input_timestep)
    output_timestep_2_above_thresh = combined_reward_2_above_thresh.process(
        self._input_timestep)

    self.assertLess(output_timestep_1_above_thresh.reward,
                    output_timestep_2_above_thresh.reward)

  @parameterized.named_parameters(
      # Count 1, value 1.
      ('_contiguous', (1., 1., 0.5, 0.8, 0.), 0.9, 0.4),
      # Count 3 despite 0.5 among them; value 0.8.
      ('_count_below_thresh', (1., 1., 0.5, 0.8, 0.), 0.7, 0.76),
      # Count 3; value 1, NOT 100.
      ('_clip_final', (1., 1., 0.5, 100, 0.), 0.7, 0.8),
      # Count 3 despite 500 among them.
      ('_clip_mid', (1., 1., 500, 0.8, 0.), 0.7, 0.76),
  )
  def test_staged_active_clipping(self, term_rewards, threshold,
                                  expected_reward):
    """Terms are clipped if too large."""
    reward_preprocessors = [_TestReward(reward_value=r) for r in term_rewards]
    staged_combination = rewards.StagedWithActiveThreshold(threshold=threshold)
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=reward_preprocessors,
        combination_strategy=staged_combination)

    combined_reward.setup_io_spec(self._input_spec)
    output_timestep = combined_reward.process(self._input_timestep)

    self.assertAlmostEqual(expected_reward, output_timestep.reward)

  @parameterized.named_parameters(
      # Should not affect the last stage, even if above thresh.
      ('_singleton_above_thresh', (0.92,), 0.9, 0.92, True),
      # Should not affect the last stage, even if below thresh.
      ('_singleton_below_thresh', (0.82,), 0.9, 0.82, True),
      # First two tasks are solved so we work on third. `assume` flag irrelevant
      ('_monotonic_cumululative', (0.92, 0.91, 0.1), 0.9, 0.7, True),
      # First two tasks are solved so we work on third. `assume` flag irrelevant
      ('_monotonic_not_cumululative', (0.92, 0.91, 0.1), 0.9, 0.7, False),
      # Second task is solved so we assume first is too and work on third.
      ('_not_monotonic_cumululative', (0.6, 0.91, 0.1), 0.9, 0.7, True),
      # Second task is solved but first isn't so we work on that.
      ('_not_monotonic_not_cumululative', (0.6, 0.91, 0.1), 0.9, 0.2, False),
      # Nothing is solved so we work on the first task.
      ('_none_solved', (0.6, 0.2, 0.6), 0.9, 0.2, False),
  )
  def test_staged_success(self, term_rewards, threshold, expected_reward,
                          assume_cumulative_success):
    """Terms are clipped if too large."""
    reward_preprocessors = [_TestReward(reward_value=r) for r in term_rewards]
    staged_combination = rewards.StagedWithSuccessThreshold(
        threshold=threshold,
        assume_cumulative_success=assume_cumulative_success)
    combined_reward = rewards.CombineRewards(
        reward_preprocessors=reward_preprocessors,
        combination_strategy=staged_combination)

    combined_reward.setup_io_spec(self._input_spec)
    output_timestep = combined_reward.process(self._input_timestep)

    self.assertAlmostEqual(expected_reward, output_timestep.reward)


class _TestReward(timestep_preprocessor.TimestepPreprocessor):

  def __init__(self, reward_value: Union[float, int, np.ndarray]):
    super().__init__()
    self._reward_value = reward_value

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _process_impl(
      self, timestep: timestep_preprocessor.PreprocessorTimestep
  ) -> timestep_preprocessor.PreprocessorTimestep:
    return timestep._replace(reward=self._reward_value)

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    if np.isscalar(self._reward_value):
      self._reward_value = input_spec.reward_spec.dtype.type(self._reward_value)
    else:
      return input_spec.replace(reward_spec=specs.Array(
          self._reward_value.shape, self._reward_value.dtype))
    return input_spec


if __name__ == '__main__':
  absltest.main()
