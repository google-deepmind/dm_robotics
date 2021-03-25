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
"""A collection of timestep preprocessors that define rewards."""

from typing import Callable, Sequence, Text, Union

from dm_env import specs
import dm_robotics.agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import overrides
import numpy as np


# All rewards should either be a single float or array of floats.
RewardVal = Union[float, np.floating, np.ndarray]
RewardCombinationStrategy = Callable[[Sequence[RewardVal]], RewardVal]


class L2Reward(af.TimestepPreprocessor):
  """Returns a continuous reward based on the L2-distance between two keypoints.

  The keypoint position are sourced from the observations.
  """

  def __init__(self,
               obs0: Text,
               obs1: Text,
               reward_scale: float = 1.0,
               reward_offset: float = 1.0):
    """Initializes L2Reward.

    Args:
      obs0: The observation key for the first keypoint.
      obs1: The observation key for the second keypoint.
      reward_scale: Scalar multiplier.
      reward_offset: Scalar offset.
    """
    super(L2Reward, self).__init__()
    self._obs0 = obs0
    self._obs1 = obs1
    self._reward_scale = reward_scale
    self._reward_offset = reward_offset
    self._output_type = None  # type: np.dtype

  @overrides(af.TimestepPreprocessor)
  def _process_impl(
      self, timestep: af.PreprocessorTimestep) -> af.PreprocessorTimestep:
    try:
      obs0_val = timestep.observation[self._obs0]
      obs1_val = timestep.observation[self._obs1]
    except KeyError:
      raise KeyError(('{} or {} not a valid observation name. Valid names are '
                      '{}').format(self._obs0, self._obs1,
                                   list(timestep.observation.keys())))

    dist = np.linalg.norm(obs0_val - obs1_val)
    reward = self._output_type.type(-1 * dist * self._reward_scale +
                                    self._reward_offset)
    return timestep._replace(reward=reward)

  @overrides(af.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    # Reward computed from observation, so dtype can change; Set accordingly.
    type0 = input_spec.observation_spec[self._obs0]
    type1 = input_spec.observation_spec[self._obs1]
    self._output_type = np.promote_types(type0, type1)
    return input_spec.replace(
        reward_spec=input_spec.reward_spec.replace(
            dtype=self._output_type.type))


class ThresholdedL2Reward(af.TimestepPreprocessor):
  """Returns a sparse reward if two keypoints are within a threshold distance.

  The keypoint position are sourced from the observations.
  """

  def __init__(self, obs0, obs1, threshold, reward=1.0):
    super(ThresholdedL2Reward, self).__init__()
    self._obs0 = obs0
    self._obs1 = obs1
    self._threshold = threshold
    self._reward = reward
    self._zero_reward = 0.0

  @overrides(af.TimestepPreprocessor)
  def _process_impl(
      self, timestep: af.PreprocessorTimestep) -> af.PreprocessorTimestep:
    try:
      obs0_val = timestep.observation[self._obs0]
      obs1_val = timestep.observation[self._obs1]
    except KeyError:
      raise KeyError(('{} or {} not a valid observation name. Valid names are '
                      '{}').format(self._obs0, self._obs1,
                                   list(timestep.observation.keys())))

    dist = np.linalg.norm(obs0_val - obs1_val)
    reward = self._reward if dist < self._threshold else self._zero_reward
    return timestep._replace(reward=reward)

  @overrides(af.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    # Verify required keys are in the spec.
    for key in [self._obs0, self._obs1]:
      if key not in input_spec.observation_spec:
        raise KeyError('Expected "{}" key in observation not found.  Existing '
                       'keys: {}'.format(
                           key, input_spec.observation_spec.keys()))

    # Reward not computed from observation, so dtype should match input_spec.
    self._reward = input_spec.reward_spec.dtype.type(self._reward)
    self._zero_reward = input_spec.reward_spec.dtype.type(self._zero_reward)
    return input_spec


def _cast_reward_to_type(reward: RewardVal, dtype: np.dtype) -> RewardVal:
  if np.isscalar(reward):
    return dtype.type(reward)
  return reward.astype(dtype)  # pytype: disable=attribute-error


class ComputeReward(af.TimestepPreprocessor):
  """Computes a reward from the observations and adds it to the timestep."""

  def __init__(
      self,
      reward_function: Callable[[spec_utils.ObservationValue], RewardVal],
      output_spec_shape: Sequence[int] = ()):
    """ComputeReward constructor.

    Args:
      reward_function: Function that takes the timestep observation as input
        and returns a reward.
      output_spec_shape: Shape of the output reward. Defaults to an empty shape
        denoting a scalar reward.
    """
    super().__init__()
    self._reward_function = reward_function
    self._output_shape = output_spec_shape

  @overrides(af.TimestepPreprocessor)
  def _process_impl(
      self, timestep: af.PreprocessorTimestep) -> af.PreprocessorTimestep:
    reward = self._reward_function(timestep.observation)
    return timestep.replace(
        reward=_cast_reward_to_type(reward, self._out_spec.reward_spec.dtype))

  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    return input_spec.replace(reward_spec=specs.Array(
        shape=self._output_shape, dtype=input_spec.reward_spec.dtype))


class CombineRewards(af.TimestepPreprocessor):
  """Preprocessor which steps multiple rewards in sequence and combines them."""

  def __init__(self,
               reward_preprocessors: Sequence[af.TimestepPreprocessor],
               combination_strategy: RewardCombinationStrategy = np.max,
               output_spec_shape: Sequence[int] = (),
               flatten_rewards: bool = True):
    """CombineRewards constructor.

    Args:
      reward_preprocessors: List of rewards preprocessor to be evaluated
        sequentially.
      combination_strategy: Callable that takes the list of rewards coming from
        the `reward_preprocessors` and outputs a new reward. Defaults to
        `np.max`, which means that it returns the maximum of all the rewards.
      output_spec_shape: The shape of the output reward from
        `combination_strategy`. Defaults to an empty shape (for scalar rewards).
      flatten_rewards: If True, flattens any reward arrays coming from the
        `reward_preprocessors` before feeding them to the
        `combination_strategy`.

    Raises:
      ValueError: If no reward_preprocessors are given.
    """
    super().__init__()
    if not reward_preprocessors:
      raise ValueError('reward_preprocessors should have non-zero length')
    self._reward_preprocessors = reward_preprocessors
    self._combination_strategy = combination_strategy
    self._flatten_rewards = flatten_rewards
    self._output_shape = output_spec_shape
    self._output_type = None  # type: np.dtype

  @overrides(af.TimestepPreprocessor)
  def _process_impl(
      self, timestep: af.PreprocessorTimestep) -> af.PreprocessorTimestep:
    rewards = []
    for reward_preprocessor in self._reward_preprocessors:
      timestep = reward_preprocessor.process(timestep)
      if not np.isscalar(timestep.reward) and self._flatten_rewards:
        rewards.extend(timestep.reward)
      else:
        rewards.append(timestep.reward)

    reward = self._combination_strategy(rewards)
    return timestep._replace(
        reward=_cast_reward_to_type(reward, self._output_type))

  @overrides(af.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    for reward_preprocessor in self._reward_preprocessors:
      input_spec = reward_preprocessor.setup_io_spec(input_spec)
    self._output_type = input_spec.reward_spec.dtype
    return input_spec.replace(reward_spec=specs.Array(
        shape=self._output_shape, dtype=self._output_type))
