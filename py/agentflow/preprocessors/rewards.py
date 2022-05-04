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
from dm_robotics.agentflow import core
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import overrides
import numpy as np
import tree

# Internal profiling


# All rewards should either be a single float or array of floats.
RewardVal = Union[float, np.floating, np.ndarray]

# Callable for reward composition for `CombineRewards`.
# This callable receives the list of rewards generated from the list of
# `reward_preprocessors` passed to `CombineRewards` and returns a single reward.
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
    super().__init__()
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
    except KeyError as key_missing:
      raise KeyError(
          f'{self._obs0} or {self._obs1} not a valid observation name. Valid '
          f'names are {list(timestep.observation.keys())}') from key_missing

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
    super().__init__()
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
    except KeyError as key_missing:
      raise KeyError(
          f'{self._obs0} or {self._obs1} not a valid observation name. Valid '
          f'names are {list(timestep.observation.keys())}') from key_missing

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
  # Profiling for .wrap_scope('ComputeReward._process_impl')
  def _process_impl(
      self, timestep: af.PreprocessorTimestep) -> af.PreprocessorTimestep:
    reward = self._reward_function(timestep.observation)
    # Cast (possibly nested) reward to expected dtype.
    reward = tree.map_structure(
        lambda r: _cast_reward_to_type(r, self._out_spec.reward_spec.dtype),
        reward)
    return timestep.replace(reward=reward)

  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    return input_spec.replace(reward_spec=specs.Array(
        shape=self._output_shape, dtype=input_spec.reward_spec.dtype))


class StagedWithActiveThreshold(RewardCombinationStrategy):
  """A RewardCombinationStrategy that stages a sequences of rewards.

  It creates a reward for following a particular sequence of tasks in order,
  given a reward value for each individual task.

  This works by cycling through the terms backwards and using the last reward
  that gives a response above the provided threshold + the number of terms
  preceding it.

  With this strategy the agent starts working on a task as soon as it's above
  the provided threshold. This was useful for the RGB stacking task in which it
  wasn't clear how close a reward could get to 1.0 for many tasks, but it was
  easy to see when a reward started to move and should therefore be switched to.

  E.g. if the threshold is 0.9 and the reward sequence is [0.95, 0.92, 0.6] it
  will output (1 + 0.92) / 3 = 0.64.

  Note: Preceding terms are given 1.0, not their current reward value.
  This assumes tasks are ordered such that success on task `i` implies all
  previous tasks `<i` are also solved, and thus removes the need to tune earlier
  rewards to remain above-threshold in all down-stream tasks.

  Use this for tasks in which it is more natural to express a threshold on which
  a task is active, vs. when it is solved. See `StagedWithSuccessThreshold` if
  the converse is true.

  Rewards must be in [0;1], otherwise they will be clipped.
  """

  def __init__(self, threshold: float = 0.1):
    """Initialize Staged.

    Args:
      threshold: A threshold that a reward must exceed for that task to be
        considered "active". All previous tasks are assumed solved.
    """
    self._thresh = threshold

  def __call__(self, rewards: Sequence[RewardVal]) -> RewardVal:
    rewards = np.clip(rewards, 0, 1)
    last_reward = 0.
    num_stages = len(rewards)
    for i, last_reward in enumerate(reversed(rewards)):
      if last_reward >= self._thresh:
        # Found a reward at/above the threshold, add number of preceding terms
        # and normalize with the number of terms.
        return (num_stages - (i + 1) + last_reward) / float(num_stages)

    # Return the accumulated rewards.
    return last_reward / num_stages


class StagedWithSuccessThreshold(RewardCombinationStrategy):
  """A RewardCombinationStrategy that stages a sequences of rewards.

  It creates a reward for following a particular sequence of tasks in order,
  given a reward value for each individual task.

  Unlike `StagedWithActiveThreshold`, which only gives reward for tasks above
  threshold, this function gives (normalized) reward 1.0 for all solved tasks,
  as well as the current shaped value for the first unsolved task.

  E.g. if the threshold is 0.9 and the reward sequence is [0.95, 0.92, 0.6] it
  will output (2 + 0.6) / 3 = 0.8666.

  With this strategy the agent starts working on a task as soon as the PREVIOUS
  task is above the provided threshold. Use this for tasks in which it is more
  natural to express a threshold on which a task is solved, vs. when it is
  active.

  E.g. a sequence of object-rearrangement tasks may have arbitrary starting
  reward due to their current positions, but the reward will always saturate
  towards 1 when the task is solved. In this case it would be difficult to set
  an "active" threshold without skipping stages.

  Rewards must be in [0;1], otherwise they will be clipped.
  """

  def __init__(self,
               threshold: float = 0.9,
               assume_cumulative_success: bool = True):
    """Initialize Staged.

    Args:
      threshold: A threshold that each reward must exceed for that task to be
        considered "solved".
      assume_cumulative_success: If True, assumes all tasks before the last task
        above threshold are also solved and given reward 1.0, regardless of
        their current value. If False, only the first K continguous tasks above
        threshold are considered solved.
    """
    self._thresh = threshold
    self._assume_cumulative_success = assume_cumulative_success

  def __call__(self, rewards: Sequence[RewardVal]) -> RewardVal:
    rewards = np.clip(rewards, 0, 1)

    num_stages = len(rewards)
    tasks_above_threshold = np.asarray(rewards) > self._thresh

    if self._assume_cumulative_success:
      if np.any(tasks_above_threshold):
        solved_task_idxs = np.argwhere(tasks_above_threshold)  # last "True"
        num_tasks_solved = solved_task_idxs.max() + 1
      else:
        num_tasks_solved = 0

    else:
      num_tasks_solved = np.argmin(tasks_above_threshold)  # first "False"

    current_task_idx = num_tasks_solved + 1
    if current_task_idx > len(rewards):
      current_task_reward = 0
    else:
      current_task_reward = rewards[num_tasks_solved]

    return (num_tasks_solved + current_task_reward) / float(num_stages)


class CombineRewards(af.TimestepPreprocessor, core.Renderable):
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
  # Profiling for .wrap_scope('CombineRewards._process_impl')
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

    # Cast (possibly nested) reward to expected dtype.
    reward = tree.map_structure(
        lambda r: _cast_reward_to_type(r, self._output_type), reward)
    return timestep.replace(reward=reward)

  @overrides(af.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    for reward_preprocessor in self._reward_preprocessors:
      input_spec = reward_preprocessor.setup_io_spec(input_spec)
    self._output_type = input_spec.reward_spec.dtype
    return input_spec.replace(reward_spec=specs.Array(
        shape=self._output_shape, dtype=self._output_type))

  def render_frame(self, canvas) -> None:
    """Callback to allow preprocessors to draw on a canvas."""
    for preprocessor in self._reward_preprocessors:
      if isinstance(preprocessor, core.Renderable):
        preprocessor.render_frame(canvas)
