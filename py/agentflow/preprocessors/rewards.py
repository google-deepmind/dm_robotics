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

import collections
from typing import Callable, Optional, Sequence, Text, Union

from absl import logging
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import overrides
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
import numpy as np
import tree
import typing_extensions

# Internal profiling


# All rewards should either be a single float or array of floats.
RewardVal = Union[float, np.floating, np.ndarray]

# Callable for reward composition for `CombineRewards`.
# This callable receives the list of rewards generated from the list of
# `reward_preprocessors` passed to `CombineRewards` and returns a single reward.
RewardCombinationStrategy = Callable[[Sequence[RewardVal]], RewardVal]


class ThresholdReward(timestep_preprocessor.TimestepPreprocessor):
  """Returns a sparse reward if reward is above a threshold.
  """

  def __init__(
      self,
      *,
      threshold: float = 0.5,
      hi: float = 1.0,
      lo: float = 0.0,
      validation_frequency: timestep_preprocessor.ValidationFrequency = (
          timestep_preprocessor.ValidationFrequency.ONCE_PER_EPISODE),
      name: Optional[str] = None,
  ):
    """Initializes ThresholdReward.

    Args:
      threshold: Reward threshold.
      hi: Value to emit in reward field if incoming reward is greater than or
        equal to `threshold`.
      lo: Value to emit in reward field if incoming reward is below `threshold`.
      validation_frequency: How often should we validate the obs specs.
      name: A name for this preprocessor.
    """
    super().__init__(validation_frequency=validation_frequency, name=name)
    self._threshold = threshold
    self._hi = hi
    self._lo = lo

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _process_impl(
      self, timestep: timestep_preprocessor.PreprocessorTimestep
  ) -> timestep_preprocessor.PreprocessorTimestep:
    reward = self._hi if timestep.reward >= self._threshold else self._lo
    return timestep._replace(reward=reward)

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    # Reward not computed from observation, so dtype should match input_spec.
    self._hi = input_spec.reward_spec.dtype.type(self._hi)
    self._lo = input_spec.reward_spec.dtype.type(self._lo)
    return input_spec


class L2Reward(timestep_preprocessor.TimestepPreprocessor):
  """Returns a continuous reward based on the L2-distance between two keypoints.

  The keypoint position are sourced from the observations.
  """

  def __init__(
      self,
      obs0: Text,
      obs1: Text,
      *,
      reward_scale: float = 1.0,
      reward_offset: float = 1.0,
      validation_frequency: timestep_preprocessor.ValidationFrequency = (
          timestep_preprocessor.ValidationFrequency.ONCE_PER_EPISODE),
      name: Optional[str] = None,
  ):
    """Initializes L2Reward.

    Args:
      obs0: The observation key for the first keypoint.
      obs1: The observation key for the second keypoint.
      reward_scale: Scalar multiplier.
      reward_offset: Scalar offset.
      validation_frequency: How often should we validate the obs specs.
      name: A name for this preprocessor.
    """
    super().__init__(validation_frequency=validation_frequency, name=name)
    self._obs0 = obs0
    self._obs1 = obs1
    self._reward_scale = reward_scale
    self._reward_offset = reward_offset
    self._output_type = None  # type: np.dtype

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _process_impl(
      self, timestep: timestep_preprocessor.PreprocessorTimestep
  ) -> timestep_preprocessor.PreprocessorTimestep:
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

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    # Reward computed from observation, so dtype can change; Set accordingly.
    type0 = input_spec.observation_spec[self._obs0]
    type1 = input_spec.observation_spec[self._obs1]
    self._output_type = np.promote_types(type0, type1)
    return input_spec.replace(
        reward_spec=input_spec.reward_spec.replace(
            dtype=self._output_type.type))


class ThresholdedL2Reward(timestep_preprocessor.TimestepPreprocessor):
  """Returns a sparse reward if two keypoints are within a threshold distance.

  The keypoint position are sourced from the observations.
  """

  def __init__(
      self,
      obs0,
      obs1,
      *,
      threshold,
      reward: float = 1.0,
      validation_frequency: timestep_preprocessor.ValidationFrequency = (
          timestep_preprocessor.ValidationFrequency.ONCE_PER_EPISODE),
      name: Optional[str] = None,
  ):
    super().__init__(validation_frequency=validation_frequency, name=name)
    self._obs0 = obs0
    self._obs1 = obs1
    self._threshold = threshold
    self._reward = reward
    self._zero_reward = 0.0

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _process_impl(
      self, timestep: timestep_preprocessor.PreprocessorTimestep
  ) -> timestep_preprocessor.PreprocessorTimestep:
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

  @overrides(timestep_preprocessor.TimestepPreprocessor)
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


class ComputeReward(timestep_preprocessor.TimestepPreprocessor):
  """Computes a reward from the observations and adds it to the timestep."""

  def __init__(
      self,
      reward_function: Callable[[spec_utils.ObservationValue], RewardVal],
      *,
      output_spec_shape: Sequence[int] = (),
      validation_frequency: timestep_preprocessor.ValidationFrequency = (
          timestep_preprocessor.ValidationFrequency.ONCE_PER_EPISODE),
      name: Optional[str] = None,
  ):
    """ComputeReward constructor.

    Args:
      reward_function: Function that takes the timestep observation as input
        and returns a reward.
      output_spec_shape: Shape of the output reward. Defaults to an empty shape
        denoting a scalar reward.
      validation_frequency: How often should we validate the obs specs.
      name: A name for this preprocessor.
    """
    super().__init__(validation_frequency=validation_frequency, name=name)
    self._reward_function = reward_function
    self._output_shape = output_spec_shape

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  # Profiling for .wrap_scope('ComputeReward._process_impl')
  def _process_impl(
      self, timestep: timestep_preprocessor.PreprocessorTimestep
  ) -> timestep_preprocessor.PreprocessorTimestep:
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


@typing_extensions.runtime_checkable
class RewardFilter(typing_extensions.Protocol):
  """An object which filters a reward signal."""

  def __call__(self, reward: RewardVal) -> RewardVal:
    raise NotImplementedError()

  def state(self) -> Optional[spec_utils.ObservationValue]:
    """Returns the internal state of the filter.

    This state should have a consistent spec regardless of the filter's state.
    """
    return None

  def state_spec(self) -> Optional[spec_utils.ObservationSpec]:
    """Returns the spec of internal state of the filter.

    Note: if not provided, a `specs.Array` will be generated from `state`.
    """
    return None

  def reset(self) -> None:
    """Resets the filter state."""
    return


class ConsecutiveValueFilter(RewardFilter):
  """A filter which changes state only after K equal readings."""

  def __init__(
      self,
      num_steps: int = 2,
      obs_prefix: str = 'consecutive_reward_filter',
  ):
    """Initializes ConsecutiveValueFilter.

    Args:
      num_steps: Number of consecutive steps that a given reward must be seen
        before updating the output to that value.
      obs_prefix: The key under which to expose the counter
    """
    super().__init__()
    self._num_steps = num_steps
    self._obs_prefix = obs_prefix

    self._cur_reward = 0.
    self._last_reward = 0.
    self._ctr = 0

  def __call__(self, reward: RewardVal) -> RewardVal:
    if np.isclose(reward, self._last_reward):
      self._ctr += 1
    else:
      self._ctr = 0
      self._last_reward = reward

    if self._ctr >= (self._num_steps - 1):
      self._cur_reward = self._last_reward
    return self._cur_reward

  def state(self) -> Optional[spec_utils.ObservationValue]:
    # Note: `_cur_reward` not needed in state bc always returned as reward.
    return {
        f'{self._obs_prefix}_ctr_normalized': np.asarray(
            float(self._ctr) / self._num_steps, dtype=np.float32
        ),
        f'{self._obs_prefix}_last_reward': np.asarray(
            self._last_reward, dtype=np.float32
        ),
    }

  def reset(self) -> None:
    self._cur_reward = 0.
    self._last_reward = 0.
    self._ctr = 0


class StackedRewardsFilter(RewardFilter):
  """A filter which stacks rewards and applies a user-defined callable."""

  def __init__(
      self,
      fn: Callable[[Sequence[RewardVal]], RewardVal],
      num_steps: int = 2,
      obs_prefix: str = 'stacked_reward_filter',
  ):
    """Initializes StackedRewardsFilter.

    Args:
      fn: A callable taking the stacked rewards and returning a reward. This
        function should accept inputs with lengths between 0 and `num_steps`.
      num_steps: Number of rewards to stack into a last-k-rewards window for
        filtering by `fn`. These rewards are accumulated each time this filter
        is called.
      obs_prefix: The key under which to expose the counter
    """
    super().__init__()
    self._fn = fn
    self._obs_prefix = obs_prefix
    self._last_k_rewards = collections.deque([0.] * num_steps, maxlen=num_steps)

  def __call__(self, reward: RewardVal) -> RewardVal:
    self._last_k_rewards.append(reward)
    return self._fn(self._last_k_rewards)

  def state(self) -> Optional[spec_utils.ObservationValue]:
    return {
        f'{self._obs_prefix}_last_k_rewards': np.asarray(
            self._last_k_rewards, dtype=np.float32
        ),
    }

  def reset(self) -> None:
    num_steps = self._last_k_rewards.maxlen
    self._last_k_rewards = collections.deque([0.] * num_steps, maxlen=num_steps)

  @classmethod
  def min_last_k_above_threshold(
      cls,
      threshold: float,
      num_steps: int = 2,
      obs_prefix: str = 'stacked_reward_filter',
  ) -> 'StackedRewardsFilter':
    """Creates a callable with logic `np.amin(last_k_rewards) >= threshold`."""
    return cls(
        fn=lambda r: np.array(np.amin(r) >= threshold),
        num_steps=num_steps,
        obs_prefix=obs_prefix,
    )


class FilterReward(timestep_preprocessor.TimestepPreprocessor):
  """Applys a user-provided callable to the reward and exposes the filter obs.
  """

  def __init__(
      self,
      *,
      filter_fn: RewardFilter,
      name: Optional[str] = None,
  ):
    """Initializes FilterReward.

    Args:
      filter_fn: A callable which consumes a reward and emits a filtered reward.
        For stateless cases e.g. thresholding this can be a simple lambda (i.e.
        `lambda x: 1 if x > 0.5 else 0`), but for more complex cases with
        internal state consider implementing the full `RewardFilter` API to
        expose the state in the observation.
      name: A name for this TimestepPreprocessor.
    """
    super().__init__(name=name)
    self._filter_fn = filter_fn

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _process_impl(
      self, timestep: timestep_preprocessor.PreprocessorTimestep
  ) -> timestep_preprocessor.PreprocessorTimestep:
    if timestep.first() and isinstance(self._filter_fn, RewardFilter):
      self._filter_fn.reset()

    reward = self._filter_fn(timestep.reward)

    # Cast (possibly nested) reward to expected dtype.
    reward = tree.map_structure(
        lambda r: _cast_reward_to_type(r, self._out_spec.reward_spec.dtype),
        reward)
    timestep = timestep._replace(reward=reward)

    try:
      # Expose filter state as observation so the task is markov.
      filter_obs = self._filter_fn.state()
      if filter_obs is not None:
        processed_obs = dict(timestep.observation)
        processed_obs.update(filter_obs)
        timestep = timestep._replace(observation=processed_obs)
    except Exception:  # pylint: disable=broad-except
      logging.log_first_n(
          logging.INFO, 'Failed to generate filter state observation.', 10
      )

    return timestep

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec
  ) -> spec_utils.TimeStepSpec:

    # If filter is stateful, generate a spec for the filter observation.
    observation_spec = dict(input_spec.observation_spec)
    try:
      filter_obs_spec = self._filter_fn.state_spec()
      if filter_obs_spec is None:
        # Generate a value to produce a spec.
        filter_obs = self._filter_fn.state()
        if filter_obs is not None:
          filter_obs_spec = {
              k: specs.Array(
                  shape=np.asarray(v).shape, dtype=np.asarray(v).dtype, name=k
              )
              for k, v in filter_obs.items()
          }

      if filter_obs_spec is not None:
        conflicting_keys = set(input_spec.observation_spec.keys()).intersection(
            filter_obs_spec.keys()
        )
        if conflicting_keys:
          raise ValueError(f'Observations {conflicting_keys} already exist.')

        observation_spec.update(filter_obs_spec)

    except Exception:  # pylint: disable=broad-except
      logging.log_first_n(
          logging.INFO, 'Failed to generate filter state observation spec.', 10
      )

    dummy_reward = input_spec.reward_spec.generate_value()
    filtered_reward = self._filter_fn(dummy_reward)
    reward_spec = specs.Array(
        shape=np.asarray(filtered_reward).shape,
        dtype=input_spec.reward_spec.dtype,
    )

    return input_spec.replace(
        observation_spec=observation_spec, reward_spec=reward_spec
    )


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

  def __init__(
      self,
      threshold: float = 0.1,
  ):
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

  def __init__(
      self,
      threshold: float = 0.9,
      *,
      assume_cumulative_success: bool = True,
  ):
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
    solved_task_idxs = np.nonzero(tasks_above_threshold)[0]

    if self._assume_cumulative_success:
      if solved_task_idxs.size:
        num_tasks_solved = solved_task_idxs[-1] + 1
      else:
        num_tasks_solved = 0

    else:
      num_tasks_solved = np.argmin(tasks_above_threshold)  # first "False"

    # The last task should never be considered "solved" because we add
    # current_task_reward. If you want to apply a reward threshold to the last
    # stage to make it sparse, do that before or after passing it to this
    # function.
    num_tasks_solved = min(num_tasks_solved, num_stages - 1)
    current_task_reward = rewards[num_tasks_solved]

    return (num_tasks_solved + current_task_reward) / float(num_stages)


class CombineRewards(timestep_preprocessor.TimestepPreprocessor,
                     core.Renderable):
  """Preprocessor which steps multiple rewards in sequence and combines them."""

  def __init__(
      self,
      reward_preprocessors: Sequence[
          timestep_preprocessor.TimestepPreprocessor],
      combination_strategy: RewardCombinationStrategy = np.max,
      *,
      output_spec_shape: Sequence[int] = (),
      flatten_rewards: bool = True,
      validation_frequency: timestep_preprocessor.ValidationFrequency = (
          timestep_preprocessor.ValidationFrequency.ONCE_PER_EPISODE),
      name: Optional[str] = None,
  ):
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
      validation_frequency: How often should we validate the obs specs.
      name: A name for this preprocessor.

    Raises:
      ValueError: If no reward_preprocessors are given.
    """
    super().__init__(validation_frequency=validation_frequency, name=name)
    if not reward_preprocessors:
      raise ValueError('reward_preprocessors should have non-zero length')
    self._reward_preprocessors = reward_preprocessors
    self._combination_strategy = combination_strategy
    self._flatten_rewards = flatten_rewards
    self._output_shape = output_spec_shape
    self._output_type = None  # type: np.dtype

  @overrides(timestep_preprocessor.TimestepPreprocessor)
  # Profiling for .wrap_scope('CombineRewards._process_impl')
  def _process_impl(
      self, timestep: timestep_preprocessor.PreprocessorTimestep
  ) -> timestep_preprocessor.PreprocessorTimestep:
    # If this processor hasn't been setup yet, infer the type from the input
    # timestep, as opposed to the input_spec (should be equivalent). This
    # typically shouldn't happen, but allows stand-alone use-cases in which the
    # processor isn't run by a subtask or environment.
    output_type = self._output_type or np.asarray(timestep.reward).dtype

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
        lambda r: _cast_reward_to_type(r, output_type), reward)
    return timestep.replace(reward=reward)

  @overrides(timestep_preprocessor.TimestepPreprocessor)
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

  def as_list(self) -> timestep_preprocessor.TimestepPreprocessorTree:
    """Recursively lists processor and any child processor lists."""
    return [self, [proc.as_list() for proc in self._reward_preprocessors]]
