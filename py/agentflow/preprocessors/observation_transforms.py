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
"""A collection of timestep preprocessors that transform observations."""

import collections
from typing import Any, Callable, FrozenSet, Mapping, Optional, Sequence, Tuple

from absl import logging
import cv2
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import overrides
from dm_robotics.agentflow.preprocessors import timestep_preprocessor as tsp
from dm_robotics.geometry import geometry
import numpy as np

# Internal profiling


class MisconfigurationError(Exception):
  """Error raised when the preprocessor is misconfigured."""


class CastPreprocessor(tsp.TimestepPreprocessor):
  """Preprocessor to cast observations, reward and discount."""

  def __init__(
      self,
      dtype: type = np.float32,  # pylint: disable=g-bare-generic
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Initialize CastPreprocessor.

    Args:
      dtype: The target dtype to cast to.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)
    self._dtype = dtype

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('CastPreprocessor._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    cast_obs = {
        k: np.asarray(v).astype(self._dtype)
        for k, v in timestep.observation.items()
    }

    return tsp.PreprocessorTimestep(
        step_type=timestep.step_type,
        reward=(self._dtype(timestep.reward) if np.isscalar(timestep.reward)
                else timestep.reward.astype(self._dtype)),
        discount=self._dtype(timestep.discount),
        observation=cast_obs,
        pterm=timestep.pterm,
        result=timestep.result)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    obs_spec = {
        k: v.replace(dtype=self._dtype)
        for k, v in input_spec.observation_spec.items()
    }
    return spec_utils.TimeStepSpec(
        observation_spec=obs_spec,
        reward_spec=input_spec.reward_spec.replace(dtype=self._dtype),
        discount_spec=input_spec.discount_spec.replace(dtype=self._dtype))


class DowncastFloatPreprocessor(tsp.TimestepPreprocessor):
  """Preprocessor to cast observations, reward and discount.

  This preprocessor downcasts all floating point observations (etc)
  with more bits than the target dtype to the target dtype.

  It does not change the dtype of non-floating point data (e.g.
  uint8 in images).
  """

  def __init__(
      self,
      max_float_dtype: type,  # pylint: disable=g-bare-generic
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Initialize DowncastFloatPreprocessor.

    Args:
      max_float_dtype: The target dtype to cast floating point types with more
        bits to, e.g. np.float32.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)
    if not np.issubdtype(max_float_dtype, np.floating):
      raise ValueError('DowncastFloatPreprocessor only supports floating point '
                       f'dtypes, not {max_float_dtype}')
    self._dtype = max_float_dtype
    self._max_bits = np.finfo(self._dtype).bits

  def _dtype_needs_downcast(self, dtype):
    return (np.issubdtype(dtype, np.floating) and
            np.finfo(dtype).bits > self._max_bits)

  def _downcast_if_necessary(self, value):
    if ((hasattr(value, 'dtype') and self._dtype_needs_downcast(value.dtype)) or
        self._dtype_needs_downcast(type(value))):
      return np.asarray(value).astype(self._dtype)
    else:
      return value

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('DowncastFloatPreprocessor._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    cast_obs = {
        k: self._downcast_if_necessary(v)
        for k, v in timestep.observation.items()
    }

    return tsp.PreprocessorTimestep(
        step_type=timestep.step_type,
        reward=self._downcast_if_necessary(timestep.reward),
        discount=self._downcast_if_necessary(timestep.discount),
        observation=cast_obs,
        pterm=timestep.pterm,
        result=timestep.result)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    obs_spec = {}
    for k, v in input_spec.observation_spec.items():
      if self._dtype_needs_downcast(v.dtype):
        obs_spec[k] = v.replace(dtype=self._dtype)
      else:
        obs_spec[k] = v

    if self._dtype_needs_downcast(input_spec.reward_spec.dtype):
      reward_spec_dtype = self._dtype
    else:
      reward_spec_dtype = input_spec.reward_spec.dtype

    if self._dtype_needs_downcast(input_spec.discount_spec.dtype):
      discount_spec_dtype = self._dtype
    else:
      discount_spec_dtype = input_spec.reward_spec.dtype

    return spec_utils.TimeStepSpec(
        observation_spec=obs_spec,
        reward_spec=input_spec.reward_spec.replace(dtype=reward_spec_dtype),
        discount_spec=input_spec.discount_spec.replace(
            dtype=discount_spec_dtype))


class ObsRelativeToEpisodeStartPreprocessor(tsp.TimestepPreprocessor):
  """Offset specified observations to be relative to initial values."""

  def __init__(
      self,
      target_obs: str,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    super().__init__(validation_frequency)
    self._target_obs = target_obs
    self._initial_values = {}

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('ObsRelativeToEpisodeStartPreprocessor._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    if timestep.first():
      self._initial_values = {}
      for k, v in timestep.observation.items():
        if k in self._target_obs:
          self._initial_values[k] = np.array(v)

    corrected_obs = {}

    for k, v in timestep.observation.items():
      if k in self._initial_values:
        corrected_obs[k] = v - self._initial_values[k]
      else:
        corrected_obs[k] = v

    return timestep._replace(observation=corrected_obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    return input_spec


class PoseRelativeToEpisodeStart(tsp.TimestepPreprocessor):
  """Change pose observations to be relative to episode start."""

  def __init__(
      self,
      pos_obs_name: str,
      quat_obs_name: str,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """PoseRelativeToEpisodeStart constructor.

    Args:
      pos_obs_name: Observation key of the pos observation.
      quat_obs_name: Observation key of the quaternion observation.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)
    self._pos_obs_name = pos_obs_name
    self._quat_obs_name = quat_obs_name
    self._initial_pose = None  # type: Optional[geometry.PoseStamped]

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('PoseRelativeToEpisodeStart._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:

    pos = np.array(timestep.observation[self._pos_obs_name])
    quat = np.array(timestep.observation[self._quat_obs_name])

    if timestep.first():
      self._initial_pose = geometry.PoseStamped(geometry.Pose(pos, quat))

    corrected_obs = {}

    cur_pose = geometry.PoseStamped(geometry.Pose(pos, quat))
    rel_pose = geometry.frame_relative_pose(cur_pose, self._initial_pose)

    for k, v in timestep.observation.items():
      if k == self._pos_obs_name:
        corrected_obs[k] = rel_pose.position.astype(pos.dtype)
      elif k == self._quat_obs_name:
        corrected_obs[k] = rel_pose.quaternion.astype(quat.dtype)
      else:
        corrected_obs[k] = v

    return timestep._replace(observation=corrected_obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:

    if self._pos_obs_name not in input_spec.observation_spec:
      raise ValueError(f'{self._pos_obs_name} not in timestep observations')

    if self._quat_obs_name not in input_spec.observation_spec:
      raise ValueError(f'{self._quat_obs_name} not in timestep observations')

    return input_spec


class ObsOffsetAndScalingPreprocessor(tsp.TimestepPreprocessor):
  """Preprocessor to offset and scale specified observations."""

  def __init__(
      self,
      obs_offsets: Mapping[str, np.floating],
      obs_scales: Mapping[str, np.floating],
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    super().__init__(validation_frequency)
    self._obs_offsets = obs_offsets
    self._obs_scales = obs_scales

  # Profiling for .wrap('ObsOffsetAndScalingPreprocessor._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    corrected_obs = {}

    for k, obs in timestep.observation.items():
      apply_offset = k in self._obs_offsets
      apply_scaling = k in self._obs_scales

      if apply_offset:
        obs -= obs.dtype.type(self._obs_offsets[k])

      if apply_scaling:
        obs /= obs.dtype.type(self._obs_scales[k])

      corrected_obs[k] = obs

    return timestep._replace(observation=corrected_obs)

  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    return input_spec


class RemoveObservations(tsp.TimestepPreprocessor):
  """Removes the specified fields from observations."""

  def __init__(
      self,
      obs_to_strip: Sequence[str],
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Initialize RemoveObs.

    Args:
      obs_to_strip: A list of strings corresponding to keys to remove from
        timestep.observation.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)
    self._obs_to_strip = obs_to_strip

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('RemoveObservations._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    retained_obs = {
        k: v
        for k, v in timestep.observation.items()
        if k not in self._obs_to_strip
    }

    return timestep._replace(observation=retained_obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    obs_spec = {
        k: v
        for k, v in input_spec.observation_spec.items()
        if k not in self._obs_to_strip
    }
    return input_spec.replace(observation_spec=obs_spec)


class RetainObservations(tsp.TimestepPreprocessor):
  """Leaves only the specified observations."""

  def __init__(
      self,
      obs_to_leave: Sequence[str],
      raise_on_missing=True,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Initialize RetainObservations.

    Args:
      obs_to_leave: A list of strings corresponding to keys to retain in
        timestep.observation.
      raise_on_missing: Whether to raise a MisconfigurationError if we are asked
        to keep a non-existent observation.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)
    self._obs_to_leave: FrozenSet[str] = frozenset(obs_to_leave)
    self._raise_on_missing = raise_on_missing

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('RetainObservations._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    retained_obs = {
        k: v for k, v in timestep.observation.items() if k in self._obs_to_leave
    }
    return timestep._replace(observation=retained_obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    obs_spec = {
        k: v
        for k, v in input_spec.observation_spec.items()
        if k in self._obs_to_leave
    }
    not_in_spec = self._obs_to_leave - set(obs_spec)
    if not_in_spec:
      log_message = ('RetainObservations asked to retain observations that do '
                     'not exist in the incoming observation spec: '
                     f'{not_in_spec}')
      if self._raise_on_missing:
        raise MisconfigurationError(log_message)
      else:
        logging.warning(log_message)
    return input_spec.replace(observation_spec=obs_spec)


class RenameObservations(tsp.TimestepPreprocessor):
  """Renames a set of observations."""

  def __init__(
      self,
      obs_mapping: Mapping[str, str],
      raise_on_missing: bool = True,
      raise_on_overwrite: bool = True,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Initialize RenameObservations.

    Args:
      obs_mapping: Mapping from old and the new observation names.
      raise_on_missing: Whether to raise a MisconfigurationError if we are asked
        to rename a non-existent observation.
      raise_on_overwrite: Whether to raise a MisconfigurationError we are asked
        to rename an observation by overwriting an existing observation.
        validation_frequency: How often should we validate the obs specs.

    Raises:
      MisconfigurationError: If the mapping has duplicate names.
    """
    super().__init__(validation_frequency)

    self._raise_on_missing = raise_on_missing
    self._raise_on_overwrite = raise_on_overwrite
    self._obs_mapping = obs_mapping
    # Check that there are no duplicates in the mapped names.
    if len(set(obs_mapping.values())) != len(obs_mapping.values()):
      log_message = (f'The new set of observation names {obs_mapping.values()}'
                     ' has duplicate elements.')
      raise MisconfigurationError(log_message)

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('RenameObservations._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    observation = self._replace_obs(timestep.observation)
    return timestep._replace(observation=observation)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    obs_spec = input_spec.observation_spec
    self._check_valid_mapping(obs_spec)
    obs_spec = self._replace_obs(obs_spec)
    return input_spec.replace(observation_spec=obs_spec)

  def _replace_obs(self, orig: Mapping[str, Any]) -> Mapping[str, Any]:
    new_dict = {}
    for obs_key in orig:
      if obs_key in self._obs_mapping:
        new_dict[self._obs_mapping[obs_key]] = orig[obs_key]
      else:
        new_dict[obs_key] = orig[obs_key]
    return new_dict

  def _check_valid_mapping(self, obs_spec):
    """Checks that the renaming of observations is valid."""

    full_mapping = {key: key for key in obs_spec}
    full_mapping.update(self._obs_mapping)

    # Check that the renamed observations exist.
    not_in_spec = set(full_mapping) - set(obs_spec)
    if not_in_spec:
      log_message = ('RenameObservations asked to rename observations that do'
                     'not exist in the incoming observation spec: '
                     f'{not_in_spec}')
      if self._raise_on_missing:
        raise MisconfigurationError(log_message)
      else:
        logging.warning(log_message)

    # Check that we do not overwrite existing observations.
    c = collections.Counter(full_mapping.values())
    overwritten_names = [key for key, count in c.items() if count > 1]
    if overwritten_names:
      log_message = ('RenameObservations asked to overwrite the following '
                     f'existing observations: {overwritten_names}')
      if self._raise_on_overwrite:
        raise MisconfigurationError(log_message)
      else:
        logging.warning(log_message)


class MergeObservations(tsp.TimestepPreprocessor):
  """Creates a single observation by merging several observations together."""

  def __init__(
      self,
      obs_to_merge: Sequence[str],
      new_obs: str,
      raise_on_missing: bool = True,
      raise_on_overwrite: bool = True,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Initialize MergeObservations.

    Args:
      obs_to_merge: Names of the observations to merge.
      new_obs: Name of the merged observation.
      raise_on_missing: Whether to raise a MisconfigurationError if we are asked
        to merge a non-existent observation.
      raise_on_overwrite: Whether to raise a MisconfigurationError if the
        new_obs name overwrites an existing observation.
      validation_frequency: How often should we validate the obs specs.
    """

    super().__init__(validation_frequency)
    self._obs_to_merge = tuple(obs_to_merge)
    self._new_obs = new_obs
    self._raise_on_missing = raise_on_missing
    self._raise_on_overwrite = raise_on_overwrite

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('MergeObs._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    obs = dict(timestep.observation)
    # Create the merged observation.
    merged_obs = np.concatenate([
        timestep.observation[obs_key]
        for obs_key in self._obs_to_merge
        if obs_key in obs
    ])

    # Remove the observations that have been merged.
    for obs_key in self._obs_to_merge:
      obs.pop(obs_key)

    obs[self._new_obs] = merged_obs

    return timestep._replace(observation=obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    obs_spec = dict(input_spec.observation_spec)
    self._check_valid_merge(obs_spec)

    # Create the merged observation.
    model_array = np.concatenate([
        input_spec.observation_spec[obs_key].generate_value()
        for obs_key in self._obs_to_merge
        if obs_key in obs_spec
    ])

    # Remove the observations that have been merged.
    for obs_key in self._obs_to_merge:
      obs_spec.pop(obs_key)

    obs_spec[self._new_obs] = specs.Array(
        shape=model_array.shape, dtype=model_array.dtype, name=self._new_obs)
    return input_spec.replace(observation_spec=obs_spec)

  def _check_valid_merge(self, obs_spec):
    """Checks if the observation merging is valid."""
    all_current_names = set(obs_spec.keys())
    merged_names = set(self._obs_to_merge)
    # Check that the merged observations exist.
    not_in_spec = merged_names - all_current_names
    if not_in_spec:
      log_message = ('MergeObservations asked to merge observations that do not'
                     f'exist in the incoming observation spec: {not_in_spec}')
      if self._raise_on_missing:
        raise MisconfigurationError(log_message)
      else:
        logging.warning(log_message)

    # Check that the merged observation name doesn't overwrite an existing one.
    available_names = all_current_names - merged_names
    if self._new_obs in available_names:
      log_message = ('MergeObservations asked to overwrite observation name: '
                     f'{self._new_obs}')
      if self._raise_on_overwrite:
        raise MisconfigurationError(log_message)
      else:
        logging.warning(log_message)


class StackObservations(tsp.TimestepPreprocessor):
  """A timestep preprocessor that stacks observations.

  This is useful for environments that are n-step markov (like a robot that
  takes a few cycles to reach the setpoints we command). On the initial
  timestep, all elements of the stack are initialized with the value of the
  first observation.
  """

  def __init__(
      self,
      obs_to_stack: Sequence[str],
      stack_depth: np.integer,
      *,
      add_leading_dim: bool = False,
      override_obs: bool = True,
      added_obs_prefix: str = 'stacked_',
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """StackObservations preprocessor constructor.

    Args:
      obs_to_stack: A list of observation to stack.
      stack_depth: How deep to stack them. The stacked observations will be
        concatenated and replace the original observation if `override_obs` is
        set to True. Otherwise, extra observations with prefix
        `added_obs_prefix` will be added.
      add_leading_dim: If False, stacks the observations along the first
        dimension. If True, stacks the observations along an extra leading
        dimension. E.g.: (7,) stacked 3 times becomes: - (21,) if
        add_leading_dim=True - (3,7) if add_leading_dim=True (4,5) stacked 3
        times becomes: - (12, 5) if add_leading_dim=False - (3, 4, 5) if
        add_leading_dim=True
      override_obs: If True, add the stacked observations and replace the
        existing ones. Otherwise, the stacked observations will be added to the
        existing ones. The name of the stacked observation is given by
        `added_obs_prefix` added to their original name.
      added_obs_prefix: The prefix to be added to the original observation name.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)
    self._obs_to_stack: FrozenSet[str] = frozenset(obs_to_stack)
    self._stack_depth = stack_depth  # type: np.integer
    self._add_leading_dim = add_leading_dim
    self._stacks = {
        name: collections.deque(maxlen=self._stack_depth)
        for name in self._obs_to_stack
    }
    self._override_obs = override_obs
    self._added_obs_prefix = added_obs_prefix

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('StackObservations._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    if self._override_obs:
      processed_obs = {
          k: self._maybe_process(timestep, k, v)
          for k, v in timestep.observation.items()
      }
    else:
      stacked_obs = {
          self._added_obs_prefix + str(k):
          self._maybe_process(timestep, k, timestep.observation[k])
          for k in self._obs_to_stack
      }
      processed_obs = {**timestep.observation, **stacked_obs}

    return timestep._replace(observation=processed_obs)

  def _maybe_process(self, timestep, key, val):
    if key not in self._obs_to_stack:
      return val

    stack = self._stacks[key]
    if timestep.first():
      stack.clear()
      stack.extend([val] * (self._stack_depth - 1))
    stack.appendleft(val)
    if self._add_leading_dim:
      return np.array(stack)
    else:
      return np.concatenate(stack)

  def _maybe_process_spec(self, key, spec):
    if key not in self._obs_to_stack:
      return spec
    if self._add_leading_dim:
      model_array = np.array([spec.generate_value()] * self._stack_depth)
    else:
      model_array = np.concatenate([spec.generate_value()] * self._stack_depth)
    return specs.Array(
        shape=model_array.shape, dtype=model_array.dtype, name=spec.name)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    if self._override_obs:
      processed_obs_spec = {
          k: self._maybe_process_spec(k, v)
          for k, v in input_spec.observation_spec.items()
      }
    else:
      stacked_obs_spec = {
          self._added_obs_prefix + str(k):
          self._maybe_process_spec(k, input_spec.observation_spec[k])
          for k in self._obs_to_stack
      }
      processed_obs_spec = {**input_spec.observation_spec, **stacked_obs_spec}

    return input_spec.replace(processed_obs_spec)


class UnstackObservations(tsp.TimestepPreprocessor):
  """A timestep preprocessor that unstacks observations."""

  def __init__(
      self,
      obs_to_unstack: Sequence[str],
      override_obs: bool = False,
      added_obs_prefix: str = 'unstacked_',
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """UnstackObservations preprocessor constructor.

    Args:
      obs_to_unstack: A list of observation to unstack.
      override_obs: If True, add the unstacked observations and replace the
        existing ones. Otherwise, the unstacked observations will be added to
        the existing ones. The name of the ustacked observation is given by
        `added_obs_prefix` added to their original name.
      added_obs_prefix: The prefix to be added to the original observation name.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)
    self._obs_to_unstack: FrozenSet[str] = frozenset(obs_to_unstack)
    self._override_obs = override_obs
    self._added_obs_prefix = added_obs_prefix

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap('UnstackObservations._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
    if self._override_obs:
      processed_obs = {
          k: self._maybe_process(timestep, k, v)
          for k, v in timestep.observation.items()
      }
    else:
      unstacked_obs = {
          self._added_obs_prefix + str(k):
          self._maybe_process(timestep, k, timestep.observation[k])
          for k in self._obs_to_unstack
      }
      processed_obs = {**timestep.observation, **unstacked_obs}

    return timestep._replace(observation=processed_obs)

  def _maybe_process(self, timestep, key, val):
    if key not in self._obs_to_unstack:
      return val
    unstack = timestep.observation[key][0]
    if not unstack.shape:
      unstack = np.asarray([unstack])
    return unstack

  def _maybe_process_spec(self, key, spec):
    if key not in self._obs_to_unstack:
      return spec
    model_array = spec.generate_value()[0]
    if not model_array.shape:
      model_array = np.asarray([model_array])
    return specs.Array(
        shape=model_array.shape, dtype=model_array.dtype, name=spec.name)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    if self._override_obs:
      processed_obs_spec = {
          k: self._maybe_process_spec(k, v)
          for k, v in input_spec.observation_spec.items()
      }
    else:
      unstacked_obs_spec = {
          self._added_obs_prefix + str(k): self._maybe_process_spec(k, v)
          for k, v in input_spec.observation_spec.items()
      }
      processed_obs_spec = {**input_spec.observation_spec, **unstacked_obs_spec}

    return input_spec.replace(processed_obs_spec)


class FoldObservations(tsp.TimestepPreprocessor):
  """Performs a fold operation and transormation some observation."""

  def __init__(
      self,
      output_obs_name: str,
      obs_to_fold: str,
      fold_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
      output_fn: Callable[[np.ndarray], np.ndarray],
      init_val: np.ndarray,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    super().__init__(validation_frequency)

    self._output_obs_name = output_obs_name
    self._obs_to_fold = obs_to_fold
    self._fold_fn = fold_fn
    self._output_fn = output_fn
    self._init_val = init_val

    self._cur_val = init_val

  @overrides(tsp.TimestepPreprocessor)
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:

    if timestep.step_type.first():
      self._cur_val = self._init_val

    step_val = timestep.observation[self._obs_to_fold]
    self._cur_val = self._fold_fn(self._cur_val, step_val)

    processed_obs = {k: v for k, v in timestep.observation.items()}

    output_val = self._output_fn(self._cur_val).astype(self._init_val.dtype)
    processed_obs[self._output_obs_name] = output_val

    return timestep._replace(observation=processed_obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    observation_spec = {k: v for k, v in input_spec.observation_spec.items()}
    observation_spec[self._output_obs_name] = specs.Array(
        shape=self._init_val.shape,
        dtype=self._init_val.dtype,
        name=self._output_obs_name)

    return input_spec.replace(observation_spec=observation_spec)


class ImageCropper(object):
  """Helper class that crops an image."""

  def __init__(
      self,
      crop_width_relative: float,
      crop_height_relative: Optional[float] = None,
      x_offset_relative: float = 0.0,
      y_offset_relative: float = 0.0,
  ):
    """This initializes internal variables that are reused for every crop operation.

    Args:
      crop_width_relative: What fraction of the original width to crop to. For
        example, for an image that is 100px wide, a value of 0.65 would crop a
        region that is 65px wide. Cannot be zero.
      crop_height_relative: Optional fraction of the original height to crop to.
        Cannot be zero. If omitted, default to a square with the side length
        implied by crop_width_relative.
      x_offset_relative: X offset for the crop. 0 means the left edge of the
        crop is aligned with the left edge of the source. 0.5 means the *center*
        of the crop is aligned with the center of the source. 1.0 means the
        *right* edge of the crop is aligned with the right edge of the source.
      y_offset_relative: Behaves like x_offset_relative, but for the y axis.
    """

    # Check parameters for limit violations (all the limits are [0,1])
    def check_limit(value: float, name: str):
      if value < 0.0 or value > 1.0:
        raise ValueError('{} must be between 0 and 1, is {}'.format(
            name, value))

    check_limit(crop_width_relative, 'Crop width')
    if crop_width_relative == 0.0:
      raise ValueError('Crop width cannot be zero!')
    if crop_height_relative is not None:
      check_limit(crop_height_relative, 'Crop height')
      if crop_height_relative == 0.0:
        raise ValueError('Crop height cannot be zero!')
    check_limit(x_offset_relative, 'X offset')
    check_limit(y_offset_relative, 'Y offset')

    self._x_offset_relative = x_offset_relative
    self._y_offset_relative = y_offset_relative
    self._crop_width_relative = crop_width_relative
    self._crop_height_relative = crop_height_relative

    self._cropped_width = None  # type: Optional[int]
    self._cropped_height = None  # type: Optional[int]
    self._x_offset = None  # type: Optional[int]
    self._y_offset = None  # type: Optional[int]

    self._last_input_width = None  # type: Optional[int]
    self._last_input_height = None  # type: Optional[int]

  def calculate_crop_params(self, input_width: int,
                            input_height: int) -> Tuple[int, int]:
    """Calculate the actual size of the crop in pixels.

    Saves the width and height used to avoid unnecessary calculations.

    Args:
      input_width: Width of the image to be cropped, in pixels.
      input_height: Height of the image to be cropped, in pixels.

    Returns:
      A tuple (output_width, output_height).

    Raises:
      ValueError if only crop width was set (in this case, crop height
      defaults to be equal to the width), and the resulting square is larger
      than the image.
    """
    # Only do the math if input_width or input_height changed from the last time
    # we were called.
    if (input_width != self._last_input_width or
        input_height != self._last_input_height):
      self._cropped_width = max(
          1, self._fraction_of_pixels(self._crop_width_relative, input_width))
      self._cropped_height = (
          self._cropped_width if self._crop_height_relative is None else max(
              1, self._fraction_of_pixels(self._crop_height_relative,
                                          input_height)))
      if self._cropped_height > input_height:
        raise ValueError(
            'Crop height is {}, but input is only {} pixels high!'.format(
                self._cropped_height, input_height))
      self._x_offset = self._fraction_of_pixels(
          self._x_offset_relative, input_width - self._cropped_width)
      self._y_offset = self._fraction_of_pixels(
          self._y_offset_relative, input_height - self._cropped_height)

    # Return the results to use outside this class.
    return (self._cropped_width, self._cropped_height)

  def _fraction_of_pixels(self, fraction: float, total_pixels: int) -> int:
    """Calculate a number of pixels based on ratio and total_pixels.

    This function exists to ensure that all conversions from relative sizes to
    pixels use the same logic.

    Args:
      fraction: ]0.0,1.0], fraction of total_pixels to calculate.
      total_pixels: Total number of pixels in the relevant dimensions.

    Returns:
      The requested fraction of the given pixel size, rounded to the next
      integer. I.e. running this with ratio=1 will always return total_pixels,
      running with ratio=0 will always return 0.

    Raises:
      ValueError if ratio is not in [0,1]
      ValueError if total_pixels is < 0
    """
    if fraction < 0.0 or fraction > 1.0:
      raise ValueError(
          'Fraction must be between 0 and 1, is {}'.format(fraction))
    if total_pixels < 0:
      raise ValueError('Total number of pixels must be positive, got {}'.format(
          total_pixels))
    return int(round(float(total_pixels) * fraction))

  def crop(self, image: np.ndarray) -> np.ndarray:
    """Crop the given image."""
    if len(image.shape) < 2:
      raise ValueError('Cropper requires at least 2 dimensions, got '
                       'shape {}'.format(image.shape))
    width = image.shape[1]
    height = image.shape[0]
    # This bails out early if we already know the parameters for this width and
    # height.
    self.calculate_crop_params(input_width=width, input_height=height)
    return image[self._y_offset:self._y_offset + self._cropped_height,
                 self._x_offset:self._x_offset + self._cropped_width]


class CropImageObservation(tsp.TimestepPreprocessor):
  """Crops an image observation to the desired shape."""

  def __init__(
      self,
      input_obs_name: str,
      output_obs_name: str,
      crop_width_relative: float,
      crop_height_relative: Optional[float] = None,
      x_offset_relative: float = 0.0,
      y_offset_relative: float = 0.0,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Build a CropImageObservation preprocessor.

    Args:
      input_obs_name: Name of the input observation. This must be a 2D array.
      output_obs_name: Name of the output observation.
      crop_width_relative: What fraction of the original width to crop to. For
        example, for an image that is 100px wide, a value of 0.65 would crop a
        region that is 65px wide. Cannot be zero.
      crop_height_relative: Optional fraction of the original height to crop to.
        Cannot be zero. If omitted, default to a square with the side length
        implied by crop_width_relative.
      x_offset_relative: X offset for the crop. 0 means the left edge of the
        crop is aligned with the left edge of the source. 0.5 means the *center*
        of the crop is aligned with the center of the source. 1.0 means the
        *right* edge of the crop is aligned with the right edge of the source.
      y_offset_relative: Behaves like x_offset_relative, but for the y axis.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)

    # Will raise a ValueError if any of the parameters are not OK.
    self._cropper = ImageCropper(
        crop_width_relative=crop_width_relative,
        crop_height_relative=crop_height_relative,
        x_offset_relative=x_offset_relative,
        y_offset_relative=y_offset_relative)

    self._input_obs_name = input_obs_name
    self._output_obs_name = output_obs_name

  def _process_image(self, image: np.ndarray):
    return self._cropper.crop(image)

  @overrides(tsp.TimestepPreprocessor)
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:

    processed_obs = dict(timestep.observation)
    processed_obs[self._output_obs_name] = self._process_image(
        timestep.observation[self._input_obs_name])

    return timestep._replace(observation=processed_obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:

    input_observation_spec = input_spec.observation_spec[self._input_obs_name]

    shape = input_observation_spec.shape
    if len(shape) < 2:
      raise ValueError(
          'CropImageObservation preprocessor expects 2D image observation, got '
          'shape {}'.format(shape))
    width = shape[1]
    height = shape[0]
    cropped_width, cropped_height = self._cropper.calculate_crop_params(
        input_width=width, input_height=height)

    observation_spec = dict(input_spec.observation_spec)

    observation_spec[self._output_obs_name] = specs.Array(
        shape=(cropped_height, cropped_width) + shape[2:],
        dtype=input_observation_spec.dtype,
        name=self._output_obs_name)

    return input_spec.replace(observation_spec=observation_spec)


class CropSquareAndResize(CropImageObservation):
  """Crop a square from an image observation and resample it to the desired size in pixels."""

  def __init__(
      self,
      input_obs_name: str,
      output_obs_name: str,
      crop_width_relative: float,
      side_length_pixels: int,
      x_offset_relative: float = 0.0,
      y_offset_relative: float = 0.0,
      interpolation=cv2.INTER_LINEAR,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Build a CropImageObservation preprocessor.

    Args:
      input_obs_name: Name of the input observation. This must be a 2D array.
      output_obs_name: Name of the output observation.
      crop_width_relative: What fraction of the original width to crop to. For
        example, for an image that is 100px wide, a value of 0.65 would crop a
        region that is 65px wide. This defines both the width and height of the
        crop, so if the image is wider than it is tall, there exist values that
        can lead to invalid crops at runtime! Cannot be zero.
      side_length_pixels: The crop will be resampled so that its side length
        matches this.
      x_offset_relative: What fraction of the original width to offset the crop
        by. Defaults to 0.0.
      y_offset_relative: What fraction of the original height to offset the crop
        by. Defaults to 0.0.
      interpolation: The interpolation method to use. Supported values are
        cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(
        input_obs_name=input_obs_name,
        output_obs_name=output_obs_name,
        crop_width_relative=crop_width_relative,
        crop_height_relative=None,
        x_offset_relative=x_offset_relative,
        y_offset_relative=y_offset_relative,
        validation_frequency=validation_frequency,
    )

    if side_length_pixels <= 0:
      raise ValueError(
          'Side length must be > 0, got {}'.format(side_length_pixels))
    self._side_length_pixels = side_length_pixels
    self._interpolation = interpolation

  def _process_image(self, image: np.ndarray):
    crop = super()._process_image(image)

    return cv2.resize(
        crop, (self._side_length_pixels, self._side_length_pixels),
        interpolation=self._interpolation)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    cropped_input_spec = super()._output_spec(input_spec)
    input_observation_spec = cropped_input_spec.observation_spec[
        self._input_obs_name]
    shape = input_observation_spec.shape
    observation_spec = dict(input_spec.observation_spec)
    observation_spec[self._output_obs_name] = specs.Array(
        shape=(self._side_length_pixels, self._side_length_pixels) + shape[2:],
        dtype=input_observation_spec.dtype,
        name=self._output_obs_name)

    return input_spec.replace(observation_spec=observation_spec)


class ResizeImage(tsp.TimestepPreprocessor):
  """Resample an image observation to the desired size in pixels.

  Resulting image is reshaped into a square if it is not already.
  """

  def __init__(
      self,
      input_obs_name: str,
      output_obs_name: str,
      side_length_pixels: int,
      interpolation=cv2.INTER_LINEAR,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """Build a ResizeImage preprocessor.

    Args:
      input_obs_name: Name of the input observation. The observation must be a
        2D array.
      output_obs_name: Name of the output observation.
      side_length_pixels: The image will be resampled so that its side length
        matches this.
      interpolation: The interpolation method to use. Supported values are
        cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)

    if side_length_pixels <= 0:
      raise ValueError(
          'Side length must be > 0, got {}'.format(side_length_pixels))

    self._input_obs_name = input_obs_name
    self._output_obs_name = output_obs_name
    self._side_length_pixels = side_length_pixels
    self._interpolation = interpolation

  def _process_image(self, image: np.ndarray):
    return cv2.resize(
        image, (self._side_length_pixels, self._side_length_pixels),
        interpolation=self._interpolation)

  @overrides(tsp.TimestepPreprocessor)
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:

    processed_obs = dict(timestep.observation)
    processed_obs[self._output_obs_name] = self._process_image(
        timestep.observation[self._input_obs_name])

    return timestep._replace(observation=processed_obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    input_observation_spec = input_spec.observation_spec[self._input_obs_name]
    shape = input_observation_spec.shape
    observation_spec = dict(input_spec.observation_spec)
    observation_spec[self._output_obs_name] = specs.Array(
        shape=(self._side_length_pixels, self._side_length_pixels) + shape[2:],
        dtype=input_observation_spec.dtype,
        name=self._output_obs_name)

    return input_spec.replace(observation_spec=observation_spec)


class AddObservation(tsp.TimestepPreprocessor):
  """Preprocessor that adds an observation."""

  def __init__(
      self,
      obs_name: str,
      obs_callable: Callable[[tsp.PreprocessorTimestep], np.ndarray],
      obs_spec: Optional[specs.Array] = None,
      validation_frequency: tsp.ValidationFrequency = (
          tsp.ValidationFrequency.ONCE_PER_EPISODE),
  ):
    """AddObservation constructor.

    Args:
      obs_name: Name of the observation to add.
      obs_callable: Callable generating the observation to be added value given
        a timestep.
      obs_spec: Specs for the output of `obs_callable`. If `None` is provided
        the specs are inferred as a `dm_env.specs.Array` with shape and dtype
        matching the output of `obs_callable` and name  set to `obs_name`.
      validation_frequency: How often should we validate the obs specs.
    """
    super().__init__(validation_frequency)

    self._obs_name = obs_name
    self._obs_callable = obs_callable
    self._obs_spec = obs_spec

  @overrides(tsp.TimestepPreprocessor)
  # Profiling for .wrap_scope('AddObservation._process_impl')
  def _process_impl(
      self, timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:

    processed_obs = dict(timestep.observation)
    processed_obs[self._obs_name] = np.asarray(self._obs_callable(timestep))

    return timestep._replace(observation=processed_obs)

  @overrides(tsp.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:

    observation_spec = dict(input_spec.observation_spec)
    if self._obs_name in observation_spec.keys():
      raise ValueError(f'Observation {self._obs_name} already exists.')

    dummy_input = tsp.PreprocessorTimestep.from_environment_timestep(
        input_spec.minimum(), pterm=0.0)
    try:
      dummy_obs = np.asarray(self._obs_callable(dummy_input))
    except Exception:
      logging.exception('Failed to run the obs_callable to add observation %s.',
                        self._obs_name)
      raise

    if self._obs_spec is None:
      self._obs_spec = specs.Array(
          shape=dummy_obs.shape, dtype=dummy_obs.dtype, name=self._obs_name)

    observation_spec[self._obs_name] = self._obs_spec

    return input_spec.replace(observation_spec=observation_spec)
