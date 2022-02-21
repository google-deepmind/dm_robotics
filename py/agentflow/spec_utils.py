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

"""Utilities for dealing with action and observation specifications.

These specifications can be nested lists and dicts of `Array` and its
subclass `BoundedArray`.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple, Type, TypeVar

from absl import logging
import dm_env
from dm_env import specs
import numpy as np

# Internal profiling


ObservationSpec = Mapping[str, specs.Array]
ObservationValue = Mapping[str, np.ndarray]
ScalarOrArray = TypeVar('ScalarOrArray', np.floating, np.ndarray)


class TimeStepSpec(object):
  """Type specification for a TimeStep."""

  def __init__(self, observation_spec: ObservationSpec,
               reward_spec: specs.Array, discount_spec: specs.Array):
    self._observation_spec = observation_spec
    self._reward_spec = reward_spec
    self._discount_spec = discount_spec

  @property
  def observation_spec(self) -> Mapping[str, specs.Array]:
    return dict(self._observation_spec)

  @property
  def reward_spec(self) -> specs.Array:
    return self._reward_spec

  @property
  def discount_spec(self) -> specs.Array:
    return self._discount_spec

  def validate(self, timestep: dm_env.TimeStep):
    validate_observation(self.observation_spec, timestep.observation)
    validate(self.reward_spec, timestep.reward)
    validate(self.discount_spec, timestep.discount)

  def minimum(self) -> dm_env.TimeStep:
    """Return a valid timestep with all minimum values."""
    reward = minimum(self._reward_spec)
    discount = minimum(self._discount_spec)
    observation = {k: minimum(v) for k, v in self._observation_spec.items()}

    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        observation=observation,
        discount=discount,
        reward=reward)

  def maximum(self) -> dm_env.TimeStep:
    """Return a valid timestep with all minimum values."""
    reward = maximum(self._reward_spec)
    discount = maximum(self._discount_spec)
    observation = {k: maximum(v) for k, v in self._observation_spec.items()}

    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        observation=observation,
        discount=discount,
        reward=reward)

  def replace(self,
              observation_spec: Optional[Mapping[str, specs.Array]] = None,
              reward_spec: Optional[specs.Array] = None,
              discount_spec: Optional[specs.Array] = None) -> 'TimeStepSpec':
    """Return a new TimeStepSpec with specified fields replaced."""
    if observation_spec is None:
      observation_spec = self._observation_spec
    if reward_spec is None:
      reward_spec = self._reward_spec
    if discount_spec is None:
      discount_spec = self._discount_spec
    return TimeStepSpec(
        observation_spec=observation_spec,
        reward_spec=reward_spec,
        discount_spec=discount_spec)

  def __eq__(self, other):
    if not isinstance(other, TimeStepSpec):
      return False
    # All the properties of the spec must be equal.
    if self.reward_spec != other.reward_spec:
      return False
    if self.discount_spec != other.discount_spec:
      return False
    if len(self.observation_spec) != len(other.observation_spec):
      return False
    for key in self.observation_spec:
      if (key not in other.observation_spec or
          self.observation_spec[key] != other.observation_spec[key]):
        return False
    return True


def minimum(spec: specs.Array):
  if hasattr(spec, 'minimum'):
    return clip(np.asarray(spec.minimum, dtype=spec.dtype), spec)
  elif np.issubdtype(spec.dtype, np.integer):
    return np.full(spec.shape, np.iinfo(spec.dtype).min)
  else:
    return np.full(spec.shape, np.finfo(spec.dtype).min)


def maximum(spec: specs.Array):
  if hasattr(spec, 'maximum'):
    return clip(np.asarray(spec.maximum, dtype=spec.dtype), spec)
  elif np.issubdtype(spec.dtype, np.integer):
    return np.full(spec.shape, np.iinfo(spec.dtype).max)
  else:
    return np.full(spec.shape, np.finfo(spec.dtype).max)


def zeros(action_spec: specs.Array) -> np.ndarray:
  """Create a zero value for this Spec."""
  return np.zeros(shape=action_spec.shape, dtype=action_spec.dtype)


def cast(spec: specs.Array, value: ScalarOrArray) -> ScalarOrArray:
  """Cast a value to conform to a spec."""
  if np.isscalar(value):
    return spec.dtype.type(value)
  else:
    return value.astype(spec.dtype)


def clip(value: np.ndarray, spec: specs.BoundedArray) -> np.ndarray:
  """Clips the given value according to the spec."""
  if value is None:
    raise ValueError('no value')

  if isinstance(spec.dtype, np.inexact):
    eps = np.finfo(spec.dtype).eps * 5.0
  else:
    eps = 0

  min_bound = np.array(spec.minimum, dtype=spec.dtype)
  max_bound = np.array(spec.maximum, dtype=spec.dtype)
  return np.clip(value, min_bound + eps, max_bound - eps)


def shrink_to_fit(
    value: np.ndarray,
    spec: specs.BoundedArray,
    ignore_nan: Optional[bool] = None,
) -> np.ndarray:
  """Scales the value towards zero to fit within spec min and max values.

  Clipping is done after scaling to ensure there are no values that are very
  slightly (say 10e-8) out of range.

  This, by nature, assumes that min <= 0 <= max for the spec.

  Args:
    value: np.ndarray to scale towards zero.
    spec: Specification for value to scale and clip.
    ignore_nan: If True, NaN values will not fail validation. If None, this is
      determined by the size of `value`, so that large values are not checked.

  Returns:
    Scaled and clipped value.

  Raises:
    ValueError: On missing values or high-dimensional values.
  """
  if value is None:
    raise ValueError('no value')
  if spec is None:
    raise ValueError('no spec')
  if not isinstance(value, np.ndarray):
    raise ValueError('value not numpy array ({})'.format(type(value)))
  if len(value.shape) > 1:
    raise ValueError('2d values not yet handled')
  if not isinstance(spec, specs.BoundedArray):
    raise ValueError('Cannot scale to spec: {})'.format(spec))
  if np.any(spec.minimum > 0) or np.any(spec.maximum < 0):
    raise ValueError('Cannot scale to spec, due to bounds: {})'.format(spec))

  factor = 1.0
  for val, min_val, max_val in zip(value, spec.minimum, spec.maximum):
    if val < min_val:
      new_factor = min_val / val
      if new_factor < factor and new_factor > 0:
        factor = new_factor

    if val > max_val:
      new_factor = max_val / val
      if new_factor < factor and new_factor > 0:
        factor = new_factor

  scaled = (value * factor).astype(spec.dtype)
  clipped = clip(scaled, spec)
  try:
    validate(spec, clipped, ignore_nan)
  except ValueError:
    logging.error('Failed to scale %s to %s.  Got: %s', value, spec, clipped)

  return clipped


def merge_specs(spec_list: Sequence[specs.BoundedArray]):
  """Merges a list of BoundedArray into one."""

  # Check all specs are flat.
  for spec in spec_list:
    if len(spec.shape) > 1:
      raise ValueError('Not merging multi-dimensional spec: {}'.format(spec))

  # Filter out no-op specs with no actuators.
  spec_list = [spec for spec in spec_list if spec.shape and spec.shape[0]]
  dtype = np.find_common_type([spec.dtype for spec in spec_list], [])

  num_actions = 0
  name = ''
  mins = np.array([], dtype=dtype)
  maxs = np.array([], dtype=dtype)

  for i, spec in enumerate(spec_list):
    num_actions += spec.shape[0]
    if name:
      name += '\t'
    name += spec.name or f'spec_{i}'
    mins = np.concatenate([mins, spec.minimum])
    maxs = np.concatenate([maxs, spec.maximum])

  return specs.BoundedArray(
      shape=(num_actions,), dtype=dtype, minimum=mins, maximum=maxs, name=name)


def merge_primitives(values: Sequence[np.ndarray],
                     default_value: Optional[float] = None) -> np.ndarray:
  """Merge the given values (arrays) where NaNs are considered missing.

  Args:
    values: The values to merge.
    default_value: A default value to replace NaNs with, after merging.

  Returns:
    A merged value.

  Raises:
    ValueError: On ambiguity, shape/dtype mismatch, or no values.
      An ambiguity means >1 arrays have a non-nan value in the same index.
  """

  if not values:
    raise ValueError('No values to merge')

  # Ignore Nones.
  shape = values[0].shape
  dtype = values[0].dtype
  result = np.ndarray(shape=shape, dtype=dtype)
  result.fill(np.nan)

  if len(shape) != 1:
    raise ValueError('Not implemented for multi-dimensional arrays')

  for value in values:
    if value.shape != shape:
      raise ValueError('Shape mismatch, expect {} got {}.  All: {}'.format(
          shape, value.shape, [v.shape for v in values]))
    if value.dtype != dtype:
      raise ValueError('dtype mismatch, expect {} got {}.  All: {}'.format(
          dtype, value.dtype, [v.dtype for v in values]))
    for i in range(shape[0]):
      if not np.isnan(value[i]):
        if np.isnan(result[i]):
          result[i] = value[i]
        else:
          raise ValueError('Ambiguous merge at index {} with values: {}'.format(
              i, values))
  if default_value is not None:
    result[np.isnan(result)] = default_value

  return result


def merge_in_default(value, default_value):
  """Fill in the given value with the parts of the default_value."""
  if value is None:
    return default_value

  if isinstance(default_value, dict):
    for key in default_value.keys():
      value[key] = merge_in_default(value.get(key, None), default_value[key])
    return value
  elif isinstance(value, list):
    for i in range(len(default_value)):
      if i >= len(value):
        value.append(default_value[i])
      else:
        value[i] = merge_in_default(value[i], default_value[i])
    return value
  else:
    return value


def validate_timestep(spec: TimeStepSpec, timestep: dm_env.TimeStep):
  validate_observation(spec.observation_spec, timestep.observation)
  validate(spec.reward_spec, timestep.reward)
  validate(spec.discount_spec, timestep.discount)


def ensure_spec_compatibility(sub_specs: TimeStepSpec,
                              full_specs: TimeStepSpec):
  """Validates compatibility of 2 timestep specs.

  For the observations we only check inclusion of sub_specs in full_specs.

  Args:
    sub_specs:
    full_specs:

  Raises:
    ValueError: If the discount_spec, the reward_spec or one of the observation
      spec do not match.
    KeyError: If an observation in sub_specs is not in full_specs.
  """
  if sub_specs.discount_spec != full_specs.discount_spec:
    raise ValueError('Non matching discount specs.\nDiscount_sub_spec : {} \n'
                     'Discount_full_specs: {}\n'.format(
                         sub_specs.discount_spec, full_specs.discount_spec))

  if sub_specs.reward_spec != full_specs.reward_spec:
    raise ValueError('Non matching reward specs.\nReward_sub_spec : {} \n'
                     'Reward_spec: {}\n'.format(sub_specs.reward_spec,
                                                full_specs.reward_spec))

  for obs_spec_key, obs_spec in sub_specs.observation_spec.items():
    if obs_spec_key not in full_specs.observation_spec:
      raise KeyError('Missing observation key {} in spec.'.format(obs_spec_key))
    if obs_spec != full_specs.observation_spec[obs_spec_key]:
      raise ValueError('Non matching observation specs for key {}. \n'
                       'sub_spec = {} \n spec = {}'.format(
                           obs_spec_key, obs_spec,
                           full_specs.observation_spec[obs_spec_key]))


def verify_specs_equal_unbounded(expected: specs.Array, actual: specs.Array):
  """Assert that two specs are equal."""
  if expected.shape != actual.shape:
    raise ValueError(f'invalid shape for spec {expected.name}: '
                     f'{expected.shape}, actual shape: {actual.shape}')
  if expected.dtype != actual.dtype:
    raise ValueError(f'invalid dtype for spec {expected.name}: '
                     f'{expected.dtype}, actual dtype: {actual.dtype}')
  if expected.name != actual.name:
    raise ValueError(f'invalid name for spec {expected.name}: '
                     f'{expected.name}, actual name: {actual.name}')


def verify_specs_equal_bounded(expected: specs.BoundedArray,
                               actual: specs.BoundedArray):
  """Check specs are equal, raise a ValueError if they are not."""
  if not isinstance(expected, specs.BoundedArray):
    raise ValueError(f'Expected BoundedArray for first spec {expected.name}, '
                     'got {str(type(expected))}')
  if not isinstance(actual, specs.BoundedArray):
    raise ValueError(f'Expected BoundedArray for second spec {actual.name}, '
                     'got {str(type(actual))}')
  if not np.allclose(expected.minimum, actual.minimum):
    raise ValueError(f'Minimum values for spec {expected.name} do not match')
  if not np.allclose(expected.maximum, actual.maximum):
    raise ValueError(f'Maximum values for spec {expected.name} do not match')
  verify_specs_equal_unbounded(expected, actual)


def validate_observation(spec: ObservationSpec,
                         value: ObservationValue,
                         check_extra_keys: bool = True,
                         ignore_nan: Optional[bool] = None,
                         ignore_ranges: Optional[bool] = None,
                         msg: Optional[str] = None):
  """Validate an observation against an observation spec.

  Args:
    spec: The spec to validate against.
    value: The value to validate (!).
    check_extra_keys: If True having extra observations will fail.
    ignore_nan: If True, NaN values will not fail validation. If None, this is
      determined by the size of `value`, so that large values are not checked.
    ignore_ranges: If True, ignore minimum and maximum of BoundedArray. If None,
      this is determined by the size of `value`, so that large values are not
      checked.
    msg: message to append to any failure message.

  Raises:
    ValueError: On a validation failure.
  """

  if check_extra_keys:
    extra_keys = set(value.keys()) - set(spec.keys())
    if extra_keys:
      raise ValueError(
          'Extra keys in observation:\nSpec keys: {}\nvalue keys: {}\n'
          'Extra keys: {}'.format(spec.keys(), value.keys(), extra_keys))

  for spec_key, sub_spec in spec.items():
    if spec_key in value:  # Assumes missing keys are allowed.
      validate(
          sub_spec,
          value[spec_key],
          ignore_nan=ignore_nan,
          ignore_ranges=ignore_ranges,
          msg='{} for observation {}'.format(msg, spec_key))


# Profiling for .wrap('spec_utils.validate')
def validate(spec: specs.Array,
             value: np.ndarray,
             ignore_nan: Optional[bool] = None,
             ignore_ranges: Optional[bool] = None,
             msg: Optional[str] = None):
  """Validates that value matches the spec.

  Args:
    spec: The spec to validate against.
    value: The value to validate (!).
    ignore_nan: If True, NaN values will not fail validation. If None, this is
      determined by the shape of `value`, so that large arrays (e.g. images) are
      not checked (for performance reasons).
    ignore_ranges: If True, ignore minimum and maximum of BoundedArray. If None,
      this is determined by the size of `value`, so that large values are not
      checked.
    msg: message to append to any failure message.

  Raises:
    ValueError: On a validation failure.
  """

  if value is None:
    return  # ASSUME this is ok.

  value = np.asarray(value)
  if not np.issubdtype(value.dtype, np.number):
    # The value is non-numeric, so skip the nan and range checks.
    ignore_nan = True
    ignore_ranges = True
  elif np.prod(spec.shape) > 128:
    # Check less, in this case.
    if ignore_nan is None:
      ignore_nan = True
    if ignore_ranges is None:
      ignore_ranges = True
  else:
    # Check more in this case, it's cheap.
    if ignore_nan is None:
      ignore_nan = False
    if ignore_ranges is None:
      ignore_ranges = False

  if not ignore_nan:
    if np.any(np.isnan(value)):
      raise ValueError('NaN in value: {}, spec: {} ({})'.format(
          value, spec, msg))

  if not ignore_ranges:
    spec.validate(value)
  else:
    if spec.shape != value.shape:
      raise ValueError('shape mismatch {}. {} vs. {}'.format(msg, spec, value))
    if value.dtype != value.dtype:
      raise ValueError('dtype mismatch {}. {} vs. {}'.format(msg, spec, value))


def assert_not_dtype(spec: specs.Array, dtype: Type[Any]):
  """Asserts that the spec is not of the given dtype.

  Args:
    spec: A spec to validate.
    dtype: The dtype to check for.
  """
  dtype = np.dtype(dtype)
  maybe_spec_name = find_dtype(spec, dtype)
  if maybe_spec_name:
    spec, name = maybe_spec_name
    raise AssertionError('type {} found in {} ({})'.format(dtype, spec, name))


def find_dtype(spec: specs.Array,
               dtype: Type[np.floating]) -> Optional[Tuple[specs.Array, str]]:
  """Finds if the given spec uses the give type.

  Args:
    spec: A spec to search.
    dtype: The dtype to find.

  Returns:
    None if no match found, else (spec, spec_name) of the spec using dtype.
  """
  dtype = np.dtype(dtype)
  match = None  # type: Optional[Tuple[specs.Array, str]]
  if isinstance(spec, specs.Array):
    if spec.dtype is dtype:
      match = (spec, '')
  elif isinstance(spec, dict):
    for name, subspec in spec.items():
      if find_dtype(subspec, dtype):
        match = (subspec, name)
  else:
    raise ValueError('Unknown spec type {}'.format(type(spec)))
  return match
