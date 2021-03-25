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
"""Timestep preprocessors.

Preprocessors exist to transform observations, define termination conditions
and define reward functions.
"""

import abc
from typing import NamedTuple, Optional, Text, Union

import dm_env
from dm_robotics.agentflow import core
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import overrides
import numpy as np
import six

# Internal profiling


class PreprocessorTimestep(NamedTuple):
  """Timestep type for subtasks.

  This timestep is equivalent to `dm_env.TimeStep`, but also has pterm and
  OptionResult. This allows us to use Timestep preprocessors to handle
  termination and exit-status.
  """
  step_type: dm_env.StepType
  reward: Union[np.floating, np.ndarray]
  discount: np.float32
  observation: spec_utils.ObservationValue
  pterm: float
  result: Optional[core.OptionResult]

  @classmethod
  def from_environment_timestep(
      cls,
      environment_timestep: dm_env.TimeStep,
      pterm: float,
      result: Optional[core.OptionResult] = None) -> 'PreprocessorTimestep':
    return cls(
        step_type=environment_timestep.step_type,
        reward=environment_timestep.reward,
        discount=environment_timestep.discount,
        observation=environment_timestep.observation,
        pterm=pterm,
        result=result)

  def to_environment_timestep(self) -> dm_env.TimeStep:
    return dm_env.TimeStep(
        step_type=self.step_type,
        reward=self.reward,
        discount=self.discount,
        observation=self.observation)

  def first(self) -> bool:
    return self.step_type == dm_env.StepType.FIRST

  def mid(self) -> bool:
    return self.step_type == dm_env.StepType.MID

  def last(self) -> bool:
    return self.step_type == dm_env.StepType.LAST

  def replace(self, **kwargs) -> 'PreprocessorTimestep':
    return self._replace(**kwargs)


@six.add_metaclass(abc.ABCMeta)
class TimestepPreprocessor(object):
  """Instances of this class update values in time steps.

  They can change observations (add, remove or modify), discount, reward and
  termination probability.

  Implementations should reset any state when a timestep is presented to them
  with a step_type of FIRST.
  """

  def __init__(self):
    self._in_spec = None  # type: spec_utils.TimeStepSpec
    self._out_spec = None  # type: spec_utils.TimeStepSpec

  def process(self, input_ts: PreprocessorTimestep) -> PreprocessorTimestep:
    """Process the timestep.

    Args:
      input_ts: Input timestep

    Returns:
      processed timestep

    This should not be overridden in subclasses.
    """
    output_ts = self._process_impl(input_ts)
    if spec_utils.debugging_flag():
      # Make sure all the required keys are present and have the correct specs
      # Ignore the extra keys in the input and output timesteps.
      self._validate(self._in_spec, input_ts, 'input timestamp')
      self._validate(self._out_spec, output_ts, 'output timestamp')

    return output_ts

  def setup_io_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    """Setup the input and output specs.

    Args:
      input_spec: Input timestep spec

    Returns:
      Timestep spec of processed output.

    This should not be overridden in subclasses.
    """
    if self._in_spec or self._out_spec:
      raise ValueError('Specs already setup')

    self._in_spec = input_spec
    self._out_spec = self._output_spec(input_spec)

    return self._out_spec

  def get_input_spec(self) -> spec_utils.TimeStepSpec:
    """Input spec getter."""
    return self._in_spec

  def get_output_spec(self) -> spec_utils.TimeStepSpec:
    """Output spec getter."""
    return self._out_spec

  def _validate(self, spec: spec_utils.TimeStepSpec,
                timestep: PreprocessorTimestep, message: Text):
    """Validate the observation against the environment specs."""
    failure_msg = '{} failed validation for {} preprocessor'.format(
        message, type(self))

    # We allow the timesteps from demonstrations to have extra keys compared to
    # the environment.
    # E.g we have collected demos with cameras but want to train a proprio agent
    # only (i.e. the environment has no more cameras)
    spec_utils.validate_observation(spec.observation_spec, timestep.observation,
                                    check_extra_keys=False, msg=failure_msg)
    spec_utils.validate(spec.reward_spec, timestep.reward, ignore_nan=True,
                        msg=failure_msg)
    spec_utils.validate(spec.discount_spec, timestep.discount, msg=failure_msg)

  @abc.abstractmethod
  def _process_impl(self,
                    timestep: PreprocessorTimestep) -> PreprocessorTimestep:
    raise NotImplementedError('This should be overridden.')

  @abc.abstractmethod
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    raise NotImplementedError('This should be overridden.')


class CompositeTimestepPreprocessor(TimestepPreprocessor):
  """Apply an ordered list of timestep preprocessors."""

  def __init__(self, *preprocessors: TimestepPreprocessor):
    super(CompositeTimestepPreprocessor, self).__init__()
    self._timestep_preprocessors = list(preprocessors)

  @overrides(TimestepPreprocessor)
  # Profiling for .wrap('CompositeTimestepPreprocessor._process_impl')
  def _process_impl(self,
                    timestep: PreprocessorTimestep) -> PreprocessorTimestep:
    for timestep_preprocessor in self._timestep_preprocessors:
      timestep = timestep_preprocessor.process(timestep)
    return timestep

  @overrides(TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:

    out_spec = input_spec
    for timestep_preprocessor in self._timestep_preprocessors:
      out_spec = timestep_preprocessor.setup_io_spec(out_spec)
    return out_spec

  def add_preprocessor(self, preprocessor: TimestepPreprocessor):
    if self._out_spec:
      raise ValueError(
          'Cannot append to an initialized CompositeTimestepPreprocessor.')
    else:
      self._timestep_preprocessors.append(preprocessor)
