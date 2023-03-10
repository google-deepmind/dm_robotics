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
"""Tests for observations_transforms."""

import copy
from typing import Mapping, Optional, Type

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import dm_env
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.preprocessors import observation_transforms
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
from dm_robotics.transformations import transformations as tr
import numpy as np

_DEFAULT_TYPE = np.float64


def scalar_array_spec(name: str, dtype: Type[np.floating] = _DEFAULT_TYPE):
  return specs.Array(shape=(), dtype=dtype, name=name)


@parameterized.parameters(
    (observation_transforms.CastPreprocessor, float, float, float),
    (observation_transforms.CastPreprocessor, np.float32, float,
     float),
    (observation_transforms.CastPreprocessor, np.float64, float,
     float),
    (observation_transforms.CastPreprocessor, float, np.float32,
     np.float32),
    (observation_transforms.CastPreprocessor, np.float32, np.float32,
     np.float32),
    (observation_transforms.CastPreprocessor, np.float64, np.float32,
     np.float32),
    (observation_transforms.CastPreprocessor, float, np.float64,
     np.float64),
    (observation_transforms.CastPreprocessor, np.float32, np.float64,
     np.float64),
    (observation_transforms.CastPreprocessor, np.float64, np.float64,
     np.float64),
    (observation_transforms.DowncastFloatPreprocessor, float, float,
     float),
    (observation_transforms.DowncastFloatPreprocessor, np.float32,
     float, np.float32),
    (observation_transforms.DowncastFloatPreprocessor, np.float64,
     float, float),
    (observation_transforms.DowncastFloatPreprocessor, float,
     np.float32, np.float32),
    (observation_transforms.DowncastFloatPreprocessor, np.float32,
     np.float32, np.float32),
    (observation_transforms.DowncastFloatPreprocessor, np.float64,
     np.float32, np.float32),
    (observation_transforms.DowncastFloatPreprocessor, float,
     np.float64, float),
    (observation_transforms.DowncastFloatPreprocessor, np.float32,
     np.float64, np.float32),
    (observation_transforms.DowncastFloatPreprocessor, np.float64,
     np.float64, np.float64),
    (observation_transforms.DowncastFloatPreprocessor, np.float128,
     np.float64, np.float64),
    # Non-floating point types should not be interefered with.
    (observation_transforms.DowncastFloatPreprocessor, np.int32,
     np.float64, np.int32),
)
class CastAndDowncastPreprocessorTest(absltest.TestCase):

  def testCastPreprocessor_Array(
      self, processor_type: timestep_preprocessor.TimestepPreprocessor,
      src_type: Type[np.number], transform_type: Type[np.number],
      expected_type: Type[np.number]):
    # Arrange:
    name = testing_functions.random_string(3)
    processor = processor_type(transform_type)
    input_observation_spec = {
        name: specs.Array(shape=(2,), dtype=src_type, name=name),
    }
    expected_observation_spec = {
        name: specs.Array(shape=(2,), dtype=expected_type, name=name),
    }

    input_reward_spec = scalar_array_spec(dtype=src_type,
                                          name='reward')
    expected_reward_spec = scalar_array_spec(dtype=expected_type,
                                             name='reward')

    input_discount_spec = scalar_array_spec(dtype=src_type,
                                            name='discount')
    expected_discount_spec = scalar_array_spec(dtype=expected_type,
                                               name='discount')

    input_timestep_spec = spec_utils.TimeStepSpec(
        observation_spec=input_observation_spec,
        reward_spec=input_reward_spec,
        discount_spec=input_discount_spec)

    input_timestep = timestep_preprocessor.PreprocessorTimestep(
        step_type=np.random.choice(list(dm_env.StepType)),
        reward=src_type(0.1),
        discount=src_type(0.2),
        observation={name: np.asarray([0.3, 0.4], dtype=src_type)},
        pterm=0.1,
        result=None)

    # Act:
    spec_utils.validate_timestep(input_timestep_spec, input_timestep)
    output_timestep_spec = processor.setup_io_spec(input_timestep_spec)

    # Assert:
    expected_timestep = timestep_preprocessor.PreprocessorTimestep(
        step_type=input_timestep.step_type,
        reward=expected_type(0.1),
        discount=expected_type(0.2),
        observation={name: np.asarray([0.3, 0.4], dtype=expected_type)},
        pterm=input_timestep.pterm,
        result=None)

    self.assertEqual(output_timestep_spec.observation_spec,
                     expected_observation_spec)
    self.assertEqual(output_timestep_spec.reward_spec, expected_reward_spec)
    self.assertEqual(output_timestep_spec.discount_spec, expected_discount_spec)

    output_timestep = processor.process(input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(output_timestep.observation[name],
                                   expected_timestep.observation[name])
    np.testing.assert_almost_equal(output_timestep.reward,
                                   expected_timestep.reward)
    np.testing.assert_almost_equal(output_timestep.discount,
                                   expected_timestep.discount)

  def testCastPreprocessor_BoundedArray(
      self, processor_type: timestep_preprocessor.TimestepPreprocessor,
      src_type: Type[np.number], transform_type: Type[np.number],
      expected_type: Type[np.number]):
    """Same as previous test, but using BoundedArray specs."""
    # Arrange:
    name = testing_functions.random_string(3)
    processor = processor_type(transform_type)

    input_minimum = np.asarray([0.3, 0.4], dtype=src_type)
    input_maximum = np.asarray([0.5, 0.6], dtype=src_type)
    input_observation_spec = {
        name:
            specs.BoundedArray(
                shape=(2,),
                dtype=src_type,
                minimum=input_minimum,
                maximum=input_maximum,
                name=name),
    }
    input_reward_spec = scalar_array_spec(name='reward', dtype=src_type)
    input_discount_spec = scalar_array_spec(name='discount', dtype=src_type)
    input_timestep_spec = spec_utils.TimeStepSpec(
        observation_spec=input_observation_spec,
        reward_spec=input_reward_spec,
        discount_spec=input_discount_spec)

    input_timestep = timestep_preprocessor.PreprocessorTimestep(
        step_type=np.random.choice(list(dm_env.StepType)),
        reward=src_type(0.1),
        discount=src_type(0.2),
        observation={name: np.asarray([0.4, 0.5], dtype=src_type)},
        pterm=0.1,
        result=None)

    # Act:
    spec_utils.validate_timestep(input_timestep_spec, input_timestep)
    output_timestep_spec = processor.setup_io_spec(input_timestep_spec)

    # Assert:
    expected_minimum = np.asarray([0.3, 0.4], dtype=expected_type)
    expected_maximum = np.asarray([0.5, 0.6], dtype=expected_type)
    expected_output_observation_spec = {
        name:
            specs.BoundedArray(
                shape=(2,),
                dtype=expected_type,
                minimum=expected_minimum,
                maximum=expected_maximum,
                name=name),
    }
    expected_output_reward_spec = scalar_array_spec(
        name='reward', dtype=expected_type)
    expected_output_discount_spec = scalar_array_spec(
        name='discount', dtype=expected_type)

    expected_output_timestep = timestep_preprocessor.PreprocessorTimestep(
        step_type=input_timestep.step_type,
        reward=expected_type(0.1),
        discount=expected_type(0.2),
        observation={name: np.asarray([0.4, 0.5], dtype=expected_type)},
        pterm=input_timestep.pterm,
        result=None)

    self.assertEqual(
        set(output_timestep_spec.observation_spec.keys()),
        set(expected_output_observation_spec.keys()))
    spec_utils.verify_specs_equal_bounded(
        output_timestep_spec.observation_spec[name],
        expected_output_observation_spec[name])
    self.assertEqual(output_timestep_spec.reward_spec,
                     expected_output_reward_spec)
    self.assertEqual(output_timestep_spec.discount_spec,
                     expected_output_discount_spec)

    output_timestep = processor.process(input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(output_timestep.observation[name],
                                   expected_output_timestep.observation[name])
    np.testing.assert_almost_equal(output_timestep.reward,
                                   expected_output_timestep.reward)
    np.testing.assert_almost_equal(output_timestep.discount,
                                   expected_output_timestep.discount)

  def testCastPreprocessor_RewardArray(
      self, processor_type: timestep_preprocessor.TimestepPreprocessor,
      src_type: Type[np.number], transform_type: Type[np.number],
      expected_type: Type[np.number]):
    # Arrange:
    name = testing_functions.random_string(3)
    processor = processor_type(transform_type)
    input_observation_spec = {
        name: specs.Array(shape=(2,), dtype=src_type, name=name),
    }
    expected_observation_spec = {
        name: specs.Array(shape=(2,), dtype=expected_type, name=name),
    }

    input_reward_spec = specs.Array(shape=(3,), dtype=src_type,
                                    name='reward')
    expected_reward_spec = specs.Array(
        shape=(3,), dtype=expected_type, name='reward')

    input_discount_spec = scalar_array_spec(dtype=src_type,
                                            name='discount')
    expected_discount_spec = scalar_array_spec(dtype=expected_type,
                                               name='discount')

    input_timestep_spec = spec_utils.TimeStepSpec(
        observation_spec=input_observation_spec,
        reward_spec=input_reward_spec,
        discount_spec=input_discount_spec)

    # Some test data that matches the src_type.
    if np.issubdtype(src_type, np.floating):
      numbers = (0.1, 0.2, 0.3, 0.4, 0.1)
    elif np.issubdtype(src_type, np.integer):
      numbers = (1, 2, 3, 4, 5)
    else:
      raise ValueError(
          'Only ints and floats are currently supported.')

    input_timestep = timestep_preprocessor.PreprocessorTimestep(
        step_type=np.random.choice(list(dm_env.StepType)),
        reward=numbers[0] * np.ones(shape=(3,), dtype=src_type),
        discount=src_type(numbers[1]),
        observation={name: np.asarray(numbers[2:4], dtype=src_type)},
        pterm=numbers[4],
        result=None)

    # Act:
    spec_utils.validate_timestep(input_timestep_spec, input_timestep)
    output_timestep_spec = processor.setup_io_spec(input_timestep_spec)

    # Assert:
    expected_timestep = timestep_preprocessor.PreprocessorTimestep(
        step_type=input_timestep.step_type,
        reward=numbers[0] * np.ones(shape=(3,), dtype=expected_type),
        discount=expected_type(numbers[1]),
        observation={name: np.asarray(numbers[2:4], dtype=expected_type)},
        pterm=input_timestep.pterm,
        result=None)

    self.assertEqual(output_timestep_spec.observation_spec,
                     expected_observation_spec)
    self.assertEqual(output_timestep_spec.reward_spec, expected_reward_spec)
    self.assertEqual(output_timestep_spec.discount_spec, expected_discount_spec)

    output_timestep = processor.process(input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(output_timestep.observation[name],
                                   expected_timestep.observation[name])
    np.testing.assert_almost_equal(output_timestep.reward,
                                   expected_timestep.reward)
    np.testing.assert_almost_equal(output_timestep.discount,
                                   expected_timestep.discount)


class RenameObservationsTest(absltest.TestCase):

  def test_rename_observations(self):
    preprocessor = observation_transforms.RenameObservations(
        obs_mapping={'foo': 'pow', 'faw': 'biz'})

    # Generate the input spec and input timestep
    input_obs_spec = {
        'foo': specs.Array(shape=(2,), dtype=np.float64, name='foo'),
        'bar': specs.Array(shape=(2,), dtype=np.float64, name='bar'),
        'faw': specs.Array(shape=(2,), dtype=np.float64, name='faw'),
    }
    input_spec = _build_unit_timestep_spec(observation_spec=input_obs_spec)
    input_obs = {'foo': [1., 2.], 'bar': [3., 4.], 'faw': [5., 6.]}
    input_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=_DEFAULT_TYPE(0.1),
        discount=_DEFAULT_TYPE(0.8),
        observation=input_obs)

    # Setup expectations.
    expected_output_spec = input_spec.replace(observation_spec={
        'pow': specs.Array(shape=(2,), dtype=np.float64, name='pow'),
        'bar': specs.Array(shape=(2,), dtype=np.float64, name='bar'),
        'biz': specs.Array(shape=(2,), dtype=np.float64, name='biz'),
    })

    # Check the spec
    output_spec = preprocessor.setup_io_spec(input_spec)
    self.assertEqual(output_spec.observation_spec,
                     expected_output_spec.observation_spec)

    # Check the timestep.
    output_timestep = preprocessor.process(input_timestep)
    spec_utils.validate_timestep(output_spec, output_timestep)
    np.testing.assert_array_equal(output_timestep.observation['pow'], [1., 2.])

  def test_failure_when_renaming_missing_observations(self):
    preprocessor = observation_transforms.RenameObservations(
        obs_mapping={'foo': 'pow', 'faw': 'biz'})

    # Generate the input spec and input timestep
    input_obs_spec = {
        'foo': specs.Array(shape=(2,), dtype=np.float64, name='foo'),
    }
    input_spec = _build_unit_timestep_spec(observation_spec=input_obs_spec)

    # Calculating the output spec should fail.
    with self.assertRaises(observation_transforms.MisconfigurationError):
      preprocessor.setup_io_spec(input_spec)

  def test_failure_for_duplicate_rename_targets(self):
    obs_mapping = {'foo': 'pow', 'bar': 'pow'}
    # Initialization should fail.
    with self.assertRaises(observation_transforms.MisconfigurationError):
      observation_transforms.RenameObservations(obs_mapping)

  def test_failure_for_conflicting_rename_targets(self):
    # Create the spec and timestep.
    preprocessor = observation_transforms.RenameObservations(
        obs_mapping={'foo': 'pow', 'faw': 'bar'})

    # Generate the input spec and input timestep
    input_obs_spec = {
        'foo': specs.Array(shape=(2,), dtype=np.float64, name='foo'),
        'faw': specs.Array(shape=(2,), dtype=np.float64, name='faw'),
        'bar': specs.Array(shape=(2,), dtype=np.float64, name='bar'),
    }
    input_spec = _build_unit_timestep_spec(observation_spec=input_obs_spec)

    # Calculating the output spec should fail.
    with self.assertRaises(observation_transforms.MisconfigurationError):
      preprocessor.setup_io_spec(input_spec)


class MergeObservationsTest(absltest.TestCase):

  def test_merge_observation(self):

    preprocessor = observation_transforms.MergeObservations(
        obs_to_merge=['foo', 'bar'], new_obs='baz')

    # Generate the input spec and input timestep
    input_obs_spec = {
        'foo': specs.Array(shape=(2,), dtype=np.float64, name='foo'),
        'bar': specs.Array(shape=(2,), dtype=np.float64, name='bar'),
        'faw': specs.Array(shape=(2,), dtype=np.float64, name='faw'),
    }
    input_spec = _build_unit_timestep_spec(observation_spec=input_obs_spec)
    input_obs = {'foo': [1., 2.], 'bar': [3., 4.], 'faw': [3., 4.]}
    input_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=_DEFAULT_TYPE(0.1),
        discount=_DEFAULT_TYPE(0.8),
        observation=input_obs)

    # Setup expectations.
    expected_output_spec = input_spec.replace(observation_spec={
        'baz': specs.Array(shape=(4,), dtype=np.float64, name='baz'),
        'faw': specs.Array(shape=(2,), dtype=np.float64, name='faw')
    })

    # Check the spec
    output_spec = preprocessor.setup_io_spec(input_spec)
    self.assertEqual(output_spec.observation_spec,
                     expected_output_spec.observation_spec)

    # Check the timestep.
    output_timestep = preprocessor.process(input_timestep)
    spec_utils.validate_timestep(output_spec, output_timestep)
    np.testing.assert_array_equal(output_timestep.observation['baz'],
                                  [1., 2., 3., 4.])

  def test_failure_when_merging_missing_observation(self):
    preprocessor = observation_transforms.MergeObservations(
        obs_to_merge=['foo', 'bar'], new_obs='baz')

    # Generate the input spec
    input_obs_spec = {
        'foo': specs.Array(shape=(2,), dtype=np.float64, name='foo')}
    input_spec = _build_unit_timestep_spec(observation_spec=input_obs_spec)

    # Calculating the output spec should fail.
    with self.assertRaises(observation_transforms.MisconfigurationError):
      preprocessor.setup_io_spec(input_spec)

  def test_failure_for_conflicting_new_name(self):

    preprocessor = observation_transforms.MergeObservations(
        obs_to_merge=['foo', 'bar'], new_obs='faw')

    # Generate the input spec and input timestep
    input_obs_spec = {
        'foo': specs.Array(shape=(2,), dtype=np.float64, name='foo'),
        'bar': specs.Array(shape=(2,), dtype=np.float64, name='bar'),
        'faw': specs.Array(shape=(2,), dtype=np.float64, name='faw'),
    }
    input_spec = _build_unit_timestep_spec(observation_spec=input_obs_spec)

    # Calculating the output spec should fail.
    with self.assertRaises(observation_transforms.MisconfigurationError):
      preprocessor.setup_io_spec(input_spec)


class CropImageObservationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._input_obs_name = 'input_obs'
    self._output_obs_name = 'output_obs'
    # This has a shape of (4,5)
    self._input_spec = testing_functions.random_array_spec(
        shape=(4, 5, 3), dtype=float, name=self._input_obs_name)
    self._input_observation_spec = {self._input_obs_name: self._input_spec}
    self._input_obs_value = testing_functions.valid_value(self._input_spec)

    self._input_timestep_spec = testing_functions.random_timestep_spec(
        observation_spec=self._input_observation_spec)
    self._input_timestep = testing_functions.random_timestep(
        spec=self._input_timestep_spec,
        observation={self._input_obs_name: self._input_obs_value})

    spec_utils.validate_timestep(self._input_timestep_spec,
                                 self._input_timestep)

  def _get_expected_spec(self, value: np.ndarray):
    return testing_functions.random_array_spec(
        shape=value.shape, dtype=value.dtype, name=self._output_obs_name)

  def testFullCrop(self):
    """Don't modify the input at all."""
    processor = observation_transforms.CropImageObservation(
        input_obs_name=self._input_obs_name,
        output_obs_name=self._output_obs_name,
        crop_width_relative=1.0,
        crop_height_relative=1.0,
        x_offset_relative=0.0,
        y_offset_relative=0.0)
    expected_value = self._input_obs_value

    output_timestep_spec = processor.setup_io_spec(self._input_timestep_spec)

    self.assertIn(self._output_obs_name, output_timestep_spec.observation_spec)
    spec_utils.verify_specs_equal_unbounded(
        self._input_spec.replace(name=self._output_obs_name),
        output_timestep_spec.observation_spec[self._output_obs_name])

    output_timestep = processor.process(self._input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(
        output_timestep.observation[self._output_obs_name], expected_value)

  def testCropNoOffset(self):
    """Crop to a region that is in a corner of the original observation."""
    processor = observation_transforms.CropImageObservation(
        input_obs_name=self._input_obs_name,
        output_obs_name=self._output_obs_name,
        crop_width_relative=0.4,
        crop_height_relative=0.75,
        x_offset_relative=0.0,
        y_offset_relative=0.0)
    expected_value = self._input_obs_value[:3, :2]

    output_timestep_spec = processor.setup_io_spec(self._input_timestep_spec)

    self.assertIn(self._output_obs_name, output_timestep_spec.observation_spec)
    spec_utils.verify_specs_equal_unbounded(
        self._get_expected_spec(expected_value),
        output_timestep_spec.observation_spec[self._output_obs_name])

    output_timestep = processor.process(self._input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(
        output_timestep.observation[self._output_obs_name], expected_value)

  def testSquareCropNoOffset(self):
    """Crop to a region that is in a corner of the original observation.

    Leaving out the height parameter should default to a square crop.
    """
    processor = observation_transforms.CropImageObservation(
        input_obs_name=self._input_obs_name,
        output_obs_name=self._output_obs_name,
        crop_width_relative=0.4,
        x_offset_relative=0.0,
        y_offset_relative=0.0)
    expected_value = self._input_obs_value[:2, :2]

    output_timestep_spec = processor.setup_io_spec(self._input_timestep_spec)

    self.assertIn(self._output_obs_name, output_timestep_spec.observation_spec)
    spec_utils.verify_specs_equal_unbounded(
        self._get_expected_spec(expected_value),
        output_timestep_spec.observation_spec[self._output_obs_name])

    output_timestep = processor.process(self._input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(
        output_timestep.observation[self._output_obs_name], expected_value)

  def testCropWithOffset(self):
    """Crop to the center of the observation."""
    processor = observation_transforms.CropImageObservation(
        input_obs_name=self._input_obs_name,
        output_obs_name=self._output_obs_name,
        crop_width_relative=0.6,
        crop_height_relative=0.5,
        x_offset_relative=0.5,
        y_offset_relative=0.5)
    expected_value = self._input_obs_value[1:3, 1:4]

    output_timestep_spec = processor.setup_io_spec(self._input_timestep_spec)

    self.assertIn(self._output_obs_name, output_timestep_spec.observation_spec)
    spec_utils.verify_specs_equal_unbounded(
        self._get_expected_spec(expected_value),
        output_timestep_spec.observation_spec[self._output_obs_name])

    output_timestep = processor.process(self._input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(
        output_timestep.observation[self._output_obs_name], expected_value)

  def testInvalidParams(self):
    """Ensure that invalid parameters cause Exceptions."""
    # Zero width and height are invalid
    with self.assertRaisesRegex(ValueError, 'zero'):
      _ = observation_transforms.CropImageObservation(
          input_obs_name=self._input_obs_name,
          output_obs_name=self._output_obs_name,
          crop_width_relative=0.,
          crop_height_relative=0.,
          x_offset_relative=0.,
          y_offset_relative=0.)

    # Negative width is invalid
    with self.assertRaisesRegex(ValueError, 'width must be between'):
      _ = observation_transforms.CropImageObservation(
          input_obs_name=self._input_obs_name,
          output_obs_name=self._output_obs_name,
          crop_width_relative=-1.,
          crop_height_relative=1.,
          x_offset_relative=0.,
          y_offset_relative=0.)

    # Height > 1.0 is invalid
    with self.assertRaisesRegex(ValueError, 'height must be between'):
      _ = observation_transforms.CropImageObservation(
          input_obs_name=self._input_obs_name,
          output_obs_name=self._output_obs_name,
          crop_width_relative=1.,
          crop_height_relative=1.5,
          x_offset_relative=0.,
          y_offset_relative=0.)

    # Offset > 1.0 is invalid
    with self.assertRaisesRegex(ValueError, 'offset must be between'):
      _ = observation_transforms.CropImageObservation(
          input_obs_name=self._input_obs_name,
          output_obs_name=self._output_obs_name,
          crop_width_relative=0.6,
          crop_height_relative=1.,
          x_offset_relative=1.5,
          y_offset_relative=0.)


class CropSquareAndResizeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._input_obs_name = 'input_obs'
    self._output_obs_name = 'output_obs'
    # This has a shape of (4,5)
    self._input_spec = testing_functions.random_array_spec(
        shape=(4, 5), dtype=float, name=self._input_obs_name)
    self._input_observation_spec = {self._input_obs_name: self._input_spec}
    self._input_obs_value = testing_functions.valid_value(self._input_spec)

    self._input_timestep_spec = testing_functions.random_timestep_spec(
        observation_spec=self._input_observation_spec)
    self._input_timestep = testing_functions.random_timestep(
        spec=self._input_timestep_spec,
        observation={self._input_obs_name: self._input_obs_value})

    spec_utils.validate_timestep(self._input_timestep_spec,
                                 self._input_timestep)

  def _get_expected_spec(self, value: np.ndarray):
    return testing_functions.random_array_spec(
        shape=value.shape, dtype=value.dtype, name=self._output_obs_name)

  def testCropNoOffset(self):
    """Crop to a region that is in a corner of the original observation."""
    processor = observation_transforms.CropSquareAndResize(
        input_obs_name=self._input_obs_name,
        output_obs_name=self._output_obs_name,
        crop_width_relative=0.8,
        side_length_pixels=4,
        x_offset_relative=0.0,
        y_offset_relative=0.0)
    expected_value = self._input_obs_value[:4, :4]

    output_timestep_spec = processor.setup_io_spec(self._input_timestep_spec)

    self.assertIn(self._output_obs_name, output_timestep_spec.observation_spec)
    spec_utils.verify_specs_equal_unbounded(
        self._get_expected_spec(expected_value),
        output_timestep_spec.observation_spec[self._output_obs_name])

    output_timestep = processor.process(self._input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(
        output_timestep.observation[self._output_obs_name], expected_value)

  def testScaledCropNoOffset(self):
    """Crop to a region that is in a corner of the original observation."""
    processor = observation_transforms.CropSquareAndResize(
        input_obs_name=self._input_obs_name,
        output_obs_name=self._output_obs_name,
        crop_width_relative=0.8,
        side_length_pixels=8,
        x_offset_relative=0.0,
        y_offset_relative=0.0,
        interpolation=cv2.INTER_NEAREST)
    # Nearest neighbor sampling should just duplicate the original pixels
    expected_value = np.repeat(
        np.repeat(self._input_obs_value[:4, :4], 2, axis=0), 2, axis=1)

    output_timestep_spec = processor.setup_io_spec(self._input_timestep_spec)

    self.assertIn(self._output_obs_name, output_timestep_spec.observation_spec)
    spec_utils.verify_specs_equal_unbounded(
        self._get_expected_spec(expected_value),
        output_timestep_spec.observation_spec[self._output_obs_name])

    output_timestep = processor.process(self._input_timestep)
    spec_utils.validate_timestep(output_timestep_spec, output_timestep)

    np.testing.assert_almost_equal(
        output_timestep.observation[self._output_obs_name], expected_value)


class PoseRelativeTest(absltest.TestCase):

  def _check_spec_float_unchanged(self, dtype):
    preprocessor = observation_transforms.PoseRelativeToEpisodeStart(
        pos_obs_name='pos', quat_obs_name='quat')

    # Generate the input spec and input timestep
    input_obs_spec = {
        'pos': specs.Array(shape=(3,), dtype=dtype, name='pos'),
        'quat': specs.Array(shape=(4,), dtype=dtype, name='quat'),
    }
    input_spec = testing_functions.random_timestep_spec(
        observation_spec=input_obs_spec)

    first_input_timestep = testing_functions.random_timestep(
        spec=input_spec,
        step_type=dm_env.StepType.FIRST)

    # Setup expectations.
    expected_output_spec = input_spec

    # Check the spec
    output_spec = preprocessor.setup_io_spec(input_spec)
    self.assertEqual(output_spec.observation_spec,
                     expected_output_spec.observation_spec)

    # Check the timestep.
    output_timestep = preprocessor.process(first_input_timestep)
    spec_utils.validate_timestep(output_spec, output_timestep)

  def test_spec_float32_unchanged(self):
    self._check_spec_float_unchanged(dtype=np.float32)

  def test_spec_float64_unchanged(self):
    self._check_spec_float_unchanged(dtype=np.float64)

  def test_initial_observations(self):
    preprocessor = observation_transforms.PoseRelativeToEpisodeStart(
        pos_obs_name='pos', quat_obs_name='quat')

    # Generate the input spec and input timestep
    input_obs_spec = {
        'pos': specs.Array(shape=(3,), dtype=np.float64, name='pos'),
        'quat': specs.Array(shape=(4,), dtype=np.float64, name='quat'),
    }
    input_spec = testing_functions.random_timestep_spec(
        observation_spec=input_obs_spec)

    input_obs = {
        'pos': [1.0, -1.5, 3.2],
        'quat': tr.euler_to_quat([0.1, 0.2, 0.3])
    }
    first_input_timestep = testing_functions.random_timestep(
        spec=input_spec, step_type=dm_env.StepType.FIRST, observation=input_obs)

    # Setup expectations.
    expected_output_spec = input_spec

    # Check the spec
    output_spec = preprocessor.setup_io_spec(input_spec)
    self.assertEqual(output_spec.observation_spec,
                     expected_output_spec.observation_spec)

    # Check the timestep.
    output_timestep = preprocessor.process(first_input_timestep)
    spec_utils.validate_timestep(output_spec, output_timestep)

    output_pos = output_timestep.observation['pos']
    np.testing.assert_array_almost_equal(output_pos, [0., 0., 0.])

    output_euler = tr.quat_to_euler(output_timestep.observation['quat'])
    np.testing.assert_array_almost_equal(output_euler, [0., 0., 0.])

  def test_relative_observations(self):
    preprocessor = observation_transforms.PoseRelativeToEpisodeStart(
        pos_obs_name='pos', quat_obs_name='quat')

    # Generate the input spec and input timestep
    input_obs_spec = {
        'pos': specs.Array(shape=(3,), dtype=np.float64, name='pos'),
        'quat': specs.Array(shape=(4,), dtype=np.float64, name='quat'),
    }
    input_spec = testing_functions.random_timestep_spec(
        observation_spec=input_obs_spec)

    input_obs = {
        'pos': np.array([1.0, -1.5, 3.2]),
        'quat': tr.euler_to_quat([0.0, 0.0, 0.0])
    }
    first_input_timestep = testing_functions.random_timestep(
        spec=input_spec,
        step_type=dm_env.StepType.FIRST,
        observation=input_obs)

    preprocessor.setup_io_spec(input_spec)
    preprocessor.process(first_input_timestep)

    pos_offset = np.array([0.1, -0.2, -0.3])

    input_obs = {
        'pos': (input_obs['pos'] + pos_offset),
        'quat': tr.euler_to_quat([0.2, 0.0, 0.0])
    }
    second_input_timestep = testing_functions.random_timestep(
        spec=input_spec,
        step_type=dm_env.StepType.MID,
        observation=input_obs)

    output_timestep = preprocessor.process(second_input_timestep)

    output_pos = output_timestep.observation['pos']
    np.testing.assert_array_almost_equal(output_pos, pos_offset)

    output_euler = tr.quat_to_euler(output_timestep.observation['quat'])
    np.testing.assert_array_almost_equal(output_euler, [0.2, 0., 0.])


class StackObservationsTest(parameterized.TestCase):

  @parameterized.parameters(
      (False, (4,), (12,)),
      (True, (4,), (3, 4)),
      (False, (1,), (3,)),
      (True, (1,), (3, 1)),
      (False, (4, 4), (12, 4)),
      (True, (4, 4), (3, 4, 4)),
  )
  def test_stack_observations_spec(
      self, add_leading_dim, input_shape, output_shape):
    # Generate the input spec and input timestep.
    input_obs_spec = {
        'pos': specs.Array(shape=input_shape, dtype=np.float32, name='pos'),
    }
    input_spec = _build_unit_timestep_spec(
        observation_spec=input_obs_spec)

    # Generate the expected stacked output spec.
    expected_output_obs_spec = {
        'pos': specs.Array(shape=output_shape, dtype=np.float32, name='pos'),
    }
    expected_output_spec = _build_unit_timestep_spec(
        observation_spec=expected_output_obs_spec)

    preprocessor = observation_transforms.StackObservations(
        obs_to_stack=['pos'],
        stack_depth=3,
        add_leading_dim=add_leading_dim)

    output_spec = preprocessor.setup_io_spec(input_spec)
    self.assertEqual(expected_output_spec, output_spec)

  @parameterized.parameters(
      (False, (4,), (12,)),
      (True, (4,), (3, 4)),
      (False, (1,), (3,)),
      (True, (1,), (3, 1)),
      (False, (4, 4), (12, 4)),
      (True, (4, 4), (3, 4, 4)),
  )
  def test_stack_observations(self, add_leading_dim, input_shape, output_shape):
    # Generate the input spec.
    input_obs_spec = {
        'pos': specs.Array(shape=input_shape, dtype=np.float32, name='pos'),
    }
    input_spec = _build_unit_timestep_spec(
        observation_spec=input_obs_spec)

    preprocessor = observation_transforms.StackObservations(
        obs_to_stack=['pos'],
        stack_depth=3,
        add_leading_dim=add_leading_dim)

    preprocessor.setup_io_spec(input_spec)

    input_pos = np.random.random(input_shape).astype(np.float32)

    if add_leading_dim:
      expected_output_pos = np.stack([input_pos for _ in range(3)], axis=0)
    else:
      expected_output_pos = np.concatenate(
          [input_pos for _ in range(3)], axis=0)

    input_timestep = testing_functions.random_timestep(
        spec=input_spec,
        step_type=dm_env.StepType.FIRST,
        observation={'pos': input_pos,})

    output_timestep = preprocessor.process(input_timestep)
    output_pos = output_timestep.observation['pos']
    np.testing.assert_allclose(expected_output_pos, output_pos)
    np.testing.assert_allclose(expected_output_pos.shape, output_shape)

  @parameterized.parameters(
      (False, (4,), (12,)),
      (True, (4,), (3, 4)),
      (False, (1,), (3,)),
      (True, (1,), (3, 1)),
      (False, (4, 4), (12, 4)),
      (True, (4, 4), (3, 4, 4)),
  )
  def test_add_stack_observations_spec(
      self, add_leading_dim, input_shape, output_shape):
    # Generate the input spec and input timestep.
    input_obs_spec = {
        'pos': specs.Array(shape=input_shape, dtype=np.float32, name='pos'),
    }
    input_spec = _build_unit_timestep_spec(
        observation_spec=input_obs_spec)

    # Generate the expected stacked output spec.
    expected_output_obs_spec = {
        'pos': specs.Array(shape=input_shape, dtype=np.float32, name='pos'),
        'stacked_pos': specs.Array(
            shape=output_shape, dtype=np.float32, name='pos'),
    }
    expected_output_spec = _build_unit_timestep_spec(
        observation_spec=expected_output_obs_spec)

    preprocessor = observation_transforms.StackObservations(
        obs_to_stack=['pos'],
        stack_depth=3,
        add_leading_dim=add_leading_dim,
        override_obs=False)

    output_spec = preprocessor.setup_io_spec(input_spec)
    self.assertEqual(expected_output_spec, output_spec)

  @parameterized.parameters(
      (False, (4,), (12,)),
      (True, (4,), (3, 4)),
      (False, (1,), (3,)),
      (True, (1,), (3, 1)),
      (False, (4, 4), (12, 4)),
      (True, (4, 4), (3, 4, 4)),
  )
  def test_add_stack_observations(self,
                                  add_leading_dim, input_shape, output_shape):
    # Generate the input spec.
    input_obs_spec = {
        'pos': specs.Array(shape=input_shape, dtype=np.float32, name='pos'),
    }
    input_spec = _build_unit_timestep_spec(
        observation_spec=input_obs_spec)

    preprocessor = observation_transforms.StackObservations(
        obs_to_stack=['pos'],
        stack_depth=3,
        add_leading_dim=add_leading_dim,
        override_obs=False)

    preprocessor.setup_io_spec(input_spec)

    input_pos = np.random.random(input_shape).astype(np.float32)

    if add_leading_dim:
      expected_output_pos = np.stack([input_pos for _ in range(3)], axis=0)
    else:
      expected_output_pos = np.concatenate(
          [input_pos for _ in range(3)], axis=0)

    input_timestep = testing_functions.random_timestep(
        spec=input_spec,
        step_type=dm_env.StepType.FIRST,
        observation={'pos': input_pos,})

    output_timestep = preprocessor.process(input_timestep)
    output_stacked_pos = output_timestep.observation['stacked_pos']
    output_pos = output_timestep.observation['pos']
    np.testing.assert_allclose(expected_output_pos, output_stacked_pos)
    np.testing.assert_allclose(input_pos, output_pos)
    np.testing.assert_allclose(expected_output_pos.shape, output_shape)


class UnstackObservationsTest(parameterized.TestCase):

  @parameterized.parameters(
      (False, (3, 4), (4,)),
      (True, (1, 7), (7,)),
  )
  def test_unstack_observations_spec(
      self, override, input_shape, output_shape):
    # Generate the input spec and input timestep.
    input_obs_spec = {
        'pos': specs.Array(shape=input_shape, dtype=np.float32, name='pos'),
    }
    input_spec = _build_unit_timestep_spec(
        observation_spec=input_obs_spec)

    # Generate the expected stacked output spec.
    if override:
      expected_output_obs_spec = {
          'pos': specs.Array(shape=output_shape, dtype=np.float32, name='pos'),
      }
    else:
      expected_output_obs_spec = {
          'pos': specs.Array(shape=input_shape, dtype=np.float32, name='pos'),
          'unstacked_pos': specs.Array(
              shape=output_shape, dtype=np.float32, name='unstacked_pos'),
      }
    expected_output_spec = _build_unit_timestep_spec(
        observation_spec=expected_output_obs_spec)

    preprocessor = observation_transforms.UnstackObservations(
        obs_to_unstack=['pos'],
        override_obs=override)

    output_spec = preprocessor.setup_io_spec(input_spec)
    self.assertEqual(expected_output_spec, output_spec)

  @parameterized.parameters(
      (False, (3, 4,), (3, 4)),
      (True, (1, 7,), (7,)),
  )
  def test_unstack_observations(
      self, override, input_shape, output_shape):
    input_obs_spec = {
        'pos': specs.Array(shape=input_shape, dtype=np.float32, name='pos'),
    }
    # Generate the input spec and input timestep.
    input_spec = _build_unit_timestep_spec(
        observation_spec=input_obs_spec)

    preprocessor = observation_transforms.UnstackObservations(
        obs_to_unstack=['pos'],
        override_obs=override)

    preprocessor.setup_io_spec(input_spec)

    input_pos = np.random.random(input_shape).astype(np.float32)

    if override:
      expected_output_pos = input_pos[0]
    else:
      expected_output_pos = input_pos

    input_timestep = testing_functions.random_timestep(
        spec=input_spec,
        step_type=dm_env.StepType.FIRST,
        observation={'pos': input_pos,})

    output_timestep = preprocessor.process(input_timestep)
    output_pos = output_timestep.observation['pos']

    np.testing.assert_allclose(expected_output_pos, output_pos)
    np.testing.assert_allclose(expected_output_pos.shape, output_shape)


class AddObservationTest(absltest.TestCase):

  def test_no_overwriting(self):
    preprocessor = observation_transforms.AddObservation(
        obs_name='pos',
        obs_callable=lambda _: [1., 1., 1.])

    # Generate the input spec and input timestep.
    input_obs_spec = {
        'pos': specs.Array(shape=(3,), dtype=np.float32, name='pos'),
        'quat': specs.Array(shape=(4,), dtype=np.float32, name='quat'),
    }
    input_spec = testing_functions.random_timestep_spec(
        observation_spec=input_obs_spec)

    error_msg = 'Observation pos already exists.'

    with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
      preprocessor.setup_io_spec(input_spec)

  def test_fail_to_run_obs_callable(self):

    preprocessor = observation_transforms.AddObservation(
        obs_name='new_obs',
        obs_callable=lambda timestep: timestep.observation['not_exist'])

    # Generate the input spec and input timestep.
    input_obs_spec = {
        'pos': specs.Array(shape=(3,), dtype=np.float32, name='pos'),
        'quat': specs.Array(shape=(4,), dtype=np.float32, name='quat'),
    }
    input_spec = testing_functions.random_timestep_spec(
        observation_spec=input_obs_spec)

    # The obs_callable is trying to use an observation named `not_exist` not
    # present.
    with self.assertRaisesRegex(KeyError, 'not_exist'):
      preprocessor.setup_io_spec(input_spec)

  def test_add_obs_correctly(self):
    preprocessor = observation_transforms.AddObservation(
        obs_name='new_obs',
        obs_callable=lambda _: np.asarray([1., 1., 1.], dtype=np.float32))

    # Generate the input spec and input timestep.
    input_obs_spec = {
        'pos': specs.Array(shape=(3,), dtype=np.float32, name='pos'),
        'quat': specs.Array(shape=(4,), dtype=np.float32, name='quat'),
    }
    input_spec = testing_functions.random_timestep_spec(
        observation_spec=input_obs_spec)

    input_obs = {
        'pos': np.array([1.0, -1.5, 3.2], dtype=np.float32),
        'quat': np.asarray(tr.euler_to_quat([0.1, 0.2, 0.3]), dtype=np.float32)
    }
    input_timestep = testing_functions.random_timestep(
        spec=input_spec, step_type=dm_env.StepType.MID, observation=input_obs)

    # Setup the expected output specs.
    expected_observation_spec = input_obs_spec.copy()
    expected_observation_spec['new_obs'] = (
        specs.Array(shape=[3,], dtype=np.float32, name='new_obs'))
    expected_output_spec = copy.deepcopy(input_spec)

    # Check the specs.
    output_spec = preprocessor.setup_io_spec(input_spec)
    self.assertEqual(output_spec.observation_spec,
                     expected_observation_spec)
    self.assertEqual(output_spec.reward_spec,
                     expected_output_spec.reward_spec)
    self.assertEqual(output_spec.discount_spec,
                     expected_output_spec.discount_spec)

    # Check the timestep.
    output_timestep = preprocessor.process(input_timestep)
    spec_utils.validate_timestep(output_spec, output_timestep)

    output_new_obs = output_timestep.observation['new_obs']
    np.testing.assert_array_almost_equal(output_new_obs, [1., 1., 1.])

  def test_add_obs_correctly_with_provided_specs(self):
    new_obs_spec = specs.BoundedArray(
        shape=(3,), dtype=np.int32, minimum=-1, maximum=3, name='new_obs')
    preprocessor = observation_transforms.AddObservation(
        obs_name='new_obs',
        obs_callable=lambda _: np.array([1, 1, 1], dtype=np.int32),
        obs_spec=new_obs_spec)

    # Generate the input spec and input timestep.
    input_obs_spec = {
        'pos': specs.Array(shape=(3,), dtype=np.float32, name='pos'),
        'quat': specs.Array(shape=(4,), dtype=np.float32, name='quat'),
    }
    input_spec = testing_functions.random_timestep_spec(
        observation_spec=input_obs_spec)

    input_obs = {
        'pos': np.array([1.0, -1.5, 3.2], dtype=np.float32),
        'quat': np.asarray(tr.euler_to_quat([0.1, 0.2, 0.3]), dtype=np.float32)
    }
    input_timestep = testing_functions.random_timestep(
        spec=input_spec, step_type=dm_env.StepType.MID, observation=input_obs)

    # Setup the expected specs.
    expected_observation_spec = dict(input_obs_spec)
    expected_observation_spec['new_obs'] = new_obs_spec
    expected_output_spec = copy.deepcopy(input_spec)

    output_spec = preprocessor.setup_io_spec(input_spec)

    self.assertEqual(output_spec.observation_spec,
                     expected_observation_spec)
    self.assertEqual(output_spec.reward_spec,
                     expected_output_spec.reward_spec)
    self.assertEqual(output_spec.discount_spec,
                     expected_output_spec.discount_spec)

    # Check the timestep.
    output_timestep = preprocessor.process(input_timestep)
    spec_utils.validate_timestep(output_spec, output_timestep)

    output_new_obs = output_timestep.observation['new_obs']
    np.testing.assert_array_almost_equal(output_new_obs, [1., 1., 1.])


def _build_unit_timestep_spec(
    observation_spec: Optional[Mapping[str, specs.Array]] = None,
    reward_spec: Optional[specs.Array] = None,
    discount_spec: Optional[specs.BoundedArray] = None):
  if observation_spec is None:
    name = 'foo'
    observation_spec = {
        name: specs.Array(shape=(2,), dtype=_DEFAULT_TYPE, name=name),
    }

  if reward_spec is None:
    reward_spec = scalar_array_spec(name='reward')

  if discount_spec is None:
    discount_spec = scalar_array_spec(name='discount')

  return spec_utils.TimeStepSpec(
      observation_spec=observation_spec,
      reward_spec=reward_spec,
      discount_spec=discount_spec)


if __name__ == '__main__':
  absltest.main()
