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
"""Tests for timestep_preprocessor."""

import functools
from unittest import mock

from absl.testing import absltest
import dm_env
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.preprocessors import timestep_preprocessor as tsp
import tree

ALWAYS = tsp.ValidationFrequency.ALWAYS
NEVER = tsp.ValidationFrequency.NEVER
ONCE = tsp.ValidationFrequency.ONCE
ONCE_PER_EPISODE = tsp.ValidationFrequency.ONCE_PER_EPISODE
FIRST = dm_env.StepType.FIRST
LAST = dm_env.StepType.LAST
MID = dm_env.StepType.MID


class CompositeTimestepPreprocessorTest(absltest.TestCase):

  def test_empty_preprocessor_list_gives_noop_preprocessor(self):

    # Check that the preprocessor doesn't change the spec
    input_spec = testing_functions.random_timestep_spec()
    preprocessor = tsp.CompositeTimestepPreprocessor()
    output_spec = preprocessor.setup_io_spec(input_spec)
    testing_functions.assert_spec(input_spec, output_spec)

    # Check that the preprocessor does not modify the timestep
    timestep = testing_functions.random_timestep(input_spec)
    input_timestep = (
        tsp.PreprocessorTimestep.from_environment_timestep(
            timestep, pterm=0.1))
    output_timestep = preprocessor.process(input_timestep)
    testing_functions.assert_timestep(input_timestep.to_environment_timestep(),
                                      output_timestep.to_environment_timestep())

  @mock.patch.object(spec_utils, 'validate_observation', autospec=True)
  def test_validation_frequency_controls_calls_to_spec_utils_validate(
      self, validate_obs_mock):
    input_spec = testing_functions.random_timestep_spec()
    timestep = testing_functions.random_timestep(input_spec)
    input_timestep = (
        tsp.PreprocessorTimestep.from_environment_timestep(timestep, pterm=0.1))

    with self.subTest('once_checks_only_once'):
      processor = tsp.CompositeTimestepPreprocessor(validation_frequency=ONCE)
      processor.setup_io_spec(input_spec)
      processor.process(input_timestep._replace(step_type=FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      processor.process(input_timestep._replace(step_type=MID))
      validate_obs_mock.assert_not_called()
      processor.process(input_timestep._replace(step_type=LAST))
      validate_obs_mock.assert_not_called()
      processor.process(input_timestep._replace(step_type=FIRST))
      validate_obs_mock.assert_not_called()
      processor.process(input_timestep._replace(step_type=MID))
      validate_obs_mock.assert_not_called()

    with self.subTest('once_per_episode_only_checks_on_first_ts'):
      validate_obs_mock.reset_mock()
      processor = tsp.CompositeTimestepPreprocessor(
          validation_frequency=ONCE_PER_EPISODE)
      processor.setup_io_spec(input_spec)
      processor.process(input_timestep._replace(step_type=FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      processor.process(input_timestep._replace(step_type=MID))
      validate_obs_mock.assert_not_called()
      processor.process(input_timestep._replace(step_type=LAST))
      validate_obs_mock.assert_not_called()
      processor.process(input_timestep._replace(step_type=FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      processor.process(input_timestep._replace(step_type=MID))
      validate_obs_mock.assert_not_called()

    with self.subTest('never_checks'):
      validate_obs_mock.reset_mock()
      processor = tsp.CompositeTimestepPreprocessor(validation_frequency=NEVER)
      processor.setup_io_spec(input_spec)
      processor.process(input_timestep._replace(step_type=FIRST))
      validate_obs_mock.assert_not_called()
      processor.process(input_timestep._replace(step_type=MID))
      validate_obs_mock.assert_not_called()
      processor.process(input_timestep._replace(step_type=LAST))
      validate_obs_mock.assert_not_called()
      processor.process(input_timestep._replace(step_type=FIRST))
      validate_obs_mock.assert_not_called()

    with self.subTest('always_checks'):
      validate_obs_mock.reset_mock()
      processor = tsp.CompositeTimestepPreprocessor(validation_frequency=ALWAYS)
      processor.setup_io_spec(input_spec)
      processor.process(input_timestep._replace(step_type=FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      processor.process(input_timestep._replace(step_type=MID))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      processor.process(input_timestep._replace(step_type=LAST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      processor.process(input_timestep._replace(step_type=FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()

  def test_as_list_allows_tree_traversal(self):
    # Tests that we can create a nested CompositeTimestepPreprocessor and use
    # the `as_list` mechanism to visit all processors.

    class DummyPreprocessor(tsp.TimestepPreprocessor):
      """Dummy processor which passes `np.prod(spec.shape) > 128` check."""

      def _process_impl(
          self,
          timestep: tsp.PreprocessorTimestep) -> tsp.PreprocessorTimestep:
        return timestep

      def _output_spec(
          self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
        return input_spec

    leaf_processors = [
        mock.MagicMock(wraps=DummyPreprocessor()),
        mock.MagicMock(wraps=DummyPreprocessor()),
        mock.MagicMock(wraps=DummyPreprocessor())
    ]
    middle_preprocessor = tsp.CompositeTimestepPreprocessor(*leaf_processors)
    middle_preprocessor = mock.MagicMock(wraps=middle_preprocessor)

    top_preprocessor = tsp.CompositeTimestepPreprocessor(middle_preprocessor)
    top_preprocessor = mock.MagicMock(wraps=top_preprocessor)

    # Disable validation for entire processor.
    def set_validation_frequency(proc, freq):
      proc.set_validation_frequency(freq)

    _ = tree.map_structure(
        functools.partial(set_validation_frequency, freq=NEVER),
        top_preprocessor.as_list())

    # Verify all validation is disabled.
    expected_validation_frequency_flattened = [NEVER] * 5
    actual_validation_frequency_flattened = [
        p.validation_frequency for p in tree.flatten(top_preprocessor.as_list())
    ]
    self.assertSequenceEqual(actual_validation_frequency_flattened,
                             expected_validation_frequency_flattened)

    # Verify the structure is preserved when traversing with `map_structure`
    actual_processor_names = tree.map_structure(lambda p: p.name,
                                                top_preprocessor.as_list())
    expected_processor_names = [
        'CompositeTimestepPreprocessor',
        [[
            'CompositeTimestepPreprocessor',
            [['DummyPreprocessor'], ['DummyPreprocessor'],
             ['DummyPreprocessor']]
        ]]
    ]
    self.assertSequenceEqual(actual_processor_names, expected_processor_names)


if __name__ == '__main__':
  absltest.main()
