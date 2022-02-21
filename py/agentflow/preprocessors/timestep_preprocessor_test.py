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

from unittest import mock

from absl.testing import absltest
import dm_env
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.preprocessors import timestep_preprocessor


class CompositeTimestepPreprocessorTest(absltest.TestCase):

  def test_empty_preprocessor_list_gives_noop_preprocessor(self):

    # Check that the preprocessor doesn't change the spec
    input_spec = testing_functions.random_timestep_spec()
    preprocessor = timestep_preprocessor.CompositeTimestepPreprocessor()
    output_spec = preprocessor.setup_io_spec(input_spec)
    testing_functions.assert_spec(input_spec, output_spec)

    # Check that the preprocessor does not modify the timestep
    timestep = testing_functions.random_timestep(input_spec)
    input_timestep = (
        timestep_preprocessor.PreprocessorTimestep.from_environment_timestep(
            timestep, pterm=0.1))
    output_timestep = preprocessor.process(input_timestep)
    testing_functions.assert_timestep(
        input_timestep.to_environment_timestep(),
        output_timestep.to_environment_timestep())

  @mock.patch.object(spec_utils, 'validate_observation', autospec=True)
  def test_validation_frequency_controls_calls_to_spec_utils_validate(
      self, validate_obs_mock):
    input_spec = testing_functions.random_timestep_spec()
    timestep = testing_functions.random_timestep(input_spec)
    input_timestep = (
        timestep_preprocessor.PreprocessorTimestep.from_environment_timestep(
            timestep, pterm=0.1))
    with self.subTest('once_checks_only_once'):
      tsp = timestep_preprocessor.CompositeTimestepPreprocessor(
          validation_frequency=timestep_preprocessor.ValidationFrequency.ONCE)
      tsp.setup_io_spec(input_spec)
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.MID))
      validate_obs_mock.assert_not_called()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.LAST))
      validate_obs_mock.assert_not_called()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.FIRST))
      validate_obs_mock.assert_not_called()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.MID))
      validate_obs_mock.assert_not_called()
    with self.subTest('once_per_episode_only_checks_on_first_ts'):
      validate_obs_mock.reset_mock()
      tsp = timestep_preprocessor.CompositeTimestepPreprocessor(
          validation_frequency=(
              timestep_preprocessor.ValidationFrequency.ONCE_PER_EPISODE))
      tsp.setup_io_spec(input_spec)
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.MID))
      validate_obs_mock.assert_not_called()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.LAST))
      validate_obs_mock.assert_not_called()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.MID))
      validate_obs_mock.assert_not_called()
    with self.subTest('never_checks'):
      validate_obs_mock.reset_mock()
      tsp = timestep_preprocessor.CompositeTimestepPreprocessor(
          validation_frequency=timestep_preprocessor.ValidationFrequency.NEVER)
      tsp.setup_io_spec(input_spec)
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.FIRST))
      validate_obs_mock.assert_not_called()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.MID))
      validate_obs_mock.assert_not_called()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.LAST))
      validate_obs_mock.assert_not_called()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.FIRST))
      validate_obs_mock.assert_not_called()
    with self.subTest('always_checks'):
      validate_obs_mock.reset_mock()
      tsp = timestep_preprocessor.CompositeTimestepPreprocessor(
          validation_frequency=timestep_preprocessor.ValidationFrequency.ALWAYS)
      tsp.setup_io_spec(input_spec)
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.MID))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.LAST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()
      tsp.process(input_timestep._replace(step_type=dm_env.StepType.FIRST))
      validate_obs_mock.assert_called()
      validate_obs_mock.reset_mock()


if __name__ == '__main__':
  absltest.main()
