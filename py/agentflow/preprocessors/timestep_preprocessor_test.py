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

from absl.testing import absltest
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


if __name__ == '__main__':
  absltest.main()
