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
"""Tests for dm_robotics.agentflow.subtasks.parameterized_subtask."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.preprocessors import timestep_preprocessor as tsp
from dm_robotics.agentflow.subtasks import parameterized_subtask


class ParameterizedSubtaskTest(parameterized.TestCase):

  def test_pterm_and_result_caching(self):
    mock_preprocessor = mock.MagicMock(tsp.TimestepPreprocessor)
    default_termination_type = core.TerminationType.FAILURE
    test_subtask = parameterized_subtask.ParameterizedSubTask(
        parent_spec=mock.MagicMock(spec=spec_utils.TimeStepSpec),
        action_space=mock.MagicMock(spec=core.ActionSpace[specs.BoundedArray]),
        timestep_preprocessor=mock_preprocessor,
        default_termination_type=default_termination_type,
        name='TestParameterizedSubTask')

    parent_timestep = testing_functions.random_timestep()

    # Verify the preprocessor's result is captured.
    test_result = core.OptionResult(core.TerminationType.SUCCESS,
                                    data='test_data')
    preprocessor_timestep = tsp.PreprocessorTimestep.from_environment_timestep(
        parent_timestep, pterm=0.123, result=test_result)
    mock_preprocessor.process.return_value = preprocessor_timestep
    test_subtask.parent_to_agent_timestep(parent_timestep)
    self.assertEqual(test_subtask.pterm(parent_timestep, 'fake_key'), 0.123)
    self.assertEqual(test_subtask.subtask_result(parent_timestep, 'fake_key'),
                     test_result)

    # If preprocessor doesn't set the result we should recover the default.
    preprocessor_timestep = tsp.PreprocessorTimestep.from_environment_timestep(
        parent_timestep, pterm=0.456, result=None)
    mock_preprocessor.process.return_value = preprocessor_timestep
    test_subtask.parent_to_agent_timestep(parent_timestep)
    self.assertEqual(test_subtask.pterm(parent_timestep, 'fake_key'), 0.456)
    self.assertEqual(test_subtask.subtask_result(parent_timestep, 'fake_key'),
                     core.OptionResult(default_termination_type, data=None))

  def test_agent_action_projection(self):
    mock_preprocessor = mock.MagicMock(tsp.TimestepPreprocessor)
    default_termination_type = core.TerminationType.FAILURE
    mock_action_space = mock.create_autospec(spec=core.ActionSpace)
    test_subtask = parameterized_subtask.ParameterizedSubTask(
        parent_spec=mock.MagicMock(spec=spec_utils.TimeStepSpec),
        action_space=mock_action_space,
        timestep_preprocessor=mock_preprocessor,
        default_termination_type=default_termination_type,
        name='TestParameterizedSubTask')

    agent_action = testing_functions.random_action()
    test_subtask.agent_to_parent_action(agent_action)
    mock_action_space.project.assert_called_once_with(agent_action)


if __name__ == '__main__':
  absltest.main()
