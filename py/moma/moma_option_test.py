# Copyright 2021 DeepMind Technologies Limited.
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

"""Tests for moma_option."""

from unittest import mock

from absl.testing import absltest
import dm_env
from dm_robotics import agentflow as af
from dm_robotics.moma import moma_option
import numpy as np


class MomaOptionTest(absltest.TestCase):

  def _create_agentflow_lambda_option(self, on_step_func):
    return af.LambdaOption(
        delegate=af.FixedOp(
            action=np.array([], dtype=np.float64),
            num_steps=1,
            name='test_option'),
        on_step_func=on_step_func)

  def test_sequence_of_options(self):
    """This tests a sequence of MomaOption and af.coreOption."""
    options_stepped = []

    def build_option_name_appender(option_name):
      nonlocal options_stepped

      def append_option_name(timestep):
        nonlocal options_stepped
        # When af.Sequence switches from one option to another, the
        # previous option gets one more "last" timestep sent to it.
        # Ignore this in our counting, since we just want to ensure
        # that both options are stepped.
        if not timestep.last():
          options_stepped.append(option_name)

      return append_option_name

    first_option = moma_option.MomaOption(
        physics_getter=mock.MagicMock(),
        effectors=[],
        delegate=self._create_agentflow_lambda_option(
            build_option_name_appender('first')))

    second_option = self._create_agentflow_lambda_option(
        build_option_name_appender('second'))

    option_to_test = moma_option.MomaOption(
        physics_getter=mock.MagicMock(),
        effectors=[],
        delegate=af.Sequence([
            first_option,
            second_option,
        ], allow_stepping_after_terminal=False))

    irrelevant_timestep_params = {
        'reward': 0,
        'discount': 0,
        'observation': np.array([], dtype=np.float64)
    }
    option_to_test.step(dm_env.TimeStep(dm_env.StepType.FIRST,
                                        **irrelevant_timestep_params))
    self.assertEqual(options_stepped, ['first'])
    option_to_test.step(dm_env.TimeStep(dm_env.StepType.MID,
                                        **irrelevant_timestep_params))
    self.assertEqual(options_stepped, ['first', 'second'])


if __name__ == '__main__':
  absltest.main()
