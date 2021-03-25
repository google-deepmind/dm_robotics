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

"""Tests for example_task.example_task."""

from absl.testing import absltest
from dm_robotics.moma.tasks.example_task import example_task
import numpy as np


class ExampleTaskTest(absltest.TestCase):

  def test_environment_stepping(self):
    np.random.seed(42)
    with example_task.build_task_environment() as env:
      action = np.zeros(env.action_spec().shape)
      env.step(action)


if __name__ == '__main__':
  absltest.main()
