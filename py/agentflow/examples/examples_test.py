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

"""Tests for dm_robotics.agentflow.examples.examples."""

from absl.testing import absltest
from dm_robotics.agentflow.examples import direct_dispatch_workflow
from dm_robotics.agentflow.examples import environment_dispatch_workflow


class ExamplesTest(absltest.TestCase):
  """Simple smoke-tests for agentflow workflow examples."""

  def test_direct_dispatch_workflow(self):
    direct_dispatch_workflow.main(None)

  def test_environment_dispatch_workflow(self):
    environment_dispatch_workflow.main(None)


if __name__ == '__main__':
  absltest.main()
