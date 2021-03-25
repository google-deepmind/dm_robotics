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
"""An Control-Flow example which builds up a simple insertion experiment."""

from typing import Optional

from absl import app
import dm_env
from dm_robotics.agentflow import core
from dm_robotics.agentflow import subtask
from dm_robotics.agentflow.meta_options.control_flow import cond
from dm_robotics.agentflow.meta_options.control_flow import loop_ops
from dm_robotics.agentflow.meta_options.control_flow import sequence
from dm_robotics.agentflow.meta_options.control_flow.examples import common
from dm_robotics.agentflow.rendering import graphviz_renderer


def near_socket(unused_timestep: dm_env.TimeStep,
                unused_result: Optional[core.OptionResult]) -> bool:
  return False


def last_option_successful(unused_timestep: dm_env.TimeStep,
                           result: core.OptionResult):
  return result.termination_reason == core.TerminationType.SUCCESS


def build():
  """Builds example graph."""
  env = common.DummyEnv()

  # Define a subtask that exposes the desired RL-environment view on `base_task`
  my_subtask = common.DummySubTask(env.observation_spec(), 'Insertion SubTask')

  # Define a regular RL agent against this task-spec.
  my_policy = common.DummyPolicy(my_subtask.action_spec(),
                                 my_subtask.observation_spec(), 'My Policy')

  # Compose the policy and subtask to form an Option module.
  learned_insert_option = subtask.SubTaskOption(
      my_subtask, my_policy, name='Learned Insertion')

  reach_option = common.DummyOption(env.action_spec(), env.observation_spec(),
                                    'Reach for Socket')
  scripted_reset = common.DummyOption(env.action_spec(), env.observation_spec(),
                                      'Scripted Reset')
  extract_option = common.DummyOption(env.action_spec(), env.observation_spec(),
                                      'Extract')
  recovery_option = common.DummyOption(env.action_spec(),
                                       env.observation_spec(), 'Recover')

  # Use some AgentFlow operators to embed the agent in a bigger agent.
  # First use Cond to op run learned-agent if sufficiently close.
  reach_or_insert_op = cond.Cond(
      cond=near_socket,
      true_branch=learned_insert_option,
      false_branch=reach_option,
      name='Reach or Insert')

  # Loop the insert-or-reach option 5 times.
  reach_and_insert_5x = loop_ops.Repeat(
      5, reach_or_insert_op, name='Retry Loop')

  loop_body = sequence.Sequence([
      scripted_reset,
      reach_and_insert_5x,
      cond.Cond(
          cond=last_option_successful,
          true_branch=extract_option,
          false_branch=recovery_option,
          name='post-insert')
  ])
  main_loop = loop_ops.While(lambda _: True, loop_body)

  graphviz_renderer.open_viewer(main_loop)

  return main_loop


def main(unused_argv):
  build()


if __name__ == '__main__':
  app.run(main)
