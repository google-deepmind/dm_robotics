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

# Lint as: python3
"""Tak builder for the dm_env interface to the example task."""
import dm_env

from dm_robotics import agentflow as af
from dm_robotics.agentflow.preprocessors import observation_transforms
from dm_robotics.agentflow.preprocessors import rewards
from dm_robotics.agentflow.subtasks import subtask_termination
from dm_robotics.moma import action_spaces
from dm_robotics.moma import subtask_env_builder
from dm_robotics.moma.tasks.example_task import task_builder


def build_task_environment() -> dm_env.Environment:
  """Returns the environment."""
  task, components = task_builder.build_task()
  del components

  env_builder = subtask_env_builder.SubtaskEnvBuilder()
  env_builder.set_task(task)
  task_env = env_builder.build_base_env()

  # Define the action space.
  parent_action_spec = task.effectors_action_spec(physics=task_env.physics)
  robot_action_spaces = []
  for rbt in task.robots:
    # Joint space control of each individual robot.
    joint_action_space = action_spaces.ArmJointActionSpace(
        af.prefix_slicer(parent_action_spec, rbt.arm_effector.prefix))
    gripper_action_space = action_spaces.GripperActionSpace(
        af.prefix_slicer(parent_action_spec, rbt.gripper_effector.prefix))

    # Gripper isn't controlled by the agent for this task.
    gripper_action_space = af.FixedActionSpace(
        gripper_action_space,
        gripper_action_space.spec().minimum)

    robot_action_spaces.extend([joint_action_space, gripper_action_space])

  env_builder.set_action_space(
      action_spaces.CompositeActionSpace(robot_action_spaces))

  env_builder.add_preprocessor(observation_transforms.CastPreprocessor())
  env_builder.add_preprocessor(
      rewards.L2Reward(obs0='robot0_tcp_pos', obs1='robot1_tcp_pos'))

  # End episodes after 100 steps.
  env_builder.add_preprocessor(subtask_termination.MaxStepsTermination(100))

  return env_builder.build()
