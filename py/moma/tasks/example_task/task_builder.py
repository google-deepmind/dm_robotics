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

"""A module for constructing an example handshake task and its dependencies."""

from typing import Sequence, Text, Tuple

import attr
from dm_control import composer
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma import base_task
from dm_robotics.moma import entity_composer
from dm_robotics.moma import entity_initializer
from dm_robotics.moma import prop
from dm_robotics.moma import robot
from dm_robotics.moma import scene_initializer
from dm_robotics.moma.effectors import arm_effector as arm_effector_module
from dm_robotics.moma.effectors import default_gripper_effector
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.models.end_effectors.wrist_sensors import robotiq_fts300
from dm_robotics.moma.models.robots.robot_arms import sawyer
from dm_robotics.moma.models.robots.robot_arms import sawyer_constants
from dm_robotics.moma.sensors import robot_arm_sensor
from dm_robotics.moma.sensors import robot_tcp_sensor
import numpy as np


@attr.s(auto_attribs=True)
class MutableTaskComponents:
  """Components that are returned with the task for easier access."""
  robot0_pose_initializer: scene_initializer.EntityPoseInitializer = None
  robot1_pose_initializer: scene_initializer.EntityPoseInitializer = None
  gripper0_initializer: entity_initializer.PoseInitializer = None
  gripper1_initializer: entity_initializer.PoseInitializer = None
  prop_initializer: entity_initializer.PoseInitializer = None


@attr.s(frozen=True)
class TaskComponents(MutableTaskComponents):
  pass


class ExampleTaskComposer(entity_composer.TaskEntitiesComposer):
  """Task composer that performs entity composition on the scene."""

  def __init__(self, task_robots: Sequence[robot.Robot],
               task_props: Sequence[prop.Prop]):
    self._robots = task_robots
    self._props = task_props

  def compose_entities(self, arena: composer.Arena) -> None:
    """Adds all of the necessary objects to the arena and composes objects."""
    for rbt in self._robots:
      arena.attach(rbt.arm)

    for p in self._props:
      frame = arena.add_free_entity(p)
      p.set_freejoint(frame.freejoint)


def build_task() -> Tuple[base_task.BaseTask, TaskComponents]:
  """Builds a BaseTask and all dependencies."""
  arena = _build_arena(name='arena')
  task_props = _build_props()

  task_robots = [
      _build_sawyer_robot(robot_name='robot0'),
      _build_sawyer_robot(robot_name='robot1')
  ]

  task_composer = ExampleTaskComposer(task_robots, task_props)
  task_composer.compose_entities(arena)

  components = MutableTaskComponents()
  _populate_scene_initializers(task_robots, components)
  _populate_gripper_initializers(task_robots, components)
  _populate_prop_initializer(task_props, components)

  # This initializer is used to place the robots before compiling the model.
  static_initializer = scene_initializer.CompositeSceneInitializer([
      components.robot0_pose_initializer,
      components.robot1_pose_initializer,
  ])

  # This initializer is used to set the state of the simulation once the
  # physics model as be compiled.
  dynamic_initializer = entity_initializer.TaskEntitiesInitializer([
      components.gripper0_initializer,
      components.gripper1_initializer,
      components.prop_initializer,
  ])

  task = base_task.BaseTask(
      task_name='handshake',
      arena=arena,
      robots=task_robots,
      props=task_props,
      extra_sensors=[],
      extra_effectors=[],
      scene_initializer=static_initializer,
      episode_initializer=dynamic_initializer,
      control_timestep=0.1)
  return task, TaskComponents(**attr.asdict(components))


def _populate_scene_initializers(task_robots: Sequence[robot.Robot],
                                 components: MutableTaskComponents) -> None:
  """Populates components with initializers that arrange the scene."""

  pose0 = pose_distribution.ConstantPoseDistribution(
      np.array([-0.75, 0., 0., 0., 0., 0.]))
  pose1 = pose_distribution.ConstantPoseDistribution(
      np.array([0.75, 0., 0.0, 0., 0., 0.]))

  components.robot0_pose_initializer = scene_initializer.EntityPoseInitializer(
      entity=task_robots[0].arm_frame, pose_sampler=pose0.sample_pose)
  components.robot1_pose_initializer = scene_initializer.EntityPoseInitializer(
      entity=task_robots[1].arm_frame, pose_sampler=pose1.sample_pose)


def _populate_gripper_initializers(task_robots: Sequence[robot.Robot],
                                   components: MutableTaskComponents) -> None:
  """Populates components with gripper initializers."""

  pose0 = pose_distribution.ConstantPoseDistribution(
      np.array([-0.1, 0., 0.4, -np.pi, 0., 0.]))
  pose1 = pose_distribution.ConstantPoseDistribution(
      np.array([0.1, 0., 0.4, np.pi, 0., 0.]))
  components.gripper0_initializer = entity_initializer.PoseInitializer(
      task_robots[0].position_gripper, pose0.sample_pose)
  components.gripper1_initializer = entity_initializer.PoseInitializer(
      task_robots[1].position_gripper, pose1.sample_pose)


def _populate_prop_initializer(task_props: Sequence[prop.Prop],
                               components: MutableTaskComponents):
  """Populates components with prop pose initializers."""
  prop_pose = pose_distribution.ConstantPoseDistribution(
      np.array([0.2, 0.2, 0.06, 0., 0., 0.]))
  components.prop_initializer = entity_initializer.PoseInitializer(
      initializer_fn=task_props[0].set_pose, pose_sampler=prop_pose.sample_pose)


def _build_arena(name: Text) -> composer.Arena:
  """Builds an arena Entity."""
  arena = empty.Arena(name)
  arena.mjcf_model.option.timestep = 0.001
  arena.mjcf_model.option.gravity = (0., 0., -1.0)
  arena.mjcf_model.size.nconmax = 1000
  arena.mjcf_model.size.njmax = 2000
  arena.mjcf_model.visual.__getattr__('global').offheight = 480
  arena.mjcf_model.visual.__getattr__('global').offwidth = 640
  arena.mjcf_model.visual.map.znear = 0.0005

  return arena


def _build_sawyer_robot(robot_name: str) -> robot.Robot:
  """Returns a Sawyer robot."""

  arm = sawyer.Sawyer(
      name=robot_name, actuation=sawyer_constants.Actuation.INTEGRATED_VELOCITY)
  gripper = robotiq_2f85.Robotiq2F85()
  wrist_ft = robotiq_fts300.RobotiqFTS300()

  # Compose the robot after its model components are constructed. This should
  # usually be done early on as some Effectors (and possibly Sensors) can only
  # be constructed after the robot components have been composed.
  robot.standard_compose(
      arm=arm, gripper=gripper, wrist_ft=wrist_ft, wrist_cameras=[])

  robot_sensors = [
      robot_arm_sensor.RobotArmSensor(
          arm=arm, name=robot_name, have_torque_sensors=True),
      robot_tcp_sensor.RobotTCPSensor(gripper=gripper, name=robot_name)
  ]

  arm_effector = arm_effector_module.ArmEffector(
      arm=arm, action_range_override=None, robot_name=robot_name)
  gripper_effector = default_gripper_effector.DefaultGripperEffector(
      gripper, robot_name)

  return robot.StandardRobot(
      arm=arm,
      arm_base_site_name='pedestal_attachment',
      gripper=gripper,
      wrist_ft=wrist_ft,
      robot_sensors=robot_sensors,
      arm_effector=arm_effector,
      gripper_effector=gripper_effector,
      name=robot_name)


def _build_props() -> Sequence[prop.Prop]:
  """Build task props."""
  block = prop.Block()
  return (block,)
