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

"""Integration test for the gripper, its sensor and effector."""

from absl.testing import absltest
from dm_control import composer
from dm_robotics import agentflow as af
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma import action_spaces
from dm_robotics.moma import base_task
from dm_robotics.moma import entity_initializer
from dm_robotics.moma import robot
from dm_robotics.moma import scene_initializer
from dm_robotics.moma import subtask_env_builder
from dm_robotics.moma.effectors import arm_effector
from dm_robotics.moma.effectors import default_gripper_effector
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.models.robots.robot_arms import sawyer
from dm_robotics.moma.models.robots.robot_arms import sawyer_constants
from dm_robotics.moma.sensors import robotiq_gripper_sensor
import numpy as np


class GripperSensorIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._arena = _build_arena(name='arena')
    self._robot = _build_sawyer_with_gripper(robot_name='robot0')
    self._arena.attach(self._robot.arm)

    robot_pose = pose_distribution.ConstantPoseDistribution(
        np.array([-0.75, 0., 0., 0., 0., 0.]))
    robot_pose_initializer = scene_initializer.EntityPoseInitializer(
        entity=self._robot.arm_frame, pose_sampler=robot_pose.sample_pose)

    gripper_pose = pose_distribution.ConstantPoseDistribution(
        np.array([-0.1, 0., 0.4, -np.pi, 0., 0.]))
    gripper_initializer = entity_initializer.PoseInitializer(
        self._robot.position_gripper, gripper_pose.sample_pose)

    static_initializer = robot_pose_initializer
    dynamic_initializer = gripper_initializer

    self._task = base_task.BaseTask(
        task_name='handshake',
        arena=self._arena,
        robots=[self._robot],
        props=[],
        extra_sensors=[],
        extra_effectors=[],
        scene_initializer=static_initializer,
        episode_initializer=dynamic_initializer,
        control_timestep=0.1)

  def test_actuator_and_sensor(self):
    env_builder = subtask_env_builder.SubtaskEnvBuilder()
    env_builder.set_task(self._task)
    task_env = env_builder.build_base_env()
    parent_action_spec = self._task.effectors_action_spec(task_env.physics)
    gripper_action_space = action_spaces.GripperActionSpace(
        af.prefix_slicer(
            parent_action_spec,
            self._robot.gripper_effector.prefix,
            default_value=0.0))

    env_builder.set_action_space(gripper_action_space)
    with env_builder.build() as env:
      timestep = env.reset()

      # Assert the shapes of the observations:
      self.assertEqual(timestep.observation['robot0_gripper_pos'].shape, (1,))
      self.assertEqual(timestep.observation['robot0_gripper_vel'].shape, (1,))
      self.assertEqual(timestep.observation['robot0_gripper_grasp'].shape, (1,))

      for i in range(5000):
        old_pos = timestep.observation['robot0_gripper_pos']
        # We send the max command.
        timestep = env.step(np.asarray([255.0]))
        cur_pos = timestep.observation['robot0_gripper_pos']
        vel = timestep.observation['robot0_gripper_vel']
        self.assertEqual(vel, cur_pos - old_pos)
        # We can never reach to position 255 because we are using a p controller
        # and there will therefore always be a steady state error. More details
        # in the actuation of the gripper.
        if cur_pos[0] >= 254:
          break
        self.assertLess(i, 4999)  # 5000 steps should be enough to get to 255.


def _build_arena(name: str) -> composer.Arena:
  """Builds an arena Entity."""
  arena = empty.Arena(name)
  arena.ground.size = (2.0, 2.0, 2.0)
  arena.mjcf_model.option.timestep = 0.001
  arena.mjcf_model.option.gravity = (0., 0., -1.0)
  arena.mjcf_model.size.nconmax = 1000
  arena.mjcf_model.size.njmax = 2000
  arena.mjcf_model.visual.__getattr__('global').offheight = 480
  arena.mjcf_model.visual.__getattr__('global').offwidth = 640
  arena.mjcf_model.visual.map.znear = 0.0005
  return arena


def _build_sawyer_with_gripper(robot_name: str) -> robot.Robot:
  """Returns a Sawyer robot."""

  arm = sawyer.Sawyer(
      name=robot_name, actuation=sawyer_constants.Actuation.INTEGRATED_VELOCITY)
  gripper = robotiq_2f85.Robotiq2F85()
  arm.attach(gripper)

  robot_sensors = [
      robotiq_gripper_sensor.RobotiqGripperSensor(
          gripper=gripper, name=f'{robot_name}_gripper')
  ]

  robot_effector = arm_effector.ArmEffector(
      arm=arm, action_range_override=None, robot_name=robot_name)
  gripper_effector = default_gripper_effector.DefaultGripperEffector(
      gripper, robot_name)

  return robot.StandardRobot(
      arm=arm,
      arm_base_site_name='pedestal_attachment',
      gripper=gripper,
      robot_sensors=robot_sensors,
      arm_effector=robot_effector,
      gripper_effector=gripper_effector,
      name=robot_name)


if __name__ == '__main__':
  absltest.main()
