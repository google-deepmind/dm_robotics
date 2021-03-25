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

"""Tests for base_task."""

import contextlib

from absl.testing import absltest
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer.observation import updater
from dm_control.rl import control
from dm_robotics.moma import base_task
from dm_robotics.moma import entity_initializer
from dm_robotics.moma import robot
from dm_robotics.moma import scene_initializer
from dm_robotics.moma.effectors import arm_effector as arm_effector_module
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.models.robots.robot_arms import sawyer
from dm_robotics.moma.sensors import robot_arm_sensor
from dm_robotics.moma.sensors import robot_tcp_sensor
import numpy as np


class BaseTaskTest(absltest.TestCase):

  def _build_sample_base_task(self):
    arena = empty.Arena('test_arena')

    arm = sawyer.Sawyer(with_pedestal=False)

    gripper = robotiq_2f85.Robotiq2F85()

    robot.standard_compose(
        arm=arm, gripper=gripper, wrist_ft=None, wrist_cameras=[])

    robot_sensors = [
        robot_arm_sensor.RobotArmSensor(
            arm=arm, name='robot0', have_torque_sensors=True),
        robot_tcp_sensor.RobotTCPSensor(gripper=gripper, name='robot0'),
    ]

    arm_effector = arm_effector_module.ArmEffector(
        arm=arm, action_range_override=None, robot_name='robot0')

    rbt = robot.StandardRobot(
        arm=arm,
        arm_base_site_name='pedestal_attachment',
        gripper=gripper,
        wrist_ft=None,
        wrist_cameras=[],
        robot_sensors=robot_sensors,
        arm_effector=arm_effector,
        gripper_effector=None,
        name='robot0')

    arena.attach(arm)

    task = base_task.BaseTask(
        task_name='test',
        arena=arena,
        robots=[rbt],
        props=[],
        extra_sensors=[],
        extra_effectors=[],
        scene_initializer=scene_initializer.CompositeSceneInitializer([]),
        episode_initializer=entity_initializer.TaskEntitiesInitializer([]),
        control_timestep=0.1)

    return task

  def test_observables(self):
    """Test that the task observables includes only sensor observables."""

    task = self._build_sample_base_task()
    task_obs = set(task.observables)

    robot_sensor_obs = []
    for s in task.robots[0].sensors:
      robot_sensor_obs.extend(list(s.observables))
    robot_sensor_obs = set(robot_sensor_obs)

    self.assertEmpty(task_obs ^ robot_sensor_obs)

  def test_observable_types(self):
    """Test that the task observables includes only sensor observables."""

    task = self._build_sample_base_task()
    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
    obs_spec = env.observation_spec()

    acceptable_types = set(
        [np.dtype(np.uint8),
         np.dtype(np.int64),
         np.dtype(np.float32)])

    for spec in obs_spec.values():
      self.assertIn(np.dtype(spec.dtype), acceptable_types)


class FakePhysics(control.Physics):
  """A fake Physics class for unit testing observations."""

  def __init__(self):
    self._step_counter = 0
    self._observables = {}

  def step(self, sub_steps=1):
    self._step_counter += 1

  @property
  def observables(self):
    return self._observables

  def time(self):
    return self._step_counter

  def timestep(self):
    return 1.0

  def set_control(self, ctrl):
    pass

  def reset(self):
    self._step_counter = 0

  def after_reset(self):
    pass

  @contextlib.contextmanager
  def suppress_physics_errors(self):
    yield


class CastObservationsTest(absltest.TestCase):

  def testCastAggregatedObservable(self):
    physics = FakePhysics()
    physics.observables['raw_value'] = observable.Generic(
        raw_observation_callable=lambda unused: np.float64(physics.time()),
        update_interval=1,
        buffer_size=2,
        aggregator=lambda arr: np.asarray([arr[0], arr[1], arr[0] + arr[1]]),
        corruptor=lambda value, random_state: value * 10.0)
    physics.observables['cast_value'] = base_task.CastObservable(
        physics.observables['raw_value'])

    for obs in physics.observables.values():
      obs.enabled = True
    physics.reset()

    physics_steps_per_control_step = 2
    observation_updater = updater.Updater(
        physics.observables,
        physics_steps_per_control_step,
        strip_singleton_buffer_dim=True)
    observation_updater.reset(physics=physics, random_state=None)

    raw_values, cast_values = [], []
    for unused_step in range(0, 3):
      observation_updater.prepare_for_next_control_step()
      for _ in range(physics_steps_per_control_step):
        physics.step()
        observation_updater.update()

      observation = observation_updater.get_observation()
      print(observation)

      raw_values.append(observation['raw_value'])
      cast_values.append(observation['cast_value'])

    np.testing.assert_equal(raw_values[0], np.asarray([10.0, 20.0, 30.0]))
    np.testing.assert_equal(cast_values[0], np.asarray([10.0, 20.0, 30.0]))
    np.testing.assert_equal(raw_values[1], np.asarray([30.0, 40.0, 70.0]))
    np.testing.assert_equal(cast_values[1], np.asarray([30.0, 40.0, 70.0]))
    np.testing.assert_equal(raw_values[2], np.asarray([50.0, 60.0, 110.0]))
    np.testing.assert_equal(cast_values[2], np.asarray([50.0, 60.0, 110.0]))
    self.assertEqual(raw_values[0].dtype, np.float64)
    self.assertEqual(cast_values[0].dtype, np.float32)


if __name__ == '__main__':
  absltest.main()
