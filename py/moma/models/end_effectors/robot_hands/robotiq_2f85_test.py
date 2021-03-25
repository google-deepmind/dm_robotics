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
"""Tests for the Robotiq 2-finger 85 adaptive gripper."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85_constants as consts
import numpy as np

# The default control time step between two agent actions.
_DEFAULT_CONTROL_TIMESTEP = 0.05
# The default physics timestep used by the simulation.
_DEFAULT_PHYSICS_TIMESTEP = 0.0005
_GRIPPER_CLOSED_ANGLE_THRESHOLD = 254
_GRIPPER_OPENED_ANGLE_THRESHOLD = 1
_OPEN_VEL = -255
_CLOSE_VEL = 255
_POS_SCALE = 255


class GripperTask(composer.Task):
  """Dummy task containing only a gripper."""

  def __init__(self, gripper):
    self.gripper = gripper
    self.set_timesteps(
        control_timestep=_DEFAULT_CONTROL_TIMESTEP,
        physics_timestep=_DEFAULT_PHYSICS_TIMESTEP)

  def get_reward(self, physics):
    return 0.

  @property
  def name(self):
    return 'gripper_task'

  @property
  def root_entity(self):
    return self.gripper


def _get_pos(environment):
  return environment.task.gripper.convert_position(
      environment.physics.bind(
          environment.task.gripper.joint_sensor).sensordata)


def measure_open_ticks_per_sec(environment, velocity):
  while _get_pos(environment) < _GRIPPER_CLOSED_ANGLE_THRESHOLD:
    environment.step(np.array([_CLOSE_VEL]))
  start_time = environment.physics.time()
  while _get_pos(environment) > _GRIPPER_OPENED_ANGLE_THRESHOLD:
    environment.step(np.array([velocity]))
  end_time = environment.physics.time()
  return -1.0 * _POS_SCALE / (end_time - start_time)


def measure_close_ticks_per_sec(environment, velocity):
  while _get_pos(environment) > _GRIPPER_OPENED_ANGLE_THRESHOLD:
    environment.step(np.array([_OPEN_VEL]))
  start_time = environment.physics.time()
  while _get_pos(environment) < _GRIPPER_CLOSED_ANGLE_THRESHOLD:
    environment.step(np.array([velocity]))
  end_time = environment.physics.time()
  return 1.0 * _POS_SCALE / (end_time - start_time)


class Robotiq2F85Test(parameterized.TestCase):
  """Tests for the Robotiq 2-finger 85 adaptive gripper."""

  def test_physics_step(self):
    gripper = robotiq_2f85.Robotiq2F85()
    physics = mjcf.Physics.from_mjcf_model(gripper.mjcf_model)
    physics.step()

  def test_ctrlrange(self):
    gripper = robotiq_2f85.Robotiq2F85()
    physics = mjcf.Physics.from_mjcf_model(gripper.mjcf_model)
    min_ctrl, max_ctrl = physics.bind(gripper.actuators[0]).ctrlrange
    self.assertEqual(255.0, max_ctrl)
    self.assertEqual(-255.0, min_ctrl)

  def test_action_spec(self):
    gripper = robotiq_2f85.Robotiq2F85()
    task = GripperTask(gripper)
    environment = composer.Environment(task)
    action_spec = environment.action_spec()
    self.assertEqual(-255.0, action_spec.minimum[0])
    self.assertEqual(255.0, action_spec.maximum[0])

  def test_grasp_fully_open(self):
    gripper = robotiq_2f85.Robotiq2F85()
    task = GripperTask(gripper)
    environment = composer.Environment(task)

    while _get_pos(environment) > _GRIPPER_OPENED_ANGLE_THRESHOLD:
      environment.step(np.array([_OPEN_VEL]))

    self.assertEqual(0, _get_pos(environment))

  def test_grasp_fully_closeed(self):
    gripper = robotiq_2f85.Robotiq2F85()
    task = GripperTask(gripper)
    environment = composer.Environment(task)

    while _get_pos(environment) < _GRIPPER_CLOSED_ANGLE_THRESHOLD:
      environment.step(np.array([_CLOSE_VEL]))

    self.assertEqual(255, _get_pos(environment))

  def test_boxes_in_fingertips(self):
    gripper = robotiq_2f85.Robotiq2F85()
    for side in ('left', 'right'):
      for idx in (1, 2):
        self.assertIsNotNone(gripper.mjcf_model.find(
            'geom', f'{side}_collision_box{idx}'))

  def test_inwards_grasp(self):
    gripper = robotiq_2f85.Robotiq2F85()
    gripper.tool_center_point.parent.add(
        'geom',
        name='inward_colliding_box',
        type='box',
        size='0.03 0.03 0.03',
        pos=gripper.tool_center_point.pos)
    task = GripperTask(gripper)
    environment = composer.Environment(task)

    # Check that we start with no grasp.
    self.assertEqual(
        gripper.grasp_sensor_callable(environment.physics), consts.NO_GRASP)

    while gripper.grasp_sensor_callable(
        environment.physics) is not consts.INWARD_GRASP:
      environment.step(np.array([_CLOSE_VEL]))

    # Check that we have inward grasp.
    self.assertEqual(
        gripper.grasp_sensor_callable(environment.physics), consts.INWARD_GRASP)

  def test_actuation_is_valid(self):
    gripper = robotiq_2f85.Robotiq2F85()
    physics = mjcf.Physics.from_mjcf_model(gripper.mjcf_model)
    act_min = 0.0
    act_max = 255.0
    actuators = physics.bind(gripper.actuators)

    # Actuator state should be clipped if the actuation goes over or below the
    # control range.
    actuators.act = [-10]
    gripper.after_substep(physics, None)
    np.testing.assert_array_equal(actuators.act, act_min)
    actuators.act = [666]
    gripper.after_substep(physics, None)
    np.testing.assert_array_equal(actuators.act, act_max)

    # Actuator state should not change the actuation
    actuators.act = [123]
    gripper.after_substep(physics, None)
    np.testing.assert_array_equal(actuators.act, 123)


if __name__ == '__main__':
  absltest.main()
