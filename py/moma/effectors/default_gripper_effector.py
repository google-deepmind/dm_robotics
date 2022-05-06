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

"""Default effector for grippers in sim."""

from dm_control import mjcf  # type: ignore
from dm_env import specs
from dm_robotics.moma import effector
from dm_robotics.moma.effectors import mujoco_actuation
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand
import numpy as np


class DefaultGripperEffector(effector.Effector):
  """An effector interface for MoMa grippers."""

  def __init__(self, gripper: robot_hand.RobotHand, robot_name: str):
    self._gripper = gripper
    self._effector_prefix = '{}_gripper'.format(robot_name)

    self._mujoco_effector = mujoco_actuation.MujocoEffector(
        self._gripper.actuators, self._effector_prefix)

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    return self._mujoco_effector.action_spec(physics)

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    self._mujoco_effector.set_control(physics, command)

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass

  @property
  def prefix(self) -> str:
    return self._effector_prefix
