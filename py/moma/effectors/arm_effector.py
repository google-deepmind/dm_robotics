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

"""Standard effector for arms in sim."""

from typing import List, Optional, Tuple

from dm_control import mjcf  # type: ignore
from dm_env import specs
from dm_robotics.moma import effector
from dm_robotics.moma.effectors import mujoco_actuation
from dm_robotics.moma.models.robots.robot_arms import robot_arm
import numpy as np


class ArmEffector(effector.Effector):
  """An effector interface for a robot arm."""

  def __init__(self, arm: robot_arm.RobotArm,
               action_range_override: Optional[List[Tuple[float, float]]],
               robot_name: str):
    self._arm = arm
    self._effector_prefix = '{}_arm_joint'.format(robot_name)
    self._mujoco_effector = mujoco_actuation.MujocoEffector(
        self._arm.actuators,
        self._effector_prefix,
        action_range_override
    )

  def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
    pass

  def initialize_episode(self, physics, random_state) -> None:
    pass

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    return self._mujoco_effector.action_spec(physics)

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    self._mujoco_effector.set_control(physics, command)

  @property
  def prefix(self) -> str:
    return self._mujoco_effector.prefix
