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

"""A module for encapsulating the robot arm, gripper, and bracelet."""

import abc
from typing import List, Optional, Sequence

from dm_control import composer
from dm_control import mjcf
from dm_control.mjcf import traversal_utils
from dm_robotics.geometry import geometry
from dm_robotics.moma import effector
from dm_robotics.moma import prop
from dm_robotics.moma import sensor as moma_sensor
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand
from dm_robotics.moma.models.robots.robot_arms import robot_arm
from dm_robotics.moma.utils import ik_solver
import numpy as np


class Robot(abc.ABC):
  """Abstract base class for MOMA robotic arms and their attachments."""

  @property
  @abc.abstractmethod
  def name(self):
    pass

  @property
  @abc.abstractmethod
  def sensors(self) -> Sequence[moma_sensor.Sensor]:
    pass

  @property
  @abc.abstractmethod
  def effectors(self) -> List[effector.Effector]:
    pass

  @property
  @abc.abstractmethod
  def arm_effector(self) -> effector.Effector:
    pass

  @property
  @abc.abstractmethod
  def gripper_effector(self) -> Optional[effector.Effector]:
    pass

  @property
  @abc.abstractmethod
  def arm(self) -> robot_arm.RobotArm:
    pass

  @property
  @abc.abstractmethod
  def gripper(self) -> robot_hand.AnyRobotHand:
    pass

  @property
  @abc.abstractmethod
  def wrist_ft(self):
    pass

  @property
  @abc.abstractmethod
  def wrist_cameras(self):
    """Returns a sequence of wrist cameras (if any)."""

  @property
  @abc.abstractmethod
  def arm_base_site(self):
    """Returns a site at the base of the arm."""

  @property
  @abc.abstractmethod
  def arm_frame(self):
    pass

  @abc.abstractmethod
  def position_gripper(self, physics: mjcf.Physics, position: np.ndarray,
                       quaternion: np.ndarray):
    """Moves the gripper ik point position to the (pos, quat) pose tuple."""

  @abc.abstractmethod
  def position_arm_joints(self, physics, joint_angles):
    """Positions the arm joints to the given angles."""


class StandardRobot(Robot):
  """A Robot class representing the union of arm, gripper, and bracelet."""

  def __init__(
      self,
      arm: robot_arm.RobotArm,
      arm_base_site_name: str,
      gripper: robot_hand.AnyRobotHand,
      robot_sensors: Sequence[moma_sensor.Sensor],
      arm_effector: effector.Effector,
      gripper_effector: Optional[effector.Effector],
      wrist_ft: Optional[composer.Entity] = None,
      wrist_cameras: Optional[Sequence[prop.Camera]] = None,
      name: str = 'robot'):
    """Robot constructor.

    Args:
      arm: The robot arm Entity.
      arm_base_site_name: The label of the base site of the arm model, so that
        we can position the robot base in the world.
      gripper: The gripper Entity to attach to the arm.
      robot_sensors: List of abstract Sensors that are associated with this
        robot.
      arm_effector: An effector for the robot arm.
      gripper_effector: An effector for the robot gripper.
      wrist_ft: Optional wrist force-torque Entity to add between the arm and
        gripper in the kinematic chain.
      wrist_cameras: Optional list of camera props attached to the robot wrist.
      name: A unique name for the robot.
    """

    self._arm = arm
    self._gripper = gripper
    self._robot_sensors = robot_sensors
    self._arm_effector = arm_effector
    self._gripper_effector = gripper_effector
    self._wrist_ft = wrist_ft
    self._wrist_cameras = wrist_cameras or []
    self._name = name

    # Site for the robot "base" for reporting wrist-site pose observations.
    self._arm_base_site = self._arm.mjcf_model.find('site', arm_base_site_name)
    self._gripper_ik_site = self._gripper.tool_center_point

  @property
  def name(self) -> str:
    return self._name

  @property
  def sensors(self) -> Sequence[moma_sensor.Sensor]:
    return self._robot_sensors

  @property
  def effectors(self) -> List[effector.Effector]:
    effectors = [self._arm_effector]
    if self.gripper_effector is not None:
      assert self.gripper_effector is not None  # This placates pytype.
      effectors.append(self.gripper_effector)
    return effectors

  @property
  def arm_effector(self) -> effector.Effector:
    return self._arm_effector

  @property
  def gripper_effector(self) -> Optional[effector.Effector]:
    return self._gripper_effector

  @property
  def arm(self) -> robot_arm.RobotArm:
    return self._arm

  @property
  def gripper(self) -> robot_hand.AnyRobotHand:
    return self._gripper

  @property
  def wrist_ft(self):
    return self._wrist_ft

  @property
  def wrist_cameras(self) -> Sequence[prop.Camera]:
    return self._wrist_cameras

  @property
  def arm_base_site(self):
    """Returns a site at the base of the arm."""
    return self._arm_base_site

  @property
  def arm_frame(self):
    return traversal_utils.get_attachment_frame(self._arm.mjcf_model)

  def position_gripper(self, physics: mjcf.Physics, position: np.ndarray,
                       quaternion: np.ndarray):
    """Moves the gripper ik point position to the (pos, quat) pose tuple.

    Args:
      physics: An MJCF physics.
      position: The cartesian position of the desired pose given in the world
        frame.
      quaternion: The quaternion (wxyz) giving the desired orientation of the
        gripper in the world frame.

    Raises:
      ValueError: If the gripper cannot be placed at the desired pose.
    """

    # Initialize the ik solver. We create a new version of the solver at every
    # solve because there is no guarantee that the mjcf_model has not been
    # modified.
    mjcf_model = self._arm.mjcf_model.root_model
    solver = ik_solver.IkSolver(
        mjcf_model, self._arm.joints, self._gripper_ik_site)

    qpos = solver.solve(ref_pose=geometry.Pose(position, quaternion))

    if qpos is None:
      if self.gripper_effector is not None:
        gripper_prefix = self.gripper_effector.prefix  # pytype: disable=attribute-error
      else:
        gripper_prefix = 'gripper'
      raise ValueError('IK Failed to converge to the desired target pose'
                       f'{geometry.Pose(position, quaternion)} '
                       f'for {gripper_prefix} of {self._arm.name}')

    self.position_arm_joints(physics, qpos)

  def position_arm_joints(self, physics, joint_angles):
    self.arm.set_joint_angles(physics, joint_angles)


def standard_compose(arm: robot_arm.RobotArm, gripper: robot_hand.AnyRobotHand,
                     wrist_ft: Optional[composer.Entity] = None,
                     wrist_cameras: Sequence[prop.Camera] = ()) -> None:
  """Creates arm and attaches gripper."""

  if wrist_ft:
    wrist_ft.attach(gripper)
    arm.attach(wrist_ft)
  else:
    arm.attach(gripper)

  for cam in wrist_cameras:
    arm.attach(cam, arm.wrist_site)
