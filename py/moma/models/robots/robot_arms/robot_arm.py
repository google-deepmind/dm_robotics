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
"""MOMA Composer Robot Arm."""

import abc
from typing import List
from dm_control import composer
from dm_control import mjcf
from dm_robotics.moma.models import types
import numpy as np


class RobotArm(abc.ABC, composer.Entity):
  """MOMA composer robot arm base class."""

  @abc.abstractmethod
  def _build(self):
    """Entity initialization method to be overridden by subclasses."""
    pass

  @property
  @abc.abstractmethod
  def joints(self) -> List[types.MjcfElement]:
    """List of joint elements belonging to the arm."""
    pass

  @property
  @abc.abstractmethod
  def actuators(self) -> List[types.MjcfElement]:
    """List of actuator elements belonging to the arm."""
    pass

  @property
  @abc.abstractmethod
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to this robot arm."""
    pass

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Name of the robot arm."""
    pass

  @property
  @abc.abstractmethod
  def wrist_site(self) -> types.MjcfElement:
    """Get the MuJoCo site of the wrist.

    Returns:
      MuJoCo site
    """
    pass

  @property
  def attachment_site(self):
    """Make the wrist site the default attachment site for mjcf elements."""
    return self.wrist_site

  @abc.abstractmethod
  def set_joint_angles(self, physics: mjcf.Physics,
                       joint_angles: np.ndarray) -> None:
    """Sets the joints of the robot to a given configuration."""
    pass

