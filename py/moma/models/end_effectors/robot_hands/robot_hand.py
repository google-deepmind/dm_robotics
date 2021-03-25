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
"""MOMA Composer Robot Hand.

Rationale and difference to `robot_base.RobotHand`:
MoMa communicates to hardware (sim and real) though the Sensor and Actuator
interfaces.  It does not intend users (for example scripted policies) to
perform these actions through the set_grasp function.  While sim hardware can
and should be reset (aka initialized) differently to real hardware, it's
expected that normal behavioural policies, learnt or not, use the Sensor and
Actuator interfaces.

In this way the same control mechanisms, e.g. collision avoidance, cartesian
to joint mapping can be used without special cases.
"""

import abc
from typing import Sequence
from typing import Union

from dm_control import composer
from dm_control.entities.manipulators import base as robot_base
from dm_robotics.moma.models import types


class RobotHand(abc.ABC, composer.Entity):
  """MOMA composer robot hand base class."""

  @abc.abstractmethod
  def _build(self):
    """Entity initialization method to be overridden by subclasses."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def joints(self) -> Sequence[types.MjcfElement]:
    """List of joint elements belonging to the hand."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def actuators(self) -> Sequence[types.MjcfElement]:
    """List of actuator elements belonging to the hand."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def mjcf_model(self) -> types.MjcfElement:
    """Returns the `mjcf.RootElement` object corresponding to the robot hand."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Name of the robot hand."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def tool_center_point(self) -> types.MjcfElement:
    """Tool center point site of the hand."""
    raise NotImplementedError


# The interfaces of moma's RobotHand and dm_control's RobotHand intersect.
# In particular:
# * tool_center_point
# * actuators
# * dm_control.composer.Entity as a common base class.
#
# Some code is intended to be compatible with either type, and can use this
# Gripper type to express that intent.
AnyRobotHand = Union[RobotHand, robot_base.RobotHand]

