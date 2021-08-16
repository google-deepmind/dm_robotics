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

"""Abstract effector interface definition."""

import abc
from dm_control import mjcf
from dm_env import specs
import numpy as np


class Effector(abc.ABC):
  """Abstract effector interface, a controllable element of the environment.

  An effector provides an interface for an agent to interact in some way with
  the environment. eg: a robot arm, a gripper, a pan tilt unit, etc. The
  effector is defined by its action spec, a control method, and a prefix that
  marks the control components of the effector in the wider task action spec.
  """

  def close(self):
    """Clean up after we are done using the effector.

    Called to clean up when we are done using the effector. This is
    mainly used for real effectors that might need to close all the connections
    to the robot.
    """
    pass

  @abc.abstractmethod
  def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
    """Method called after the MJCF model has been compiled and finalized.

    Args:
      mjcf_model: The root element of the scene MJCF model.
    """
    pass

  @abc.abstractmethod
  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass

  @abc.abstractmethod
  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    pass

  @abc.abstractmethod
  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    pass

  @property
  @abc.abstractmethod
  def prefix(self) -> str:
    pass
