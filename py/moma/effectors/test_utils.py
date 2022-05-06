# Copyright 2021 DeepMind Technologies Limited.
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

"""Test utilities for MoMa effectors."""

from dm_control import mjcf
from dm_env import specs
from dm_robotics.geometry import geometry
from dm_robotics.moma import effector
import numpy as np

_MjcfElement = mjcf.element._ElementImpl  # pylint: disable=protected-access


# Joint positions that put the robot TCP pointing down somewhere in front of
# the basket.
SAFE_SAWYER_JOINTS_POS = np.array(
    [-0.0835859, -0.82730523, -0.24968541, 1.75960196, 0.27188317, 0.67231963,
     1.26143456])


class SpyEffector(effector.Effector):
  """An effector that allows retrieving the most recent action."""

  def __init__(self, dofs: int):
    self._previous_action = np.zeros(dofs)

  def initialize_episode(self, physics, random_state) -> None:
    pass

  def action_spec(self, physics) -> specs.BoundedArray:
    return specs.BoundedArray(
        self._previous_action.shape, self._previous_action.dtype,
        minimum=-1.0, maximum=1.0)

  def set_control(self, physics, command: np.ndarray) -> None:
    self._previous_action = command[:]

  @property
  def prefix(self) -> str:
    return 'spy'

  @property
  def previous_action(self) -> np.ndarray:
    return self._previous_action


class SpyEffectorWithControlFrame(SpyEffector):
  """A spy effector with a control frame property."""

  def __init__(self, element: _MjcfElement, dofs: int):
    self._control_frame = geometry.HybridPoseStamped(
        pose=None,
        frame=element,
        quaternion_override=geometry.PoseStamped(None, None))
    super().__init__(dofs)

  @property
  def control_frame(self) -> geometry.Frame:
    return self._control_frame
