# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Physics implementation that gets Poses from observation values."""

from typing import Any, Callable, Dict, Mapping, Text
from dm_robotics.geometry import geometry
import numpy as np

_IDENTITY_QUATERNION = np.array([1, 0, 0, 0])


class ObservationPhysics(geometry.Physics):
  """A `geometry.Physics` backed by environment observations."""

  def __init__(self,
               observation_to_pose: Callable[[np.ndarray], geometry.Pose]):
    """Initializes ObservationPhysics.

    Args:
      observation_to_pose: A function to convert an observation to a Pose.
    """
    super().__init__()
    self._obs = None  # type: Dict[Text, Any]
    self._parser = observation_to_pose

  def set_observation(self, observation: Mapping[str, np.ndarray]):
    """Sets the dict as the current observation."""
    self._obs = observation  # pytype: disable=annotation-type-mismatch

  def world_pose(self,
                 frame: geometry.Grounding,
                 get_pos: bool = True,
                 get_rot: bool = True) -> geometry.Pose:
    """Return world pose of the provided frame.

    Args:
      frame: A frame identifier.
      get_pos: If False, zero out position entries.
      get_rot: If False, make the rotation an identity quaternion.

    Returns:
      A `geometry.Pose` containing the requested pose.
    """
    if not isinstance(frame, str):
      raise ValueError(f"bad frame: {frame}, expected identifier")
    raw_value = self._obs.get(frame, None)
    if raw_value is None:
      raise ValueError(f"{frame} not found in {self._obs.keys()}")
    full_pose = self._parser(raw_value)
    if not get_pos:
      return full_pose.with_position([0, 0, 0])
    elif not get_rot:
      return full_pose.with_quaternion(_IDENTITY_QUATERNION)
    return full_pose
