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
"""Utilities for working in safety with robot setup."""

from typing import Mapping, Optional

from dmr_vision import types
import numpy as np


class PoseValidator:
  """Helper class for validating poses inside pre-defined limits and deadzones."""

  def __init__(self,
               limits: types.PositionLimit,
               deadzones: Optional[Mapping[str, types.PositionLimit]] = None):
    """Constructs a `PoseValidator` instance.

    Args:
      limits: A range of Cartesian position in terms of lower and upper bounds.
      deadzones: A mapping specifying deadzones with their limits, specified in
        the same terms of `limits`.
    """
    if len(limits.lower) != 3 or len(limits.upper) != 3:
      raise ValueError("Upper/lower limits need to be of length 3 (cartesian)")

    self._limits = limits
    self._deadzones = deadzones

  def is_valid(self, pose: np.ndarray) -> bool:
    """Checks if a pose is valid by checking it against limits and deadzones."""
    position = pose[0:3]
    if not self._within_zone(position, self._limits):
      return False
    if self._deadzones is not None:
      for zone in self._deadzones.values():
        if self._within_zone(position, zone):
          return False
    return True

  def _within_zone(self, position: np.ndarray,
                   limits: types.PositionLimit) -> bool:
    """Checks if position is within a zone defined by limits."""
    if ((position < limits.lower).any() or (position > limits.upper).any()):
      return False
    return True
