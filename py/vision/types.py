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

"""Common types used in DeepMind Robotics vision package.

This file is maintained for simplifying data creation and manipulation as well
as facilitating type hints.
"""

from typing import Mapping, Optional, Sequence, Tuple

import dataclasses
import numpy as np


MaskPoints = Sequence[Sequence[Tuple[int, int]]]
Centers = Mapping[str, Optional[np.ndarray]]
Detections = Mapping[str, Optional[np.ndarray]]

Extrinsics = Mapping[str, Sequence[float]]


@dataclasses.dataclass(frozen=True)
class Camera:
  """Camera parameters.

  Attributes:
    width: image width.
    height: image height.
    extrinsics: camera extrinsics expressed as [x, y, z] `position` and
      [x, y, z, w] `orientation` quaternion.
  """
  width: int
  height: int
  extrinsics: Extrinsics


@dataclasses.dataclass(frozen=True)
class Blob:
  """An image blob.

  Attributes:
    center: (u, v) coordintes of the blob barycenter.
    contour: Matrix of (u, v) coordinates of the blob contour.
  """
  center: np.ndarray
  contour: np.ndarray


@dataclasses.dataclass(frozen=True)
class ValueRange:
  """A generic N-dimensional range of values in terms of lower and upper bounds.

  Attributes:
    lower: A ND array with the lower values of the range.
    upper: A ND array with the upper values of the range.
  """
  lower: np.ndarray
  upper: np.ndarray


@dataclasses.dataclass(frozen=True)
class ColorRange(ValueRange):
  """A range of colors in terms of lower and upper bounds.

  Typical usage example:
    # A YUV color range (cuboid)
    ColorRange(lower=np.array[0., 0.25, 0.25],
               upper=np.array[1., 0.75, 0.75])

  Attributes:
    lower: A 3D array with the lower values of the color range.
    upper: A 3D array with the upper values of the color range.
  """


@dataclasses.dataclass(frozen=True)
class PositionLimit(ValueRange):
  """A range of Cartesian position in terms of lower and upper bounds.

  Typical usage example:
    # Define a position limit in Cartesian space (cuboid)
    Limits(lower=np.array[-0.5, -0.5, 0.],
           upper=np.array[0.5, 0.5, 0.5])

  Attributes:
    lower: An [x, y, z] array with the lower values of the position limit.
    upper: An [x, y, z] array with the upper values of the position limit.
  """


@dataclasses.dataclass(frozen=True)
class RobotConfig:
  """Robot parameters.

  Attributes:
    name: unique robot name.
    cameras: collection of cameras.
    basket_center: center of playground relative to the robot base frame.
    basket_height: displacement of the playground from the robot base frame.
    base_frame_name: the name (or id) that identifies the robot ROS base frame.
  """
  name: str
  cameras: Mapping[str, Camera]
  basket_center: Optional[Tuple[float, float]] = None
  basket_height: Optional[float] = None
  base_frame_name: Optional[str] = None
