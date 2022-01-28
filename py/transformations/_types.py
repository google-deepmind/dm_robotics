# Copyright 2021 DeepMind Technologies Limited.
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

"""Type alias definitions for dm_robotics.transformations."""
from typing import Any, Union

# pylint:disable=g-import-not-at-top
try:
  # This is only available for NumPy >= 1.20.
  import numpy.typing
  ArrayLike = numpy.typing.ArrayLike
except ImportError:
  ArrayLike = Any

QuatArray = ArrayLike  # [...,4] array of quaternions (w, i, j, k)

# [...,3] arrays of Euler angles

EulerArray = {
    'XYZ': ArrayLike,
    'XYX': ArrayLike,
    'XZY': ArrayLike,
    'ZYX': ArrayLike,
    'YZX': ArrayLike,
    'ZXY': ArrayLike,
    'YXZ': ArrayLike,
    'XZX': ArrayLike,
    'YXY': ArrayLike,
    'YZY': ArrayLike,
    'ZXZ': ArrayLike,
    'ZYZ': ArrayLike,
}
SomeEulerArray = Union[EulerArray['XYZ'], EulerArray['XYX'], EulerArray['XZY'],
                       EulerArray['ZYX'], EulerArray['YZX'], EulerArray['ZXY'],
                       EulerArray['YXZ'], EulerArray['XZX'], EulerArray['YXY'],
                       EulerArray['YZY'], EulerArray['ZXZ'], EulerArray['ZYZ'],]

AxisAngleArray = ArrayLike  # [...,3] array of axis-angle rotations
PositionArray = ArrayLike  # [...,3] array of 3d position vectors
AngVelArray = ArrayLike  # [...,3] array of 3d angular velocity vectors
RotationMatrix = ArrayLike  # [3,3] rotation matrix
RotationMatrix2d = ArrayLike  # [2,2] rotation matrix
HomogeneousMatrix = ArrayLike  # [4,4] homogeneous transformation matrix
HomogeneousMatrix2d = ArrayLike  # [3,3] homogeneous matrix
Twist = ArrayLike  # [6] twist
