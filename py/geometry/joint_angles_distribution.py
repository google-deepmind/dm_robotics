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
"""A module for sampling joint angles distributions."""

import abc
from dm_robotics.geometry import geometry
import numpy as np
import six


class JointAnglesDistribution(six.with_metaclass(abc.ABCMeta, object)):
  """An interface for joint angles distributions."""

  @abc.abstractmethod
  def sample_angles(self, random_state,
                    physics: geometry.Physics) -> np.ndarray:
    """Returns angles sampled from some distribution."""
    pass


class ConstantPanTiltDistribution(JointAnglesDistribution):
  """A distribution with only a single angle with probability 1."""

  def __init__(self, joint_angles):
    super().__init__()
    self._joint_angles = joint_angles

  def sample_angles(self, random_state,
                    physics: geometry.Physics) -> np.ndarray:
    return np.array(self._joint_angles, dtype=np.float32)


class NormalOffsetJointAnglesDistribution(JointAnglesDistribution):
  """Distribution for angles distributed normally around a mean."""

  def __init__(self, mean_angles, angles_sd, clip_sd=3.0):
    super().__init__()
    self._mean_angles = mean_angles
    self._angles_sd = angles_sd
    self._clip_sd = clip_sd

  def sample_angles(self, random_state,
                    physics: geometry.Physics) -> np.ndarray:
    offset = random_state.normal(scale=self._angles_sd)
    clip_range = self._angles_sd * self._clip_sd
    return self._mean_angles + np.clip(offset, -clip_range, clip_range)


class UniformJointAnglesDistribution(JointAnglesDistribution):
  """Uniform random distribution for joint angles."""

  def __init__(self, min_angles, max_angles):
    super().__init__()
    self._min_angles = min_angles
    self._max_angles = max_angles

  def sample_angles(self, random_state,
                    physics: geometry.Physics) -> np.ndarray:
    joint_angles = random_state.uniform(
        low=self._min_angles, high=self._max_angles)
    return np.array(joint_angles, dtype=np.float32)
