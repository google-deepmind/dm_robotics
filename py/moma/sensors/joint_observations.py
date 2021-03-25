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

"""Enum to help key into a timestep's observations dictionary for joint obs."""

import enum


@enum.unique
class Observations(enum.Enum):
  """Joint state observations exposed by a MoMa sensor."""
  # The joint angles, in radians.
  JOINT_POS = '{}_joint_pos'
  # The joint velocities, in rad/s.
  JOINT_VEL = '{}_joint_vel'
  # The joint torques of the arm.
  JOINT_TORQUES = '{}_joint_torques'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)
