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

"""Enum to help key into a timestep's observations for the Robotiq gripper."""

import enum


@enum.unique
class Observations(enum.Enum):
  """Observations exposed by a Robotiq gripper sensor."""
  # The finger position of the gripper.
  POS = '{}_pos'
  # The finger velocity of the gripper.
  VEL = '{}_vel'
  # Whether an object is grasped by the gripper.
  GRASP = '{}_grasp'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)
