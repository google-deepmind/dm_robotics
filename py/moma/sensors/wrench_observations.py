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

"""The observation enum used by all wrench sensors."""

import enum


@enum.unique
class Observations(enum.Enum):
  """Observations exposed by a wrench sensor."""
  # The 3D force sensed by the sensor.
  FORCE = '{}_force'
  # The 3D torque sensed by the sensor.
  TORQUE = '{}_torque'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)
