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

"""Util functions for dealing with poses."""

import numpy as np


def positive_leading_quat(quat: np.ndarray) -> np.ndarray:
  """Returns the quaternion with a positive leading scalar (wxyz)."""
  quat = np.copy(quat)
  if quat[0] < 0:
    quat *= -1
  return quat
