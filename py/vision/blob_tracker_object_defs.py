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

"""Common definitions of blob detector based objects."""

import enum

from dmr_vision import types
import numpy as np


@enum.unique
class Props(enum.Enum):
  RGB30_RED = "rgb30_red"
  RGB30_GREEN = "rgb30_green"
  RGB30_BLUE = "rgb30_blue"


PROP_SPEC = {
    Props.RGB30_RED:
        types.ColorRange(
            lower=np.array([0., 0., 0.669]), upper=np.array([1., 0.518, 1.])),
    Props.RGB30_GREEN:
        types.ColorRange(
            lower=np.array([0., 0., 0.]), upper=np.array([1., 0.427, 0.479])),
    Props.RGB30_BLUE:
        types.ColorRange(
            lower=np.array([0., 0.568, 0.]), upper=np.array([1., 1., 0.590])),
}

ROS_PROPS = {
    Props.RGB30_RED: "/blob/rgb30_red/pose",
    Props.RGB30_GREEN: "/blob/rgb30_green/pose",
    Props.RGB30_BLUE: "/blob/rgb30_blue/pose",
}
