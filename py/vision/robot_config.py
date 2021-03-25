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

"""Module defining robot configurations."""

import enum

from dmr_vision import types

DEFAULT_SAWYER_BASKET_CENTER = (0.6, 0.)
DEFAULT_BASKET_HEIGHT = 0.0498


# TODO(b/183080853): data from R28
@enum.unique
class RobotType(enum.Enum):
  STANDARD_SAWYER = types.RobotConfig(
      name="STANDARD_SAWYER",
      cameras={
          "basket_front_left":
              types.Camera(
                  width=1920,
                  height=1200,
                  extrinsics={
                      "pos_xyz": [0.973, -0.375, 0.299],
                      "quat_xyzw": [0.783, 0.329, -0.196, -0.489]
                  }),
          "basket_front_right":
              types.Camera(
                  width=1920,
                  height=1200,
                  extrinsics={
                      "pos_xyz": [0.978, 0.375, 0.294],
                      "quat_xyzw": [0.332, 0.774, -0.496, -0.213]
                  }),
          "basket_back_left":
              types.Camera(
                  width=1920,
                  height=1200,
                  extrinsics={
                      "pos_xyz": [0.059, -0.251, 0.441],
                      "quat_xyzw": [0.759, -0.482, 0.235, -0.370]
                  })
      },
      basket_center=DEFAULT_SAWYER_BASKET_CENTER,
      basket_height=DEFAULT_BASKET_HEIGHT,
      base_frame_name="base",
  )


def get_robot_config(robot_type: str) -> types.RobotConfig:
  """Retrieves robot configration."""
  try:
    return RobotType[robot_type.upper()].value
  except KeyError as ke:
    raise ValueError("No robot configuration available for given robot type. "
                     f"Can be one among {[robot.name for robot in RobotType]}. "
                     f"Provided robot type: {robot_type}.") from ke
