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
"""Configuration for the blob triangulation node."""

import dataclasses
from typing import Mapping, Optional

from dmr_vision import robot_config
from dmr_vision import types
import numpy as np

ROS_TOPIC_CAM_PREFIX = "/pylon_"


@dataclasses.dataclass(init=False)
class BlobTriangulationConfig:
  """Data class for the blob triangulation node.

  Attributes:
    node_name: the name of the ROS node.
    rate: the desired frame rate.
    input_queue_size: the input data queue size (see ROS documentation).
    output_queue_size: the output data queue size (see ROS documentation).
    fuse_tolerance: time in seconds after which data is considered outdated and
      not used for triangulation.
    extrinsics: extrinsic camera parameters.
    limits: Cartesian limits over which points are considered as outliers.
    base_frame: the name of the robot base frame.
    deadzones: additional Cartesian limits excluding volumes of the robot
      operative space where points are discarded.
  """
  node_name: str
  rate: int
  input_queue_size: int
  output_queue_size: int
  fuse_tolerance: float
  extrinsics: Mapping[str, types.Extrinsics]
  limits: types.PositionLimit
  base_frame: str
  deadzones: Optional[Mapping[str, types.PositionLimit]] = None


def get_config(robot_type: str) -> BlobTriangulationConfig:
  """Returns the parameters for running ROS blob triangulation node.

  Args:
    robot_type: the name of a robot among the ones listed in `robot_config`.

  Returns:
    The configuration parameters for the blob triangulation ROS node.
  """
  ## Base configs
  r_config = robot_config.get_robot_config(robot_type)
  config = BlobTriangulationConfig()

  ## ROS node configuration
  config.node_name = "blob_triangulation"
  config.rate = 60
  config.input_queue_size = 3
  config.output_queue_size = 1
  config.fuse_tolerance = 0.2

  ## Robot configuration
  config.extrinsics = {
      ROS_TOPIC_CAM_PREFIX + cam_name: cam.extrinsics
      for cam_name, cam in r_config.cameras.items()
  }

  center = np.append(r_config.basket_center, r_config.basket_height)
  config.limits = types.PositionLimit(
      upper=center + np.array([0.45, 0.45, 0.20]),
      lower=center + np.array([-0.45, -0.45, -0.02]),
  )

  config.deadzones = {
      "front_left":
          types.PositionLimit(
              upper=center + np.array([0.45, 0.27, 0.20]),
              lower=center + np.array([0.27, -0.45, -0.02]),
          ),
      "front_right":
          types.PositionLimit(
              upper=center + np.array([0.45, 0.45, 0.20]),
              lower=center + np.array([0.27, 0.27, -0.02]),
          ),
  }

  config.base_frame = r_config.base_frame_name

  return config
