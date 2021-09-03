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
"""Configuration for the blob detector node."""

import dataclasses
from typing import Mapping

from dmr_vision import types


@dataclasses.dataclass(init=False)
class BlobDetectorConfig:
  """Data class for the blob triangulation node.

  Attributes:
    node_name: the name of the ROS node.
    rate: the desired frame rate.
    input_queue_size: the input data queue size (see ROS documentation).
    output_queue_size: the output data queue size (see ROS documentation).
    topic_by_camera_name: a camera name to ROS topic mapping.
    mask_by_camera_name: (u, v) coordinates defining closed regions of interest
      in the image where the blob detector will not look for blobs.
    scale: image scaling factor to increase speed and frame rate.
    min_area: minimum size in pixels above which a blob is deemed valid.
  """
  node_name: str
  input_queue_size: int
  output_queue_size: int
  topic_by_camera_name: Mapping[str, str]
  mask_by_camera_name: Mapping[str, types.MaskPoints]
  scale: float
  min_area: int


def get_config() -> BlobDetectorConfig:
  """Returns the parameters for running ROS blob detector node."""
  ## Base configs
  config = BlobDetectorConfig()

  ## ROS node configuration
  config.node_name = "blob_detector"
  config.input_queue_size = 1
  config.output_queue_size = 1

  config.topic_by_camera_name = {
      "basket_front_left": "/pylon_basket_front_left/image_raw",
      "basket_front_right": "/pylon_basket_front_right/image_raw",
      "basket_back_left": "/pylon_basket_back_left/image_raw",
  }

  config.mask_by_camera_name = {
      "basket_front_left": [[
          (0, 0),
          (0, 320),
          (976, 176),
          (1920, 360),
          (1920, 0),
      ]],
      "basket_front_right": [[
          (0, 0),
          (0, 360),
          (944, 176),
          (1920, 400),
          (1920, 0),
      ]],
      "basket_back_left": [],
  }

  config.scale = 1. / 8.

  config.min_area = 1000

  return config
