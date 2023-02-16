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
"""Launch a ROS node to traingulate the barycenter of colored blobs from images."""

import math

from absl import app
from absl import flags
from absl import logging
from dmr_vision import blob_tracker_object_defs
from dmr_vision import blob_triangulation_node
from dmr_vision import config_blob_triangulation
from dmr_vision import detector_node
from dmr_vision import ros_utils
from dmr_vision import types
import numpy as np
import rospy

_ROBOT = flags.DEFINE_string(
    name="robot",
    default="STANDARD_SAWYER",
    help=(
        "The name of the robot."
        "Must be one of the enums in the robot configuration file."
    ),
)

_PROPS = flags.DEFINE_list(
    name="props",
    default=[
        blob_tracker_object_defs.Props.GREEN.value,
        blob_tracker_object_defs.Props.RED.value,
        blob_tracker_object_defs.Props.BLUE.value,
    ],
    help="The names of the objects to track.",
)

_MAX_INVALID_CAMINFO = flags.DEFINE_float(
    name="max_invalid_caminfo",
    default=math.inf,
    help=(
        "Number of tentatives after receiving an invalid camera info message "
        "before raising an exception. Setting this to `1` means to never "
        "wait for a valid message after receiving a wrong one, while setting "
        "it to `math.inf` means to never raise an exception and keep on "
        "trying until a healthy message is received."
    ),
)


def main(_):
  logging.info("Collecting configuration parameters.")
  config = config_blob_triangulation.get_config(_ROBOT.value)
  point_topics_by_camera_prop_name = {}
  intrinsics = {}
  for cam_name in config.extrinsics:
    cam_namespace = f"pylon_{cam_name}"
    point_topics_by_camera_prop_name[cam_name] = {
        prop: detector_node.POINT_PUB_TOPIC_FSTR.format(
            namespace=cam_namespace, blob_name=prop
        )
        for prop in _PROPS.value
    }
    camera_info = ros_utils.CameraInfoHandler(
        topic=f"/{cam_namespace}/camera_info",
        queue_size=config.input_queue_size,
    )
    invalid_caminfo_counter = 0
    while (
        intrinsics.get(cam_name) is None
        and invalid_caminfo_counter < _MAX_INVALID_CAMINFO.value
    ):
      with camera_info:
        if np.count_nonzero(camera_info.camera_matrix) == 0:
          invalid_caminfo_counter += 1
          logging.log_every_n_seconds(
              logging.INFO,
              (
                  "Received all-zero camera matrix from topic /%s. Tentative"
                  " number %d. Discarding the message and not updating camera"
                  " matrix and distortion parameters. If the problem persists,"
                  " consider restarting the camera driver, checking the"
                  " camera's calibration file, or the provided intrinsics."
              ),
              12,
              cam_namespace,
              invalid_caminfo_counter,
          )
          camera_info.wait()
          continue
        else:
          intrinsics[cam_name] = types.Intrinsics(
              camera_matrix=camera_info.camera_matrix,
              distortion_parameters=camera_info.distortion_parameters,
          )
          camera_info.close()
    if invalid_caminfo_counter >= _MAX_INVALID_CAMINFO.value:
      camera_info.close()
      raise ValueError(
          "Received an all-zero camera matrix for more than "
          f"{_MAX_INVALID_CAMINFO.value} time(s). Please restart the camera "
          "driver, check the camera's calibration file, or the provided "
          "intrinsics if the issue persists."
      )

  logging.info("Initializing blob triangulation ROS node.")
  rospy.init_node(name=config.node_name, anonymous=True)
  ros_node = blob_triangulation_node.BlobTriangulationNode(
      prop_names=_PROPS.value,
      point_topics_by_camera_prop_name=point_topics_by_camera_prop_name,
      extrinsics=config.extrinsics,
      intrinsics=intrinsics,
      limits=config.limits,
      deadzones=config.deadzones,
      base_frame=config.base_frame,
      rate=config.rate,
      input_queue_size=config.input_queue_size,
      output_queue_size=config.output_queue_size,
  )

  logging.info("Spinning ROS node.")
  ros_node.spin()
  logging.info("ROS node terminated.")


if __name__ == "__main__":
  app.run(main)
