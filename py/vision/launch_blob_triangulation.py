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

import collections

from absl import app
from absl import flags
from absl import logging
from dmr_vision import blob_tracker_object_defs
from dmr_vision import blob_triangulation_node
from dmr_vision import config_blob_triangulation
import numpy as np
import rospy

FLAGS = flags.FLAGS

flags.DEFINE_list(
    name="props",
    default=[
        blob_tracker_object_defs.Props.RGB30_GREEN.value,
        blob_tracker_object_defs.Props.RGB30_RED.value,
        blob_tracker_object_defs.Props.RGB30_BLUE.value,
    ],
    help="The names of the objects to track.")


def main(_):
  logging.info("Collecting configuration parameters.")
  config = config_blob_triangulation.get_config("STANDARD_SAWYER")
  extrinsics = collections.defaultdict(dict)
  for cam_topic, cam_extrinsics in config.extrinsics.items():
    extrinsics[cam_topic] = np.append(cam_extrinsics["pos_xyz"],
                                      cam_extrinsics["quat_xyzw"])

  logging.info("Initializing blob triangulation ROS node.")
  rospy.init_node(name=config.node_name, anonymous=True)
  ros_node = blob_triangulation_node.BlobTriangulationNode(
      prop_names=FLAGS.props,
      extrinsics=extrinsics,
      limits=config.limits,
      deadzones=config.deadzones,
      base_frame=config.base_frame,
      rate=config.rate,
      input_queue_size=config.input_queue_size,
      output_queue_size=config.output_queue_size)

  logging.info("Spinning ROS node.")
  ros_node.spin()
  logging.info("ROS node terminated.")


if __name__ == "__main__":
  app.run(main)
