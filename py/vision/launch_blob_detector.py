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

"""Launch a ROS node to detect colored blobs from images."""

from absl import app
from absl import flags
from absl import logging
from dmr_vision import blob_detector
from dmr_vision import blob_tracker_object_defs
from dmr_vision import config_blob_detector
from dmr_vision import detector_node
import rospy

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name="camera",
    default=None,
    help=("The camera to use."
          "Must be one of the keys in `cameras` in the configuration file."))

flags.DEFINE_list(
    name="props",
    default=[
        blob_tracker_object_defs.Props.RGB30_GREEN.value,
        blob_tracker_object_defs.Props.RGB30_RED.value,
        blob_tracker_object_defs.Props.RGB30_BLUE.value,
    ],
    help="The names of the props to track.")

flags.DEFINE_boolean(
    name="visualize",
    default=False,
    help="Publish the visualizations provided by the blob detector.")

flags.DEFINE_boolean(
    name="toolkit",
    default=False,
    help=("Displays a YUV GUI toolkit to fine tune parameters. "
          "Sets `visualize = True`."))


def main(_):
  logging.info("Collecting configuration parameters.")
  config = config_blob_detector.get_config()
  try:
    namespace = config.camera_namespaces[FLAGS.camera]
  except KeyError as ke:
    raise ValueError("Please provide the name of one of the cameras listed in "
                     "the config `camera_namespaces` attribute. "
                     f"Provided: {FLAGS.camera}. "
                     f"Available: {[cam for cam in config.camera_namespaces]}.")
  color_ranges = {}
  for name in FLAGS.props:
    prop = blob_tracker_object_defs.Props(name.lower())
    color_ranges[name] = blob_tracker_object_defs.PROP_SPEC[prop]

  logging.info("Initializing blob detector ROS node.")
  rospy.init_node(name=config.node_name, anonymous=True)
  detector = blob_detector.BlobDetector(
      color_ranges=color_ranges,
      scale=config.scale,
      min_area=config.min_area,
      mask_points=config.masks[FLAGS.camera],
      visualize=FLAGS.visualize,
      toolkit=FLAGS.toolkit)
  ros_node = detector_node.DetectorNode(
      namespace=namespace,
      detector=detector,
      input_queue_size=config.input_queue_size,
      output_queue_size=config.output_queue_size)

  logging.info("Spinning ROS node.")
  ros_node.spin()
  logging.info("ROS node terminated.")


if __name__ == "__main__":
  flags.mark_flag_as_required("camera")
  app.run(main)
