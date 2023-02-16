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

_CAMERA = flags.DEFINE_string(
    name="camera",
    default=None,
    help=("The camera to use."
          "Must be one of the keys in `cameras` in the configuration file."))

_PROPS = flags.DEFINE_list(
    name="props",
    default=[
        blob_tracker_object_defs.Props.GREEN.value,
        blob_tracker_object_defs.Props.RED.value,
        blob_tracker_object_defs.Props.BLUE.value,
    ],
    help="The names of the props to track.")

_VISUALISE = flags.DEFINE_boolean(
    name="visualize",
    default=False,
    help="Whether to publish helper images of the detected blobs or not.",
)

_TOOLKIT = flags.DEFINE_boolean(
    name="toolkit",
    default=False,
    help=("Whether to display a YUV GUI toolkit to find good YUV parameters to "
          "detect blobs or not. Sets `visualize = True`."),
)


def main(_):
  logging.info("Collecting configuration parameters.")
  config = config_blob_detector.get_config()
  try:
    topic = config.topic_by_camera_name[_CAMERA.value]
  except KeyError as ke:
    raise ValueError(
        "Please provide the name of one of the cameras listed in "
        "the config `camera_namespaces` attribute. "
        f"Provided: {_CAMERA.value}. Available: "
        f"{[cam for cam in config.topic_by_camera_name]}."
    ) from ke
  color_ranges = {}
  for name in _PROPS.value:
    prop = blob_tracker_object_defs.Props(name.lower())
    color_ranges[name] = blob_tracker_object_defs.PROP_SPEC[prop]

  logging.info("Initializing blob detector ROS node.")
  rospy.init_node(name=config.node_name, anonymous=True)
  detector = blob_detector.BlobDetector(
      color_ranges=color_ranges,
      scale=config.scale,
      min_area=config.min_area,
      mask_points=config.mask_by_camera_name[_CAMERA.value],
      visualize=_VISUALISE.value,
      toolkit=_TOOLKIT.value,
  )
  ros_node = detector_node.DetectorNode(
      camera_topic=topic,
      detector=detector,
      input_queue_size=config.input_queue_size,
      output_queue_size=config.output_queue_size,
  )

  logging.info("Spinning ROS node.")
  ros_node.spin()
  logging.info("ROS node terminated.")
  ros_node.close()


if __name__ == "__main__":
  flags.mark_flag_as_required("camera")
  app.run(main)
