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
"""Module defining a ROS node to detect blobs on camera images."""

from typing import Callable, Mapping, Optional

from absl import logging
from dmr_vision import detector as vision_detector
from dmr_vision import ros_utils
from dmr_vision import types
import numpy as np
import rospy


class DetectorNode:
  """A ROS node for generic image-based detection."""

  def __init__(
      self,
      topic: str,
      detector: vision_detector.Signature,
      input_queue_size: int = 1,
      output_queue_size: int = 1,
      image_optimizer: Optional[Callable[[], bool]] = None,
  ):
    """Constructs a `DetectorNode` instance.

    Args:
      topic: the camera ROS topic.
      detector: the detector to use.
      input_queue_size: the size of input queues.
      output_queue_size: the size of output queues.
      image_optimizer: a function that can be used to trigger specific options
        on the camera. For example, this function may call camera APIs to adjust
        the brightness, gamma values, etc.

    Raises:
      EnvironmentError: if `image_optimizer` fails and return `False`.
    """
    self._topic = topic
    self._namespace = "/" + topic.split("/")[1]
    self._detector = detector
    self._input_queue_size = input_queue_size
    self._output_queue_size = output_queue_size
    self._point_publishers = {}
    self._visualization_publishers = {}

    if image_optimizer and not image_optimizer():
      raise EnvironmentError("Provided `image_optimizer` failed execution.")

    self._image_handler = ros_utils.ImageHandler(
        topic=self._topic,
        queue_size=input_queue_size,
    )

  def spin(self) -> None:
    """Loops the node until shutdown."""
    stamp = None
    while not rospy.is_shutdown():
      # Get the most recent data.
      with self._image_handler:
        while self._image_handler.stamp == stamp:
          self._image_handler.wait()
        image = self._image_handler.data
        frame_id = self._image_handler.frame_id
        stamp = self._image_handler.stamp
      # Run the blob detector.
      centers, detections = self._detector(image)
      # Publish detection results.
      self._publish_centers(centers, frame_id, stamp)
      self._publish_detections(detections, frame_id, stamp)

  def close(self) -> None:
    """Gently cleans up DetectorNode and closes ROS topics."""
    logging.info("Closing ROS nodes.")
    self._image_handler.close()
    for point_publisher in self._point_publishers.values():
      point_publisher.close()
    for visualization_publisher in self._visualization_publishers.values():
      visualization_publisher.close()

  def _publish_centers(self, centers: Mapping[str, Optional[np.ndarray]],
                       frame_id: str, stamp: rospy.Time) -> None:
    for blob_name, center in centers.items():
      if center is not None:
        if blob_name not in self._point_publishers:
          self._setup_point_publisher(blob_name, "center", frame_id)
        self._point_publishers[blob_name].publish(center, stamp=stamp)

  def _publish_detections(self, detections: types.Detections, frame_id: str,
                          stamp: rospy.Time) -> None:
    for blob_name, visualization in detections.items():
      if visualization is not None:
        if blob_name not in self._visualization_publishers:
          self._setup_image_publisher(blob_name, "visualization", frame_id)
        publisher = self._visualization_publishers[blob_name]
        publisher.publish(visualization, stamp=stamp)

  def _setup_point_publisher(self, blob_name: str, topic: str,
                             frame_id: str) -> None:
    topic = f"{self._namespace}/blob/{blob_name}/{topic}"
    self._point_publishers[blob_name] = ros_utils.PointPublisher(
        topic=topic, frame_id=frame_id, queue_size=self._output_queue_size)

  def _setup_image_publisher(self, blob_name: str, topic: str,
                             frame_id: str) -> None:
    topic = f"{self._namespace}/blob/{blob_name}/{topic}"
    self._visualization_publishers[blob_name] = ros_utils.ImagePublisher(
        topic=topic, frame_id=frame_id, queue_size=self._output_queue_size)
