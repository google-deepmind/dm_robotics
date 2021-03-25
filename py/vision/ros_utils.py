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

"""Module defining utility classes for ROS."""

import threading
from typing import Optional

from absl import logging
import dataclasses
import numpy as np
import rospy

from geometry_msgs import msg as geometry_msgs
from sensor_msgs import msg as geometry_msgs
import cv_bridge
from std_msgs import msg as std_msgs


@dataclasses.dataclass(frozen=True)
class PointData:
  """ROS data representing a point.

  Attributes:
    data: a Cartesian [x, y, z] point.
    frame_id: the frame id associated to the data.
    stamp: the time stamp associated to the data.
  """
  data: np.ndarray
  frame_id: str
  stamp: rospy.Time


class CameraInfoHandler:
  """Handler for receiving camera info."""

  def __init__(self, topic: str, queue_size: int = 1):
    """Constructs a `CameraInfoHandler` instance.

    Args:
      topic: The ROS topic to subscribe to.
      queue_size: The ROS subscriber queue size.
    """
    self._lock = threading.RLock()
    self._camera_matrix = None
    self._distortions = None
    self._subscriber = rospy.Subscriber(
        name=topic,
        data_class=sensor_msgs.CameraInfo,
        callback=self.__call__,
        queue_size=queue_size,
        tcp_nodelay=True)
    logging.info('Waiting for message on topic %s', topic)
    rospy.wait_for_message(topic, sensor_msgs.CameraInfo)

  @property
  def matrix(self) -> np.ndarray:
    """Returns the most current camera matrix."""
    with self._lock:
      return self._camera_matrix

  @property
  def distortions(self) -> np.ndarray:
    """Returns the most current camera distortions."""
    with self._lock:
      return self._distortions

  # TODO(b/183081482): is this required?
  def __call__(self, camera_info_msg: sensor_msgs.CameraInfo) -> None:
    camera_matrix = np.array(camera_info_msg.K).reshape((3, 3))
    distortions = np.array(camera_info_msg.D)
    with self._lock:
      self._camera_matrix = camera_matrix
      self._distortions = distortions

  def __enter__(self) -> bool:
    return self._lock.__enter__()

  def __exit__(self, *args, **kwargs) -> Optional[bool]:
    return self._lock.__exit__(*args, **kwargs)


class ImageHandler:
  """Handler for receiving and decoding images."""

  def __init__(self, topic: str, encoding: str = 'rgb8', queue_size: int = 1):
    """Constructs a `ImageHandler` instance.

    Args:
      topic: The topic to subscribe to.
      encoding: The desired encoding of the image.
      queue_size: The queue size to use.
    """
    self._encoding = encoding
    self._bridge = cv_bridge.CvBridge()
    self._lock = threading.Condition(threading.RLock())
    self._data = None
    self._frame_id = None
    self._stamp = None
    self._subscriber = rospy.Subscriber(
        name=topic,
        data_class=sensor_msgs.Image,
        callback=self.__call__,
        queue_size=queue_size,
        tcp_nodelay=True)
    rospy.wait_for_message(topic, sensor_msgs.Image)

  def wait(self) -> None:
    """Waits for the next image to be available."""
    with self._lock:
      self._lock.wait()

  @property
  def data(self) -> np.ndarray:
    """Returns the most current image data."""
    with self._lock:
      return self._data

  @property
  def frame_id(self) -> str:
    """Returns the most current frame id."""
    with self._lock:
      return self._frame_id

  @property
  def stamp(self) -> rospy.Time:
    """Returns the most current timestamp."""
    with self._lock:
      return self._stamp

  def __call__(self, image_msg: sensor_msgs.Image):
    data = self._bridge.imgmsg_to_cv2(image_msg, self._encoding)
    with self._lock:
      self._data = data
      self._frame_id = image_msg.header.frame_id
      self._stamp = image_msg.header.stamp
      self._lock.notify()

  def __enter__(self) -> bool:
    return self._lock.__enter__()

  def __exit__(self, *args, **kwargs) -> Optional[bool]:
    return self._lock.__exit__(*args, **kwargs)


class ImagePublisher:
  """Publisher for OpenCV images."""

  def __init__(self,
               topic: str,
               encoding: str = 'rgb8',
               frame_id: Optional[str] = None,
               queue_size: int = 1):
    """Constructs an `ImagePublisher` instance.

    Args:
      topic: The topic to publish to.
      encoding: The desired encoding.
      frame_id: The associated frame id.
      queue_size: The queue size to use.
    """
    self._encoding = encoding
    self._frame_id = frame_id
    self._bridge = cv_bridge.CvBridge()
    self._publisher = rospy.Publisher(
        name=topic,
        data_class=sensor_msgs.Image,
        queue_size=queue_size,
        tcp_nodelay=True)

  def publish(self,
              image: np.ndarray,
              stamp: Optional[rospy.Time] = None) -> None:
    """Publishes the image.

    Args:
      image: The image to publish.
      stamp: A ROS timestamp.
    """
    message = self._bridge.cv2_to_imgmsg(image, encoding=self._encoding)
    message.header.frame_id = self._frame_id
    message.header.stamp = stamp
    self._publisher.publish(message)


class PointHandler:
  """Handler for receiving point data."""

  def __init__(self, topic: str, queue_size: int = 1):
    """Constructs a `PointHandler` instance.

    Args:
      topic: The ROS topic to subscribe to.
      queue_size: The ROS subscriber queue size.
    """
    self._lock = threading.Lock()
    self._point_data: Optional[PointData] = None
    self._subscriber = rospy.Subscriber(
        name=topic,
        data_class=geometry_msgs.PointStamped,
        callback=self.__call__,
        queue_size=queue_size,
        tcp_nodelay=True)
    logging.info('Waiting for message on topic %s', topic)
    try:
      rospy.wait_for_message(topic, geometry_msgs.PointStamped, timeout=10.)
    except rospy.exceptions.ROSException:
      logging.warning(
          'Did not reveive a message on topic %s, object may be '
          'occluded or colors may be poorly calibrated.', topic)

  @property
  def point_data(self) -> Optional[PointData]:
    """Returns the most current point."""
    with self._lock:
      return self._point_data

  def __call__(self, point_msg: geometry_msgs.PointStamped) -> None:
    """Callback used by ROS subscriber."""
    data = np.array([point_msg.point.x, point_msg.point.y, point_msg.point.z])
    with self._lock:
      self._point_data = PointData(
          data=data,
          frame_id=point_msg.header.frame_id,
          stamp=point_msg.header.stamp)


class PointMessage(geometry_msgs.PointStamped):
  """Simplifies constructions of `PointStamped` messages."""

  def __init__(self,
               point: np.ndarray,
               frame_id: Optional[str] = None,
               stamp: Optional[rospy.Time] = None):
    """Constructs a `PointMessage` instance.

    Args:
      point: The point.
      frame_id: The associated frame id.
      stamp: A timestamp.
    """
    super().__init__()
    self.header = std_msgs.Header()
    self.header.frame_id = frame_id
    self.header.stamp = stamp
    if len(point) == 2:
      (self.point.x, self.point.y) = point
    else:
      (self.point.x, self.point.y, self.point.z) = point


class PointPublisher:
  """Publisher for 2D / 3D points."""

  def __init__(self,
               topic: str,
               frame_id: Optional[str] = None,
               queue_size: int = 1):
    """Constructs a `PointPublisher` instance.

    Args:
      topic: The topic to publish to.
      frame_id: The associated frame id.
      queue_size: The queue size to use.
    """
    self._frame_id = frame_id
    self._publisher = rospy.Publisher(
        name=topic,
        data_class=geometry_msgs.PointStamped,
        queue_size=queue_size,
        tcp_nodelay=True)

  def publish(self,
              point: np.ndarray,
              stamp: Optional[rospy.Time] = None) -> None:
    """Publishes the point.

    Args:
      point: The point.
      stamp: A ROS timestamp.
    """
    message = PointMessage(point, frame_id=self._frame_id, stamp=stamp)
    self._publisher.publish(message)


class PoseMessage(geometry_msgs.PoseStamped):
  """Simplifies constructions of `PoseStamped` messages."""

  def __init__(self,
               pose: np.ndarray,
               frame_id: Optional[str] = None,
               stamp: Optional[rospy.Time] = None):
    """Constructs a `PoseMessage` instance.

    Args:
      pose: The pose.
      frame_id: The associated frame id.
      stamp: A ROS timestamp.
    """
    super().__init__()
    self.header = std_msgs.Header()
    self.header.frame_id = frame_id
    self.header.stamp = stamp
    (self.pose.position.x, self.pose.position.y, self.pose.position.z,
     self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z,
     self.pose.orientation.w) = pose


class PosePublisher:
  """Publisher for object poses."""

  def __init__(self,
               topic: str,
               frame_id: Optional[str] = None,
               queue_size: int = 1):
    """Constructs a `PosePublisher` instance.

    Args:
      topic: The topic to publish to.
      frame_id: The associated frame id.
      queue_size: The queue size to use.
    """
    self._publisher = rospy.Publisher(
        name=topic,
        data_class=geometry_msgs.PoseStamped,
        queue_size=queue_size,
        tcp_nodelay=True)
    self._frame_id = frame_id

  def publish(self,
              pose: np.ndarray,
              stamp: Optional[rospy.Time] = None) -> None:
    """Publishes the pose.

    Args:
      pose: The pose.  #TODO(b/183081484): dimensions.
      stamp: A ROS timestamp.
    """
    message = PoseMessage(pose, frame_id=self._frame_id, stamp=stamp)
    self._publisher.publish(message)
