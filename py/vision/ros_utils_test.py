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
"""Tests for `ros_utils.py`."""

from unittest import mock

from absl.testing import absltest
from dmr_vision import ros_utils
import rospy


class RosUtilsTest(absltest.TestCase):
  def test_image_handler_times_out_waiting_for_initial_message(self):
    mock_wait_for_message = self.enter_context(
        mock.patch.object(rospy, "wait_for_message", autospec=True)
    )
    mock_wait_for_message.side_effect = rospy.exceptions.ROSException()
    with self.assertRaises(TimeoutError):
      ros_utils.ImageHandler(topic="/test/foo")

  def test_point_handler_times_out_waiting_for_initial_message(self):
    mock_wait_for_message = self.enter_context(
        mock.patch.object(rospy, "wait_for_message", autospec=True)
    )
    mock_wait_for_message.side_effect = rospy.exceptions.ROSException()
    with self.assertRaises(TimeoutError):
      ros_utils.PointHandler(topic="/test/foo")


if __name__ == "__main__":
  absltest.main()
