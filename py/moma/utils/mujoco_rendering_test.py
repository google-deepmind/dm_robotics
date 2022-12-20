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

"""Tests for mujoco_rendering."""

from unittest import mock

from absl.testing import absltest
from dm_robotics.moma.utils import mujoco_rendering
import numpy as np

_GUI_PATH = mujoco_rendering.__name__ + '.gui'

_WINDOW_PARAMS = (640, 480, 'title')


class BasicRenderingObserverTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = mock.MagicMock()
    self.env.physics = mock.MagicMock()

    with mock.patch(_GUI_PATH):
      self.observer = mujoco_rendering.Observer(self.env, *_WINDOW_PARAMS)
      self.observer._viewer = mock.MagicMock()
      self.observer._render_surface = mock.MagicMock()

  def test_deinitializing_viewer_on_episode_end(self):
    self.observer.end_episode(
        agent_id=0, termination_reason=1, agent_time_step=2)
    self.observer._viewer.deinitialize.assert_called_once()

  def test_set_camera_config_full_dict(self):
    initial_camera_cfg = {
        'distance': 1.0,
        'azimuth': 30.0,
        'elevation': -45.0,
        'lookat': [0.0, 0.1, 0.2],
    }

    self.observer.camera_config = initial_camera_cfg

    self.observer._viewer.camera.settings.lookat = np.zeros(3)

    self.observer.step(0, None, None)

    self.assertEqual(self.observer._viewer.camera.settings.distance, 1.0)
    self.assertEqual(self.observer._viewer.camera.settings.azimuth, 30.0)
    self.assertEqual(self.observer._viewer.camera.settings.elevation, -45.0)
    self.assertTrue(np.array_equal(
        self.observer._viewer.camera.settings.lookat, [0.0, 0.1, 0.2]))

  def test_set_camera_config_single_field(self):
    initial_camera_cfg = {
        'distance': 1.0,
        'azimuth': 30.0,
        'elevation': -45.0,
        'lookat': [0.0, 0.1, 0.2],
    }

    self.observer.camera_config = initial_camera_cfg

    self.observer._viewer.camera.settings.lookat = np.zeros(3)

    self.observer.step(0, None, None)

    self.assertEqual(self.observer._viewer.camera.settings.distance, 1.0)

    self.observer.camera_config = {
        'distance': 3.0,
    }

    self.observer.step(0, None, None)

    self.assertEqual(self.observer._viewer.camera.settings.distance, 3.0)

  def test_set_camera_config_fails(self):
    with self.assertRaises(ValueError):
      self.observer.camera_config = {'not_a_valid_key': 0}


if __name__ == '__main__':
  absltest.main()
