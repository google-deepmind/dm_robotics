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

"""Stripped down Mujoco environment renderer implemented as an Observer."""

from dm_control import _render
from dm_control.viewer import gui
from dm_control.viewer import renderer
from dm_control.viewer import viewer

_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_MAX_FRONTBUFFER_SIZE = 2048
_DEFAULT_WINDOW_TITLE = 'Rendering Observer'


class Observer:
  """A stripped down 3D renderer for Mujoco environments.

  Attributes:
    camera_config: A dict.  The camera configuration for the observer.
  """

  def __init__(self, env, width, height, name):
    """Observer constructor.

    Args:
      env: The environment.
      width: Window width, in pixels.
      height: Window height, in pixels.
      name: Window name.
    """
    self._env = env
    self._viewport = renderer.Viewport(width, height)
    self._viewer = None
    self._camera_config = {
        'lookat': None,
        'distance': None,
        'azimuth': None,
        'elevation': None
    }
    self._camera_config_dirty = False

    self._render_surface = _render.Renderer(
        max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
    self._renderer = renderer.NullRenderer()
    self._window = gui.RenderWindow(width, height, name)

  @classmethod
  def build(cls, env, height=_DEFAULT_HEIGHT, width=_DEFAULT_WIDTH,
            name=_DEFAULT_WINDOW_TITLE):
    """Returns a Observer with a default platform.

    Args:
      env: The environment.
      height: Window height, in pixels.
      width: Window width, in pixels.
      name: Window name.

    Returns:
      Newly constructor Observer.
    """
    return cls(env, width, height, name)

  def _apply_camera_config(self):
    for key, value in self._camera_config.items():
      if value is not None:
        if key == 'lookat':  # special case since we can't just set this attr.
          self._viewer.camera.settings.lookat[:] = self._camera_config['lookat']
        else:
          setattr(self._viewer.camera.settings, key, value)

    self._camera_config_dirty = False

  @property
  def camera_config(self):
    """Retrieves the current camera configuration."""
    if self._viewer:
      for key, value in self._camera_config.items():
        self._camera_config[key] = getattr(self._viewer.camera.settings, key,
                                           value)
    return self._camera_config

  @camera_config.setter
  def camera_config(self, camera_config):
    for key, value in camera_config.items():
      if key not in self._camera_config:
        raise ValueError(('Key {} is not a valid key in camera_config. '
                          'Valid keys are: {}').format(
                              key, list(camera_config.keys())))
      self._camera_config[key] = value
    self._camera_config_dirty = True

  def begin_episode(self, *unused_args, **unused_kwargs):
    """Notifies the observer that a new episode is about to begin.

    Args:
      *unused_args: ignored.
      **unused_kwargs: ignored.
    """
    if not self._viewer:
      self._viewer = viewer.Viewer(
          self._viewport, self._window.mouse, self._window.keyboard)
    if self._viewer:
      self._renderer = renderer.OffScreenRenderer(
          self._env.physics.model, self._render_surface)
      self._viewer.initialize(self._env.physics, self._renderer, False)

  def end_episode(self, *unused_args, **unused_kwargs):
    """Notifies the observer that an episode has ended.

    Args:
      *unused_args: ignored.
      **unused_kwargs: ignored.
    """
    if self._viewer:
      self._viewer.deinitialize()

  def _render(self):
    self._viewport.set_size(*self._window.shape)
    self._viewer.render()
    return self._renderer.pixels

  def step(self, *unused_args, **unused_kwargs):
    """Notifies the observer that an agent has taken a step.

    Args:
      *unused_args: ignored.
      **unused_kwargs: ignored.
    """
    if self._viewer:
      if self._camera_config_dirty:
        self._apply_camera_config()
    self._window.update(self._render)
