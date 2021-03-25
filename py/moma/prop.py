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

# Lint as: python3
"""Base classes for props."""

from dm_control import composer
from dm_control import mjcf
from dm_robotics.transformations import transformations as tr
import numpy as np


class Prop(composer.Entity):
  """Base class for MOMA props."""

  def _build(self,
             name: str,
             mjcf_root: mjcf.RootElement,
             prop_root: str = 'prop_root'):  # TODO(b/186731907): rename asap.
    """Base constructor for props.

    This constructor sets up the common observables and element access
    properties.

    Args:
      name: The unique name of this prop.
      mjcf_root: (mjcf.Element) The root element of the MJCF model.
      prop_root: (string) Name of the prop root body MJCF element.

    Raises:
      ValueError: If the model does not contain the necessary elements.
    """

    self._name = name
    self._mjcf_root = mjcf_root
    self._prop_root = mjcf_root.find('body', prop_root)  # type: mjcf.Element
    if self._prop_root is None:
      raise ValueError(f'model does not contain prop root {prop_root}.')
    self._freejoint = None  # type: mjcf.Element

  @property
  def name(self) -> str:
    return self._name

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to this prop."""
    return self._mjcf_root

  def set_pose(self, physics: mjcf.Physics, position: np.ndarray,
               quaternion: np.ndarray) -> None:
    """Sets the pose of the prop wrt to where it was defined.

    This function overrides `Entity.set_pose`, which has the annoying property
    that it doesn't consider where the prop was originally attached.  EG if you
    do `pinch_site.attach(prop)`, the prop will be a sibling of pinch_site with
    the pinch_site's pose parameters, and calling
      `set_pose([0, 0, 0], [1, 0, 0, 0])`
    will clobber these params and move the prop to the parent-body origin.

    Oleg's fix uses an extra `prop_root` body that's a child of the sibling
    body, and sets the pose of this guy instead.

    Args:
      physics: An instance of `mjcf.Physics`.
      position: A NumPy array of size 3.
      quaternion: A NumPy array of size [w, i, j, k].

    Raises:
      RuntimeError: If the entity is not attached.
      Exception: If oleg isn't happy
    """

    if self._prop_root is None:
      raise Exception('prop {} missing root element'.format(
          self.mjcf_model.model))

    if self._freejoint is None:
      physics.bind(self._prop_root).pos = position  # pytype: disable=not-writable
      physics.bind(self._prop_root).quat = quaternion  # pytype: disable=not-writable
    else:
      # If we're attached via a freejoint then bind().pos or quat does nothing,
      # as the pose is controlled by qpos directly.
      physics.bind(self._freejoint).qpos = np.hstack([position, quaternion])  # pytype: disable=not-writable

  def set_freejoint(self, joint: mjcf.Element):
    """Associates a freejoint with this prop if attached to arena."""
    joint_type = joint.tag  # pytype: disable=attribute-error
    if joint_type != 'freejoint':
      raise ValueError(f'Expected a freejoint but received {joint_type}')
    self._freejoint = joint

  def disable_collisions(self) -> None:
    for geom in self.mjcf_model.find_all('geom'):
      geom.contype = 0
      geom.conaffinity = 0


class WrapperProp(Prop):

  def _build(self, wrapped_entity: composer.Entity, name: str):
    root = mjcf.element.RootElement(model=name)
    body_elem = root.worldbody.add('body', name='prop_root')
    site_elem = body_elem.add('site', name='prop_root_site')
    site_elem.attach(wrapped_entity.mjcf_model)
    super()._build(name=name, mjcf_root=root, prop_root='prop_root')


class Camera(Prop):
  """Base class for Moma camera props."""

  def _build(self,
             name: str,
             mjcf_root: mjcf.RootElement,
             camera_element: str,
             prop_root: str = 'prop_root',
             width: int = 480,
             height: int = 640,
             fovy: float = 90.0):
    """Camera  constructor.

    Args:
      name: The unique name of this prop.
      mjcf_root: The root element of the MJCF model.
      camera_element: Name of the camera MJCF element.
      prop_root: Name of the prop root body MJCF element.
      width: Width of the camera image
      height: Height of the camera image
      fovy: Field of view, in degrees.
    """
    super(Camera, self)._build(name, mjcf_root, prop_root)

    self._camera_element = camera_element
    self._width = width
    self._height = height
    self._fovy = fovy

    # Sub-classes should extend `_build` to construct the appropriate mjcf, and
    # over-ride the `rgb_camera` and `depth_camera` properties.

  @property
  def camera(self) -> mjcf.Element:
    """Returns an mjcf.Element representing the camera."""
    return self._mjcf_root.find('camera', self._camera_element)

  def get_camera_pos(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self.camera).xpos  # pytype: disable=attribute-error

  def get_camera_quat(self, physics: mjcf.Physics) -> np.ndarray:
    return tr.mat_to_quat(
        np.reshape(physics.bind(self.camera).xmat, [3, 3]))  # pytype: disable=attribute-error

  def render_rgb(self, physics: mjcf.Physics) -> np.ndarray:
    return np.atleast_3d(
        physics.render(
            height=self._height,
            width=self._width,
            camera_id=self.camera.full_identifier,  # pytype: disable=attribute-error
            depth=False))

  def render_depth(self, physics: mjcf.Physics) -> np.ndarray:
    return np.atleast_3d(physics.render(
        height=self._height,
        width=self._width,
        camera_id=self.camera.full_identifier,  # pytype: disable=attribute-error
        depth=True))

  def get_intrinsics(self, physics: mjcf.Physics) -> np.ndarray:
    focal_len = self._height / 2 / np.tan(self._fovy / 2 * np.pi / 180)
    return np.array([[focal_len, 0, (self._height - 1) / 2, 0],
                     [0, focal_len, (self._height - 1) / 2, 0],
                     [0, 0, 1, 0]])


class Block(Prop):
  """A block prop."""

  def _build(self, name: str = 'box', width=0.04, height=0.04, depth=0.04):
    mjcf_root, site = _make_block_model(name, width, height, depth)
    super()._build(name, mjcf_root, 'prop_root')
    del site


def _make_block_model(name,
                      width,
                      height,
                      depth,
                      color=(1, 0, 0, 1),
                      solimp=(0.95, 0.995, 0.001),
                      solref=(0.002, 0.7)):
  """Makes a plug model: the mjcf element, and reward sites."""

  mjcf_root = mjcf.element.RootElement(model=name)
  prop_root = mjcf_root.worldbody.add('body', name='prop_root')
  box = prop_root.add(
      'geom',
      name='body',
      type='box',
      pos=(0, 0, 0),
      size=(width / 2., height / 2., depth / 2.),
      mass=0.050,
      solref=solref,
      solimp=solimp,
      condim=1,
      rgba=color)
  site = prop_root.add(
      'site',
      name='box_centre',
      type='sphere',
      rgba=(0.1, 0.1, 0.1, 0.8),
      size=(0.002,),
      pos=(0, 0, 0),
      euler=(0, 0, 0))  # Was (np.pi, 0, np.pi / 2)
  del box

  return mjcf_root, site
