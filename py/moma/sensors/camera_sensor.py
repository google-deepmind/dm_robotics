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

"""Sensor for gathering pose and image observations from mjcf cameras."""

import dataclasses
import enum
from typing import Dict, Mapping, Optional, Tuple, Union, List

from absl import logging
from dm_control import mjcf
from dm_control import mujoco
from dm_control.composer.observation import observable
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.moma import sensor as moma_sensor
from dm_robotics.transformations import transformations as tr
import numpy as np

CameraSensorBundle = Tuple['CameraPoseSensor', 'CameraImageSensor']

# A quatenion representing a rotation from the mujoco camera, which has the -z
# axis towards the scene, to the opencv camera, which has +z towards the scene.
_OPENGL_TO_OPENCV_CAM_QUAT = np.array([0., 1., 0., 0.])


@enum.unique
class PoseObservations(enum.Enum):
  """Pose observations exposed by a camera pose sensor."""
  # The position of the camera.
  POS = '{}_pos'
  # The orientation of the camera.
  QUAT = '{}_quat'
  # The full pose of the camera, as 7-dim posquat.
  POSE = '{}_pose'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)


@enum.unique
class ImageObservations(enum.Enum):
  """Image observations exposed by a camera."""
  # The rgb image sensed by the camera.
  RGB_IMAGE = '{}_rgb_img'
  # The depth image sensed by the camera.
  DEPTH_IMAGE = '{}_depth_img'
  # The segmentation image sensed by the camera.
  SEGMENTATION_IMAGE = '{}_segmentation_img'
  # The intrinsics of the cameras.
  INTRINSICS = '{}_intrinsics'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)


@dataclasses.dataclass
class CameraConfig:
  """Represents a camera config for the arena.

  Attributes:
    width: Width of the camera image.
    height: Height of the camera image.
    fovy: Vertical field of view of the camera, expressed in degrees. See
      http://www.mujoco.org/book/XMLreference.html#camera for more details. Note
        that we do not use the fovy defined in the `mjcf.Element` as it only
        contained the fovy information. And not the height which are linked (see
        `_get_instrinsics`).
    has_rgb: Is the camera producing rgb channels.
    has_depth: Is the camera producing a depth channel.
    has_segmentation: Is the camera producing a segmentation image. If `True`,
      adds a 2-channel NumPy int32 array of label values where the pixels of
      each object are labeled with the pair (mjModel ID, mjtObj object_type).
      Background pixels are labeled (-1, -1)
    render_shadows: If true the camera will render the shadows of the scene.
      Note that when using CPU for rendering, shadow rendering increases
      significantly the amount of processing time.
  """
  width: int = 128
  height: int = 128
  fovy: float = 90.0
  has_rgb: bool = True
  has_depth: bool = False
  has_segmentation: bool = False
  render_shadows: bool = False


class CameraPoseSensor(moma_sensor.Sensor[PoseObservations]):
  """Sensor providing the camera pos observations."""

  def __init__(self, camera_element: mjcf.Element, name: str):
    """Init.

    Args:
      camera_element: MJCF camera.
      name: Name of the pose sensor.
    """
    self.element = camera_element
    self._name = name

    self._observables = {
        self.get_obs_key(PoseObservations.POS):
            observable.Generic(self._camera_pos),
        self.get_obs_key(PoseObservations.QUAT):
            observable.Generic(self._camera_quat),
        self.get_obs_key(PoseObservations.POSE):
            observable.Generic(self._camera_pose),
    }

    for obs in self._observables.values():
      obs.enabled = True

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    return self._observables

  @property
  def name(self) -> str:
    return self._name

  def get_obs_key(self, obs: PoseObservations) -> str:
    return obs.get_obs_key(self._name)

  def _camera_pos(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self.element).xpos  # pytype: disable=attribute-error

  def _camera_quat(self, physics: mjcf.Physics) -> np.ndarray:
    # Rotate the camera to have +z towards the scene to be consistent with
    # real-robot camera calibration.
    mujoco_cam_quat = tr.mat_to_quat(
        np.reshape(physics.bind(self.element).xmat, (3, 3)))  # pytype: disable=attribute-error
    return tr.quat_mul(mujoco_cam_quat, _OPENGL_TO_OPENCV_CAM_QUAT)

  def _camera_pose(self, physics: mjcf.Physics) -> np.ndarray:
    # TODO(jscholz): rendundant with pos & quat; remove pos & quat?
    return np.concatenate(
        (self._camera_pos(physics), self._camera_quat(physics)), axis=-1)


class CameraImageSensor(moma_sensor.Sensor[ImageObservations]):
  """Camera sensor providing the image observations."""

  def __init__(self, camera_element: mjcf.Element, config: CameraConfig,
               name: str):
    """Init.

    Args:
      camera_element: MJCF camera.
      config: Configuration of the camera sensor.
      name: Name of the image sensor.
    """
    self.element = camera_element
    self._cfg = config
    self._name = name
    self._camera: Optional[mujoco.Camera] = None

    self._observables = {
        self.get_obs_key(ImageObservations.INTRINSICS):
            observable.Generic(self._camera_intrinsics),
    }

    if self._cfg.has_rgb:
      self._observables[self.get_obs_key(
          ImageObservations.RGB_IMAGE)] = observable.Generic(self._camera_rgb)
    if self._cfg.has_depth:
      self._observables[self.get_obs_key(
          ImageObservations.DEPTH_IMAGE)] = observable.Generic(
              self._camera_depth)
    if self._cfg.has_segmentation:
      self._observables[self.get_obs_key(
          ImageObservations.SEGMENTATION_IMAGE)] = observable.Generic(
              self._camera_segmentation)

    for obs in self._observables.values():
      obs.enabled = True

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass

  def after_compile(self, mjcf_model: mjcf.RootElement,
                    physics: mjcf.Physics) -> None:
    # We create the camera at the beginning of every episode. This is to improve
    # speed of rendering. Previously we used `physics.render` which creates a
    # new camera every single time you render.
    self._create_camera(physics)

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    return self._observables

  @property
  def name(self) -> str:
    return self._name

  def get_obs_key(self, obs: ImageObservations) -> str:
    return obs.get_obs_key(self._name)

  def _camera_intrinsics(self, physics: mjcf.Physics) -> np.ndarray:
    del physics  # unused.
    return pinhole_intrinsics(
        img_shape=(self._cfg.height, self._cfg.width), fovy=self._cfg.fovy)

  def _camera_rgb(self, physics: mjcf.Physics) -> np.ndarray:
    return np.atleast_3d(self._camera.render(depth=False, segmentation=False))

  def _camera_depth(self, physics: mjcf.Physics) -> np.ndarray:
    return np.atleast_3d(self._camera.render(depth=True))

  def _camera_segmentation(self, physics: mjcf.Physics) -> np.ndarray:
    return np.atleast_3d(self._camera.render(segmentation=True))

  def _create_camera(self, physics: mjcf.Physics) -> None:
    self._camera = mujoco.Camera(
        physics=physics,
        height=self._cfg.height,
        width=self._cfg.width,
        camera_id=self.element.full_identifier)
    if self._cfg.render_shadows:
      self._camera.scene.flags[enums.mjtRndFlag.mjRND_SHADOW] = 1
    else:
      self._camera.scene.flags[enums.mjtRndFlag.mjRND_SHADOW] = 0


def pinhole_intrinsics(img_shape: Tuple[int, int],
                       fovy: float,
                       negate_x_focal_len: bool = False) -> np.ndarray:
  """Builds camera intrinsic matrix for simple pinhole model.

  This function returns a camera intrinsic matrix for projecting 3D camera-
  frame points to the image plane.

  For background see:
  https://en.wikipedia.org/wiki/Pinhole_camera_model

  Args:
    img_shape: (2,) array containing image height and width.
    fovy: Field-of-view in Y, in degrees.
    negate_x_focal_len: If True, negate the X focal_len following mujoco's -z
      convention. This matrix will be identical to mujoco's camera intrinsics,
      but will be inconsistent with the +z-towards-scene achieved by
      CameraPoseSensor.

  Returns:
    (3, 4) array containing camera intrinsics
  """
  height, width = img_shape

  # Calculate the focal length to get the requested image height given the
  # field of view.
  half_angle = fovy / 2
  half_angle_rad = half_angle * np.pi / 180
  focal_len = height / 2 / np.tan(half_angle_rad)

  # Note: By default these intrinsics do not include the negation of the x-
  # focal-length that mujoco uses in its camera matrix. To utilize this camera
  # matrix for projection and back-projection you must rotate the camera xmat
  # from mujoco by 180- degrees around the Y-axis. This is performed by
  # CameraPoseSensor.
  #
  # Background: Mujoco cameras view along the -z-axis, and require fovx and
  # depth-negation to do reprojection. This camera matrix follows the OpenCV
  # convention of viewing along +z, which does not require these hacks.
  x_focal_len = -focal_len if negate_x_focal_len else focal_len
  return np.array([[x_focal_len, 0, (width - 1) / 2, 0],
                   [0, focal_len, (height - 1) / 2, 0],
                   [0, 0, 1, 0]])


def get_sensor_bundle(
    camera_element: mjcf.Element, config: CameraConfig, name: str
) -> CameraSensorBundle:
  return (CameraPoseSensor(camera_element, name),
          CameraImageSensor(camera_element, config, name))


def build_camera_sensors(
    camera_configurations: Mapping[str, CameraConfig],
    mjcf_root: mjcf.element.RootElement,
    mjcf_full_identifiers: Optional[Mapping[str, str]] = None,
    ) -> List[Union[CameraPoseSensor, CameraImageSensor]]:
  """Create the camera sensors for a list of cameras.

  Args:
    camera_configurations: Configurations of the cameras. Maps from the camera
      sensor name to the configuration of the camera.
    mjcf_root: MJCf root element in which the cameras are present.
    mjcf_full_identifiers: Mapping of the `camera_configurations`
      sensor names to the full identifiers of the cameras used in the MJCF
      model. If `None`, it is set as an identity mapping of
      `camera_configurations` names.

  Returns:
    A list of camera pose and camera image sensor for each camera.

  Raise:
    ValueError if the camera_configurations and mjcf_full_identifiers do not
    have the same set of keys.
  """
  if mjcf_full_identifiers is None:
    mjcf_full_identifiers = {
        name: name for name in camera_configurations.keys()}

  if mjcf_full_identifiers.keys() != camera_configurations.keys():
    raise ValueError(
        f'mjcf_full_identifiers: {mjcf_full_identifiers.keys()} and '
        f'camera_configurations: {camera_configurations.keys()} '
        'do not contain the same set of keys.')

  camera_sensors = []
  for name, identifier in mjcf_full_identifiers.items():
    camera_prop = mjcf_root.find('camera', identifier)

    all_cameras = [
        elmt.full_identifier for elmt in mjcf_root.find_all('camera')]
    if camera_prop is None:
      logging.warning('Could not find camera with identifier %s '
                      'in the workspace. Available cameras: %s. '
                      'There will be no camera sensor for %s.',
                      identifier, all_cameras, identifier)
    else:
      config = camera_configurations[name]
      pose_sensor, image_sensor = get_sensor_bundle(camera_prop, config, name)

      camera_sensors.append(pose_sensor)
      camera_sensors.append(image_sensor)

  return camera_sensors
