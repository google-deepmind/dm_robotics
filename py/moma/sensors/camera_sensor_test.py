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

"""Tests for camera sensors."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_robotics.moma.sensors import camera_sensor


class CameraPoseSensorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mjcf_root = mjcf.element.RootElement(model='camera')
    mjcf_root.worldbody.add('body', name='prop_root')
    prop_root = mjcf_root.find('body', 'prop_root')

    prop_root.add(
        'camera',
        name='test_camera',
        pos=[0., 0., 0.],
        quat=[0., 0., 0., 0.],
        fovy=90.0)

    self._camera_element = mjcf_root.find('camera', 'test_camera')

  def test_pose_camera(self):
    sensor_name = 'pose_test'

    sensor = camera_sensor.CameraPoseSensor(
        camera_element=self._camera_element, name=sensor_name)

    expected_obs = [
        f'{sensor_name}_pos',
        f'{sensor_name}_quat',
        f'{sensor_name}_pose',
    ]

    self.assertCountEqual(sensor.observables.keys(), expected_obs)


class CameraImageSensorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    mjcf_root = mjcf.element.RootElement(model='camera')
    mjcf_root.worldbody.add('body', name='prop_root')
    prop_root = mjcf_root.find('body', 'prop_root')

    prop_root.add(
        'camera',
        name='test_camera',
        pos=[0., 0., 0.],
        quat=[0., 0., 0., 0.],
        fovy=90.0)

    self._camera_element = mjcf_root.find('camera', 'test_camera')

  @parameterized.parameters(
      itertools.product([True, False], [True, False], [True, False]))
  def test_camera(self, has_rgb, has_depth, has_segmentation):
    sensor_name = 'camera_sensor_test'
    camera_config = camera_sensor.CameraConfig(
        width=128,
        height=128,
        fovy=45.,
        has_rgb=has_rgb,
        has_depth=has_depth,
        has_segmentation=has_segmentation,
    )

    sensor = camera_sensor.CameraImageSensor(
        camera_element=self._camera_element,
        config=camera_config,
        name=sensor_name,
    )

    expected_obs = [f'{sensor_name}_intrinsics']
    if has_rgb:
      expected_obs.append(f'{sensor_name}_rgb_img')
    if has_depth:
      expected_obs.append(f'{sensor_name}_depth_img')
    if has_segmentation:
      expected_obs.append(f'{sensor_name}_segmentation_img')

    self.assertCountEqual(sensor.observables.keys(), expected_obs)


class CameraSensorBundleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mjcf_root = mjcf.element.RootElement(model='camera')
    mjcf_root.worldbody.add('body', name='prop_root')
    prop_root = mjcf_root.find('body', 'prop_root')

    prop_root.add(
        'camera',
        name='test_camera',
        pos=[0., 0., 0.],
        quat=[0., 0., 0., 0.],
        fovy=90.0)

    self._camera_element = mjcf_root.find('camera', 'test_camera')

  def test_rgb_camera(self):
    sensor_name = 'rgb_test'

    camera_config = camera_sensor.CameraConfig(
        width=128,
        height=128,
        fovy=45.,
        has_rgb=True,
        has_depth=False,
    )

    sensor_bundle = camera_sensor.get_sensor_bundle(
        camera_element=self._camera_element,
        config=camera_config,
        name=sensor_name)

    self.assertIsInstance(sensor_bundle[0],
                          camera_sensor.CameraPoseSensor)
    self.assertIsInstance(sensor_bundle[1],
                          camera_sensor.CameraImageSensor)


class SensorCreationTest(absltest.TestCase):

  def test_sensor_creation(self):
    mjcf_root = mjcf.element.RootElement(model='camera')
    mjcf_root.worldbody.add('body', name='prop_root')
    prop_root = mjcf_root.find('body', 'prop_root')
    prop_root.add('camera', name='camera1', pos=[0., 0., 0.],
                  quat=[0., 0., 0., 0.], fovy=90.0)
    prop_root.add('camera', name='camera2', pos=[0., 0., 0.],
                  quat=[0., 0., 0., 0.], fovy=90.0)

    camera_configurations = {
        'camera1': camera_sensor.CameraConfig(
            width=128, height=128, fovy=90.0, has_rgb=True, has_depth=False),
        'camera2': camera_sensor.CameraConfig(
            width=128, height=128, fovy=90.0, has_rgb=True, has_depth=False)
    }

    mjcf_full_identifiers = {'camera1': 'camera1', 'camera2': 'camera2'}

    sensors = camera_sensor.build_camera_sensors(
        camera_configurations, mjcf_root, mjcf_full_identifiers)
    self.assertLen(sensors, 4)
    self.assertIsInstance(sensors[0], camera_sensor.CameraPoseSensor)
    self.assertIsInstance(sensors[1], camera_sensor.CameraImageSensor)
    self.assertIsInstance(sensors[2], camera_sensor.CameraPoseSensor)
    self.assertIsInstance(sensors[3], camera_sensor.CameraImageSensor)

  def test_failure_for_non_matching_camera_keys(self):
    mjcf_root = mjcf.element.RootElement(model='camera')
    mjcf_root.worldbody.add('body', name='prop_root')
    prop_root = mjcf_root.find('body', 'prop_root')
    prop_root.add('camera', name='foo', pos=[0., 0., 0.],
                  quat=[0., 0., 0., 0.], fovy=90.0)

    camera_configurations = {
        'foo': camera_sensor.CameraConfig(
            width=128, height=128, fovy=90.0, has_rgb=True, has_depth=False),
    }
    mjcf_full_identifiers = {'not_foo': 'not_foo'}

    with self.assertRaises(ValueError):
      camera_sensor.build_camera_sensors(
          camera_configurations, mjcf_root, mjcf_full_identifiers)


if __name__ == '__main__':
  absltest.main()
