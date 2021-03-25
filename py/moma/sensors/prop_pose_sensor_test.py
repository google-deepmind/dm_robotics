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

"""Tests for prop_pose_sensor."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_robotics.moma import prop as moma_prop
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.sensors import prop_pose_sensor
import numpy as np

# Absolute tolerance parameter.
_A_TOL = 5e-03
# Relative tolerance parameter.
_R_TOL = 0.01


class PropPoseSensorTest(parameterized.TestCase):

  def test_sensor_returns_pose(self):
    arena = empty.Arena()
    name = 'prop'

    prop = moma_prop.Block()
    frame = arena.add_free_entity(prop)
    prop.set_freejoint(frame.freejoint)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    sensor = prop_pose_sensor.PropPoseSensor(prop=prop, name=name)
    self.assertIn(
        sensor.get_obs_key(prop_pose_sensor.Observations.POSE),
        sensor.observables)
    expected_pose = [1., 2., 3., 1., 0., 0., 0.]
    prop.set_pose(physics, expected_pose[:3], expected_pose[3:])
    physics.step()
    observable = sensor.observables[sensor.get_obs_key(
        prop_pose_sensor.Observations.POSE)]
    np.testing.assert_allclose(
        observable(physics), expected_pose, rtol=_R_TOL, atol=_A_TOL)


class SensorCreationTest(absltest.TestCase):

  def test_sensor_creation(self):
    mjcf_root = mjcf.element.RootElement(model='arena')
    names = ['foo', 'baz', 'wuz']
    moma_props = []
    for name in names:
      mjcf_root.worldbody.add('body', name=name)
      moma_props.append(
          moma_prop.Prop(name=name, mjcf_root=mjcf_root, prop_root=name))

    sensors = prop_pose_sensor.build_prop_pose_sensors(moma_props)
    self.assertLen(sensors, 3)
    for sensor in sensors:
      self.assertIsInstance(sensor, prop_pose_sensor.PropPoseSensor)


if __name__ == '__main__':
  absltest.main()
