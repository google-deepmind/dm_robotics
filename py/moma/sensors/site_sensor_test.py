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

"""Tests for site_sensor."""

from absl.testing import absltest
from dm_control import mjcf
from dm_robotics.moma.sensors import site_sensor
from dm_robotics.transformations import transformations as tr
import numpy as np


class SiteSensorTest(absltest.TestCase):

  def test_read_value_from_sensor(self):
    # We create a mjcf body with a site we want to measure.
    mjcf_root = mjcf.RootElement()
    box_body = mjcf_root.worldbody.add(
        "body", pos="0 0 0", axisangle="0 0 1 0",
        name="box")
    box_body.add("inertial", pos="0. 0. 0.", mass="1", diaginertia="1 1 1")
    box_body.add("freejoint")
    site = box_body.add("site", pos="0 0 0")

    # Get expected values
    expected_pos = np.array([1., 2., 3.])
    expected_quat = np.array([4.0, 5.0, 6.0, 7.0])
    expected_quat = expected_quat/ np.linalg.norm(expected_quat)
    expected_rmat = np.reshape(tr.quat_to_mat(expected_quat)[:3, :3], (9,))
    expected_vel = np.array([8., 9., 10., 11., 12., 13.])

    # We then set the position and velocity of the body
    physics = mjcf.Physics.from_mjcf_model(mjcf_root)
    physics.data.qpos[:] = np.hstack((expected_pos, expected_quat))
    physics.data.qvel[:] = expected_vel
    physics.forward()

    # Read the measurements of the sensors and ensure everything is correct
    sensor = site_sensor.SiteSensor(site, "test_site")

    pos_callable = sensor.observables[
        sensor.get_obs_key(site_sensor.Observations.POS)]
    np.testing.assert_allclose(expected_pos, pos_callable(physics))

    quat_callable = sensor.observables[
        sensor.get_obs_key(site_sensor.Observations.QUAT)]
    np.testing.assert_allclose(expected_quat, quat_callable(physics))

    rmat_callable = sensor.observables[
        sensor.get_obs_key(site_sensor.Observations.RMAT)]
    np.testing.assert_allclose(expected_rmat, rmat_callable(physics))

    # The qvel that is set has the linear velocity expressed in the world
    # frame orientation and the angular velocity expressed in the body frame
    # orientation. We therefore test that the values appear where they should.
    vel_world_callable = sensor.observables[
        sensor.get_obs_key(site_sensor.Observations.VEL_WORLD)]
    np.testing.assert_allclose(
        expected_vel[:3], vel_world_callable(physics)[:3])

    vel_relative_callable = sensor.observables[
        sensor.get_obs_key(site_sensor.Observations.VEL_RELATIVE)]
    np.testing.assert_allclose(
        expected_vel[3:], vel_relative_callable(physics)[3:])

  def test_passing_a_non_site_raise(self):
    # We create a mjcf body with a site we want to measure.
    mjcf_root = mjcf.RootElement()
    box_body = mjcf_root.worldbody.add(
        "body", pos="0 0 0", axisangle="0 0 1 0",
        name="box")
    with self.assertRaises(ValueError):
      site_sensor.SiteSensor(box_body, "error")


if __name__ == "__main__":
  absltest.main()
