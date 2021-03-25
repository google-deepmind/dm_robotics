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

"""Tests for dm_robotics.moma.models.end_effectors.wrist_sensors.robotiq_fts300."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf

from dm_robotics.moma.models.end_effectors.wrist_sensors import robotiq_fts300


class RobotiqFTS300Test(parameterized.TestCase):
  """Tests for the Robotiq FTS300 force/torque sensor."""

  def test_load_sensor(self):
    """Check RobotiqFTS300 can be instantiated and physics step() executed."""
    entity = robotiq_fts300.RobotiqFTS300()
    physics = mjcf.Physics.from_mjcf_model(entity.mjcf_model)
    physics.step()

  def test_zero_gravity_readings(self):
    """Measure the force applied to F/T sensor when gravity is disabled."""
    entity = robotiq_fts300.RobotiqFTS300()
    physics = mjcf.Physics.from_mjcf_model(entity.mjcf_model)
    with physics.model.disable("gravity"):
      physics.forward()
    force_z = physics.bind(entity.force_sensor).sensordata[2]
    self.assertEqual(force_z, 0.)


if __name__ == "__main__":
  absltest.main()
