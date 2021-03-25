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

"""Tests for empty."""

from absl.testing import absltest
from dm_control import mjcf
from dm_robotics.moma.models.arenas import empty


class TestArena(absltest.TestCase):

  def test_initialize(self):

    arena = empty.Arena()

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    physics.step()


if __name__ == '__main__':
  absltest.main()
