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

"""Tests for mujoco_collisions."""

from absl.testing import absltest
from dm_control import mjcf
from dm_robotics.moma.utils import mujoco_collisions


class MujocoCollisionsTest(absltest.TestCase):

  def test_collision_are_excluded_when_contype_is_zero(self):
    # We create a mjcf model with 3 boxes and disable the contype for 2 of them.
    mjcf_root = mjcf.RootElement()
    box1_body = mjcf_root.worldbody.add(
        'body', pos='0 0 0', axisangle='0 0 1 0', name='box1')
    box1_body.add(
        'geom', name='box1', type='box', size='.2 .2 .2', contype='0')

    box2_body = mjcf_root.worldbody.add(
        'body', pos='0 1 0', axisangle='0 0 1 0', name='box2',)
    box2_body.add(
        'geom', name='box2', type='box', size='.2 .2 .2', contype='0')

    # For the third body disable conaffinity and ensure it does not exclude the
    # contacts between this box and the first two.
    box3_body = mjcf_root.worldbody.add(
        'body', pos='0 0 1', axisangle='0 0 1 0', name='box3')
    box3_body.add(
        'geom', name='box3', type='box', size='.2 .2 .2', conaffinity='0')

    mujoco_collisions.exclude_bodies_based_on_contype_conaffinity(mjcf_root)

    # We assert that the contact array only excludes contacts between box 1 and
    # box 2.
    [child] = mjcf_root.contact.all_children()
    self.assertEqual('exclude', child.spec.name)
    self.assertEqual('box2', child.body1)
    self.assertEqual('box1', child.body2)

  def test_collision_are_excluded_when_conaffinity_is_zero(self):
    # We create a mjcf model with 3 boxes and disable the conaffinity for
    # 2 of them.
    mjcf_root = mjcf.RootElement()
    box1_body = mjcf_root.worldbody.add(
        'body', pos='0 0 0', axisangle='0 0 1 0', name='box1')
    box1_body.add(
        'geom', name='box1', type='box', size='.2 .2 .2', conaffinity='0')

    box2_body = mjcf_root.worldbody.add(
        'body', pos='0 1 0', axisangle='0 0 1 0', name='box2',)
    box2_body.add(
        'geom', name='box2', type='box', size='.2 .2 .2', conaffinity='0')

    # For the third body disable contype and ensure it does not exclude the
    # contacts between this box and the first two.
    box3_body = mjcf_root.worldbody.add(
        'body', pos='0 0 1', axisangle='0 0 1 0', name='box3')
    box3_body.add(
        'geom', name='box3', type='box', size='.2 .2 .2', contype='0')

    mujoco_collisions.exclude_bodies_based_on_contype_conaffinity(mjcf_root)

    # We assert that the contact array only excludes contacts between box 1 and
    # box 2.
    [child] = mjcf_root.contact.all_children()
    self.assertEqual('exclude', child.spec.name)
    self.assertEqual('box2', child.body1)
    self.assertEqual('box1', child.body2)


if __name__ == '__main__':
  absltest.main()
