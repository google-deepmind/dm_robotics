# Copyright 2022 DeepMind Technologies Limited.
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

"""Tests for action_spaces."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.geometry import geometry as geo
from dm_robotics.moma import action_spaces
import numpy as np

SEED = 0


class ReframeVelocityActionSpaceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_full', None),
      ('_pos_dims', [0, 1, 2]),
      ('_rot_dims', [3, 4, 5]),
      ('_mixed_dims', [0, 2, 4, 5]),
  )
  def test_various_slices(self, velocity_dims):
    random_state = np.random.RandomState(SEED)

    action_dim = len(velocity_dims) if velocity_dims else 6
    base_spec = specs.BoundedArray(
        shape=(action_dim,),
        dtype=np.float32,
        minimum=[-1] * action_dim,
        maximum=[1] * action_dim,
        name='\t'.join([f'a{i}' for i in range(action_dim)]))
    physics_getter = lambda: mock.MagicMock(spec=mjcf.Physics)

    for _ in range(10):
      parent_frame = geo.PoseStamped(
          geo.Pose.from_poseuler(random_state.randn(6)), frame=None)
      input_frame = geo.PoseStamped(
          geo.Pose.from_poseuler(random_state.randn(6)), parent_frame)
      output_frame = geo.PoseStamped(
          geo.Pose.from_poseuler(random_state.randn(6)), parent_frame)

      reframe_space = action_spaces.ReframeVelocityActionSpace(
          spec=base_spec,
          physics_getter=physics_getter,
          input_frame=input_frame,
          output_frame=output_frame,
          velocity_dims=velocity_dims,
          name='test ReframeVelocity')

      input_action = random_state.randn(action_dim)
      if velocity_dims is not None:
        full_action = np.zeros(6)
        full_action[velocity_dims] = input_action
      else:
        full_action = input_action

      input_twist = geo.TwistStamped(geo.Twist(full_action), input_frame)
      expected_output_action = input_twist.get_relative_twist(output_frame).full
      if velocity_dims is not None:
        expected_output_action = expected_output_action[velocity_dims]

      output_action = reframe_space.project(input_action)

      expected_output_action = spec_utils.shrink_to_fit(
          value=expected_output_action, spec=base_spec)

      np.testing.assert_almost_equal(output_action, expected_output_action)


if __name__ == '__main__':
  absltest.main()
