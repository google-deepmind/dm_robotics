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

# python3
"""Tests for dm_robotics.agentflow.action_spaces."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
from dm_robotics.agentflow import action_spaces
from dm_robotics.agentflow import core
from dm_robotics.agentflow import testing_functions
import numpy as np


class ActionSpacesTest(parameterized.TestCase):

  def test_constrained_action_spec(self):
    spec = specs.BoundedArray(
        shape=(2,), dtype=float, minimum=[-50.0, 0.0], maximum=[50.0, 100.0])
    space = core.IdentityActionSpace(spec)

    constrained_spec = action_spaces.constrained_action_spec(
        minimum=[-10.0, 20.0], maximum=[40.0, 90.0], base=spec)
    constrained_space = core.IdentityActionSpace(constrained_spec)

    good_base_input = np.asarray([0.0, 10.0])
    np.testing.assert_almost_equal(
        space.project(good_base_input), good_base_input)

    # This action is within the new min/max bounds, should pass.
    good_smaller_input = np.asarray([0.0, 25.0])
    np.testing.assert_almost_equal(
        constrained_space.project(good_smaller_input), good_smaller_input)

    # The original action that passed the base space should fail in the smaller
    # action space.
    with self.assertRaises(ValueError):
      constrained_space.project(good_base_input)

    # Check handling of scalar min/max
    spec = specs.BoundedArray(
        shape=(3,), dtype=float, minimum=-50.0, maximum=50.0)
    constrained_spec = action_spaces.constrained_action_spec(
        minimum=-10.0, maximum=40.0, base=spec)
    constrained_space = core.IdentityActionSpace(constrained_spec)

    good_constrained_input = np.asarray([0.0] * 3)
    np.testing.assert_almost_equal(
        constrained_space.project(good_constrained_input),
        good_constrained_input)

    bad_constrained_input = np.asarray([90.0] * 3)
    with self.assertRaises(ValueError):
      constrained_space.project(bad_constrained_input)

  def test_constrained_action_space(self):
    spec = specs.BoundedArray(
        shape=(2,), dtype=float, minimum=[-50.0, 0.0], maximum=[50.0, 100.0])
    space = core.IdentityActionSpace(spec)

    constrained_space = action_spaces.constrained_action_space(
        minimum=[-10.0, 20.0], maximum=[40.0, 90.0], base=space)

    good_base_input = np.asarray([0.0, 10.0])
    np.testing.assert_almost_equal(
        space.project(good_base_input), good_base_input)

    # This action is within the new min/max bounds, should pass.
    good_smaller_input = np.asarray([0.0, 25.0])
    np.testing.assert_almost_equal(
        constrained_space.project(good_smaller_input), good_smaller_input)

    # The original action that passed the base space should fail in the smaller
    # action space.
    with self.assertRaises(ValueError):
      constrained_space.project(good_base_input)

  def test_simple_fixed_action_space(self):
    base = specs.Array(shape=(2,), dtype=np.float32, name='a1\ta2')
    base_space = action_spaces.prefix_slicer(base, 'a')
    fixed_spec = action_spaces.FixedActionSpace(
        base_space, np.asarray([1, 2], dtype=np.float32))

    self.assertEqual(base_space.spec().shape, (2,))
    self.assertEqual(fixed_spec.spec().shape, (0,))

    np.testing.assert_almost_equal(
        fixed_spec.project(np.asarray([], np.float32)),
        np.asarray([1, 2], dtype=np.float32))

    self.assertIsNotNone(fixed_spec.spec().name)

  def test_exclusion_slicer(self):
    base = specs.Array(shape=(4,), dtype=np.float32,
                       name='a1\ta2\texclude_action1\texclude_action2')
    base_space = action_spaces.prefix_slicer(base,
                                             '^(?!exclude)[[a-zA-Z0-9-_.]+$')
    fixed_spec = action_spaces.FixedActionSpace(
        base_space, np.asarray([1, 2], dtype=np.float32))

    self.assertEqual(base_space.spec().shape, (2,))
    self.assertEqual(fixed_spec.spec().shape, (0,))

    np.testing.assert_almost_equal(
        fixed_spec.project(np.asarray([], np.float32)),
        np.asarray([1, 2, np.nan, np.nan], dtype=np.float32))

    self.assertIsNotNone(fixed_spec.spec().name)

  def test_shrink_to_fit_action_space(self):
    # Corresponds to `spec_utils_test.test_primitive`.
    spec = specs.BoundedArray(
        shape=(3,),
        dtype=float,
        minimum=[0.0, 0.0, 0.0],
        maximum=[20.0, 100.0, 20.0])
    action_space = action_spaces.ShrinkToFitActionSpace(spec)

    val1 = np.asarray([21.0, 5.0, 21.0])  # over-max, under-min, over-max
    factor1 = 20.0 / 21.0
    expected1 = np.asarray([20.0, 5.0 * factor1, 20.0])

    testing_functions.assert_value(action_space.project(val1), expected1)

    val2 = np.asarray([1.0, 200.0, 21.0])  # ok, over-max, over-max
    expected2 = np.asarray([0.5, 100.0, 10.5])

    testing_functions.assert_value(action_space.project(val2), expected2)

  def test_identity_action_space_output(self):
    spec = specs.BoundedArray(
        shape=(2,), dtype=float, minimum=[-50.0, 0.0], maximum=[50.0, 100.0])
    space = core.IdentityActionSpace(spec)
    good_input = np.asarray([0.0, 10.0])
    bad_input = np.asarray([0.0, 110.0])

    np.testing.assert_almost_equal(space.project(good_input), good_input)

    try:
      space.project(bad_input)
      self.fail('Should fail validation')
    except ValueError as expected:
      del expected

  def test_cast_action_space_output(self):
    spec = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=[-1.0, -2.0], maximum=[1.0, 2.0])

    # Should pass validation if action has NaN and ignore_nan is True.
    space = action_spaces.CastActionSpace(spec, ignore_nan=True)
    _ = space.project(np.asarray([0.0, np.nan]))

    # Should raise an exception if action has NaN and ignore_nan is False.
    space = action_spaces.CastActionSpace(spec, ignore_nan=False)
    with self.assertRaises(ValueError):
      space.project(np.asarray([0.0, np.nan]))

    # Should raise an exception if action has wrong shape.
    with self.assertRaises(ValueError):
      space.project(np.asarray([0.0, 0.0, 0.0]))

    # Should raise an exception if action is out of bounds.
    with self.assertRaises(ValueError):
      space.project(np.asarray([0.0, 3.0]))

    # Should cast a float64 to float32 and pass validation.
    good_input = np.asarray([0.0, 1.0], dtype=np.float64)
    expected_result = np.asarray([0.0, 1.0], dtype=np.float32)
    actual_result = space.project(good_input)
    np.testing.assert_array_almost_equal(expected_result, actual_result)
    self.assertEqual(expected_result.dtype, actual_result.dtype)

  @parameterized.parameters(
      specs.Array(shape=(3,), dtype=np.float32, name='a11\ta12\ta2'),
      specs.BoundedArray(shape=(3,), dtype=np.float32, name='a11\ta12\ta2',
                         minimum=[-1., -2., -3.], maximum=[1., 2., 3.]),
  )
  def test_sequential_action_space(self, base_spec):
    base_space = action_spaces.prefix_slicer(base_spec, 'a')
    subspace1 = action_spaces.prefix_slicer(base_space.spec(), 'a1')
    subspace2 = action_spaces.prefix_slicer(subspace1.spec(), 'a12')
    sequential_spec = action_spaces.SequentialActionSpace(
        [subspace2, subspace1, base_space], 'Sequential space')

    self.assertEqual(base_space.spec().shape, (3,))
    self.assertEqual(sequential_spec.spec().shape, subspace2.spec().shape)

    expected_result = np.asarray([np.nan, 3., np.nan], dtype=base_spec.dtype)

    np.testing.assert_almost_equal(
        sequential_spec.project(np.asarray([3.], np.float32)),
        expected_result)

    self.assertIsNotNone(sequential_spec.spec().name)


if __name__ == '__main__':
  absltest.main()
