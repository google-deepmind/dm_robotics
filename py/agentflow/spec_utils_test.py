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
"""Tests for dm_robotics.agentflow.spec_utils."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
from dm_robotics.agentflow import action_spaces
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import testing_functions
import numpy as np

_rand_spec = testing_functions.random_array_spec
_valid_val = testing_functions.valid_value

random_array_spec = testing_functions.random_array_spec
random_observation_spec = testing_functions.random_observation_spec
random_discount_spec = testing_functions.random_discount_spec
random_reward_spec = testing_functions.random_reward_spec
random_timestep_spec = testing_functions.random_timestep_spec
valid_value = testing_functions.valid_value


class ValidationTest(parameterized.TestCase):

  def assert_invalid_observation(self, spec, value, msg_substring=None):
    try:
      spec_utils.validate_observation(spec, value)
      self.fail('Expected validation failure')
    except ValueError as expected:
      actual_msg = str(expected)
      if msg_substring:
        if msg_substring not in actual_msg:
          self.fail('Got message "{}", expected to find "{}" in it'.format(
              actual_msg, msg_substring))

  def test_Float32(self):
    spec = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0., maximum=1., name='foo')
    spec_utils.validate(
        spec, np.float32(0.5), ignore_nan=False, ignore_ranges=False)

    try:
      spec_utils.validate(
          spec, float(0.5), ignore_nan=False, ignore_ranges=False)
      self.fail('Expected exception')
    except ValueError as unused_but_expected:
      del unused_but_expected

    try:
      spec_utils.validate(
          spec, np.float64(0.5), ignore_nan=False, ignore_ranges=False)
      self.fail('Expected exception')
    except ValueError as unused_but_expected:
      del unused_but_expected

  @absltest.skip('dm_env StringArray incompatible with numpy 1.24')
  def test_StringArray(self):
    test_string = 'test string'
    spec = specs.StringArray(shape=(), string_type=str, name='foo')
    spec_utils.validate(
        spec, test_string, ignore_nan=False, ignore_ranges=False)

    with self.assertRaises(ValueError):
      spec_utils.validate(
          spec,
          test_string.encode('ASCII'),
          ignore_nan=False,
          ignore_ranges=False)

    # Test that StringArray is amenable to maximum/minimum. This is not of
    # obvious utility, but arises due to the occasional need to derive a spec
    # from a sample value, e.g. in AddObservation.
    string_minimum = spec_utils.minimum(spec)
    string_maximum = spec_utils.maximum(spec)
    spec.validate(string_minimum)
    spec.validate(string_maximum)

  def test_NoneAlwaysAccepted(self):
    spec_utils.validate(
        random_array_spec(), None, ignore_nan=False, ignore_ranges=False)

  def test_Observation_MissingKeysOk(self):
    spec1 = random_array_spec()
    spec2 = random_array_spec()
    value1 = valid_value(spec1)
    value2 = valid_value(spec2)

    spec_utils.validate_observation({'foo': spec1}, {'foo': value1})
    spec_utils.validate_observation({'foo': spec1}, {'foo': None})
    spec_utils.validate_observation({'foo': spec1}, {})
    spec_utils.validate_observation({
        'foo': spec1,
        'bar': spec2
    }, {'bar': value2})

  def test_Observation_ExtraKeysFail(self):
    spec = random_array_spec()
    value = valid_value(spec)

    spec_utils.validate_observation({'foo': spec}, {'foo': value})
    self.assert_invalid_observation({'foo': spec}, {'bar': value})
    self.assert_invalid_observation({}, {'': value})

  def test_Observation_Success(self):
    spec1, spec2 = random_array_spec(), random_array_spec()
    val1, val2 = valid_value(spec1), valid_value(spec2)

    spec_utils.validate_observation({'a': spec1}, {'a': val1})
    spec_utils.validate_observation({'a': spec1, 'b': spec2}, {'a': val1})
    spec_utils.validate_observation({'a': spec1, 'b': spec2}, {'b': val2})
    spec_utils.validate_observation({
        'a': spec1,
        'b': spec2
    }, {
        'a': val1,
        'b': val2
    })

  @parameterized.parameters(float, np.float32, np.float64)
  def test_IgnoreNan_Scalar_ArraySpec(self, dtype):
    spec = specs.Array(shape=(), dtype=dtype)
    value = dtype('nan')
    spec_utils.validate(spec, value, ignore_nan=True, ignore_ranges=False)
    try:
      spec_utils.validate(spec, value, ignore_nan=False, ignore_ranges=False)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  @parameterized.parameters(float, np.float32, np.float64)
  def test_IgnoreNan_Scalar_BoundedArraySpec(self, dtype):
    # In this case, validation against the min and max will fail for nans.
    spec = specs.BoundedArray(
        shape=(),
        dtype=dtype,
        minimum=np.asarray(dtype(0.0)),
        maximum=np.asarray(dtype(1.0)))
    value = dtype('nan')
    spec_utils.validate(spec, value, ignore_nan=True, ignore_ranges=False)
    try:
      spec_utils.validate(spec, value, ignore_nan=False, ignore_ranges=False)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  @parameterized.parameters(float, np.float32, np.float64)
  def test_IgnoreNan_Array(self, dtype):
    spec = specs.Array(shape=(2,), dtype=dtype)
    value = np.asarray([dtype('nan'), dtype('nan')], dtype=dtype)

    spec_utils.validate(spec, value, ignore_nan=True, ignore_ranges=False)
    try:
      spec_utils.validate(spec, value, ignore_nan=False, ignore_ranges=False)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  @parameterized.parameters(float, np.float32, np.float64)
  def test_IgnoreNan_BoundedArray(self, dtype):
    # Since unbounded, arrays don't care about nan.
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=dtype,
        minimum=np.asarray([dtype(0.0), dtype(1.0)]),
        maximum=np.asarray([dtype(2.0), dtype(3.0)]))
    value = np.asarray([dtype('nan'), dtype('nan')], dtype=dtype)

    spec_utils.validate(spec, value, ignore_nan=True, ignore_ranges=False)
    try:
      spec_utils.validate(spec, value, ignore_nan=False, ignore_ranges=False)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  @parameterized.parameters(float, np.float32, np.float64)
  def test_IgnoreNan_BoundedArray_RespectsLimits(self, dtype):
    # Since unbounded, arrays don't care about nan.
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=dtype,
        minimum=np.asarray([dtype(0.0), dtype(1.0)]),
        maximum=np.asarray([dtype(2.0), dtype(3.0)]))
    oob_value = np.asarray([dtype('nan'), dtype(4.0)], dtype=dtype)

    try:
      spec_utils.validate(spec, oob_value, ignore_nan=True, ignore_ranges=False)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

    try:
      spec_utils.validate(
          spec, oob_value, ignore_nan=False, ignore_ranges=False)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  @parameterized.parameters(float, np.float32, np.float64)
  def test_IgnoreRanges_And_IgnoreNan(self, dtype):
    # Since unbounded, arrays don't care about nan.
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=dtype,
        minimum=np.asarray([dtype(0.0), dtype(1.0)]),
        maximum=np.asarray([dtype(2.0), dtype(3.0)]))
    oob_value = np.asarray([dtype('nan'), dtype(4.0)], dtype=dtype)

    spec_utils.validate(spec, oob_value, ignore_nan=True, ignore_ranges=True)

    try:
      spec_utils.validate(spec, oob_value, ignore_nan=True, ignore_ranges=False)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  @parameterized.parameters(float, np.float32, np.float64)
  def test_IgnoreRanges(self, dtype):
    # Since unbounded, arrays don't care about nan.
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=dtype,
        minimum=np.asarray([dtype(0.0), dtype(1.0)]),
        maximum=np.asarray([dtype(2.0), dtype(3.0)]))
    oob_value = np.asarray([dtype(1.0), dtype(4.0)], dtype=dtype)

    spec_utils.validate(spec, oob_value, ignore_ranges=True, ignore_nan=False)

    try:
      spec_utils.validate(
          spec, oob_value, ignore_ranges=False, ignore_nan=False)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  @parameterized.parameters(float, np.float32, np.float64)
  def test_TypeCheck_Scalar(self, dtype):
    spec = specs.Array(shape=(), dtype=dtype)
    spec_utils.validate(spec, dtype(1.0), ignore_ranges=False, ignore_nan=False)

  def test_EmptySpec(self):
    empty_spec = spec_utils.merge_specs([])
    spec_utils.validate(empty_spec, [])


class EnsureSpecCompatibilityTest(absltest.TestCase):

  def test_EqualSpecs(self):
    timestep_spec = random_timestep_spec()
    spec_utils.ensure_spec_compatibility(
        sub_specs=timestep_spec, full_specs=timestep_spec)

  def test_DifferentReward(self):
    observation_spec = random_observation_spec()
    discount_spec = random_discount_spec()
    reward_subspec = random_reward_spec(shape=(2,), dtype=np.float32)
    reward_fullspec = random_reward_spec(shape=(5,), dtype=np.int8)

    timestep_subspec = random_timestep_spec(
        observation_spec=observation_spec,
        discount_spec=discount_spec,
        reward_spec=reward_subspec)
    timestep_fullspec = random_timestep_spec(
        observation_spec=observation_spec,
        discount_spec=discount_spec,
        reward_spec=reward_fullspec)

    try:
      spec_utils.ensure_spec_compatibility(timestep_subspec, timestep_fullspec)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  def test_DifferentDiscount(self):
    observation_spec = random_observation_spec()
    reward_spec = random_reward_spec()
    discount_subspec = random_discount_spec(minimum=-5., maximum=5.)
    discount_fullspec = random_discount_spec(minimum=0., maximum=1.)

    timestep_subspec = random_timestep_spec(
        observation_spec=observation_spec,
        discount_spec=discount_subspec,
        reward_spec=reward_spec)
    timestep_fullspec = random_timestep_spec(
        observation_spec=observation_spec,
        discount_spec=discount_fullspec,
        reward_spec=reward_spec)

    try:
      spec_utils.ensure_spec_compatibility(timestep_subspec, timestep_fullspec)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected

  def test_SubsetObservation(self):
    reward_spec = random_reward_spec()
    discount_spec = random_discount_spec()
    observation_subspec = random_observation_spec()
    observation_fullspec = copy.deepcopy(observation_subspec)
    observation_fullspec['extra_obs'] = random_array_spec((8, 2), 'extra_obs',
                                                          np.float64)

    timestep_subspec = random_timestep_spec(
        observation_spec=observation_subspec,
        discount_spec=discount_spec,
        reward_spec=reward_spec)
    timestep_fullspec = random_timestep_spec(
        observation_spec=observation_fullspec,
        discount_spec=discount_spec,
        reward_spec=reward_spec)

    spec_utils.ensure_spec_compatibility(timestep_subspec, timestep_fullspec)

  def test_MissingObservation(self):
    reward_spec = random_reward_spec()
    discount_spec = random_discount_spec()
    observation_subspec = random_observation_spec()
    observation_fullspec = copy.deepcopy(observation_subspec)
    observation_fullspec.pop(list(observation_fullspec.keys())[-1])

    timestep_subspec = random_timestep_spec(
        observation_spec=observation_subspec,
        discount_spec=discount_spec,
        reward_spec=reward_spec)
    timestep_fullspec = random_timestep_spec(
        observation_spec=observation_fullspec,
        discount_spec=discount_spec,
        reward_spec=reward_spec)

    try:
      spec_utils.ensure_spec_compatibility(timestep_subspec, timestep_fullspec)
      self.fail('Validation failure expected.')
    except KeyError as unused_but_expected:
      del unused_but_expected

  def test_WrongObservation(self):
    reward_spec = random_reward_spec()
    discount_spec = random_discount_spec()
    observation_subspec = random_observation_spec()
    observation_fullspec = copy.deepcopy(observation_subspec)

    # Modify the first op:
    obs0 = observation_fullspec[list(observation_fullspec.keys())[0]]
    new_shape = tuple((d + 1 for d in obs0.shape))
    new_obs = random_array_spec(
        shape=new_shape, name=obs0.name, dtype=obs0.dtype)

    observation_fullspec[obs0.name] = new_obs

    timestep_subspec = random_timestep_spec(
        observation_spec=observation_subspec,
        discount_spec=discount_spec,
        reward_spec=reward_spec)
    timestep_fullspec = random_timestep_spec(
        observation_spec=observation_fullspec,
        discount_spec=discount_spec,
        reward_spec=reward_spec)

    try:
      spec_utils.ensure_spec_compatibility(timestep_subspec, timestep_fullspec)
      self.fail('Validation failure expected.')
    except ValueError as unused_but_expected:
      del unused_but_expected


class CastTest(absltest.TestCase):

  def _assert_array(self, actual: np.ndarray, expected: np.ndarray):
    self.assertEqual(actual.dtype, expected.dtype)
    np.testing.assert_almost_equal(actual, expected)

  def test_CastArrayToFloat64(self):
    spec = specs.Array(shape=(2,), dtype=np.float64, name='64bitspec')
    expected = np.array([1, 2], dtype=np.float64)

    input_float = np.array([1, 2], dtype=float)
    input_float32 = np.array([1, 2], dtype=np.float32)
    input_float64 = np.array([1, 2], dtype=np.float64)

    self._assert_array(spec_utils.cast(spec, input_float), expected)
    self._assert_array(spec_utils.cast(spec, input_float32), expected)
    self._assert_array(spec_utils.cast(spec, input_float64), expected)

  def test_CastArrayToFloat32(self):
    spec = specs.Array(shape=(2,), dtype=np.float32, name='64bitspec')
    expected = np.array([1, 2], dtype=np.float32)

    input_float = np.array([1, 2], dtype=float)
    input_float32 = np.array([1, 2], dtype=np.float32)
    input_float64 = np.array([1, 2], dtype=np.float64)

    self._assert_array(spec_utils.cast(spec, input_float), expected)
    self._assert_array(spec_utils.cast(spec, input_float32), expected)
    self._assert_array(spec_utils.cast(spec, input_float64), expected)

  def test_CastArrayToInt32(self):
    spec = specs.Array(shape=(2,), dtype=np.int32, name='32bitspec')
    expected = np.array([1, 2], dtype=np.int32)

    input_int = np.array([1, 2], dtype=int)
    input_int32 = np.array([1, 2], dtype=np.int32)
    input_int64 = np.array([1, 2], dtype=np.int64)

    self._assert_array(spec_utils.cast(spec, input_int), expected)
    self._assert_array(spec_utils.cast(spec, input_int32), expected)
    self._assert_array(spec_utils.cast(spec, input_int64), expected)

  def test_CastScalarToFloat64(self):
    spec = specs.Array(shape=(), dtype=np.float64)

    def check_value(value):
      assert type(value) == np.float64  # pylint: disable=unidiomatic-typecheck

    check_value(spec_utils.cast(spec, float(1.2)))
    check_value(spec_utils.cast(spec, np.float32(1.2)))
    check_value(spec_utils.cast(spec, np.float64(1.2)))
    check_value(spec_utils.cast(spec, float('nan')))
    check_value(spec_utils.cast(spec, np.float32('nan')))
    check_value(spec_utils.cast(spec, np.float64('nan')))

  def test_CastScalarToFloat32(self):
    spec = specs.Array(shape=(), dtype=np.float32)

    def check_value(value):
      assert type(value) == np.float32  # pylint: disable=unidiomatic-typecheck

    check_value(spec_utils.cast(spec, float(1.2)))
    check_value(spec_utils.cast(spec, np.float32(1.2)))
    check_value(spec_utils.cast(spec, np.float64(1.2)))
    check_value(spec_utils.cast(spec, float('nan')))
    check_value(spec_utils.cast(spec, np.float32('nan')))
    check_value(spec_utils.cast(spec, np.float64('nan')))

  def test_CastScalarToInt32(self):
    spec = specs.Array(shape=(), dtype=np.int32)

    def check_value(value):
      assert type(value) == np.int32  # pylint: disable=unidiomatic-typecheck

    check_value(spec_utils.cast(spec, int(12)))
    check_value(spec_utils.cast(spec, np.int32(12)))
    check_value(spec_utils.cast(spec, np.int64(12)))


class MergeTest(absltest.TestCase):

  def test_MergePrimitives(self):
    # Can only merge a value with None, cannot merge two primitives
    # Well, we /could/ do, but that would require an aggregation function
    # E.g. SUM, MIN, MAX, MEAN etc.
    val1 = np.asarray([1, 2, np.nan, np.nan, 5, 6])
    val2 = np.asarray([np.nan, np.nan, 3, 4, np.nan, np.nan])

    testing_functions.assert_value(spec_utils.merge_primitives([val1]), val1)

    testing_functions.assert_value(
        np.asarray([1, 2, 3, 4, 5, 6]),
        spec_utils.merge_primitives([val1, val2]))

    try:
      spec_utils.merge_primitives(
          [np.asarray([np.nan, 1, 2]),
           np.asarray([np.nan, np.nan, 2])])
      self.fail('Exception expected')
    except ValueError as unused_but_expected:
      pass

  def test_MergeSpecs(self):
    with self.subTest('with_same_dtypes'):
      spec1 = specs.BoundedArray(shape=(3,), dtype=np.int32,
                                 minimum=np.zeros((3,), np.int32),
                                 maximum=np.ones((3,), np.int32))
      spec2 = specs.BoundedArray(shape=(2,), dtype=np.int32,
                                 minimum=np.ones((2,), np.int32),
                                 maximum=np.ones((2,), np.int32) * 2)
      expected_spec = specs.BoundedArray(
          shape=(5,), dtype=np.int32,
          minimum=np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
          maximum=np.asarray([1, 1, 1, 2, 2], dtype=np.int32))
      self.assertEqual(spec_utils.merge_specs([spec1, spec2]), expected_spec)
    with self.subTest('with_different_dtypes'):
      spec1 = specs.BoundedArray(shape=(1,), dtype=np.int32,
                                 minimum=np.zeros((1,), np.int32),
                                 maximum=np.ones((1,), np.int32))
      spec2 = specs.BoundedArray(shape=(1,), dtype=np.float32,
                                 minimum=np.ones((1,), np.float32),
                                 maximum=np.ones((1,), np.float32) * 2)
      # Defaults to float64 if there are no matching dtypes.
      expected_spec = specs.BoundedArray(
          shape=(2,), dtype=np.float64, minimum=np.asarray([0., 1.]),
          maximum=np.asarray([1., 2.]))
      self.assertEqual(spec_utils.merge_specs([spec1, spec2]), expected_spec)
    with self.subTest('skips_empty_specs'):
      spec1 = specs.BoundedArray(shape=(1,), dtype=np.int32,
                                 minimum=np.zeros((1,), np.int32),
                                 maximum=np.ones((1,), np.int32))
      empty_spec = specs.BoundedArray(shape=(), dtype=np.int32,
                                      minimum=0, maximum=0)
      self.assertEqual(spec_utils.merge_specs([spec1, empty_spec]), spec1)
    with self.subTest('returns_empty_spec_if_no_inputs'):
      empty_spec = specs.BoundedArray(shape=(0,), dtype=np.float64,
                                      minimum=[], maximum=[])
      self.assertEqual(spec_utils.merge_specs([]), empty_spec)


class ShrinkToFitTest(absltest.TestCase):

  def test_primitive(self):
    spec = specs.BoundedArray(
        shape=(3,),
        dtype=float,
        minimum=[0.0, 0.0, 0.0],
        maximum=[20.0, 100.0, 20.0])

    val1 = np.asarray([21.0, 5.0, 21.0])  # over-max, under-min, over-max
    factor1 = 20.0 / 21.0
    expected1 = np.asarray([20.0, 5.0 * factor1, 20.0])

    testing_functions.assert_value(
        spec_utils.shrink_to_fit(val1, spec), expected1)

    val2 = np.asarray([1.0, 200.0, 21.0])  # ok, over-max, over-max
    # factor2 = 0.5  # 100 / 200
    expected2 = np.asarray([0.5, 100.0, 10.5])

    testing_functions.assert_value(
        spec_utils.shrink_to_fit(val2, spec), expected2)

  def test_zero_inside_bounds(self):
    spec = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=[-10.0], maximum=[10.0])

    val1 = np.asarray([0.0])
    expected1 = np.copy(val1)

    testing_functions.assert_value(
        spec_utils.shrink_to_fit(val1, spec), expected1)

  def test_negative(self):
    spec = specs.BoundedArray(
        shape=(3,),
        dtype=float,
        minimum=[-10.0, 0.0, 0.0],
        maximum=[10.0, 100.0, 20.0])

    val = np.asarray([-20.0, 50.0, 10.0])
    # Values are halved to make -20 -> -10.
    expected = np.asarray([-10.0, 25.0, 5.0])

    testing_functions.assert_value(
        spec_utils.shrink_to_fit(val, spec), expected)


class ClipTest(absltest.TestCase):

  def test_primitive(self):
    spec = specs.BoundedArray(
        shape=(3,),
        dtype=float,
        minimum=[0.0, 10.0, 20.0],
        maximum=[20.0, 100.0, 20.0])

    val1 = np.asarray([21.0, 5.0, 21.0])  # over-max, under-min, over-max
    expected1 = np.asarray([20.0, 10.0, 20.0])
    testing_functions.assert_value(spec_utils.clip(val1, spec), expected1)

    val2 = np.asarray([1.0, 200.0, 21.0])  # ok, over-max, over-max
    expected2 = np.asarray([1.0, 100.0, 20.0])
    testing_functions.assert_value(spec_utils.clip(val2, spec), expected2)


class PrefixSlicerTest(absltest.TestCase):

  def test_ArmJoints(self):
    spec = specs.BoundedArray(
        shape=(3,),
        dtype=np.float32,
        minimum=np.asarray([1.0, 2.0, 3.0]),
        maximum=np.asarray([11.0, 22.0, 33.0]),
        name='name/sawyer/j0\tname/sawyer/j1\tname/sawyer/gripper')

    action_space = action_spaces.prefix_slicer(spec, prefix='.*/j[0-9]+$')

    # Verify the retuurned spec.
    expected_spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=np.asarray([1.0, 2.0]),
        maximum=np.asarray([11.0, 22.0]),
        name='name/sawyer/j0\tname/sawyer/j1')
    spec_utils.verify_specs_equal_bounded(expected_spec, action_space.spec())

    # Verify the returned action space.
    np.testing.assert_array_almost_equal(
        np.asarray([1.0, 2.0, np.nan]),
        action_space.project(np.asarray([1.0, 2.0])))

  def test_Gripper(self):
    spec = specs.BoundedArray(
        shape=(3,),
        dtype=np.float32,
        minimum=np.asarray([1.0, 2.0, 3.0]),
        maximum=np.asarray([11.0, 22.0, 33.0]),
        name='name/sawyer/j0\tname/sawyer/j1\tname/sawyer/gripper')

    action_space = action_spaces.prefix_slicer(spec, prefix='.*/gripper$')

    # Verify the retuurned spec.
    expected_spec = specs.BoundedArray(
        shape=(1,),
        dtype=np.float32,
        minimum=np.asarray([3.0]),
        maximum=np.asarray([33.0]),
        name='name/sawyer/gripper')
    spec_utils.verify_specs_equal_bounded(expected_spec, action_space.spec())

    # Verify the returned action space.
    np.testing.assert_array_almost_equal(
        np.asarray([np.nan, np.nan, 3.0]),
        action_space.project(np.asarray([3.0])))

  def test_NonContiguousMatches(self):
    spec = specs.BoundedArray(
        shape=(4,),
        dtype=np.float32,
        minimum=np.asarray([1.0, 2.0, 3.0, 4.0]),
        maximum=np.asarray([11.0, 22.0, 33.0, 44.0]),
        name='h1\tm1\th2\tm2')

    action_space = action_spaces.prefix_slicer(spec, prefix='h.$')

    expected_spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=np.asarray([1.0, 3.0]),
        maximum=np.asarray([11.0, 33.0]),
        name='h1\th2')
    spec_utils.verify_specs_equal_bounded(expected_spec, action_space.spec())

    np.testing.assert_array_almost_equal(
        np.asarray([1.0, np.nan, 2.0, np.nan]),
        action_space.project(np.asarray([1.0, 2.0])))

  def test_EmptySpec(self):
    spec = specs.BoundedArray(
        shape=(0,),
        dtype=np.float32,
        minimum=np.asarray([]),
        maximum=np.asarray([]),
        name='')

    action_space = action_spaces.prefix_slicer(spec, prefix='a')

    spec_utils.verify_specs_equal_bounded(spec, action_space.spec())
    np.testing.assert_array_almost_equal(
        np.asarray([]), action_space.project(np.asarray([], dtype=np.float32)))

  def test_AllMatch(self):
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=np.asarray([1.0, 2.0]),
        maximum=np.asarray([11.0, 22.0]),
        name='h1\th2')

    action_space = action_spaces.prefix_slicer(spec, prefix='h.$')
    spec_utils.verify_specs_equal_bounded(spec, action_space.spec())

    np.testing.assert_array_almost_equal(
        np.asarray([1.0, 2.0]), action_space.project(np.asarray([1.0, 2.0])))

  def test_NoneMatch(self):
    spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=np.asarray([1.0, 2.0]),
        maximum=np.asarray([11.0, 22.0]),
        name='h1\th2')

    action_space = action_spaces.prefix_slicer(spec, prefix='m.$')
    expected_spec = specs.BoundedArray(
        shape=(0,),
        dtype=np.float32,
        minimum=np.asarray([]),
        maximum=np.asarray([]),
        name='')

    spec_utils.verify_specs_equal_bounded(expected_spec, action_space.spec())

    np.testing.assert_array_almost_equal(
        np.asarray([np.nan, np.nan]), action_space.project(np.asarray([])))

  def test_Defaulting(self):
    spec = specs.BoundedArray(
        shape=(4,),
        dtype=np.float32,
        minimum=np.asarray([1.0, 2.0, 3.0, 4.0]),
        maximum=np.asarray([11.0, 22.0, 33.0, 44.0]),
        name='h1\tm1\th2\tm2')

    action_space = action_spaces.prefix_slicer(
        spec, prefix='h.$', default_value=99.0)

    np.testing.assert_array_almost_equal(
        np.asarray([1.0, 99.0, 2.0, 99.0]),
        action_space.project(np.asarray([1.0, 2.0])))


class TimeStepSpecTest(parameterized.TestCase):

  def test_EqualSpecs(self):
    array = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=np.asarray([-1.0, -2.0]),
        maximum=np.asarray([1.0, 2.0]),
        name='bounded_array')
    obs_spec = {'obs1': array}
    spec = spec_utils.TimeStepSpec(
        observation_spec=obs_spec, reward_spec=array, discount_spec=array)
    self.assertEqual(spec, copy.deepcopy(spec))

  def test_NonEqualSpecs(self):
    array = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=np.asarray([-1.0, -2.0]),
        maximum=np.asarray([1.0, 2.0]),
        name='bounded_array')
    obs_spec = {'obs1': array}
    spec1 = spec_utils.TimeStepSpec(
        observation_spec=obs_spec, reward_spec=array, discount_spec=array)
    array2 = specs.BoundedArray(
        shape=(2,),
        dtype=np.float32,
        minimum=np.asarray([-3.0, -4.0]),
        maximum=np.asarray([3.0, 4.0]),
        name='bounded_array2')
    obs_spec2 = {'obs2': array2}
    spec2 = spec_utils.TimeStepSpec(
        observation_spec=obs_spec2, reward_spec=array2, discount_spec=array2)
    self.assertNotEqual(spec1, spec2)

  @parameterized.parameters(float, np.int8)
  def test_minimum(self, dtype):
    array = specs.BoundedArray(
        shape=(2,),
        dtype=dtype,
        minimum=np.asarray([-1.0, -2.0]),
        maximum=np.asarray([1.0, 2.0]),
        name='bounded_array')
    obs_spec = {'obs1': array}
    spec = spec_utils.TimeStepSpec(
        observation_spec=obs_spec, reward_spec=array, discount_spec=array)

    expected_minimum_obs = {'obs1': np.array([-1.0, -2.0], dtype=dtype)}
    expected_minimum_reward = np.array([-1.0, -2.0], dtype=dtype)
    expected_minimum_discount = np.array([-1.0, -2.0], dtype=dtype)

    minimum_timestep = spec.minimum()

    if issubclass(dtype, np.inexact):
      assert_fn = np.testing.assert_almost_equal
    else:
      assert_fn = np.testing.assert_equal
    assert_fn(minimum_timestep.reward, expected_minimum_reward)
    assert_fn(minimum_timestep.discount, expected_minimum_discount)

    self.assertEqual(set(minimum_timestep.observation.keys()),
                     set(expected_minimum_obs.keys()))

    for key in expected_minimum_obs:
      assert_fn(minimum_timestep.observation[key],
                expected_minimum_obs[key])

  @parameterized.parameters(float, np.int8)
  def test_maximum(self, dtype):
    array = specs.BoundedArray(
        shape=(2,),
        dtype=dtype,
        minimum=np.asarray([-1.0, -2.0]),
        maximum=np.asarray([1.0, 2.0]),
        name='bounded_array')
    obs_spec = {'obs1': array}
    spec = spec_utils.TimeStepSpec(
        observation_spec=obs_spec, reward_spec=array, discount_spec=array)

    expected_maximum_obs = {'obs1': np.array([1.0, 2.0], dtype=dtype)}
    expected_maximum_reward = np.array([1.0, 2.0], dtype=dtype)
    expected_maximum_discount = np.array([1.0, 2.0], dtype=dtype)

    maximum_timestep = spec.maximum()

    if issubclass(dtype, np.inexact):
      assert_fn = np.testing.assert_almost_equal
    else:
      assert_fn = np.testing.assert_equal
    assert_fn(maximum_timestep.reward, expected_maximum_reward)
    assert_fn(maximum_timestep.discount, expected_maximum_discount)

    self.assertEqual(set(maximum_timestep.observation.keys()),
                     set(expected_maximum_obs.keys()))

    for key in expected_maximum_obs:
      assert_fn(maximum_timestep.observation[key],
                expected_maximum_obs[key])


if __name__ == '__main__':
  absltest.main()
