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

"""Tests for action_spaces."""

from typing import Text

from absl import flags
from absl.testing import absltest
from dm_robotics.agentflow import action_spaces
from dm_robotics.agentflow import testing_functions
import numpy as np

FLAGS = flags.FLAGS


class CompositeActionSpaceTest(absltest.TestCase):

  def _create_spec(self, *names: Text):
    name = "\t".join(names)
    return testing_functions.random_array_spec(shape=(len(names),), name=name)

  def test_single_action_space(self):
    outer_spec = self._create_spec("a_1", "a_2", "b_1")
    primitive = action_spaces.prefix_slicer(outer_spec, "a_")
    composite = action_spaces.CompositeActionSpace([primitive])

    self.assertEqual(primitive.spec(), composite.spec())

    value = testing_functions.valid_value(primitive.spec())
    np.testing.assert_array_almost_equal(
        primitive.project(value), composite.project(value))

  def test_adjacent_action_spaces_full(self):
    # Testing two action spaces that project to values that are adjacent in
    # the output value and that cover the entire output value
    outer_spec = self._create_spec("a_1", "a_2", "b_1", "b_2")
    primitive_1 = action_spaces.prefix_slicer(outer_spec, "a_")
    primitive_2 = action_spaces.prefix_slicer(outer_spec, "b_")
    composite = action_spaces.CompositeActionSpace([primitive_1, primitive_2])

    self.assertEqual(outer_spec, composite.spec())

    # The composite spec is the two sub specs in order, so project should be
    # an identity - the output should be the input.
    value = testing_functions.valid_value(composite.spec())
    np.testing.assert_array_almost_equal(composite.project(value), value)

  def test_adjacent_action_spaces_full_reverse(self):
    # As above, but the primitive action spaces are in reverse order, so the
    # composite spec should not match and should rearrange values in project.
    outer_spec = self._create_spec("a_1", "a_2", "b_1", "b_2")
    primitive_1 = action_spaces.prefix_slicer(outer_spec, "a_")
    primitive_2 = action_spaces.prefix_slicer(outer_spec, "b_")
    composite = action_spaces.CompositeActionSpace([primitive_2, primitive_1])

    self.assertEqual(outer_spec.name, "\t".join(["a_1", "a_2", "b_1", "b_2"]))
    self.assertEqual(composite.spec().name,
                     "\t".join(["b_1", "b_2", "a_1", "a_2"]))

    input_value = testing_functions.valid_value(composite.spec())
    expected_output_value = np.concatenate([input_value[2:], input_value[:2]])

    np.testing.assert_array_almost_equal(
        composite.project(input_value), expected_output_value)

  def test_adjacent_action_spaces_partial(self):
    # Testing two action spaces that project to values that are adjacent in
    # the output value but do not cover the entire output value
    outer_spec = self._create_spec("a_1", "a_2", "b_1", "b_2", "c_1")
    primitive_1 = action_spaces.prefix_slicer(outer_spec, "a_")
    primitive_2 = action_spaces.prefix_slicer(outer_spec, "b_")
    composite = action_spaces.CompositeActionSpace([primitive_1, primitive_2])

    input_value = testing_functions.valid_value(composite.spec())
    expected_output_value = np.concatenate([input_value,
                                            np.asarray([np.nan])
                                           ]).astype(input_value.dtype)

    np.testing.assert_array_almost_equal(
        composite.project(input_value), expected_output_value)

  def test_separated_action_spaces(self):
    # like test_adjacent_action_spaces_partial, but the gap is in the middle.
    outer_spec = self._create_spec("a_1", "a_2", "c_1", "b_1", "b_2")
    primitive_1 = action_spaces.prefix_slicer(outer_spec, "a_")
    primitive_2 = action_spaces.prefix_slicer(outer_spec, "b_")
    composite = action_spaces.CompositeActionSpace([primitive_1, primitive_2])

    input_value = testing_functions.valid_value(composite.spec())
    expected_output_value = np.concatenate(
        [input_value[:2], [np.nan], input_value[2:]]).astype(input_value.dtype)

    np.testing.assert_array_almost_equal(
        composite.project(input_value), expected_output_value)

  def test_composite_action_spaces(self):
    # Compose composite action spaces.
    name = "\t".join(["a1", "a2", "b1", "b2", "c1", "c2"] +
                     ["d1", "d2", "e1", "e2", "f1", "f2"])
    outer_spec = testing_functions.random_array_spec(shape=(12,), name=name)

    # Make specs for [a, c, d, f], I.e. b and e are missing.
    primitive_1 = action_spaces.prefix_slicer(outer_spec, "a")
    primitive_2 = action_spaces.prefix_slicer(outer_spec, "c")
    primitive_3 = action_spaces.prefix_slicer(outer_spec, "d")
    primitive_4 = action_spaces.prefix_slicer(outer_spec, "f")

    # Make specs for [a, c] and [d, f]
    composite_1 = action_spaces.CompositeActionSpace([primitive_1, primitive_2])
    composite_2 = action_spaces.CompositeActionSpace([primitive_3, primitive_4])

    composite = action_spaces.CompositeActionSpace([composite_1, composite_2])

    input_value = testing_functions.valid_value(composite.spec())
    two_nans = [np.nan, np.nan]
    expected_1 = [input_value[0:2], two_nans, input_value[2:4]]
    expected_2 = [input_value[4:6], two_nans, input_value[6:8]]
    expected_output_value = np.concatenate(expected_1 + expected_2)
    expected_output_value = expected_output_value.astype(input_value.dtype)

    np.testing.assert_array_almost_equal(
        composite.project(input_value), expected_output_value)

  def test_zero_action_spaces(self):
    composite = action_spaces.CompositeActionSpace([])

    self.assertEqual(composite.spec().shape, (0,))
    composite.spec().validate(np.asarray([], dtype=np.float32))

  def test_zero_sized_space(self):
    # Testing two action spaces that project to values that are adjacent in
    # the output value and that cover the entire output value
    outer_spec = self._create_spec("a_1", "a_2", "b_1", "b_2")
    primitive_1 = action_spaces.prefix_slicer(outer_spec, "a_")
    primitive_2 = action_spaces.prefix_slicer(outer_spec, "EMPTY")
    primitive_3 = action_spaces.prefix_slicer(outer_spec, "b_")
    composite = action_spaces.CompositeActionSpace(
        [primitive_1, primitive_2, primitive_3])

    self.assertEqual(outer_spec, composite.spec())
    self.assertEqual(outer_spec.name, composite.spec().name)

    # The composite spec is the two sub specs in order, so project should be
    # an identity - the output should be the input.
    value = testing_functions.valid_value(composite.spec())
    np.testing.assert_array_almost_equal(composite.project(value), value)

  def test_with_fixed_space(self):
    # Testing two action spaces that project to values that are adjacent in
    # the output value and that cover the entire output value
    outer_spec = self._create_spec("a_1", "a_2", "b_1", "b_2")
    primitive_1 = action_spaces.prefix_slicer(outer_spec, "a_")
    primitive_2 = action_spaces.prefix_slicer(outer_spec, "b_")

    value_2 = testing_functions.valid_value(primitive_2.spec())
    fixed_2 = action_spaces.FixedActionSpace(primitive_2, value_2)
    composite = action_spaces.CompositeActionSpace([primitive_1, fixed_2])

    self.assertEqual(primitive_1.spec(), composite.spec())

    input_value = testing_functions.valid_value(composite.spec())
    output_value = composite.project(input_value)

    np.testing.assert_array_almost_equal(output_value[:2], input_value)
    np.testing.assert_array_almost_equal(output_value[2:], value_2)


if __name__ == "__main__":
  absltest.main()
