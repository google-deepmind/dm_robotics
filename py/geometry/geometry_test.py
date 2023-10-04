# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""Tests for geometry."""

import itertools
import operator

from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.geometry import geometry
from dm_robotics.transformations import transformations as tr
import numpy as np

_N_RANDOM = 100
random_state = np.random.RandomState(1)


class GeometryTest(parameterized.TestCase):

  def test_pos_mul_random(self):
    for _ in range(_N_RANDOM):
      pose1 = geometry.Pose.from_poseuler(
          random_state.uniform(size=6), ordering='XYZ')
      pose2 = geometry.Pose.from_poseuler(
          random_state.uniform(size=6), ordering='XYZ')

      hmat_world_pose1 = pose1.hmat
      hmat_pose1_pose2 = pose2.hmat
      hmat_world_pose2_true = hmat_world_pose1.dot(hmat_pose1_pose2)
      hmat_world_pose2_test = pose1.mul(pose2).hmat
      np.testing.assert_array_almost_equal(hmat_world_pose2_true,
                                           hmat_world_pose2_test)

  def test_pose_inv_random(self):
    for _ in range(_N_RANDOM):
      pose = geometry.Pose.from_poseuler(
          random_state.uniform(size=6), ordering='XYZ')
      np.testing.assert_array_almost_equal(
          tr.hmat_inv(pose.hmat),
          pose.inv().hmat)
      np.testing.assert_array_almost_equal(
          np.linalg.inv(pose.hmat),
          pose.inv().hmat)

  def test_pose_stamped_to_frame_simple(self):
    pose1 = geometry.PoseStamped(
        pose=geometry.Pose.from_poseuler([1, 2, 3, 0.1, 0.2, 0.3],
                                         ordering='XYZ'),
        frame=None)
    pose2 = geometry.PoseStamped(
        pose=geometry.Pose.from_poseuler([-0.5, -1, 2, -0.3, 0.2, -0.1],
                                         ordering='XYZ'),
        frame=None)

    hmat_world_pose1 = pose1.get_world_pose().hmat
    hmat_world_pose2 = pose2.get_world_pose().hmat
    np.testing.assert_array_almost_equal(hmat_world_pose1, pose1.pose.hmat)
    np.testing.assert_array_almost_equal(hmat_world_pose2, pose2.pose.hmat)

    pose2 = pose2.to_frame(pose1)
    hmat_pose1_pose2 = tr.hmat_inv(hmat_world_pose1).dot(hmat_world_pose2)
    np.testing.assert_array_almost_equal(pose2.pose.hmat, hmat_pose1_pose2)

  def test_pose_stamped_to_world_simple(self):
    pose1 = geometry.PoseStamped(
        pose=geometry.Pose.from_poseuler([1, 2, 3, 0.1, 0.2, 0.3],
                                         ordering='XYZ'),
        frame=None)
    pose2 = geometry.PoseStamped(
        pose=geometry.Pose.from_poseuler([-0.5, -1, 2, -0.3, 0.2, -0.1],
                                         ordering='XYZ'),
        frame=pose1)

    hmat_world_pose1 = pose1.pose.hmat
    hmat_pose1_pose2 = pose2.pose.hmat
    hmat_world_pose2_true = hmat_world_pose1.dot(hmat_pose1_pose2)
    pose2 = pose2.to_world()
    self.assertIsNone(pose2.frame)
    hmat_world_pose2_test = pose2.pose.hmat
    np.testing.assert_array_almost_equal(hmat_world_pose2_true,
                                         hmat_world_pose2_test)

  def test_world_pose_with_hybrid_frame(self):
    pose1 = geometry.Pose.from_poseuler(
        [0.1, 0.2, 0.3, np.pi, np.pi / 2, -np.pi])
    pose2 = geometry.Pose.from_poseuler([0.1, 0.2, 0.3, 0, np.pi, np.pi / 2])

    non_hybrid = geometry.PoseStamped(
        pose=pose1, frame=geometry.PoseStamped(pose=pose2))

    identity_quaternion = [1, 0, 0, 0]
    hybrid = geometry.PoseStamped(
        pose=pose1,
        frame=geometry.HybridPoseStamped(
            pose=geometry.Pose.from_poseuler(
                np.hstack((pose2.position, identity_quaternion))),
            quaternion_override=geometry.PoseStamped(pose2, None)))

    non_hybrid_world_pose = non_hybrid.get_world_pose()
    hybrid_world_pose = hybrid.get_world_pose()
    self.assertEqual(non_hybrid_world_pose, hybrid_world_pose)

  @parameterized.named_parameters(
      ('Vec6', geometry.Vec6),
      ('Twist', geometry.Twist),
      ('Wrench', geometry.Wrench),
      ('Accel', geometry.Accel),
  )
  def test_default_construction_of_base_types(self, base_cls):
    value = base_cls()
    np.testing.assert_array_almost_equal(np.zeros(6), value.data)

  @parameterized.named_parameters(
      ('VectorStamped', geometry.VectorStamped),
      ('TwistStamped', geometry.TwistStamped),
      ('WrenchStamped', geometry.WrenchStamped),
      ('AccelStamped', geometry.AccelStamped),
  )
  def test_default_construction_of_stamped_tyes(self, stamped_cls):
    value = stamped_cls(None, None)
    np.testing.assert_array_almost_equal(np.zeros(6), value.data.data)


BASE_TYPE_SPECS = (('Accel', geometry.Accel, {
    'full': 6,
    'linear': 3,
    'angular': 3
}), ('Twist', geometry.Twist, {
    'full': 6,
    'linear': 3,
    'angular': 3
}), ('Wrench', geometry.Wrench, {
    'full': 6,
    'force': 3,
    'torque': 3
}))


class BaseImmutabiliyTest(parameterized.TestCase):
  """Test of immutability for Accel, Twist, Wrench, Vector and Pose."""

  def _copy_properties(self, value, property_map):
    value_map = {}
    for name in property_map:
      value_map[name] = np.copy(getattr(value, name))
    return value_map

  def assertPropertiesEqual(self, expected, actual):
    self.assertEqual(list(expected.keys()), list(actual.keys()))
    for key in expected.keys():
      try:
        np.testing.assert_array_almost_equal(expected[key], actual[key])
      except AssertionError as failure:
        failure.args += (key)
        raise

  @parameterized.named_parameters(*BASE_TYPE_SPECS)
  def test_construction(self, geometry_type, property_map):
    # We should not be able to modify the object through its constructor param.

    # Test with a numpy array (must be copied).
    input_array = np.asarray(list(range(6)))
    value = geometry_type(input_array)
    initial_values = self._copy_properties(value, property_map)

    input_array[:] = list(range(10, 16, 1))
    current_values = self._copy_properties(value, property_map)

    self.assertPropertiesEqual(initial_values, current_values)

    # Test with a list (a new numpy array will be created).
    input_list = list(range(6))
    value = geometry_type(input_list)
    initial_values = self._copy_properties(value, property_map)

    input_list[:] = list(range(10, 16, 1))
    current_values = self._copy_properties(value, property_map)

    self.assertPropertiesEqual(initial_values, current_values)

  @parameterized.named_parameters(*BASE_TYPE_SPECS)
  def test_property_cannot_set(self, geometry_type, property_map):
    value = geometry_type(list(range(6)))

    for property_name, property_size in property_map.items():
      with self.assertRaises(AttributeError):
        setattr(value, property_name, list(range(property_size)))

  @parameterized.named_parameters(*BASE_TYPE_SPECS)
  def test_property_cannot_setitem(self, geometry_type, property_map):
    value = geometry_type(list(range(6)))

    for property_name, property_size in property_map.items():
      with self.assertRaises(
          ValueError,
          msg='{}.{}[:] allowed'.format(geometry_type,
                                        property_name)) as expected:
        property_value = getattr(value, property_name)
        property_value[:] = list(range(property_size))
      self.assertIn('read-only', str(expected.exception))

  def test_pose_construction(self):
    # Test that the arguments we give to the Pose constructor do not permit
    # modifications to the Pose instance value.
    property_map = {'position': 3, 'quaternion': 4}

    # Test with the input being a numpy array.
    position = np.asarray(list(range(3)))
    quaternion = np.asarray(list(range(4)))

    pose = geometry.Pose(position, quaternion)
    initial_values = self._copy_properties(pose, property_map)
    position[:] = list(range(10, 13, 1))
    quaternion[:] = list(range(10, 14, 1))
    current_values = self._copy_properties(pose, property_map)

    self.assertPropertiesEqual(initial_values, current_values)

    # Test with the input being a list.
    position = list(range(3))
    quaternion = list(range(4))

    pose = geometry.Pose(position, quaternion)
    initial_values = self._copy_properties(pose, property_map)
    position[:] = list(range(10, 13, 1))
    quaternion[:] = list(range(10, 14, 1))
    current_values = self._copy_properties(pose, property_map)

    self.assertPropertiesEqual(initial_values, current_values)

  def test_pose_property_cannot_set(self):
    pose = geometry.Pose(list(range(3)), list(range(4)))
    with self.assertRaisesRegex(
        AttributeError, "can't set attribute|object has no setter"
    ):
      pose.position = list(range(10, 13, 1))

    with self.assertRaisesRegex(
        AttributeError, "can't set attribute|object has no setter"
    ):
      pose.quaternion = list(range(10, 14, 1))

  def test_pose_property_cannot_setitem(self):
    pose = geometry.Pose(list(range(3)), list(range(4)))
    with self.assertRaises(ValueError) as expected:
      pose.position[:] = list(range(10, 13, 1))
    self.assertIn('read-only', str(expected.exception))

    with self.assertRaises(ValueError) as expected:
      pose.quaternion[:] = list(range(10, 14, 1))
    self.assertIn('read-only', str(expected.exception))

  def test_pose_with_position(self):
    # Check that the pose from with_position has a new position.
    first_pose = geometry.Pose(list(range(3)), list(range(4)))
    second_pose = first_pose.with_position(list(range(10, 13, 1)))

    self.assertTrue(np.array_equal(first_pose.position, list(range(3))))
    self.assertTrue(
        np.array_equal(second_pose.position, list(range(10, 13, 1))))
    self.assertTrue(
        np.array_equal(first_pose.quaternion, second_pose.quaternion))

  def test_pose_with_quaternion(self):
    # Check that the pose from with_quaternion has a new quaternion.
    first_pose = geometry.Pose(list(range(3)), list(range(4)))
    second_pose = first_pose.with_quaternion(list(range(10, 14, 1)))

    self.assertTrue(np.array_equal(first_pose.quaternion, list(range(4))))
    self.assertTrue(
        np.array_equal(second_pose.quaternion, list(range(10, 14, 1))))
    self.assertTrue(np.array_equal(first_pose.position, second_pose.position))

  def test_vec6_construction(self):
    # Test copy of numpy object.
    input_array = np.asarray(list(range(6)))
    vec = geometry.Vec6(input_array)
    input_array[0] = 1
    self.assertSequenceEqual(list(vec.data), list(range(6)))

    input_list = list(range(6))
    vec = geometry.Vec6(input_list)
    input_list[0] = 1
    self.assertSequenceEqual(list(vec.data), list(range(6)))

  def test_vec6_cannot_setitem(self):
    vec = geometry.Vec6(list(range(6)))

    # Test we can __getitem__
    for i in range(6):
      self.assertEqual(vec[i], i)

    # but not __setitem__
    with self.assertRaises(TypeError) as expected:
      vec[0] = 1
    self.assertIn('does not support item assignment', str(expected.exception))

  def test_vec6_cannot_setitem_on_full(self):
    vec = geometry.Vec6(list(range(6)))

    # Test we can __getitem__
    for i in range(6):
      self.assertEqual(float(vec.data[i]), float(i))

    # but not __setitem__
    with self.assertRaises(ValueError) as expected:
      vec.data[0] = 1.0
    self.assertIn('read-only', str(expected.exception))

  @parameterized.named_parameters(
      ('Accel', geometry.Accel(list(range(6)))),
      ('Twist', geometry.Twist(list(range(6)))),
      ('Wrench', geometry.Wrench(list(range(6)))),
      ('Pose', geometry.Pose(list(range(3)), list(range(4)))),
      ('Vec6', geometry.Vec6(list(range(6)))))
  def test_no_dict(self, obj):
    # __dict__ allows us to add arbitrary attributes to objects, which we don't
    # want for immutable types.
    self.assertFalse(hasattr(obj, '__dict__'))

  # The immutability of the stamped types is assured by:
  # 1: Knowing that all attributes of these types are themselves immutable,
  # 2: Public properties being read-only, returning immutable objects.
  def test_accel_stamped_immutable(self):
    accel_stamped = geometry.AccelStamped(list(range(6)), None)
    with self.assertRaises(AttributeError):
      accel_stamped.accel = geometry.Accel(list(range(1, 7)))

    with self.assertRaises(AttributeError):
      accel_stamped.frame = 'Not allowed'

  def test_pose_stamped_immutable(self):
    pose_stamped = geometry.PoseStamped(geometry.Pose(), None)
    with self.assertRaises(AttributeError):
      pose_stamped.pose = geometry.Pose(list(range(3)), list(range(4)))

    with self.assertRaises(AttributeError):
      pose_stamped.frame = 'Not allowed'

  def test_twist_stamped_immutable(self):
    twist = geometry.Twist(list(range(6)))
    twist_stamped = geometry.TwistStamped(twist, None)
    with self.assertRaises(AttributeError):
      twist_stamped.twist = geometry.Twist(list(range(6)))

    with self.assertRaises(AttributeError):
      twist_stamped.frame = 'Not allowed'

  def test_vector_stamped_immutable(self):
    vector = geometry.Vec6(list(range(6)))
    vector_stamped = geometry.VectorStamped(vector, None)
    with self.assertRaises(AttributeError):
      vector_stamped.vector = geometry.Vec6(list(range(6)))

    with self.assertRaises(AttributeError):
      vector_stamped.frame = 'Not allowed'

  def test_wrench_stamped_immutable(self):
    wrench = geometry.Wrench(list(range(6)))
    wrench_stamped = geometry.WrenchStamped(wrench, None)
    with self.assertRaises(AttributeError):
      wrench_stamped.wrench = geometry.Wrench(list(range(6)))

    with self.assertRaises(AttributeError):
      wrench_stamped.frame = 'Not allowed'


class VecSubtype(geometry.Vec6):
  pass


class FancyGains(geometry.Vec6):

  def __mul__(self, other):
    return other.__rmul__(self)

  def __add__(self, other):
    return other.__radd__(self)


class GeometryArithmeticTest(parameterized.TestCase):

  wrench_1_6 = geometry.Wrench(list(range(1, 7)))
  twist_1_6 = geometry.Twist(list(range(1, 7)))
  accel_1_6 = geometry.Accel(list(range(1, 7)))
  vec_10_60 = geometry.Vec6(list(range(10, 70, 10)))

  def test_vec6_scalar_addition_and_subtraction(self):
    target = geometry.Vec6(list(range(0, 6)))
    result = target + 1
    self.assertSequenceEqual(list(target.data), list(range(0, 6)))
    self.assertSequenceEqual(list(result.data), list(range(1, 7)))

    result -= 1
    self.assertSequenceEqual(list(target.data), list(range(0, 6)))
    self.assertSequenceEqual(list(result.data), list(range(0, 6)))

    result = target - 1
    self.assertSequenceEqual(list(target.data), list(range(0, 6)))
    self.assertSequenceEqual(list(result.data), list(range(-1, 5)))

    result += 1
    self.assertSequenceEqual(list(target.data), list(range(0, 6)))
    self.assertSequenceEqual(list(result.data), list(range(0, 6)))

  def test_vec6_scalar_multiplication_and_true_division(self):
    target = geometry.Vec6(np.arange(0.0, 6.0, 1.0))
    result = target * 2
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 12.0, 2.0))

    result /= 2
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0, 1.0))

    result = target / 2
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 3.0, 0.5))

    result *= 2
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))

  def test_vec6_vector_addition_and_subtraction(self):
    target = geometry.Vec6(np.arange(0.0, 6.0, 1.0))
    result = target + np.arange(0.0, 6.0, 1.0)
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 12.0, 2.0))

    result -= np.arange(0.0, 6.0, 1.0)
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0))
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0))

    result = target - np.arange(0.0, 6.0, 1.0)
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0))
    np.testing.assert_array_almost_equal(result.data, np.zeros(6))

    result += np.arange(0.0, 6.0, 1.0)
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0))
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0))

  def test_vec6_vector_multiplication_and_true_division(self):
    target = geometry.Vec6(np.arange(0.0, 6.0, 1.0))
    result = target * np.arange(0.0, 6.0, 1.0)
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(result.data,
                                         np.asarray([0, 1, 4, 9, 16, 25]))

    result /= np.arange(1.0, 7.0, 1.0)
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(
        result.data, np.asarray([0, 1 / 2, 4 / 3, 9 / 4, 16 / 5, 25 / 6]))

    result = target / np.arange(1.0, 7.0, 1.0)
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(
        result.data, np.asarray([0, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6]))

    result *= np.arange(1.0, 7.0, 1.0)
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))

  def test_vec6_broadcast_vector_addition_and_subtraction(self):
    target = geometry.Vec6(np.arange(0.0, 6.0, 1.0))
    result = target + np.asarray([2])
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(2.0, 8.0))

    result -= np.asarray([2])
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0))
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0))

    result = target - np.asarray([2])
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(-2.0, 4.0))

    result += np.asarray([2])
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0))
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0))

  def test_vec6_broadcast_vector_multiplication_and_true_division(self):
    target = geometry.Vec6(np.arange(0.0, 6.0, 1.0))
    result = target * np.asarray([2])
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 12.0, 2.0))

    result /= np.asarray([2])
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0, 1.0))

    result = target / np.asarray([2])
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 3.0, 0.5))

    result *= np.asarray([2])
    np.testing.assert_array_almost_equal(result.data, np.arange(0.0, 6.0, 1.0))
    np.testing.assert_array_almost_equal(target.data, np.arange(0.0, 6.0, 1.0))

  @parameterized.parameters(
      itertools.product(
          [wrench_1_6, twist_1_6, accel_1_6],  # Improve formatting.
          [vec_10_60],  # Improve formatting.
          [operator.mul, operator.add]))
  def test_commutative_operators(self, geometry_obj, vec, operation):
    # Test Vec6 OP some-geometry-type and some-geometry-type OP Vec6
    # This operation should return a some-geometry-type instance.

    vec_op_obj = operation(vec, geometry_obj)
    obj_op_vec = operation(geometry_obj, vec)

    # Assert commutativity.
    self.assertEqual(
        vec_op_obj,
        obj_op_vec,
        msg=(f'{vec} {operation} {geometry_obj} not commutative'))
    self.assertIsInstance(vec_op_obj, type(geometry_obj))
    self.assertIsInstance(obj_op_vec, type(geometry_obj))

    # Assert correctness.
    expected = type(geometry_obj)(operation(geometry_obj.data, vec.data))
    self.assertEqual(obj_op_vec, expected)

  @parameterized.parameters(
      itertools.product(
          [wrench_1_6, twist_1_6, accel_1_6],  # Formatting.
          [vec_10_60],  # Formatting.
          [operator.truediv, operator.sub]))
  def test_non_commutative_operators(self, geometry_obj, vec, operation):
    obj_op_vec = operation(geometry_obj, vec)
    self.assertIsInstance(obj_op_vec, type(geometry_obj))
    expected = type(geometry_obj)(operation(geometry_obj.data, vec.data))
    self.assertEqual(obj_op_vec, expected)

  @parameterized.parameters([wrench_1_6, twist_1_6, accel_1_6])
  def test_operators_with_vec6_subtype(self, geometry_obj):
    # Users may want their own subtypes of Vec6.
    # This tests that a basic version can be used for arithmetic.
    vec_sub_10_60 = VecSubtype(list(range(10, 70, 10)))
    obj_add = geometry_obj + vec_sub_10_60
    obj_sub = geometry_obj - vec_sub_10_60
    obj_mul = geometry_obj * vec_sub_10_60
    obj_div = geometry_obj / vec_sub_10_60

    self.assertIsInstance(obj_add, type(geometry_obj))
    self.assertIsInstance(obj_sub, type(geometry_obj))
    self.assertIsInstance(obj_mul, type(geometry_obj))
    self.assertIsInstance(obj_div, type(geometry_obj))

    expected_add = type(geometry_obj)(geometry_obj.data + vec_sub_10_60.data)
    expected_sub = type(geometry_obj)(geometry_obj.data - vec_sub_10_60.data)
    expected_mul = type(geometry_obj)(geometry_obj.data * vec_sub_10_60.data)
    expected_div = type(geometry_obj)(geometry_obj.data / vec_sub_10_60.data)

    self.assertEqual(obj_add, expected_add)
    self.assertEqual(obj_sub, expected_sub)
    self.assertEqual(obj_mul, expected_mul)
    self.assertEqual(obj_div, expected_div)

  @parameterized.parameters([wrench_1_6, twist_1_6, accel_1_6])
  def test_operators_with_fancy_vec6_subtype(self, geometry_obj):
    # Users may want their own subtypes of Vec6.
    # This tests that a better version can improve on mul and add.
    # returning the geometry type from those operations.
    fancy_vec = FancyGains(list(range(10, 70, 10)))
    obj_add_vec = geometry_obj + fancy_vec
    obj_mul_vec = geometry_obj * fancy_vec
    vec_add_obj = fancy_vec + geometry_obj
    vec_mul_obj = fancy_vec * geometry_obj

    self.assertIsInstance(obj_add_vec, type(geometry_obj))
    self.assertIsInstance(obj_mul_vec, type(geometry_obj))
    self.assertIsInstance(vec_add_obj, type(geometry_obj))
    self.assertIsInstance(vec_mul_obj, type(geometry_obj))

    expected_add = type(geometry_obj)(geometry_obj.data + fancy_vec.data)
    expected_mul = type(geometry_obj)(geometry_obj.data * fancy_vec.data)

    self.assertEqual(obj_add_vec, expected_add)
    self.assertEqual(vec_add_obj, expected_add)
    self.assertEqual(obj_mul_vec, expected_mul)
    self.assertEqual(vec_mul_obj, expected_mul)

  @parameterized.parameters(
      itertools.product(
          [wrench_1_6, twist_1_6, accel_1_6],  # Formatting.
          [wrench_1_6, twist_1_6, accel_1_6],  # Formatting.
          [operator.mul, operator.add, operator.truediv, operator.sub]))
  def test_invalid_operations(self, lhs, rhs, operation):
    # This tests that you can't apply arithmetic operations to two instances
    # of a geometric type.  What is a Twist added to an Accel?
    with self.assertRaises(TypeError):
      operation(lhs, rhs)


class HashEqualsTest(parameterized.TestCase):

  @parameterized.parameters(
      [geometry.Accel, geometry.Twist, geometry.Wrench, geometry.Vec6])
  def test_vec6_types(self, geometry_type):
    data = np.random.random(6)
    value1 = geometry_type(data)
    value2 = geometry_type(data)
    self.assertEqual(value1, value2)
    self.assertEqual(hash(value1), hash(value2))

    data2 = data + np.arange(6)
    value3 = geometry_type(data2)
    self.assertNotEqual(value1, value3)

  def test_pose(self):
    position = np.random.random(3)
    quaternion = np.random.random(4)
    pose1 = geometry.Pose(position.copy(), quaternion.copy(), name='name')
    pose2 = geometry.Pose(position.copy(), quaternion.copy(), name='name')
    self.assertEqual(pose1, pose2)
    self.assertEqual(hash(pose1), hash(pose2))

  @parameterized.parameters([(geometry.Accel, geometry.AccelStamped),
                             (geometry.Twist, geometry.TwistStamped),
                             (geometry.Wrench, geometry.WrenchStamped),
                             (geometry.Vec6, geometry.VectorStamped)])
  def test_stamped(self, base_type, stamped_type):
    base1 = base_type(np.random.random(6))
    pose1 = geometry.Pose.from_poseuler(base1)
    pose_stamped_1 = geometry.PoseStamped(pose1, frame=None)

    pose2 = geometry.Pose.from_poseuler(base1)
    pose_stamped_2 = geometry.PoseStamped(pose2, frame=None)

    stamped1 = stamped_type(base1, pose_stamped_1)
    stamped2 = stamped_type(base1, pose_stamped_2)

    self.assertEqual(stamped1, stamped2)
    self.assertEqual(hash(stamped1), hash(stamped2))


class BatchedGeometryTest(parameterized.TestCase):
  BATCH = 5
  IDENTITY_QUAT_BATCH = np.tile((1, 0, 0, 0), (BATCH, 1))
  IDENTITY_POS_BATCH = np.zeros((BATCH, 3))

  def test_batched_pose_mul(self):
    pose1 = geometry.Pose(position=self.IDENTITY_POS_BATCH,
                          quaternion=self.IDENTITY_QUAT_BATCH)
    pose2 = geometry.Pose(position=np.ones((self.BATCH, 3)),
                          quaternion=np.tile((0, 1, 0, 0),
                                             (self.BATCH, 1)))
    world_pose2_test = pose1.mul(pose2)
    for i in range(self.BATCH):
      np.testing.assert_array_almost_equal(
          world_pose2_test.position[i],
          pose2.position[i])
      np.testing.assert_array_almost_equal(
          world_pose2_test.quaternion[i],
          pose2.quaternion[i])

  def test_pose_inv_random(self):
    pose = geometry.Pose(
        position=np.zeros((5, 3)), quaternion=np.tile((1, 0, 0, 0), (5, 1)))
    inv = pose.inv()
    identity = pose.mul(inv)
    np.testing.assert_array_almost_equal(
        identity.position,
        self.IDENTITY_POS_BATCH)
    np.testing.assert_array_almost_equal(
        identity.quaternion,
        self.IDENTITY_QUAT_BATCH)

if __name__ == '__main__':
  absltest.main()
