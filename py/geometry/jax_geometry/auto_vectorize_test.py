# Copyright 2024 DeepMind Technologies Limited.
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
# limitations under the License.

"""Tests for geometry.jax.auto_vectorize."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_robotics.geometry.jax_geometry import auto_vectorize
import jax.numpy as jnp
import numpy as np


class AutobatchingTest(chex.TestCase, parameterized.TestCase):

  @chex.all_variants
  @parameterized.parameters([((),), ((2,),), ((3, 3),), ((2, 2, 2),)])
  def test_broadcast_to_scalar(self, batch_shape):

    @chex.dataclass
    class Scalar:
      raw_value: chex.Array

      @property
      def batch_shape(self):
        return self.raw_value.shape

      @auto_vectorize.batched_property
      def value(self):
        chex.assert_shape(self.raw_value, ())
        return self.raw_value

      @auto_vectorize.batched_method
      def method(self, arg):
        chex.assert_shape(self.raw_value, ())
        chex.assert_shape(arg, ())
        return self.raw_value * arg

    raw_value = jnp.zeros(batch_shape)
    scalar = Scalar(raw_value=raw_value)
    self.assertSequenceEqual(scalar.batch_shape, batch_shape)

    # Test that property batches properly.
    @self.variant
    def get_value():
      return scalar.value

    chex.assert_shape(get_value(), batch_shape)

    # Test that method batches properly.
    @self.variant
    def call_method(arg):
      return scalar.method(arg)

    arg = jnp.zeros(batch_shape)
    chex.assert_shape(call_method(arg), batch_shape)

  @chex.all_variants
  @parameterized.parameters([((),), ((2,),), ((3, 3),), ((2, 2, 2),)])
  def test_broadcast_to_vector(self, batch_shape):

    @chex.dataclass
    class Vector:
      value: chex.Array

      @property
      def batch_shape(self):
        return self.value.shape[:-1]

      @auto_vectorize.batched_property
      def norm(self):
        return jnp.linalg.norm(self.value)

      @auto_vectorize.batched_method
      def mul(self, arg):
        return self.value @ arg

    value = jnp.ones(batch_shape + (3,))
    vector = Vector(value=value)
    self.assertSequenceEqual(vector.batch_shape, batch_shape)

    # Test that property batches properly.
    @self.variant
    def get_norm():
      return vector.norm

    chex.assert_shape(get_norm(), batch_shape)
    chex.assert_trees_all_close(get_norm(), jnp.sqrt(3))

    # Test that method batches properly.
    @self.variant
    def get_prod(arg):
      return vector.mul(arg)

    arg = jnp.ones(batch_shape + (3,))
    chex.assert_shape(get_prod(arg), batch_shape)
    chex.assert_trees_all_close(get_prod(arg), 3)

  @chex.all_variants
  def test_missing_batch_shape(self):

    @chex.dataclass
    class MissingBatchShape:
      value: chex.Array

      @auto_vectorize.batched_property
      def norm(self):
        return jnp.linalg.norm(self.value)

      @auto_vectorize.batched_method
      def mul(self, arg):
        return self.value @ arg

    value = jnp.ones((3, 3, 3))
    bad = MissingBatchShape(value=value)

    # Verify that `batched_property` fails without `self.batch_shape`.
    @self.variant
    def get_norm():
      return bad.norm

    with self.assertRaises(ValueError):
      get_norm()

    # Verify that `batched_method` fails without `self.batch_shape`.
    @self.variant
    def get_prod(arg):
      return bad.mul(arg)

    with self.assertRaises(ValueError):
      get_prod(jnp.ones((3, 3, 3)))

  @chex.all_variants
  def test_indexable(self):

    @auto_vectorize.indexable
    @chex.dataclass
    class TestDataclass:
      a: chex.Array
      b: chex.Array

    # Verify that both fields can be indexed simultaneously.
    data = self.variant(TestDataclass)(a=jnp.ones((3, 4)), b=jnp.ones((3, 2)))
    # All attributes should have this reduced shape.
    self.assertSequenceEqual(data.index[0].a.shape, (4,))
    self.assertSequenceEqual(data.index[0].b.shape, (2,))

    # Verify that both fields can be sliced simultaneously.
    data = self.variant(TestDataclass)(
        a=jnp.ones((5, 3, 4)), b=jnp.ones((5, 3, 2)))
    # All attributes should have this reduced shape.
    self.assertSequenceEqual(data.index[1:3, 2, :].a.shape, (2, 4))
    self.assertSequenceEqual(data.index[1:3, 2, :].b.shape, (2, 2))


class MultiVmapTest(chex.TestCase, parameterized.TestCase):

  @chex.all_variants
  def test_multi_vmap_jax_vs_np(self):
    def func(x):
      return x[0] + x[1] + x[2]

    func_mapped = auto_vectorize.multi_vmap(func, n_times=-1)
    func_mapped_np = auto_vectorize.multi_vmap(
        func, n_times=-1, map_fn=auto_vectorize.vmap_np)

    test_input = np.arange(24).reshape((2, 4, 3))
    test_input_jax = jnp.array(test_input)
    expected_result = self.variant(func_mapped)(test_input_jax)
    actual_result = func_mapped_np(test_input)
    chex.assert_trees_all_close(actual_result, expected_result)

  def test_n_times_inference_immutable(self):
    # This test was motivated by a bug in the initial implementation of
    # multi_vmap for the `n_times < 0` case, in which the batch shape is
    # inferred from the input. The initial implementation re-assigned `n_times`
    # in-place using the "nonlocal" keyword (otherwise we'd received
    # UnboundLocalError: local variable 'n_times' referenced before assignment).
    # However, nonlocal actually allows mutation of the variable at the outer
    # scope, which led to subtle bugs when invoking `multi_vmap` on differing
    # batch shapes. This was missed in early implementation due to jax.vmap
    # caching mechanism, but arose when vmap_np support was added. The key
    # lesson here was:
    #  "Whenever a value is assigned to a variable inside a function, python
    #   considers that variable a local variable of that function."
    #   (stackoverflow link: http://shortn/_LRGiq3APwz)
    # The solution was simply to use a different variable name for the inferred
    # value of `n_times`, rather than reassigning it. This test verifies that
    # this solution allows independence of `n_times` across calls to
    # `multi_vmap`. This test also verifies that `multi_vmap` can be applied to
    # functions over scalar arguments (i.e. n_times=0).
    def test_fn(a, b, c):
      assert a.ndim == 0  # Verifies we see scalar args and aren't broadcasting.
      return a + b + c

    test_fn_mapped = auto_vectorize.multi_vmap(
        test_fn, n_times=0, map_fn=auto_vectorize.vmap_np)
    batch_shape1 = (2, 3, 4)
    batch_shape2 = (5, 2, 3, 4)

    batch1_output = test_fn_mapped(
        np.ones(batch_shape1) * 1,
        np.ones(batch_shape1) * 2,
        np.ones(batch_shape1) * 3)

    # In the original implementation this call would fail with `AssertionError`
    # from the above assert. This happened becaused the previous call to
    # `test_fn_mapped` set the value of `n_times` to 3, which broke the
    # assumption `n_times <= 0` for the shape-inference step of
    # _batched_method_wrapper.
    batch2_output = test_fn_mapped(
        np.ones(batch_shape2) * 1,
        np.ones(batch_shape2) * 2,
        np.ones(batch_shape2) * 3)

    self.assertEqual(batch1_output.shape, batch_shape1)
    self.assertEqual(batch2_output.shape, batch_shape2)

    np.testing.assert_allclose(batch1_output, 6)
    np.testing.assert_allclose(batch2_output, 6)


if __name__ == '__main__':
  absltest.main()
