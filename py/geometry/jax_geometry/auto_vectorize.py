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

"""Utilities for automatically batching functions."""

import functools
import itertools
from typing import Callable, Any

import jax
import numpy as np

AnyFn = Callable[..., Any]
NumpyFn = Callable[..., np.ndarray]


def _get_reference_arg(*args, **kwargs):
  """Returns the first valid array argument from args (preferred) or kwargs."""

  def is_valid(arg):
    return arg is not None and hasattr(arg, 'shape')

  try:
    return next(
        itertools.chain(
            filter(is_valid, args), filter(is_valid, kwargs.values())))
  except StopIteration as no_arr_arg:
    raise ValueError('Could not find valid array argument.') from no_arr_arg


def vmap_np(fn: NumpyFn, in_axis: int = 0) -> NumpyFn:
  """Maps a python-function over the leading dimension of all inputs.

  This is an API-equivalent of jax.vmap(fn), but for numpy functions. It does
  not support `in_axes`, currently only supports a single output array, and
  offers no performance benefit. The intended use-case is to allow `multi-vmap`
  to apply a function to batches of data with an arbitrary number of leading
  dimensions.

  The main benefit to `vmap_np` vs. `np.vectorize` is that it does not require a
  `signature=(...)` argument when the pyfunc takes non-scalar arguments. These
  can become unweildy, especially when nesting map expressions.

  Note: Another reason this was preferred vs. np.vectorize was that in the
    original use-case of PNG-encoding batches of images, np.vectorize was
    randomly generating invalid PNG headers for >1 leading batch dimension.

  Example:
  >>> def myfunc(a, b):
  ...   "Return a-b if a>b, otherwise return a+b"
  ...   if a > b:
  ...       return a - b
  ...   else:
  ...       return a + b

  >>> mapped_fn = np.vectorize(myfunc)
  >>> mapped_fn([1, 2, 3, 4], 2)
  array([3, 4, 1, 2])

  >>> mapped_fn = auto_vectorize.vmap_np(functools.partial(myfunc, b=2))
  >>> mapped_fn(np.array([1, 2, 3, 4]))
  array([3, 4, 1, 2])

  multi_vmap example:
  >>> def encode_rgb_img(img: np.ndarray) -> np.ndarray:
  ...   <encodes a single RGB image to bytes>
  >>> encode_rgb_mapped = multi_vmap(encode_rgb_img, n_times=-3, map_fn=vmap_np)
  >>> encoded_images = encode_rgb_mapped(rgb_img_batch)  # (32, 64, 222, 296, 3)
  >>> encoded_images.shape, encoded_images.dtype
  ((32, 64), dtype('S6768'))

  Args:
    fn: An arbitrary python function that takes one or more np.ndarray arguments
      and returns an np.ndarray
    in_axis: An integer specifying which input array axes to map over. Analog of
      `in_axes` in jax.vmap -- this function currently only supports a single
      axis common to all arguments.

  Returns:
    A function which wraps `fn`, calling it on each input along the leading
    dimension of the input and stacking the result.
  """

  @functools.wraps(fn)
  def _call_n_times(*args, **kwargs):
    # Ensure all args are numpy types, not jax.
    args = jax.tree_util.tree_map(np.asarray, args)
    kwargs = jax.tree_util.tree_map(np.asarray, kwargs)

    reference_arg = _get_reference_arg(*args, **kwargs)
    axis_len = reference_arg.shape[in_axis]
    res = []
    for i in range(axis_len):
      args_i = [np.take(arg, i, axis=in_axis) for arg in args]
      kwargs_i = {
          k: np.take(arg, i, axis=in_axis) for k, arg in kwargs.items()
      }
      res.append(fn(*args_i, **kwargs_i))
    # TODO(jscholz) generalize to multiple outputs (naive tree_map doesn't
    # capture output-signature properly).
    return np.stack(res)

  return _call_n_times


def multi_vmap(fn: ...,
               n_times: int = -1,
               map_fn: Callable[..., AnyFn] = jax.vmap) ->...:
  """A wrapper for auto-batching module-level functions.

  This method provides a mechanism for vmapping `fn` over an arbitrary number of
  leading batch dimensions. It also allows lazy inference of the number of
  leading batch dimensions from the function input.

  For decorating member-methods & properties of supporting objects, see
  `batched_method` and `batched_property` below.

  Args:
    fn: The method to be mapped.
    n_times: The number of leading axes of all inputs to vmap over. E.g. if `fn`
      expects an RGB image of shape (H, W, C) and you want to call it on a batch
      of images with shape (10, 50, H, W, C), then `n_times=2`. Alternatively,
      can be 0 or a negative number indicating the content shape, i.e. which
      axes NOT to map over. E.g. for above RGB example, setting `n_times=-3`
      will vmap as many times as needed given an input to map all but the last 3
      axes (so, once for input shape (B, H, W, C), twice for shape (B, T, H, W,
      C), etc.) Default is -1, i.e. assume 1-dimensional content. This
      shape-inference is based on the first non-None argument, falling back to
      the first non-None keyword argument obtained from `kwargs.values` (no
      necessarily first in function call).
    map_fn: A callable which maps the function to be mapped. Can be `jax.vmap`
      or `vmap_np`. Default is jax.vmap

  Returns:
    An autobatched method version of fn.
  """
  def _map_n_times(fn: ..., n_times: int) -> ...:
    for _ in range(n_times):
      fn = map_fn(fn)
    return fn

  if n_times > 0:
    return _map_n_times(fn, n_times)

  else:
    # Create a wrapper to allow input-dependent mapping.
    @functools.wraps(fn)
    def _batched_method_wrapper(*args, **kwargs):
      # If `n_times` is negative then it indicates content dimension, so
      # inspect the argument to infer number of vmaps required.
      ref_arg = _get_reference_arg(*args, **kwargs)
      batch_shape = ref_arg.shape if n_times == 0 else ref_arg.shape[:n_times]
      inferred_n_times = len(batch_shape)
      batched_fn = _map_n_times(fn, inferred_n_times)
      return batched_fn(*args, **kwargs)

    return _batched_method_wrapper


def batched_method(fn):
  """A decorator for auto-batching member-functions.

  This decorator is designed for use on `chex.dataclass`s to implement
  methods whose implementations batch automatically.

  Any `chex.dataclass` that uses this decorator must define a `batch_shape`
  property that is computed from the raw (unbatched) members of the dataclass.
  Semantically, `batch_shape` indicates all the batch dimensions *excluding* the
  content dimension.

  E.g. a `Camera` class might define an intrinsics matrix with shape 3x3. Using
  `batched_method` we can write methods on `Camera` that operate on
  `self.intrinsics` in terms of this nominal shape which continue to work even
  on batches of `Camera` in which `Camera.intrinsics` is ([...], B, 3, 3).

  Example usage:

  ```
    @auto_vectorize.indexable  # Not needed by `batched_method`.
    @chex.dataclass
    class Foo:
      a: chex.Array  # Nominally a 1-dim array, with leading batch-dim(s)
      b: chex.Array  # Nominally a 1-dim array, with leading batch-dim(s)

      @property
      def batch_shape(self) -> Sequence[int]:
        return self.a.shape[:-1]  # assume a and b have same leading shape.

      @property
      @auto_vectorize.batched_method
      def sum(self) -> chex.Array:
        return jnp.sum(self.a) + jnp.sum(self.b)

      @auto_vectorize.batched_method
      def mul(self, quantity: chex.Array) -> chex.Array:
        return self.a @ quantity + self.b @ quantity
    ...
    # `mul` works on batches via internal vmap:
    foo = Foo(a=jnp.ones((3, 3)), b=jnp.ones((3, 3)) * 2)
    foo.mul(jnp.tri(3))
    >>> DeviceArray([3., 6., 9.], dtype=float32)

    # `mul` works with multiple leading batch-dimensions (w/o einsum):
    foo = Foo(a=jnp.ones((2, 4, 3)), b=jnp.ones((2, 4, 3)) * 2)
    foo.mul(jnp.ones((2, 4, 3)))
    >>> DeviceArray([[9., 9., 9., 9.],
                     [9., 9., 9., 9.]], dtype=float32)

    # `index` and `batched_method` are compatible:
    foo = Foo(a=jnp.ones((4, 3)), b=jnp.ones((4, 3)) * 2)
    foo.index[0].mul(jnp.ones(3))
    >>> DeviceArray(9., dtype=float32)

    # Property operates on nominal shape --> Result will be `batch_shape`:
    foo = Foo(a=jnp.ones((4, 3)), b=jnp.ones((4, 3)) * 2)
    foo.sum
    >>> DeviceArray([9., 9., 9., 9.], dtype=float32)
  ```

  Args:
    fn: The method to be decorated.

  Returns:
    An autobatched method version of fn.
  """
  @functools.wraps(fn)
  def _batched_method_wrapper(self, *args, **kwargs):
    if not hasattr(self, 'batch_shape'):
      raise ValueError(
          'You are trying to use `@batched_method` on a class that does not '
          'define `batch_shape`. This is not permitted.')

    batched_fn = fn
    for _ in self.batch_shape:
      batched_fn = jax.vmap(batched_fn)
    return batched_fn(self, *args, **kwargs)
  return _batched_method_wrapper


def batched_property(fn):
  """A decorator for auto-batching properties.

  This decorator is shorthand for:
  ```
    @property
    @batched_method
    def fn(self):
      ...
  ```

  See `batched_method` for details.
  Any `chex.dataclass` that uses this decorator must define a `batch_shape`
  property that is computed from the raw (unbatched) members of the dataclass.

  Example usage:

  ```
    @chex.dataclass
    class Vector:
      value: chex.Array

      @property
      def batch_shape(self):
        return self.value.shape[:-1]

      @auto_vectorize.batched_property
      def norm(self):
        return jnp.linalg.norm(self.value)
    ...
    vector = Vector(value=jnp.ones((2, 4, 3)))
    assert vector.norm.shape == (2, 4)
  ```

  Args:
    fn: The property to be decorated.

  Returns:
    An autobatched property version of fn.
  """
  return property(batched_method(fn))


def indexable(cls, name: str = 'index'):
  """Adds an `index` property to a chex.dataclass.

  This can be used to index or slice all fields of a chex.dataclass
  simultaneously.

  E.g.:
    @indexable
    @chex.dataclass(frozen=True)
    class Foo:
      a: chex.Array
      b: chex.Array

    foo = Foo(a=jnp.ones((3, 4, 5)), b=jnp.ones((3, 8, 10)))
    bar = foo.index[0, 1:3, :]
    bar.a.shape, bar.b.shape
    >>> ((2, 5), (2, 10))

  Args:
    cls: A class to add the `index` property to.
    name: A name for the property add. Defaults to "index".

  Returns:
    `cls` with the method `name` added which can be used for indexing.
  """

  class _IndexHelper:
    # Note: this docstring will appear as the docstring for the `index` property
    """Indexable helper object to call indexed index function.

    The ``index`` property is syntactic sugar for indexing all fields of a
    `chex.dataclass` simultaneously. It is a simplified version of the Jax `at`
    mechanism:
    https://jax.readthedocs.io/en/latest/jax.ops.html#indexed-update-operators

    In particular:
    - `obj = obj.index[0:3]` is shorthand for
      `jax.tree_util.tree_map(lambda x: x[0:3], obj)`.

    Note: this implementation is recursive -- if your dataclass has container-
    type attributes they will be indexed as well!
    """

    def __init__(self, obj):
      self.obj = obj

    def __getitem__(self, index):
      return jax.tree_util.tree_map(lambda x: x[index], self.obj)

  setattr(cls, name, property(_IndexHelper))

  return cls
