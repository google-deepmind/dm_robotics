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
"""Manage collections of props."""
import collections
from typing import Any, Callable, Union


class Singleton(type):
  _instances = {}

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]


VersionedSequence = collections.namedtuple('VersionedSequence',
                                           ['version', 'ids'])


class PropSetDict(dict):
  """A dictionary that supports a function evaluation on every key access.

  Extends the standard dictionary to provide dynamic behaviour for object sets.
  """

  def __getitem__(self, key: Any) -> VersionedSequence:
    # The method is called during [] access.
    # Provides a collection of prop names.
    return self._evaluate(dict.__getitem__(self, key))

  def __repr__(self) -> str:
    return f'{type(self).__name__}({super().__repr__()})'

  def get(self, key) -> VersionedSequence:
    return self.__getitem__(key)

  def values(self):
    values = super().values()
    return [self._evaluate(x) for x in values]

  def items(self):
    new_dict = {k: self._evaluate(v) for k, v in super().items()}
    return new_dict.items()

  def _evaluate(
      self, sequence_or_function: Union[VersionedSequence,
                                        Callable[[], VersionedSequence]]
  ) -> VersionedSequence:
    """Based on the type of an argument, execute different actions.

    Supports static sequence containers or functions that create such. When the
    argument is a contrainer, the function returns the argument "as is". In case
    a callable is provided as an argument, it will be evaluated to create a
    container.

    Args:
      sequence_or_function: A sequence or a function that creates a sequence.

    Returns:
      A versioned set of names.
    """
    if isinstance(sequence_or_function, VersionedSequence):
      return sequence_or_function
    new_sequence = sequence_or_function()
    return new_sequence
