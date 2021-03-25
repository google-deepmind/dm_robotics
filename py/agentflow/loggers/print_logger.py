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
"""A rudimentary logging class that outputs data as strings."""

from typing import Any, Callable, Mapping, Optional
import numpy as np


class PrintLogger:
  """Serializes logging vales to strings and prints them."""

  def __init__(
      self,
      print_fn: Callable[[str], None] = print,
      serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = None,
  ):
    """Initializes the logger.

    Args:
      print_fn: function to call which acts like print.
      serialize_fn: function to call which formats a values dict.
    """

    self._print_fn = print_fn
    self._serialize_fn = serialize_fn or _serialize

  def write(self, values: Mapping[str, Any]):
    self._print_fn(self._serialize_fn(values))


def _format_value(value: Any) -> str:
  if isinstance(value, (float, np.number)):
    return f'{value:0.3f}'
  return str(value)


def _serialize(values: Mapping[str, Any]) -> str:
  return ', '.join(
      f'{k} = {_format_value(v)}' for k, v in sorted(values.items()))
