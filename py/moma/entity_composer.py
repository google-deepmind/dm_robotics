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

# Lint as: python3
"""Composes Entities."""

import abc
from dm_control import composer


class TaskEntitiesComposer(abc.ABC):

  @abc.abstractmethod
  def compose_entities(self, arena: composer.Arena) -> None:
    """Adds all of the necessary objects to the arena and composes objects."""
    pass
