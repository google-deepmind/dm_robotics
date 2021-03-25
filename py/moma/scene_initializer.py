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
"""Module for scene initializers.

Scene initializers are callables that can change the MJCF of a scene. They are
called before each episode, and the MJCF is recompiled afterwards. See
`base_task.py` for more information on how they are called.
"""

from typing import Callable, Iterable, Tuple

import dataclasses
from dm_control import mjcf
from dm_robotics.moma import base_task
import numpy as np


@dataclasses.dataclass(frozen=True)
class CompositeSceneInitializer:
  initializers: Iterable[base_task.SceneInitializer]

  def __call__(self, random_state: np.random.RandomState) -> None:
    for initializer in self.initializers:
      initializer(random_state)


NO_INIT = CompositeSceneInitializer(initializers=tuple())


@dataclasses.dataclass
class EntityPoseInitializer:
  """Pose initializer."""

  entity: mjcf.Element
  pose_sampler: Callable[[np.random.RandomState], Tuple[np.ndarray, np.ndarray]]

  def __call__(self, random_state: np.random.RandomState) -> None:
    pos, quat = self.pose_sampler(random_state)
    # pytype: disable=not-writable
    self.entity.pos = pos
    self.entity.quat = quat
    # pytype: enable=not-writable
