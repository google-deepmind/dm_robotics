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

"""Module for scene composers for the insertion task."""

import dataclasses
from typing import Callable, Iterable, Tuple

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.initializers import prop_initializer
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
from dm_robotics.moma import initializer as base_initializer
import numpy as np

PropPlacer = prop_initializer.PropPlacer


@dataclasses.dataclass
class TaskEntitiesInitializer(base_initializer.Initializer):
  """An initializer composed of other initializers."""
  initializers: Iterable[composer.Initializer]

  def __call__(self, physics: mjcf.Physics,
               random_state: np.random.RandomState) -> bool:
    """Runs initializers sequentially, until all done or one fails."""
    for initializer in self.initializers:
      # pylint: disable=singleton-comparison,g-explicit-bool-comparison
      # Explicitly check for false because an initializer returning `None`
      # should be counted as success.
      if initializer(physics, random_state) == False:
        return False
      # pylint: enable=singleton-comparison,g-explicit-bool-comparison
    return True


NO_INIT = TaskEntitiesInitializer(initializers=tuple())


@dataclasses.dataclass
class PoseInitializer(base_initializer.Initializer):
  """Initialize entity pose.

  This can be used to initialize things like entities that are attached to a
  scene with a free joint.

  Attributes:
    initializer_fn: A function that will initialize the pose of an entity, the
      array arguments passed are pos and quat.
    pose_sampler: A function that will provide a new pos and quat on each
      invocation, to pass to the `initializer_fn`.
  """
  initializer_fn: Callable[
      [mjcf.Physics, np.ndarray, np.ndarray, np.random.RandomState], None
  ]
  pose_sampler: Callable[[np.random.RandomState, geometry.Physics],
                         Tuple[np.ndarray, np.ndarray]]

  def __call__(self, physics: mjcf.Physics,
               random_state: np.random.RandomState) -> bool:
    pos, quat = self.pose_sampler(random_state, mujoco_physics.wrap(physics))
    self.initializer_fn(physics, pos, quat, random_state)
    return True


@dataclasses.dataclass
class JointsInitializer(base_initializer.Initializer):
  """Initialize joint angles.

  This can be used to initialize things like robot arm joint angles.

  Attributes:
    initializer_fn: A function that will set joint angles.
    pose_sampler: A function that will provide new joint angles on each
      invocation, to pass to the `initializer_fn`.
  """
  initializer_fn: Callable[[mjcf.Physics, np.ndarray], None]
  joints_sampler: Callable[[np.random.RandomState, geometry.Physics],
                           np.ndarray]

  def __call__(self, physics: mjcf.Physics,
               random_state: np.random.RandomState) -> bool:
    joint_angles = self.joints_sampler(random_state,
                                       mujoco_physics.wrap(physics))
    self.initializer_fn(physics, joint_angles)
    return True


@dataclasses.dataclass
class CallableInitializer(base_initializer.Initializer):
  """An initializer composed of a single callable."""
  initializer: Callable[[mjcf.Physics, np.random.RandomState], None]

  def __call__(self, physics: mjcf.Physics,
               random_state: np.random.RandomState) -> bool:
    """Invokes the initializer."""
    self.initializer(physics, random_state)
    return True
