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

"""Effector for MuJoCo actuators."""

from typing import List, Optional, Sequence, Tuple

from dm_control import mjcf  # type: ignore
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.moma import effector
from dm_robotics.moma.models import types
import numpy as np


class MujocoEffector(effector.Effector):
  """A generic effector for multiple MuJoCo actuators."""

  def __init__(self,
               actuators: Sequence[types.MjcfElement],
               prefix: str = '',
               action_range_override: Optional[List[Tuple[float,
                                                          float]]] = None):
    self._actuators = actuators
    self._prefix = prefix
    self._action_range_override = action_range_override
    self._action_spec = None

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    if self._action_spec is None:
      self._action_spec = create_action_spec(physics, self._actuators,
                                             self._prefix,
                                             self._action_range_override)
    return self._action_spec

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    spec_utils.validate(self.action_spec(physics), command)
    physics.bind(self._actuators).ctrl = command

  @property
  def prefix(self) -> str:
    return self._prefix

  def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
    pass

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass


def create_action_spec(
    physics: mjcf.Physics,
    actuators: Sequence[types.MjcfElement],
    prefix: str = '',
    action_range_override: Optional[List[Tuple[float, float]]] = None
) -> specs.BoundedArray:
  """Creates an action range for the given actuators.

  The control range (ctrlrange) of the actuators determines the action range
  unless `action_range_override` is supplied.

  The action range name is the tab-separated names of the actuators, each
  prefixed by `prefix`

  Args:
    physics: Used to get the ctrlrange of the actuators.
    actuators: The MuJoCo actuators get get an action spec for.
    prefix: A name prefix to prepend to each actuator name.
    action_range_override: Optional override (min, max) sequence to use instead
      of the actuator ctrlrange.

  Returns:
    An action spec for the actuators.
  """
  num_actuators = len(actuators)
  actuator_names = [f'{prefix}{i}' for i in range(num_actuators)]
  if action_range_override is not None:
    action_min, action_max = _action_range_from_override(
        actuators, action_range_override)
  else:
    action_min, action_max = _action_range_from_actuators(physics, actuators)

  return specs.BoundedArray(
      shape=(num_actuators,),
      dtype=np.float32,
      minimum=action_min,
      maximum=action_max,
      name='\t'.join(actuator_names))


def _action_range_from_override(
    actuators: Sequence[types.MjcfElement],
    override_range: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
  """Returns the action range min, max using the values from override_range."""
  num_actions = len(actuators)
  assert (len(override_range) == 1 or len(override_range) == num_actions)

  if len(override_range) == 1:
    range_min = np.array([override_range[0][0]] * num_actions, dtype=np.float32)
    range_max = np.array([override_range[0][1]] * num_actions, dtype=np.float32)
  else:
    range_min = np.array([r[0] for r in override_range], dtype=np.float32)
    range_max = np.array([r[1] for r in override_range], dtype=np.float32)

  return range_min, range_max


def _action_range_from_actuators(
    physics: mjcf.Physics,
    actuators: Sequence[types.MjcfElement]) -> Tuple[np.ndarray, np.ndarray]:
  """Returns the action range min, max for the actuators."""
  num_actions = len(actuators)

  control_range = physics.bind(actuators).ctrlrange
  is_limited = physics.bind(actuators).ctrllimited.astype(bool)

  minima = np.full(num_actions, fill_value=-np.inf, dtype=np.float32)
  maxima = np.full(num_actions, fill_value=np.inf, dtype=np.float32)
  minima[is_limited], maxima[is_limited] = control_range[is_limited].T

  return minima, maxima
