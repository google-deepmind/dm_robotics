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

"""Moma SubTask Option."""

import re
from typing import Callable, Iterable, List, Optional

from dm_control import mjcf
import dm_env
from dm_env import specs
from dm_robotics import agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.moma import effector
import numpy as np


def _find_actuator_indices(prefix: str, spec: specs.Array) -> List[bool]:
  actuator_names = spec.name.split('\t')
  prefix_expr = re.compile(prefix)
  return [re.match(prefix_expr, name) is not None for name in actuator_names]


class MomaOption(af.DelegateOption):
  """Adapts an option to control a specific set of effectors.

  The Moma BaseTask is not actuated through the before_step method. Rather,
  individual effectors that comprise the base task are controlled individually.
  This is done by calling their set_control method. This is done so that each
  Subtask/Option can determine which effectors it controls explicitly, rather
  than being forced to control all of them. The MomaOption wrapper allows
  adapting an Option to run against a Moma BaseTask by facilitating the control
  of a subset of effectors.
  """

  def __init__(self,
               physics_getter: Callable[[], mjcf.Physics],
               effectors: Iterable[effector.Effector],
               delegate: af.Option,
               name: Optional[str] = None):
    super().__init__(delegate, name)

    self._physics_getter = physics_getter
    self._effectors = list(effectors)
    self._action_spec = None

  def step(self, parent_timestep: dm_env.TimeStep) -> np.ndarray:
    if self._action_spec is None:
      aspecs = [a.action_spec(self._physics_getter()) for a in self._effectors]
      self._action_spec = spec_utils.merge_specs(aspecs)

    action = self.step_delegate(parent_timestep)
    self._set_control(physics=self._physics_getter(), command=action)

    # Moma environment step takes zero-length actions. The actuators are driven
    # directly through effectors rather than through the monolithic per-step
    # action.
    return np.array([], dtype=np.float32)

  def step_delegate(self, parent_timestep: dm_env.TimeStep) -> np.ndarray:
    return super().step(parent_timestep)

  def _set_control(self, physics: mjcf.Physics, command: np.ndarray):
    if len(command) == 0:  # pylint: disable=g-explicit-length-test
      # If the command is empty, there isn't anything to do.
      # Empty MomaOptions can be used to wrap other MomaOptions that are in
      # charge of controlling individual effectors.
      return

    if self._action_spec is None:
      raise ValueError('self._action_spec is None.')

    spec_utils.validate(self._action_spec, command)

    for ef in self._effectors:
      e_cmd = command[_find_actuator_indices(ef.prefix, self._action_spec)]
      ef.set_control(physics, e_cmd)
