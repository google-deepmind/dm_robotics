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

"""Effector that sends only the min or max command to a delegate effector."""
from typing import Optional

from dm_control import mjcf
from dm_robotics.moma import effector
import numpy as np


class MinMaxEffector(effector.Effector):
  """Effector to send minimum or maximum command values to a base effector.

  This effector will ensure that only the min and max commands are sent to the
  effector. This is useful for example when controlling a gripper and
  we want to either have a fully open or fully closed actuation mode.

  The effector will take the input command and send:
    - The minimum command for all input values <= midrange
    - The maximum command for all input values > midrange.

  Example:
    min_action: [1., 1., 1,]
    max_action: [3., 3., 3.]
    input_command: [1.2, 2., 2.3]
    output_command: [1., 1., 3.]
  """

  def __init__(
      self, base_effector: effector.Effector,
      min_action: Optional[np.ndarray] = None,
      max_action: Optional[np.ndarray] = None):
    """Constructor.

    Args:
      base_effector: Effector to wrap.
      min_action: Array containing the minimum actions.
      max_action: Array containing the maximum actions.
    """
    self._effector = base_effector
    self._min_act = min_action
    self._max_act = max_action
    self._action_spec = None

  def action_spec(self, physics):
    if self._action_spec is None:
      self._action_spec = self._effector.action_spec(physics)

      # We get the minimum actions for each DOF. If the user provided no value,
      # we use the one from the environment.
      if self._min_act is None:
        self._min_act = self._action_spec.minimum
        if self._min_act.size == 1:
          self._min_act = np.full(
              self._action_spec.shape, self._min_act, self._action_spec.dtype)
      if self._min_act.shape != self._action_spec.shape:
        raise ValueError('The provided minimum action does not have the same '
                         'shape as the action expected by the base effector '
                         f'expected shape {self._action_spec.shape} but '
                         f'{self._min_act} was provided')

      if self._max_act is None:
        self._max_act = self._action_spec.maximum
        if self._max_act.size == 1:
          self._max_act = np.full(
              self._action_spec.shape, self._max_act, self._action_spec.dtype)
      if self._max_act.shape != self._action_spec.shape:
        raise ValueError('The provided maximum action does not have the same '
                         'shape as the action expected by the base effector '
                         f'expected shape {self._action_spec.shape} but '
                         f'{self._max_act} was provided')

    return self._action_spec

  def set_control(self, physics, command):
    if self._action_spec is None:
      self.action_spec(physics)
    new_cmd = np.zeros(
        shape=self._action_spec.shape, dtype=self._action_spec.dtype)
    mid_point = (self._action_spec.minimum + self._action_spec.maximum) / 2
    min_idxs = command <= mid_point
    max_idxs = command > mid_point
    new_cmd[min_idxs] = self._min_act[min_idxs]
    new_cmd[max_idxs] = self._max_act[max_idxs]
    self._effector.set_control(physics, new_cmd)

  def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
    self._effector.after_compile(mjcf_model)

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    self._effector.initialize_episode(physics, random_state)

  def close(self):
    self._effector.close()

  @property
  def prefix(self) -> str:
    return self._effector.prefix
