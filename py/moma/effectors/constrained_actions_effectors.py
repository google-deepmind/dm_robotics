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

"""Effectors that constrain actions to joint limits or Cartesian bounds."""

from typing import Callable, Optional, TypeVar, Generic

from dm_control import mjcf
from dm_env import specs
from dm_robotics.moma import effector
from dm_robotics.moma.models.robots.robot_arms import robot_arm
import numpy as np


# The state which we check in order to determine whether to constrain certain
# DOFs.
_State = np.ndarray

# The limits (upper or lower) of the state.
_StateLimits = np.ndarray

# The command which will be modified based on the state, if the state goes
# outside the bounds.
_Command = np.ndarray

# Callable that returns the state indices that are NOT valid, i.e. which DOFs
# are outside the limits.
_StateValidityChecker = Callable[[_State, _StateLimits, _Command], np.ndarray]

T = TypeVar('T', bound=effector.Effector)


class ConstrainedActionEffector(effector.Effector, Generic[T]):
  """Effector wrapper that limits certain DOFs based on their state.

  For instance, if you want to limit a joint torque command based on whether
  certain joints are close to their velocity limits, you may use this effector
  like so:

  ```
  my_safe_effector = ConstrainedActionEffector(
      delegate=my_raw_joint_torque_effector,
      min_limits=min_arm_joint_velocities,
      max_limits=max_arm_joint_velocities,
      state_getter=lambda physics: physics.bind(arm.joints).qvel)
  ```

  Any command DOFs whose corresponding state surpasses the provided limits will
  be set to 0.
  """

  def __init__(
      self,
      delegate: T,
      min_limits: np.ndarray,
      max_limits: np.ndarray,
      state_getter: Callable[[mjcf.Physics], np.ndarray],
      min_state_checker: Optional[_StateValidityChecker] = None,
      max_state_checker: Optional[_StateValidityChecker] = None):
    """Constructor for ConstrainedActionEffector.

    Args:
      delegate: Underlying effector which actually actuates the command.
      min_limits: The lower limits of the state of whatever is being actuated.
        If the state goes below this limit, the command gets set to 0. For
        instance, if the delegate is a joint velocity effector, and the state is
        the joint positions, if the 3rd joint position is below the 3rd limit,
        then the 3rd action will be set to 0.
      max_limits: The upper limits of the state of whatever is being actuated.
        See `min_limits` description for how these limits are used.
      state_getter: Callable that takes a physics object and returns the
        relevant "state" of the actuated entity. The limits will be applied to
        this state. When the state falls outside the bounds of the limits, the
        commanded action will be set to 0.
      min_state_checker: Optional callable that takes the state as returned by
        `state_getter`, the `min_limits`, and the input command to the effector,
        and determines which controllable DOFs are not valid. Returns a boolean
        np.ndarray mask that has the same shape as the input command. `True`
        DOFs in the mask are set to 0. If not provided, this defaults to a
        simple min bounds check.
      max_state_checker: Optional callable that takes the state as returned by
        `state_getter`, the `max_limits`, and the input command to the effector,
        and determines which controllable DOFs are not valid. Returns a boolean
        np.ndarray that has the same shape as the input command. `True`
        DOFs in the mask are set to 0. If not provided, this defaults to a
        simple max bounds check.
    """
    if min_limits.shape != max_limits.shape:
      raise ValueError('The min and max limits must have the same shape. '
                       f'Min: {min_limits.shape}, max: {max_limits.shape}.')
    self._delegate = delegate
    self._min_limits = min_limits
    self._max_limits = max_limits
    self._get_state = state_getter
    self._min_state_checker = (min_state_checker or
                               self._default_min_state_checker)
    self._max_state_checker = (max_state_checker or
                               self._default_max_state_checker)

  def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
    self._delegate.after_compile(mjcf_model)

  def initialize_episode(self, physics, random_state) -> None:
    self._delegate.initialize_episode(physics, random_state)

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    # Make sure that the delegate effector and limits are compatible.
    if self._delegate.action_spec(physics).shape != self._min_limits.shape:
      raise ValueError('The delegate effector action spec and the provided '
                       'limits have different shapes. Delegate action spec: '
                       f'{self._delegate.action_spec(physics)}. Limits shape: '
                       f'{self._min_limits.shape}')
    return self._delegate.action_spec(physics)

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    constrained_action = self._get_contstrained_action(physics, command)
    self._delegate.set_control(physics, constrained_action)

  def _get_contstrained_action(
      self, physics: mjcf.Physics, command: np.ndarray) -> np.ndarray:
    # Limit any DOFs whose state falls outside the provided limits.
    constrained_action = command[:]
    state = self._get_state(physics)
    constrained_action[
        self._min_state_checker(state, self._min_limits, command)] = 0.
    constrained_action[
        self._max_state_checker(state, self._max_limits, command)] = 0.
    return constrained_action

  @property
  def delegate(self) -> T:
    return self._delegate

  @property
  def prefix(self) -> str:
    return self._delegate.prefix

  def _default_min_state_checker(
      self, state: np.ndarray, limits: np.ndarray, command: np.ndarray
      ) -> np.ndarray:
    """Returns a bool mask for `command` for which DOFs are invalid."""
    return (state < limits) & (command < 0.)

  def _default_max_state_checker(
      self, state: np.ndarray, limits: np.ndarray, command: np.ndarray
      ) -> np.ndarray:
    """Returns a bool mask for `command` for which DOFs are invalid."""
    return (state > limits) & (command > 0.)


class LimitJointPositions(ConstrainedActionEffector):
  """Limits joint actions to stay within a safe joint position range.

  The current implementation assumes all joints are controllable.

  NOTE: Do NOT use this effector with a joint position effector. This is meant
  to be used with joint velocity, torque, etc. effectors.
  """

  def __init__(self,
               joint_effector: effector.Effector,
               min_joint_limits: np.ndarray,
               max_joint_limits: np.ndarray,
               arm: robot_arm.RobotArm):
    if len(min_joint_limits) != len(arm.joints):
      raise ValueError('The joint limits must match the number of joints. '
                       f'Length of joint limits: {len(min_joint_limits)}. '
                       f'Number of joints: {len(arm.joints)}')
    super().__init__(
        delegate=joint_effector,
        min_limits=min_joint_limits,
        max_limits=max_joint_limits,
        state_getter=lambda physics: physics.bind(arm.joints).qpos)
