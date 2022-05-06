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

"""Cartesian effector that controls linear XYZ and rotation around Z motions."""

import copy
from typing import Callable, Optional, Sequence

from dm_control import mjcf
from dm_env import specs
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
from dm_robotics.moma import effector
from dm_robotics.moma.effectors import constrained_actions_effectors
from dm_robotics.transformations import transformations
import numpy as np


_MjcfElement = mjcf.element._ElementImpl  # pylint: disable=protected-access

# Orientation corresponding to the endeffector pointing downwards.
DOWNFACING_EE_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])


class _OrientationController():
  """Proportional controller for maintaining the desired orientation."""

  def __init__(self, rotation_gain: float, align_z_orientation: bool = False):
    """Constructor.

    Args:
      rotation_gain: Proportional gain used on the orientation error to compute
        the desired angular velocity.
      align_z_orientation: If true, the z rotation is constrained. If false, the
        z rotation is left as a DOF and will not be constrained.
    """
    self._p = rotation_gain
    self._align_z = align_z_orientation

  def step(
      self, current_quat: np.ndarray, desired_quat: np.ndarray
  ) -> np.ndarray:
    """Computes angular velocity to orient with a target quat."""

    error_quat = transformations.quat_diff_active(
        source_quat=current_quat, target_quat=desired_quat)

    if not self._align_z:
      # Ignore the z component of the error because the effector controls that.
      error_euler = transformations.quat_to_euler(error_quat)
      z_correction = transformations.euler_to_quat(
          np.array([0, 0, error_euler[2]]))

      # Compute a new target quat whose Z orientation is already aligned with
      # the current frame.
      new_desired_quat = transformations.quat_mul(desired_quat, z_correction)

      # Recompute the error with this new aligned target quat.
      error_quat = transformations.quat_diff_active(
          current_quat, new_desired_quat)

    # Normalize the error.
    error_quat = error_quat / np.linalg.norm(error_quat)
    return self._p * transformations.quat_to_axisangle(error_quat)


class Cartesian4dVelocityEffector(effector.Effector):
  """Effector for XYZ translational vel and Z angular vel.

  The X and Y angular velocities are also controlled internally in order to
  maintain the desired XY orientation, but those components are not exposed via
  the action spec.

  Also, we can change the orientation that is maintained by the robot by
  changing the `target_alignement` parameter. By default the robot is
  facing downards.
  """

  def __init__(self,
               effector_6d: effector.Effector,
               element: _MjcfElement,
               effector_prefix: str,
               control_frame: Optional[geometry.Frame] = None,
               rotation_gain: float = 0.8,
               target_alignment: np.ndarray = DOWNFACING_EE_QUAT_WXYZ):
    """Initializes a QP-based 4D Cartesian velocity effector.

    Args:
      effector_6d: Cartesian 6D velocity effector controlling the linear and
        angular velocities of `element`. The action spec of this effector
        should have a shape of 6 elements corresponding to the 6D velocity.
        The effector should have a property `control_frame` that returns the
        geometry.Frame in which the command should be expressed.
      element: the `mjcf.Element` being controlled. The 4D Cartesian velocity
        commands are expressed about the element's origin in the world
        orientation, unless overridden by `control_frame`. Only site elements
        are supported.
      effector_prefix: Prefix to the actuator names in the action spec.
      control_frame: `geometry.Frame` in which to interpet the Cartesian 4D
        velocity command. If `None`, assumes that the control command is
        expressed about the element's origin in the world orientation. Note that
        this is different than expressing the Cartesian 6D velocity about the
        element's own origin and orientation. Note that you will have to update
        the `target_alignment` quaternion if the frame does not have a downward
        z axis.
      rotation_gain: Gain applied to the rotational feedback control used to
          maintain a downward-facing orientation. A value too low of this
          parameter will result in the robot not being able to maintain the
          desired orientation. A value too high will lead to oscillations.
      target_alignment: Unit quat [w, i, j, k] denoting the desired alignment
          of the element's frame, represented in the world frame. Defaults to
          facing downwards.
    """
    self._effector_6d = effector_6d
    self._element = element
    self._effector_prefix = effector_prefix
    self._target_quat = target_alignment
    self._orientation_ctrl = _OrientationController(rotation_gain)

    # The control frame is either provided by the user or defaults to the one
    # centered at the element with the world orientation.
    self._control_frame = control_frame or geometry.HybridPoseStamped(
        pose=None,
        frame=self._element,
        quaternion_override=geometry.PoseStamped(None, None))

    # We use the target frame to compute a "stabilizing angular velocity" such
    # that the z axis of the target frame aligns with the z axis of the
    # element's frame.
    self._target_frame = geometry.HybridPoseStamped(
        pose=None,
        frame=self._element,
        quaternion_override=geometry.PoseStamped(
            pose=geometry.Pose(position=None, quaternion=target_alignment)))

    self._element_frame = geometry.PoseStamped(pose=None, frame=element)

  def after_compile(self, mjcf_model: mjcf.RootElement,
                    physics: mjcf.Physics) -> None:
    self._effector_6d.after_compile(mjcf_model, physics)

  def initialize_episode(self, physics, random_state) -> None:
    self._effector_6d.initialize_episode(
        physics=physics, random_state=random_state)

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    action_spec_6d = self._effector_6d.action_spec(physics=physics)

    # We ensure that the effector_6d is indeed 6d.
    if action_spec_6d.shape != (6,):
      raise ValueError('The effector passed to cartesian_4d_velocity_effector` '
                       'should have an action spec of shape 6. Provided spec: '
                       f'{action_spec_6d.shape}')
    actuator_names = [(self.prefix + str(i)) for i in range(4)]
    return specs.BoundedArray(
        shape=(4,),
        dtype=action_spec_6d.dtype,
        minimum=action_spec_6d.minimum[[0, 1, 2, 5]],
        maximum=action_spec_6d.maximum[[0, 1, 2, 5]],
        name='\t'.join(actuator_names))

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    """Sets a 4 DoF Cartesian velocity command at the current timestep.

    Args:
      physics: The physics object with the updated environment state at the
        current timestep.
      command: Array of size 4 describing the desired Cartesian target vel
        [lin_x, lin_y, lin_z, rot_z] expressed in the control frame.
    """
    if command.size != 4:
      raise ValueError('set_control: command must be an np.ndarray of size 4. '
                       f'Got {command.size}.')

    # Turn the input into a 6D velocity expressed in the control frame.
    twist = np.zeros(6)
    twist[0:3] = command[0:3]
    twist[5] = command[3]
    twist_stamped = geometry.TwistStamped(twist, self._control_frame)

    # We then project the velocity to the target frame.
    twist_target_frame = copy.copy(twist_stamped.to_frame(
        self._target_frame, physics=mujoco_physics.wrap(physics)).twist.full)

    # We compute the angular velocity that aligns the element's frame with
    # the target frame.
    stabilizing_vel = self._compute_stabilizing_ang_vel(physics)

    # We only use the x and y components of the stabilizing velocity and keep
    # the z rotation component of the command.
    twist_target_frame[3:5] = stabilizing_vel[0:2]
    twist_stamped_target_frame = geometry.TwistStamped(
        twist_target_frame, self._target_frame)

    # Transform the command to the frame expected by the underlying 6D effector.
    try:
      twist = twist_stamped_target_frame.to_frame(
          self._effector_6d.control_frame,  # pytype: disable=attribute-error
          mujoco_physics.wrap(physics)).twist.full
    except AttributeError as error:
      raise AttributeError(
          'The 6D effector does not have a `control_frame` attribute.'
      ) from error

    self._effector_6d.set_control(physics=physics, command=twist)

  @property
  def prefix(self) -> str:
    return self._effector_prefix

  @property
  def control_frame(self) -> geometry.Frame:
    return self._control_frame

  def _compute_stabilizing_ang_vel(self, physics: mjcf.Physics) -> np.ndarray:
    """Returns the angular velocity to orient element frame with target quat.

    The returned velocity is expressed in the target frame.

    Args:
      physics: An instance of physics.
    """
    # Express the quaternion of both the element and the target in the target
    # frame. This ensures that the velocity returned by the orientation
    # controller will align the z axis of the element frame with the z axis of
    # the target frame.
    element_quat_target_frame = self._element_frame.to_frame(
        self._target_frame, mujoco_physics.wrap(physics)).pose.quaternion

    # The target quaternion is expressed in the target frame (the result is the
    # unit quaternion)
    target_quat_target_frame = self._target_frame.pose.quaternion

    return self._orientation_ctrl.step(
        current_quat=element_quat_target_frame,
        desired_quat=target_quat_target_frame)


def limit_to_workspace(
    cartesian_effector: Cartesian4dVelocityEffector,
    element: _MjcfElement,
    min_workspace_limits: np.ndarray,
    max_workspace_limits: np.ndarray,
    wrist_joint: Optional[_MjcfElement] = None,
    wrist_limits: Optional[Sequence[float]] = None,
    reverse_wrist_range: bool = False,
    pose_getter: Optional[Callable[[mjcf.Physics], np.ndarray]] = None,
    ) -> effector.Effector:
  """Returns an effector that restricts the 4D actions to a workspace.

  If wrist limits are provided, this effector will also restrict the Z rotation
  action to those limits.

  Args:
    cartesian_effector: 4D cartesian effector.
    element: `mjcf.Element` that defines the Cartesian frame about which the
      Cartesian velocity is defined.
    min_workspace_limits: Lower bound of the Cartesian workspace. Must be 3D.
    max_workspace_limits: Upper bound of the Cartesian workspace. Must be 3D.
    wrist_joint: Optional wrist joint of the arm being controlled by the
      effectors. If provided along with `wrist_limits`, then the Z rotation
      component of the action will be zero'd out when the wrist is beyond the
      limits. You can typically get the wrist joint with `arm.joints[-1]`.
    wrist_limits: Optional 2D list/tuple (min wrist limit, max wrist limit). If
      provided along with `wrist_joint`, the Z rotation component of the action
      will be set to 0 when the wrist is beyond these limits.
    reverse_wrist_range: For some arms, a positive Z rotation action actually
      decreases the wrist joint position. For these arms, set this param to
      True.
    pose_getter: Optional function that returns the pose we want to constrain
      to the workspace. If `None`, defaults to the `xpos` of the `element`.
  """
  if len(min_workspace_limits) != 3 or len(max_workspace_limits) != 3:
    raise ValueError('The workspace limits must be 3D (X, Y, Z). Provided '
                     f'min: {min_workspace_limits} and max: '
                     f'{max_workspace_limits}')
  pose_getter = pose_getter or (lambda phys: phys.bind(element).xpos)
  def state_getter(physics):
    pos = pose_getter(physics)
    if wrist_joint is not None and wrist_limits is not None:
      wrist_state = physics.bind(wrist_joint).qpos
      if reverse_wrist_range:
        wrist_state = -wrist_state
    else:
      # Even when no wrist limits are provided, we need to supply a 4D state to
      # match the action spec of the cartesian effector.
      wrist_state = [0.0]
    return np.concatenate((pos, wrist_state))
  if wrist_joint is not None and wrist_limits is not None:
    if reverse_wrist_range:
      wrist_limits = [-wrist_limits[1], -wrist_limits[0]]
    min_limits = np.concatenate((min_workspace_limits, [wrist_limits[0]]))
    max_limits = np.concatenate((max_workspace_limits, [wrist_limits[1]]))
  else:
    # Provide unused wrist limits. They will be compared to a constant 0.0.
    min_limits = np.concatenate((min_workspace_limits, [-1.]))
    max_limits = np.concatenate((max_workspace_limits, [1.]))
  return constrained_actions_effectors.ConstrainedActionEffector(
      delegate=cartesian_effector, min_limits=min_limits, max_limits=max_limits,
      state_getter=state_getter)
