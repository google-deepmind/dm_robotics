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

"""IK solver for initialization of robot arms."""

import copy
from typing import List, NamedTuple, Optional, Sequence, Union

from absl import logging
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.wrapper.mjbindings.enums import mjtObj
from dm_robotics.controllers import cartesian_6d_to_joint_velocity_mapper
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
from dm_robotics.transformations import transformations as tr
import numpy as np

# Default value for the nullspace gain parameter.
_NULLSPACE_GAIN = 0.4

# Gain for the linear and angular twist computation, these values should always
# be between 0 and 1. 0 corresponds to not move and 1 corresponds to move to the
# target in a single integration timestep.
_LINEAR_VELOCITY_GAIN = 0.95
_ANGULAR_VELOCITY_GAIN = 0.95

# Integration timestep used when solving the IK.
_INTEGRATION_TIMESTEP_SEC = 1.0


# At each step of the solve, we measure how much the tracked element
# translated (linear progress) and rotated (angular progress). We compare this
# progress to the total linear and angular error and if not enough progress is
# made stop the solve before the maximum number of steps is reached.
_ERR_TO_PROGRESS_THRESHOLD = 20.0

### ---------------PARAMETERS USED FOR THE QP MAPPER: START----------------- ###
# Regularisation parameter used by the qp to compute joint velocities.
_REGULARIZATION_WEIGHT = 0.01

# Ensure that the joint limits are respected.
_ENABLE_JOINT_POSITION_LIMITS = True

# Gain that scales the joint velocities down when close to the joint limits.
_JOINT_POSITION_LIMIT_VELOCITY_SCALE = 1.0

# The minimal distance to joint limits the IK solution can have.
_MINIMUM_DISTANCE_FROM_JOINT_POSITION_LIMIT = 0.0

# Maximum number of iteration to find a joint velocity command that applies the
# desired twist to the element.
_MAX_CARTESIAN_VELOCITY_CONTROL_ITERATIONS = 300

# Number of iterations for the nullspace control problem.
_MAX_NULLSPACE_CONTROL_ITERATIONS = 300

# Maximum error allowed for the nullspace problem.
_NULLSPACE_PROJECTION_SLACK = 1e-5

# Maximum error allowed between the requested twist command and the actual one.
_SOLUTION_TOLERANCE = 1e-4

# Remove the logging when the nullspace cannot find a solution as this
# clutters the logging.
_LOG_NULLSPACE_FAILURE_WARNINGS = False
### -----------------PARAMETERS USED FOR THE QP MAPPER: END----------------- ###

_Binding = Union[mjcf.physics.Binding, mjcf.physics._EmptyBinding]  # pylint: disable=protected-access
_MjcfElement = mjcf.element._ElementImpl  # pylint: disable=protected-access


class _Solution(NamedTuple):
  """Return value of an ik solution.

  Attributes:
    qpos: The joint configuration.
    linear_err: The linear error between the target pose and desired pose.
    angular_err: The angular error between the target pose and desired pose.
  """
  qpos: np.ndarray
  linear_err: float
  angular_err: float


class IkSolver():
  """Inverse kinematics solver.

  This class computes a joint configuration that brings an element to a certain
  pose.
  """

  # The cartesian velocity controller used to solve the IK.
  _qp_mapper: cartesian_6d_to_joint_velocity_mapper.Mapper

  # Array of indices that sorts the joints in ascending order. The qp_mapper
  # returns values in joint-ID ascending order which could be different than
  # the order of the joints provided by the user.
  _joints_argsort: List[int]

  # The desired joint configuration that is set as the nullspace goal. This
  # corresponds to the mid-range of each joint. The user can override this
  # reference configuration in the `solve` method.
  _nullspace_joint_position_reference: List[float]

  def __init__(
      self,
      model: mjcf.RootElement,
      controllable_joints: List[_MjcfElement],
      element: _MjcfElement,
      nullspace_gain: float = _NULLSPACE_GAIN,
      ):
    """Constructor.

    Args:
      model: The MJCF model root.
      controllable_joints: The joints that can be controlled to achieve
        the desired target pose. Only 1 DoF joints are supported.
      element: The MJCF element that is being placed by the inverse kinematics
        solver. Only body, geoms, and sites are supported
      nullspace_gain: Scales the nullspace velocity bias. If the gain is set to
        0, there will be no nullspace optimization during the solve process.
    """
    self._physics = mjcf.Physics.from_mjcf_model(model)
    self._geometry_physics = mujoco_physics.wrap(self._physics)
    self._joints_binding = _binding(self._physics, controllable_joints)
    self._num_joints = len(controllable_joints)
    self._element = element
    self._nullspace_gain = nullspace_gain
    self._create_qp_mapper()

  def solve(self,
            ref_pose: geometry.Pose,
            linear_tol: float = 1e-3,
            angular_tol: float = 1e-3,
            max_steps: int = 100,
            early_stop: bool = False,
            num_attempts: int = 30,
            stop_on_first_successful_attempt: bool = False,
            inital_joint_configuration: Optional[np.ndarray] = None,
            nullspace_reference: Optional[np.ndarray] = None
            ) -> Optional[np.ndarray]:
    """Attempts to solve the inverse kinematics.

    This method computes joint configuration that solves the inverse kinematics
    problem. Returns None if no solution is found. If multiple solutions are
    found, the solver will return the one where the joints are closer to the
    `nullspace_reference`. If none is provided uses the center of the joint
    ranges

    Args:
      ref_pose: Target pose of the controlled element, it must be
        in the world frame.
      linear_tol: The linear tolerance, in meters, that determines if the
        solution found is valid.
      angular_tol: The angular tolerance, in radians, to determine if the
        solution found is valid.
      max_steps: Maximum number of integration steps that can be used. The
        larger the number of steps the more likely it is a solution will be
        found but a larger number of steps increases computation time.
      early_stop: If true, stops the attempt as soon as the configuration is
        within the linear and angular tolerances. If false, it will always run
        `max_steps` iterations per attempt and return the last configuration.
      num_attempts: The number of different attempts the solver should do.
        For a given target pose, there exists an infinite number of possible
        solutions, having more attempts allows to compare different joint
        configurations. The solver will return the solution where the joints are
        closer to the `nullspace_reference`. Note that not all attempts
        are successful, and thus, having more attempts gives better chances of
        finding a correct solution.
      stop_on_first_successful_attempt: If true, the method will return the
        first solution that meets the tolerance criteria. If false, returns the
        solution where the joints are closer the center of their respective
        range.
      inital_joint_configuration: A joint configuration that will be used for
        the first attempt. This can be useful in the case of a complex pose,
        a user could provide the initial guess that is close to the desired
        solution. If None, all the joints will be set to 0 for the first
        attempt.
      nullspace_reference: The desired joint configuration. When the controlled
       element is in the desired pose, the solver will try and bring the joint
       configuration closer to the nullspace reference without moving the
       element. If no nullspace reference is provided, the center of the joint
       ranges is used as reference.

    Returns:
      If a solution is found, returns the corresponding joint configuration.
      If the inverse kinematics failed, returns None.

    Raises:
      ValueError: If the `nullspace_reference` does not have the correct length.
      ValueError: If the `inital_joint_configuration` does not have the correct
        length.
    """

    nullspace_reference = (
        nullspace_reference or self._nullspace_joint_position_reference)
    if len(nullspace_reference) != self._num_joints:
      raise ValueError(
          'The provided nullspace reference does not have the right number of '
          f'elements expected length of {self._num_joints}.'
          f' Got {nullspace_reference}')

    if inital_joint_configuration is not None:
      if len(inital_joint_configuration) != self._num_joints:
        raise ValueError(
            'The provided inital joint configuration does not have the right '
            f'number of elements expected length of {self._num_joints}.'
            f' Got {inital_joint_configuration}')

    inital_joint_configuration = inital_joint_configuration or np.zeros(
        self._num_joints)

    nullspace_jnt_qpos_min_err = np.inf
    sol_qpos = None
    success = False

    # Each iteration of this loop attempts to solve the inverse kinematics.
    # If a solution is found, it is compared to previous solutions.
    for attempt in range(num_attempts):

      # Use the user provided joint configuration for the first attempt.
      if attempt == 0:
        self._joints_binding.qpos[:] = inital_joint_configuration
      else:
        # Randomize the initial joint configuration so that the IK can find
        # different solutions.
        qpos_new = np.random.uniform(
            self._joints_binding.range[:, 0], self._joints_binding.range[:, 1])
        self._joints_binding.qpos[:] = qpos_new

      # Solve the IK.
      joint_qpos, linear_err, angular_err = self._solve_ik(
          ref_pose, linear_tol, angular_tol, max_steps,
          early_stop, nullspace_reference)

      # Check if the attempt was successful. The solution is saved if the joints
      # are closer to the nullspace reference.
      if (linear_err <= linear_tol and angular_err <= angular_tol):
        success = True
        nullspace_jnt_qpos_err = np.linalg.norm(
            joint_qpos - nullspace_reference)
        if nullspace_jnt_qpos_err < nullspace_jnt_qpos_min_err:
          nullspace_jnt_qpos_min_err = nullspace_jnt_qpos_err
          sol_qpos = joint_qpos

      if success and stop_on_first_successful_attempt:
        break

    if not success:
      logging.warning('Unable to solve the inverse kinematics for ref_pose: '
                      '%s', ref_pose)
    return sol_qpos

  def _create_qp_mapper(self):
    """Instantiates the cartesian velocity controller used by the ik solver."""

    qp_params = cartesian_6d_to_joint_velocity_mapper.Parameters()
    qp_params.model = self._physics.model
    qp_params.joint_ids = self._joints_binding.jntid
    qp_params.object_type = _get_element_type(self._element)
    qp_params.object_name = self._element.full_identifier

    qp_params.integration_timestep = _INTEGRATION_TIMESTEP_SEC
    qp_params.enable_joint_position_limits = _ENABLE_JOINT_POSITION_LIMITS
    qp_params.joint_position_limit_velocity_scale = (
        _JOINT_POSITION_LIMIT_VELOCITY_SCALE)
    qp_params.minimum_distance_from_joint_position_limit = (
        _MINIMUM_DISTANCE_FROM_JOINT_POSITION_LIMIT)

    qp_params.regularization_weight = _REGULARIZATION_WEIGHT
    qp_params.max_cartesian_velocity_control_iterations = (
        _MAX_CARTESIAN_VELOCITY_CONTROL_ITERATIONS)
    if self._nullspace_gain > 0:
      qp_params.enable_nullspace_control = True
    else:
      qp_params.enable_nullspace_control = False
    qp_params.max_nullspace_control_iterations = (
        _MAX_NULLSPACE_CONTROL_ITERATIONS)
    qp_params.nullspace_projection_slack = _NULLSPACE_PROJECTION_SLACK
    qp_params.solution_tolerance = _SOLUTION_TOLERANCE
    qp_params.log_nullspace_failure_warnings = _LOG_NULLSPACE_FAILURE_WARNINGS

    self._qp_mapper = cartesian_6d_to_joint_velocity_mapper.Mapper(qp_params)
    self._joints_argsort = np.argsort(self._joints_binding.jntid)
    self._nullspace_joint_position_reference = 0.5 * np.sum(
        self._joints_binding.range, axis=1)

  def _solve_ik(self,
                ref_pose: geometry.Pose,
                linear_tol: float,
                angular_tol: float,
                max_steps: int,
                early_stop: bool,
                nullspace_reference: np.ndarray
                ) -> _Solution:
    """Finds a joint configuration that brings element pose to target pose."""

    cur_frame = geometry.PoseStamped(pose=None, frame=self._element)
    linear_err = np.inf
    angular_err = np.inf
    cur_pose = cur_frame.get_world_pose(self._geometry_physics)
    previous_pose = copy.copy(cur_pose)

    # Each iteration of this loop attempts to reduce the error between the
    # element's pose and the target pose.
    for _ in range(max_steps):
      # Find the twist that will bring the element's pose closer to the desired
      # one.
      twist = _compute_twist(
          cur_pose, ref_pose, _LINEAR_VELOCITY_GAIN,
          _ANGULAR_VELOCITY_GAIN, _INTEGRATION_TIMESTEP_SEC)

      # Computes the joint velocities to achieve the desired twist.
      # The joint velocity vector passed to mujoco's integration
      # needs to have a value for all the joints in the model. The velocity
      # for all the joints that are not controlled is set to 0.
      qdot_sol = np.zeros(self._physics.model.nv)
      joint_vel = self._compute_joint_velocities(
          twist.full, nullspace_reference)

      # If we are unable to compute joint velocities we stop the iteration
      # as the solver is stuck and cannot make any more progress.
      if joint_vel is not None:
        qdot_sol[self._joints_binding.dofadr] = joint_vel
      else:
        break

      # The velocity vector is passed to mujoco to be integrated.
      mjbindings.mjlib.mj_integratePos(
          self._physics.model.ptr, self._physics.data.qpos,
          qdot_sol, _INTEGRATION_TIMESTEP_SEC)
      self._update_physics_data()

      # Get the distance and the angle between the current pose and the
      # target pose.
      cur_pose = cur_frame.get_world_pose(self._geometry_physics)
      linear_err = np.linalg.norm(ref_pose.position - cur_pose.position)
      angular_err = np.linalg.norm(_get_orientation_error(
          ref_pose.quaternion, cur_pose.quaternion))

      # Stop if the pose is close enough to the target pose.
      if (early_stop and
          linear_err <= linear_tol and angular_err <= angular_tol):
        break

      # We measure the progress made during this step. If the error is not
      # reduced fast enough the solve is stopped to save computation time.
      linear_change = np.linalg.norm(
          cur_pose.position - previous_pose.position)
      angular_change = np.linalg.norm(_get_orientation_error(
          cur_pose.quaternion, previous_pose.quaternion))
      if (linear_err / (linear_change + 1e-10) > _ERR_TO_PROGRESS_THRESHOLD and
          angular_err / (angular_change + 1e-10) > _ERR_TO_PROGRESS_THRESHOLD):
        break
      previous_pose = copy.copy(cur_pose)

    qpos = np.array(self._joints_binding.qpos)
    return _Solution(qpos=qpos, linear_err=linear_err, angular_err=angular_err)

  def _compute_joint_velocities(
      self, cartesian_6d_target: np.ndarray, nullspace_reference: np.ndarray
      ) -> Optional[np.ndarray]:
    """Maps a Cartesian 6D target velocity to joint velocities.

    Args:
      cartesian_6d_target: array of size 6 describing the desired 6 DoF
        Cartesian target [(lin_vel), (ang_vel)]. Must be expressed about the
        element's origin in the world orientation.
      nullspace_reference: The desired joint configuration used to compute
        the nullspace bias.

    Returns:
      Computed joint velocities in the same order as the `joints` sequence
      passed during construction. If a solution could not be found,
      returns None.
    """

    joint_velocities = np.empty(self._num_joints)

    nullspace_bias = None
    if self._nullspace_gain > 0:
      nullspace_bias = self._nullspace_gain * (
          nullspace_reference
          - self._joints_binding.qpos) / _INTEGRATION_TIMESTEP_SEC

      # Sort nullspace_bias by joint ID, ascending. The QP requires this.
      nullspace_bias = nullspace_bias[self._joints_argsort]

    # Compute joint velocities. The Python bindings throw an exception whenever
    # the mapper fails to find a solution, in which case we return None.
    # We need to catch a general exception because the StatusOr->Exception
    # conversion can result in a wide variety of different exceptions.
    try:
      # Reorder the joint velocities to be in the same order as the joints
      # sequence. The QP returns joints by ascending joint ID which could be
      # different.
      joint_velocities[self._joints_argsort] = np.array(
          self._qp_mapper.compute_joint_velocities(
              self._physics.data, cartesian_6d_target.tolist(), nullspace_bias))
    except Exception:  # pylint: disable=broad-except
      joint_velocities = None
      logging.warning('Failed to compute joint velocities, returning None.')

    return joint_velocities

  def _update_physics_data(self):
    """Updates the physics data following the integration of the velocities."""

    # Clip joint positions; the integration done previously can make joints
    # out of range.
    qpos = self._joints_binding.qpos
    min_range = self._joints_binding.range[:, 0]
    max_range = self._joints_binding.range[:, 1]
    qpos = np.clip(qpos, min_range, max_range)
    self._joints_binding.qpos[:] = qpos

    # Forward kinematics to update the pose of the tracked element.
    mjbindings.mjlib.mj_normalizeQuat(
        self._physics.model.ptr, self._physics.data.qpos)
    mjbindings.mjlib.mj_kinematics(
        self._physics.model.ptr, self._physics.data.ptr)
    mjbindings.mjlib.mj_comPos(self._physics.model.ptr, self._physics.data.ptr)


def _get_element_type(element: _MjcfElement):
  """Returns the MuJoCo enum corresponding to the element type."""
  if element.tag == 'body':
    return mjtObj.mjOBJ_BODY
  elif element.tag == 'geom':
    return mjtObj.mjOBJ_GEOM
  elif element.tag == 'site':
    return mjtObj.mjOBJ_SITE
  else:
    raise ValueError('Element must be a MuJoCo body, geom, or site. Got '
                     f'[{element.tag}].')


def _binding(physics: mjcf.Physics,
             elements: Union[Sequence[mjcf.Element], mjcf.Element]
             ) -> _Binding:
  """Binds the elements with physics and returns a non optional object."""
  physics_elements = physics.bind(elements)
  if physics_elements is None:
    raise ValueError(f'Calling physics.bind with {elements} returns None.')
  return physics_elements


def _get_orientation_error(
    to_quat: np.ndarray, from_quat: np.ndarray) -> np.ndarray:
  """Returns error between the two quaternions."""
  err_quat = tr.quat_diff_active(from_quat, to_quat)
  return tr.quat_to_axisangle(err_quat)


def _compute_twist(init_pose: geometry.Pose,
                   ref_pose: geometry.Pose,
                   linear_velocity_gain: float,
                   angular_velocity_gain: float,
                   control_timestep_seconds: float,
                   ) -> geometry.Twist:
  """Returns the twist to apply to the end effector to reach ref_pose.

  This function returns the twist that moves init_pose closer to ref_pose.
  Both poses need to be expressed in the same frame. The returned twist is
  expressed in the frame located at the initial pose and with the same
  orientation as the world frame.

  Args:
    init_pose: The inital pose.
    ref_pose: The target pose that we want to reach.
    linear_velocity_gain: Scales the linear velocity. The value should be
      between 0 and 1. A value of 0 corresponds to not moving. A value of 1
      corresponds to moving from the inital pose to the target pose in a
      single timestep.
    angular_velocity_gain: Scales the angualr velocity. The value should be
      between 0 and 1. A value of 0 corresponds to not rotating. A value of 1
      corresponds to rotating from the inital pose to the target pose in a
      single timestep.
    control_timestep_seconds: Duration of the control timestep. The outputed
      twist is intended to be used over that duration.

  Returns:
    The twist to be applied to `init_pose` to move in closer to `ref_pose`.
    The twist is expressed in the frame located at `init_pose` and with the
      same orientation as the world frame.
  """

  position_error = ref_pose.position - init_pose.position
  orientation_error = _get_orientation_error(
      from_quat=init_pose.quaternion, to_quat=ref_pose.quaternion)

  linear = linear_velocity_gain * position_error / control_timestep_seconds
  angular = angular_velocity_gain * orientation_error / control_timestep_seconds
  return geometry.Twist(np.concatenate((linear, angular)))
