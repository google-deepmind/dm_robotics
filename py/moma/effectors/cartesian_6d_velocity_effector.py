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

"""Cartesian 6D velocity (linear and angular) effector."""

from typing import Optional, Sequence, Tuple

from absl import logging
import dataclasses
from dm_control import mjcf
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings.enums import mjtJoint
from dm_env import specs
from dm_robotics.controllers import cartesian_6d_to_joint_velocity_mapper
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
from dm_robotics.moma import effector
from dm_robotics.moma.effectors import constrained_actions_effectors
import numpy as np

_MjcfElement = mjcf.element._ElementImpl  # pylint: disable=protected-access
_CartesianVelocityMapper = (cartesian_6d_to_joint_velocity_mapper.Mapper)
_CartesianVelocityMapperParams = (
    cartesian_6d_to_joint_velocity_mapper.Parameters)


def _get_joint_ids(mj_model: mujoco.wrapper.MjModel,
                   joints: Sequence[_MjcfElement]):
  """Returns the (unsorted) IDs for a list of joints. Joints must be 1 DoF."""
  joint_ids = []
  for joint in joints:
    joint_id = mj_model.name2id(joint.full_identifier, 'joint')
    joint_type = mj_model.jnt_type[joint_id]
    if not (joint_type == mjtJoint.mjJNT_HINGE or
            joint_type == mjtJoint.mjJNT_SLIDE):
      raise ValueError(
          'Only 1 DoF joints are supported at the moment. Joint with name '
          f'[{joint.full_identifier}] is not a 1 DoF joint.')
    joint_ids.append(joint_id)
  return joint_ids


def _get_element_type(element: _MjcfElement):
  """Returns the MuJoCo enum corresponding to the element type."""
  if element.tag == 'body':
    return mujoco.wrapper.mjbindings.enums.mjtObj.mjOBJ_BODY
  elif element.tag == 'geom':
    return mujoco.wrapper.mjbindings.enums.mjtObj.mjOBJ_GEOM
  elif element.tag == 'site':
    return mujoco.wrapper.mjbindings.enums.mjtObj.mjOBJ_SITE
  else:
    raise ValueError('Element must be a MuJoCo body, geom, or site. Got '
                     f'[{element.tag}].')


def _scale_cartesian_6d_velocity(cartesian_6d_vel: np.ndarray,
                                 max_lin_vel: float, max_rot_vel: float):
  """Scales down the linear and angular magnitudes of the cartesian_6d_vel."""
  lin_vel = cartesian_6d_vel[:3]
  rot_vel = cartesian_6d_vel[3:]
  lin_vel_norm = np.linalg.norm(lin_vel)
  rot_vel_norm = np.linalg.norm(rot_vel)
  if lin_vel_norm > max_lin_vel:
    lin_vel = lin_vel * max_lin_vel / lin_vel_norm
  if rot_vel_norm > max_rot_vel:
    rot_vel = rot_vel * max_rot_vel / rot_vel_norm
  return np.concatenate((lin_vel, rot_vel))


@dataclasses.dataclass
class ModelParams:
  """Helper class for the model parameters of Cartesian6dVelocityEffector.

  Attributes:
    element: the `mjcf.Element` being controlled. Cartesian velocity commands
      are expressed about the element's origin in the world orientation, unless
      overridden by `control_frame`. Only elements with tags `body`, `geom`, and
      `site` are supported.
    joints: sequence of `mjcf.Element` joint entities of the joints being
      controlled. Every element must correspond to a valid MuJoCo joint in
      `mjcf_model`. Only 1 DoF joints are supported. Velocity limits,
      acceleration limits, and nullspace references must be in the same order as
      this sequence.
    control_frame: `geometry.Frame` in which to interpet the Cartesian 6D
      velocity command. If `None`, assumes that the control command is expressed
      about the element's origin in the world orientation. Note that this is
      different than expressing the Cartesian 6D velocity about the element's
      own origin and orientation.
  """

  element: _MjcfElement
  joints: Sequence[_MjcfElement]
  control_frame: Optional[geometry.Frame] = None

  def set_qp_params(self, mjcf_model: mjcf.RootElement,
                    qp_params: _CartesianVelocityMapperParams):
    xml_string = mjcf_model.to_xml_string()
    assets = mjcf_model.get_assets()
    qp_params.model = mujoco.wrapper.MjModel.from_xml_string(
        xml_string, assets=assets)
    qp_params.joint_ids = _get_joint_ids(qp_params.model, self.joints)
    qp_params.object_type = _get_element_type(self.element)
    qp_params.object_name = self.element.full_identifier


@dataclasses.dataclass
class ControlParams:
  """Helper class for the control parameters of Cartesian6dVelocityEffector.

  Attributes:
    control_timestep_seconds: expected amount of time that the computed joint
      velocities will be held by the effector. If unsure, higher values are more
      conservative.
    max_lin_vel: (optional) linear velocity maximum magnitude.
    max_rot_vel: (optional) rotational velocity maximum magnitude.
    enable_joint_position_limits: (optional) whether to enable active joint
      limit avoidance. Joint limits are deduced from the mjcf_model passed to
      the after_compose function.
    joint_position_limit_velocity_scale: (optional) value (0,1] that defines how
      fast each joint is allowed to move towards the joint limits in each
      iteration. Values lower than 1 are safer but may make the joints move
      slowly. 0.95 is usually enough since it is not affected by Jacobian
      linearization. Ignored if `enable_joint_position_limits` is false.
    minimum_distance_from_joint_position_limit: (optional) offset in meters
      (slide joints) or radians (hinge joints) to be added to the limits.
      Positive values decrease the range of motion, negative values increase it
      (i.e. negative values allow penetration). Ignored if
      `enable_joint_position_limits` is false.
    joint_velocity_limits: (optional) array of maximum allowed magnitudes of
      joint velocities for each joint, in m/s (slide joints) or rad/s (hinge
      joints). Must be ordered according to the `joints` parameter passed to the
      Cartesian6dVelocityEffector during construction. If not specified, joint
      velocity magnitudes will not be limited. Tune this if you see the robot
      trace non-linear Cartesian paths for a constant Cartesian velocity
      command.
    joint_acceleration_limits: (optional) array of maximum allowed magnitudes of
      joint acceleration for each controllable joint, in m/s^2 (slide joints) or
      rad/s^2 (hinge joints). Must be ordered according to the `joints`
      parameter passed to the Cartesian6dVelocityEffector during construction.
      If limits are specified, the user must ensure that the `physics` object
      used by the Cartesian6dVelocityEffector has accurate joint velocity
      information at every timestep. If None, the joint acceleration will not be
      limited. Note that collision avoidance and joint position limits, if
      enabled, take precedence over these limits. This means that the joint
      acceleration limits may be violated if it is necessary to come to an
      immediate full-stop in order to avoid collisions.
    regularization_weight: (optional) scalar regularizer for damping the
      Jacobian solver.
    nullspace_joint_position_reference: preferred joint positions, if
      unspecified then the mid point of the joint ranges is used. Must be
      ordered according to the `joints` parameter passed to the
      Cartesian6dVelocityEffector during construction.
    nullspace_gain: (optional) a gain (0, 1] for the secondary control
      objective. Scaled by a factor of `1/control_timestep_seconds` internally.
      Nullspace control will be disabled if the gain is None.
    max_cartesian_velocity_control_iterations: maximum number of iterations that
      the internal LSQP solver is allowed to spend on the Cartesian velocity
      optimization problem (first hierarchy). If the internal solver is unable
      to find a feasible solution to the first hierarchy (i.e. without
      nullspace) within the specified number of iterations, it will set the
      joint effector command to zero.
    max_nullspace_control_iterations: maximum number of iterations that the
      internal LSQP solver is allowed to spend on the nullspace optimization
      problem (second hierarchy). If the internal solver is unable to find a
      feasible solution to the second hierarchy within the specified number of
      iterations, it will set the joint effector command to the solution of the
      first hierarchy. Ignored if nullspace control is disabled.
  """

  control_timestep_seconds: float
  max_lin_vel: float = 0.5
  max_rot_vel: float = 0.5

  enable_joint_position_limits: bool = True
  joint_position_limit_velocity_scale: float = 0.95
  minimum_distance_from_joint_position_limit: float = 0.01
  joint_velocity_limits: Optional[np.ndarray] = None
  joint_acceleration_limits: Optional[np.ndarray] = None

  regularization_weight: float = 0.01
  nullspace_joint_position_reference: Optional[np.ndarray] = None
  nullspace_gain: Optional[float] = 0.025
  max_cartesian_velocity_control_iterations: int = 300
  max_nullspace_control_iterations: int = 300

  def set_qp_params(self, qp_params: _CartesianVelocityMapperParams):
    """Configures `qp_params` with the ControlParams fields.

    Args:
      qp_params: QP parameters structure on which to set the parameters. The
        `model` and `joint_ids` must have been set.
    """
    joint_argsort = np.argsort(qp_params.joint_ids)

    qp_params.integration_timestep = self.control_timestep_seconds

    # Set joint limit avoidance if enabled.
    if self.enable_joint_position_limits:
      qp_params.enable_joint_position_limits = True
      qp_params.joint_position_limit_velocity_scale = (
          self.joint_position_limit_velocity_scale)
      qp_params.minimum_distance_from_joint_position_limit = (
          self.minimum_distance_from_joint_position_limit)
    else:
      qp_params.enable_joint_position_limits = False

    # Set velocity limits, if enabled.
    # Note that we have to pass them in joint-ID ascending order to the mapper.
    if self.joint_velocity_limits is not None:
      qp_params.enable_joint_velocity_limits = True
      qp_params.joint_velocity_magnitude_limits = (
          self.joint_velocity_limits[joint_argsort].tolist())
    else:
      qp_params.enable_joint_velocity_limits = False

    # Set acceleration limits, if enabled.
    # Note that we have to pass them in joint-ID ascending order to the mapper.
    if self.joint_acceleration_limits is not None:
      qp_params.enable_joint_acceleration_limits = True
      qp_params.remove_joint_acceleration_limits_if_in_conflict = True
      qp_params.joint_acceleration_magnitude_limits = (
          self.joint_acceleration_limits[joint_argsort].tolist())
    else:
      qp_params.enable_joint_acceleration_limits = False

    # We always check the solution validity, and return a zero-vector if no
    # valid solution was found.
    qp_params.check_solution_validity = True

    # Set Cartesian control iterations.
    qp_params.max_cartesian_velocity_control_iterations = (
        self.max_cartesian_velocity_control_iterations)

    # Set regularization weight to prevent high joint velocities near singular
    # configurations.
    qp_params.regularization_weight = self.regularization_weight

    # We always set our tolerance to 1.0e-3, as any value smaller than that is
    # unlikely to make a difference.
    qp_params.solution_tolerance = 1.0e-3

    # Set nullspace control if gain is valid. If reference is None, set to
    # middle of joint range. If nullspace fails, we simply return the
    # minimum-norm least-squares solution to the Cartesian problem.
    # Note that the nullspace reference is not sorted in ascending order yet, as
    # the nullspace bias needs to be sorted after computing the velocities.
    if self.nullspace_gain is not None and self.nullspace_gain > 0.0:
      qp_params.enable_nullspace_control = True
      qp_params.return_error_on_nullspace_failure = False
      qp_params.nullspace_projection_slack = 1.0e-4
      if self.nullspace_joint_position_reference is None:
        self.nullspace_joint_position_reference = 0.5 * np.sum(
            qp_params.model.jnt_range[qp_params.joint_ids, :], axis=1)
      qp_params.max_nullspace_control_iterations = (
          self.max_nullspace_control_iterations)
    else:
      qp_params.enable_nullspace_control = False


@dataclasses.dataclass
class CollisionParams:
  """Helper class for the collision parameters of Cartesian6dVelocityEffector.

  Attributes:
    collision_pairs: (optional) a sequence of collision pairs in which to
      perform active collision avoidance. A collision pair is defined as a tuple
      of two geom groups. A geom group is a sequence of geom names. For each
      collision pair, the controller will attempt to avoid collisions between
      every geom in the first pair with every geom in the second pair. Self
      collision is achieved by adding a collision pair with the same geom group
      in both tuple positions.
    collision_avoidance_normal_velocity_scale: (optional) value between (0, 1]
      that defines how fast each geom is allowed to move towards another in each
      iteration. Values lower than 1 are safer but may make the geoms move
      slower towards each other. In the literature, a common starting value is
      0.85. Ignored if collision_pairs is None.
    minimum_distance_from_collisions: (optional) defines the minimum distance
      that the solver will attempt to leave between any two geoms. A negative
      distance would allow the geoms to penetrate by the specified amount.
    collision_detection_distance: (optional) defines the distance between two
      geoms at which the active collision avoidance behaviour will start. A
      large value will cause collisions to be detected early, but may incur high
      computational costs. A negative value will cause the geoms to be detected
      only after they penetrate by the specified amount.
  """

  collision_pairs: Optional[Sequence[Tuple[Sequence[str],
                                           Sequence[str]]]] = None
  collision_avoidance_normal_velocity_scale: float = 0.85
  minimum_distance_from_collisions: float = 0.05
  collision_detection_distance: float = 0.5

  def set_qp_params(self, qp_params: _CartesianVelocityMapperParams):
    """Configures `qp_params` with the CollisionParams fields."""
    if self.collision_pairs:
      qp_params.enable_collision_avoidance = True
      qp_params.collision_avoidance_normal_velocity_scale = (
          self.collision_avoidance_normal_velocity_scale)
      qp_params.minimum_distance_from_collisions = (
          self.minimum_distance_from_collisions)
      qp_params.collision_detection_distance = (
          qp_params.collision_detection_distance)
      qp_params.collision_pairs = self.collision_pairs
    else:
      qp_params.enable_collision_avoidance = False


class Cartesian6dVelocityEffector(effector.Effector):
  """A Cartesian 6D velocity effector interface for a robot arm."""

  def __init__(self,
               robot_name: str,
               joint_velocity_effector: effector.Effector,
               model_params: ModelParams,
               control_params: ControlParams,
               collision_params: Optional[CollisionParams] = None,
               log_nullspace_failure_warnings: bool = False):
    """Initializes a QP-based 6D Cartesian velocity effector.

    Args:
      robot_name: name of the robot the Cartesian effector controls.
      joint_velocity_effector: `Effector` on the joint velocities being
        controlled to achieve the target Cartesian velocity. This class takes
        ownership of this effector, i.e. it will call `initialize_episode`
        automatically.
      model_params: parameters that describe the object being controlled.
      control_params: parameters that describe how the element should be
        controlled.
      collision_params: parameters that describe the active collision avoidance
        behaviour, if any.
      log_nullspace_failure_warnings: if true, a warning will be logged
        if the internal LSQP solver is unable to solve the nullspace
        optimization problem (second hierarchy). Ignored if nullspace control is
        disabled.
    """
    self._effector_prefix = f'{robot_name}_twist'
    self._joint_velocity_effector = joint_velocity_effector
    self._joints = model_params.joints
    self._model_params = model_params
    self._control_params = control_params
    self._collision_params = collision_params
    self._control_frame = model_params.control_frame
    self._log_nullspace_failure_warnings = log_nullspace_failure_warnings

    # These are created in after_compose, once the mjcf_model is finalized.
    self._qp_mapper = None
    self._qp_frame = None
    self._joints_argsort = None

  def after_compile(self, mjcf_model: mjcf.RootElement) -> None:
    # Construct the QP-based mapper.
    qp_params = _CartesianVelocityMapperParams()

    self._model_params.set_qp_params(mjcf_model, qp_params)
    self._control_params.set_qp_params(qp_params)
    if self._collision_params:
      self._collision_params.set_qp_params(qp_params)
    qp_params.log_nullspace_failure_warnings = (
        self._log_nullspace_failure_warnings)

    self._qp_mapper = _CartesianVelocityMapper(qp_params)

    # Array of indices that would sort the joints in ascending order.
    # This is necessary because the mapper's inputs and outputs are always in
    # joint-ID ascending order, but the effector control should be passed
    # in the same order as the joints.
    self._joints_argsort = np.argsort(qp_params.joint_ids)

    # The mapper always expects the Cartesian velocity target to be expressed in
    # the element's origin in the world's orientation.
    self._qp_frame = geometry.HybridPoseStamped(
        pose=None,
        frame=self._model_params.element,
        quaternion_override=geometry.PoseStamped(None, None))

  def initialize_episode(self, physics, random_state) -> None:
    # Initialize the joint velocity effector.
    self._joint_velocity_effector.initialize_episode(physics, random_state)

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    lin = abs(self._control_params.max_lin_vel)
    rot = abs(self._control_params.max_rot_vel)
    max_6d_vel = np.asarray([lin, lin, lin, rot, rot, rot])
    actuator_names = [(self.prefix + str(i)) for i in range(6)]
    return specs.BoundedArray(
        shape=(6,),
        dtype=np.float32,
        minimum=-1.0 * max_6d_vel,
        maximum=max_6d_vel,
        name='\t'.join(actuator_names))

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    """Sets a 6 DoF Cartesian velocity command at the current timestep.

    Args:
      physics: `mjcf.Physics` object with the updated environment state at the
        current timestep.
      command: array of size 6 describing the desired 6 DoF Cartesian target
        [(lin_vel), (ang_vel)].
    """
    if command.size != 6:
      raise ValueError('set_control: command must be an np.ndarray of size 6. '
                       f'Got {command.size}.')

    cartesian_6d_target = np.copy(command)

    # If `control_frame` is None, we assume its frame to be the same as the
    # QP, and thus no transformation is needed.
    if self._control_frame is not None:
      # Transform the command from the target frame to the QP frame.
      stamped_command = geometry.TwistStamped(cartesian_6d_target,
                                              self._control_frame)
      cartesian_6d_target = stamped_command.get_relative_twist(
          self._qp_frame, mujoco_physics.wrap(physics)).full

    # Scale the Cartesian 6D velocity target if outside of Cartesian velocity
    # limits.
    cartesian_6d_target = _scale_cartesian_6d_velocity(
        cartesian_6d_target, self._control_params.max_lin_vel,
        self._control_params.max_rot_vel)

    # Compute the joint velocities and set the control on underlying velocity
    # effector.
    self._joint_velocity_effector.set_control(
        physics,
        self._compute_joint_velocities(
            physics=physics, cartesian_6d_target=cartesian_6d_target))

  @property
  def prefix(self) -> str:
    return self._effector_prefix

  @property
  def control_frame(self) -> geometry.Frame:
    """Returns the frame in which actions are expected."""
    return self._control_frame or self._qp_frame

  def _compute_joint_velocities(self, physics: mjcf.Physics,
                                cartesian_6d_target: np.ndarray) -> np.ndarray:
    """Maps a Cartesian 6D target velocity to joint velocities.

    Args:
      physics: `mjcf.Physics` object with the updated environment state at the
        current timestep.
      cartesian_6d_target: array of size 6 describing the desired 6 DoF
        Cartesian target [(lin_vel), (ang_vel)]. Must be expressed about the
        element's origin in the world orientation.

    Returns:
      Computed joint velocities in the same order as the `joints` sequence
      passed during construction.
    """
    joints_binding = physics.bind(self._joints)
    if joints_binding is None:
      raise ValueError(
          '_compute_joint_velocities: could not bind the joint elements passed '
          'on construction to the physics object.')

    joint_velocities = np.empty(len(self._joints), dtype=np.float32)

    # Compute nullspace bias if gain is positive.
    qdot_nullspace = None
    if (self._control_params.nullspace_gain is not None and
        self._control_params.nullspace_gain > 0.0):
      qdot_nullspace = self._control_params.nullspace_gain * (
          self._control_params.nullspace_joint_position_reference -
          joints_binding.qpos) / self._control_params.control_timestep_seconds
      # Reorder qdot_nullspace such that the nullspace bias is in ascending
      # order relative to the joint IDs.
      qdot_nullspace = qdot_nullspace[self._joints_argsort]

    # Compute joint velocities. The Python bindings throw an exception whenever
    # the mapper fails to find a solution, in which case we set the joint
    # velocities to zero.
    # We need to catch a general exception because the StatusOr->Exception
    # conversion can result in a wide variety of different exceptions.
    # The only special case is when the user calls CTRL-C, in which case we
    # re-raise the KeyboardInterrupt exception arising from SIGINT.
    try:
      # Note that we need to make sure that the joint velocities are in the same
      # order as the joints sequence, which may be different from that the QP
      # returns, i.e. in ascending order.
      joint_velocities[self._joints_argsort] = np.array(
          self._qp_mapper.compute_joint_velocities(physics.data,
                                                   cartesian_6d_target.tolist(),
                                                   qdot_nullspace),
          dtype=np.float32)
    except KeyboardInterrupt:
      logging.warning('_compute_joint_velocities: Computation interrupted!')
      raise
    except Exception as e:  # pylint: disable=broad-except
      joint_velocities.fill(0.0)
      logging.warning(
          ('_compute_joint_velocities: Failed to compute joint velocities. '
           'Setting joint velocities to zero. Error: [%s]'), str(e))

    return joint_velocities


def limit_to_workspace(
    cartesian_effector: Cartesian6dVelocityEffector,
    element: _MjcfElement,
    min_workspace_limits: np.ndarray,
    max_workspace_limits: np.ndarray,
) -> effector.Effector:
  """Returns an effector that restricts the 6D actions to a workspace.

  Constraining the rotation of the end effector is currently not
  supported.

  Args:
    cartesian_effector: 6D cartesian effector.
    element: `mjcf.Element` that defines the Cartesian frame about which the
      Cartesian velocity is defined.
    min_workspace_limits: Lower bound of the Cartesian workspace. Must be 3D.
    max_workspace_limits: Upper bound of the Cartesian workspace. Must be 3D.
  """
  if len(min_workspace_limits) != 3 or len(max_workspace_limits) != 3:
    raise ValueError('The workspace limits must be 3D (X, Y, Z). Provided '
                     f'min: {min_workspace_limits} and max: '
                     f'{max_workspace_limits}')

  def state_getter(physics):
    pos = physics.bind(element).xpos
    # Even when no wrist limits are provided, we need to supply a 6D state to
    # match the action spec of the cartesian effector.
    wrist_state = [0.0] * 3
    return np.concatenate((pos, wrist_state))

  # Provide unused wrist limits. They will be compared to a constant 0.0.
  min_limits = np.concatenate((min_workspace_limits, [-1.] * 3))
  max_limits = np.concatenate((max_workspace_limits, [1.] * 3))
  return constrained_actions_effectors.ConstrainedActionEffector(
      delegate=cartesian_effector,
      min_limits=min_limits,
      max_limits=max_limits,
      state_getter=state_getter)
