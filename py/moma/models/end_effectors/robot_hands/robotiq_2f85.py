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
"""Robotiq 2-finger 85 adaptive gripper class."""

from typing import List, Tuple, Optional

from dm_control import mjcf
from dm_robotics.moma.models import types
from dm_robotics.moma.models import utils as models_utils
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85_constants as consts
import numpy as np


_GRIPPER_SITE_NAME = 'pinch_site'
_ACTUATOR_NAME = 'fingers_actuator'
_SENSOR_NAME = 'fingers_pos'
_JOINT_NAME = 'left_driver_joint'
_COLLISION_CLASS = 'reinforced_fingertip'
_PAD_GEOM_NAMES = [
    'reinforced_right_fingertip_geom',
    'reinforced_left_fingertip_geom',
]
_PAD_COLOR = (1., 1., 1., 1.)
_POS_SCALE = 255.
_VELOCITY_CTRL_TOL = 2
_DEFAULT_GRIPPER_FRICTION = (0.5, 0.1, 0.01)
_LEGACY_GRIPPER_FRICTION = (1.5, 0.1, 0.001)

# The torque tau that is applied to the actuator is:
# tau = gainprm[0] * act + bias[1] * q + bias[2] * q_dot
# where `act` is the current target reference of the actuator,
# `q` is the current position of the joint and `q_dot` is the joint velocity.
# `q_dot` is used for damping but we add it to the joint instead of the
# actuator. This has to do with the underlying simulation as joint damping is
# more stable.
# The control range of the actuator is [0, 255] but the actual joint range is
# [0 0.8] so we need to ensure that there is a mapping that is done from one
# range to the other. We want an effective torque of bias[0] Nm when the
# actuator is fully closed. Therefore we have the following equality:
# gainprm[0] * 255 + x * 0.8 = 0
# gainprm[0] = - bias[0] * 0.8 / 255
_GAINPRM = (100. * 0.8 / 255, 0.0, 0.0)
_BIASPRM = (0.0, -100, 0.0)

_BASE_COLLISION_KWARGS = [{
    'name': 'base_CollisionGeom_1',
    'type': 'cylinder',
    'pos': '0 0 0.01',
    'size': '0.04 0.024',
}, {
    'name': 'base_CollisionGeom_2',
    'type': 'sphere',
    'pos': '0 0 0.05',
    'size': '0.045',
}]
_RIGHT_DRIVER_COLLISION_KWARGS = [{
    'name': 'right_driver_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 0 0 0 0.027 0.0018',
    'size': '0.015',
}]
_RIGHT_COUPLER_COLLISION_KWARGS = [{
    'name': 'right_coupler_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 0 0 0 0.005 0.045',
    'size': '0.015',
}]
_RIGHT_SPRING_LINK_COLLISION_KWARGS = [{
    'name': 'right_spring_link_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 0 0 0 0.031 0.036',
    'size': '0.02',
}]
_RIGHT_FOLLOWER_COLLISION_KWARGS = [{
    'name': 'right_follower_CollisionGeom',
    'type': 'sphere',
    'pos': '0 -0.01 0.005',
    'size': '0.015',
}]
_LEFT_DRIVER_COLLISION_KWARGS = [{
    'name': 'left_driver_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 0 0 0 0.027 0.0018',
    'size': '0.015',
}]
_LEFT_COUPLER_COLLISION_KWARGS = [{
    'name': 'left_coupler_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 0 0 0 0.005 0.045',
    'size': '0.015',
}]
_LEFT_SPRING_LINK_COLLISION_KWARGS = [{
    'name': 'left_spring_link_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 0 0 0 0.031 0.036',
    'size': '0.02',
}]
_LEFT_FOLLOWER_COLLISION_KWARGS = [{
    'name': 'left_follower_CollisionGeom',
    'type': 'sphere',
    'pos': '0 -0.01 0.005',
    'size': '0.015',
}]

_RIGHT_PAD_COLLISION_KWARGS = [{
    'name': 'right_pad_CollisionGeom',
    'type': 'box',
    'pos': '0 0.004 0.019',
    'size': '0.012 0.01 0.019',
}]
_LEFT_PAD_COLLISION_KWARGS = [{
    'name': 'left_pad_CollisionGeom',
    'type': 'box',
    'pos': '0 0.004 0.019',
    'size': '0.012 0.01 0.019',
}]

# Dictionary mapping body names to a list of their collision geoms
_COLLISION_GEOMS_DICT = {
    'base': _BASE_COLLISION_KWARGS,
    'right_driver': _RIGHT_DRIVER_COLLISION_KWARGS,
    'right_coupler': _RIGHT_COUPLER_COLLISION_KWARGS,
    'right_spring_link': _RIGHT_SPRING_LINK_COLLISION_KWARGS,
    'right_follower': _RIGHT_FOLLOWER_COLLISION_KWARGS,
    'left_driver': _LEFT_DRIVER_COLLISION_KWARGS,
    'left_coupler': _LEFT_COUPLER_COLLISION_KWARGS,
    'left_spring_link': _LEFT_SPRING_LINK_COLLISION_KWARGS,
    'left_follower': _LEFT_FOLLOWER_COLLISION_KWARGS,
    'right_pad': _RIGHT_PAD_COLLISION_KWARGS,
    'left_pad': _LEFT_PAD_COLLISION_KWARGS,
}


class Robotiq2F85(robot_hand.RobotHand):
  """Robotiq 2-finger 85 adaptive gripper."""

  _mjcf_root: mjcf.RootElement

  def _build(self,
             name: str = 'robotiq_2f85',
             gainprm: Tuple[float, float, float] = _GAINPRM,
             biasprm: Tuple[float, float, float] = _BIASPRM,
             tcp_orientation: Optional[np.ndarray] = None,
             use_realistic_friction: bool = True):
    """Initializes the Robotiq 2-finger 85 gripper.

    Args:
      name: The name of this robot. Used as a prefix in the MJCF name
      gainprm: The gainprm of the finger actuator.
      biasprm: The biasprm of the finger actuator.
      tcp_orientation: Quaternion [w, x, y, z] representing the orientation of
        the tcp frame of the gripper. This is needed for compatibility between
        sim and real. This depends on which robot is being used so we need it to
        be parametrizable. If None, use the original tcp site.
      use_realistic_friction: If true will use friction parameters which result
        in a more realistic. Should only be set to False for backwards
        compatibility.
    """
    self._mjcf_root = mjcf.from_path(consts.XML_PATH)
    self._mjcf_root.model = name

    # If the user provided a quaternion, rotate the tcp site. Otherwise use the
    # default one.
    if tcp_orientation is not None:
      gripper_base = self.mjcf_model.find('body', 'base')
      gripper_base.add(
          'site',
          type='sphere',
          name='aligned_gripper_tcp',
          pos=consts.TCP_SITE_POS,
          quat=tcp_orientation)

      self._tool_center_point = self.mjcf_model.find(
          'site', 'aligned_gripper_tcp')
    else:
      self._tool_center_point = self._mjcf_root.find('site', _GRIPPER_SITE_NAME)

    self._finger_actuator = self._mjcf_root.find('actuator', _ACTUATOR_NAME)
    self._joint_sensor = self._mjcf_root.find('sensor', _SENSOR_NAME)
    self._joints = [self._mjcf_root.find('joint', _JOINT_NAME)]

    # Use integrated velocity control.
    self._define_integrated_velocity_actuator(gainprm, biasprm)

    # Cache the limits for the finger joint.
    joint_min, joint_max = self._finger_actuator.tendon.joint[
        0].joint.dclass.joint.range
    self._joint_offset = joint_min
    self._joint_scale = joint_max - joint_min

    self._color_pads()
    self._add_collision_geoms()
    self._add_collisions_boxes()
    self._set_physics_properties(use_realistic_friction)

  def _set_physics_properties(self, use_realistic_friction: bool):
    """Set physics related properties."""
    # Set collision and friction parameter to the same values as in the jaco
    # hand - as we know they work very stable.
    padbox_class = self._mjcf_root.find('default', _COLLISION_CLASS)
    if use_realistic_friction:
      padbox_class.geom.friction = _DEFAULT_GRIPPER_FRICTION
    else:
      padbox_class.geom.friction = _LEGACY_GRIPPER_FRICTION
    padbox_class.geom.solimp = (0.9, 0.95, 0.001)
    padbox_class.geom.solref = (-100000, -200)

    # Adapt spring link stiffness to allow proper initialisation and more
    # realistic behaviour. The original value will cause one of the links
    # to get stuck in a bent position after initialisation sometimes.
    spring_link_class = self._mjcf_root.find('default', 'spring_link')
    spring_link_class.joint.stiffness = 0.01

    # Adapt the driver joint to make movement of gripper slower, similar to the
    # real hardware.
    driver_class = self._mjcf_root.find('default', 'driver')
    driver_class.joint.armature = 0.1
    # Add in the damping on the joint level instead of on the actuator level
    # this results in a more stable damping.
    driver_class.joint.damping = 1

  def _add_collisions_boxes(self):
    """Adds two boxes to each of the fingertips to improve physics stability."""
    for side in ('left', 'right'):
      pad = self._mjcf_root.find('body', f'{side}_pad')
      pad.add(
          'geom',
          name=f'{side}_collision_box1',
          dclass='reinforced_fingertip',
          size=[0.007, 0.0021575, 0.005],
          type='box',
          rgba=[0.0, 0.0, 0.0, 0.0],
          pos=[0.0, 0.0117, 0.03])
      pad.add(
          'geom',
          name=f'{side}_collision_box2',
          dclass='reinforced_fingertip',
          size=[0.007, 0.0021575, 0.005],
          type='box',
          rgba=[0.0, 0.0, 0.0, 0.0],
          pos=[0.0, 0.0117, 0.015])

  def _add_collision_geoms(self):
    """Add collision geoms use by the QP velocity controller for avoidance."""
    self._collision_geoms = models_utils.attach_collision_geoms(
        self.mjcf_model, _COLLISION_GEOMS_DICT)

  def _color_pads(self) -> None:
    """Define the color for the gripper pads."""
    for geom_name in _PAD_GEOM_NAMES:
      geom = self._mjcf_root.find('geom', geom_name)
      geom.rgba = _PAD_COLOR

  def _define_integrated_velocity_actuator(self,
                                           gainprm: Tuple[float, float, float],
                                           biasprm: Tuple[float, float, float]):
    """Define integrated velocity actuator."""
    self._finger_actuator.ctrlrange = (-255.0, 255.0)
    self._finger_actuator.dyntype = 'integrator'
    self._finger_actuator.gainprm = gainprm
    self._finger_actuator.biasprm = biasprm

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState):
    """Function called at the beginning of every episode."""
    del random_state  # Unused.

    # Apply gravity compensation
    body_elements = self.mjcf_model.find_all('body')
    gravity = np.hstack([physics.model.opt.gravity, [0, 0, 0]])
    physics_bodies = physics.bind(body_elements)
    if physics_bodies is None:
      raise ValueError('Calling physics.bind with bodies returns None.')
    physics_bodies.xfrc_applied[:] = -gravity * physics_bodies.mass[..., None]

  @property
  def joints(self) -> List[types.MjcfElement]:
    """List of joint elements belonging to the hand."""
    if not self._joints:
      raise AttributeError('Robot joints is None.')
    return self._joints

  @property
  def actuators(self) -> List[types.MjcfElement]:
    """List of actuator elements belonging to the hand."""
    if not self._finger_actuator:
      raise AttributeError('Robot actuators is None.')
    return [self._finger_actuator]

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to the robot hand."""
    if not self._mjcf_root:
      raise AttributeError('Robot mjcf_root is None.')
    return self._mjcf_root

  @property
  def name(self) -> str:
    """Name of the robot hand."""
    return self.mjcf_model.model

  @property
  def tool_center_point(self) -> types.MjcfElement:
    """Tool center point site of the hand."""
    return self._tool_center_point

  @property
  def joint_sensor(self) -> types.MjcfElement:
    """Joint sensor of the hand."""
    return self._joint_sensor

  def after_substep(self, physics: mjcf.Physics,
                    random_state: np.random.RandomState) -> None:
    """A callback which is executed after a simulation step.

    This function is necessary when using the integrated velocity mujoco
    actuator. Mujoco will limit the incoming velocity but the hidden state of
    the integrated velocity actuators must be clipped to the actuation range.

    Args:
      physics: An instance of `mjcf.Physics`.
      random_state: An instance of `np.random.RandomState`.
    """
    del random_state  # Unused.

    # Clip the actuator.act with the actuator limits.
    physics_actuators = models_utils.binding(physics, self.actuators)
    physics_actuators.act[:] = np.clip(
        physics_actuators.act[:],
        a_min=0.,
        a_max=255.)

  def convert_position(self, position, **unused_kwargs):
    """Converts raw joint position to sensor output."""
    normed_pos = (position - self._joint_offset) / self._joint_scale  # [0, 1]
    rescaled_pos = np.clip(normed_pos * _POS_SCALE, 0, _POS_SCALE)
    return np.round(rescaled_pos)

  def grasp_sensor_callable(self, physics) -> int:
    """Simulate the robot's gOBJ object detection flag."""
    # No grasp when no collision.
    collision_geoms_colliding = _are_all_collision_geoms_colliding(
        physics, self.mjcf_model)
    if not collision_geoms_colliding:
      return consts.NO_GRASP

    # No grasp when no velocity ctrl command.
    desired_vel = physics.bind(self.actuators[0]).ctrl
    if np.abs(desired_vel) < _VELOCITY_CTRL_TOL:
      return consts.NO_GRASP

    # If ctrl is positive, the gripper is closing. Hence, inward grasp.
    if desired_vel > 0:
      return consts.INWARD_GRASP
    else:
      return consts.OUTWARD_GRASP

  @property
  def collision_geom_group(self):
    collision_geom_group = [
        geom.full_identifier for geom in self._collision_geoms
    ]
    return collision_geom_group


def _is_geom_in_collision(physics: mjcf.Physics,
                          geom_name: str,
                          geom_exceptions: Optional[List[str]] = None) -> bool:
  """Returns true if a geom is in collision in the physics object."""
  for contact in physics.data.contact:
    geom1_name = physics.model.id2name(contact.geom1, 'geom')
    geom2_name = physics.model.id2name(contact.geom2, 'geom')
    if contact.dist > 1e-8:
      continue
    if (geom1_name == geom_name and geom2_name not in geom_exceptions) or (
        geom2_name == geom_name and geom1_name not in geom_exceptions):
      return True
  return False


def _are_all_collision_geoms_colliding(physics: mjcf.Physics,
                                       mjcf_root: mjcf.RootElement) -> bool:
  """Returns true if the collision geoms in the model are colliding."""
  collision_geoms = [
      mjcf_root.find('geom', name).full_identifier
      for name in _PAD_GEOM_NAMES
  ]
  return all([
      _is_geom_in_collision(physics, geom, collision_geoms)
      for geom in collision_geoms
  ])
