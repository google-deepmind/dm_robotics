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
"""Sawyer robot arm."""

import re
from typing import List, Tuple

import dataclasses
from dm_control import mjcf
from dm_robotics.moma.models import types
from dm_robotics.moma.models import utils as models_utils
from dm_robotics.moma.models.robots.robot_arms import robot_arm
from dm_robotics.moma.models.robots.robot_arms import sawyer_constants as consts
import numpy as np


@dataclasses.dataclass(frozen=True)
class _ActuatorParams:
  # Gain parameters for MuJoCo actuator.
  gainprm: Tuple[float]
  # Bias parameters for MuJoCo actuator.
  biasprm: Tuple[float, float, float]

_SAWYER_ACTUATOR_PARAMS = {
    consts.Actuation.INTEGRATED_VELOCITY: {
        'large_joint': _ActuatorParams((500.0,), (0.0, -500.0, -50.0)),
        'medium_joint': _ActuatorParams((500.0,), (0.0, -500.0, -50.0)),
        'small_joint': _ActuatorParams((500.0,), (0.0, -500.0, -50.0)),
    }
}
# The size of each sawyer joint, from the base (j0) to the wrist (j6).
_JOINT_SIZES = ('large_joint', 'large_joint', 'medium_joint', 'medium_joint',
                'small_joint', 'small_joint', 'small_joint')
_INTEGRATED_VELOCITY_DEFAULT_DCLASS = {
    'large_joint': {
        'joint': {
            'frictionloss': 0.3,
            'armature': 1,
            'damping': 0.1,
        }
    },
    'medium_joint': {
        'joint': {
            'frictionloss': 0.3,
            'armature': 1,
            'damping': 0.1,
        }
    },
    'small_joint': {
        'joint': {
            'frictionloss': 0.1,
            'armature': 1,
            'damping': 0.1,
        }
    },
}

_ARM_BASE_LINK_COLLISION_KWARGS = [{
    'name': 'arm_base_link_CollisionGeom',
    'type': 'cylinder',
    'fromto': '0 0 -0.005 0 0 0.08',
    'size': '0.11'
}]

_L0_COLLISION_KWARGS = [{
    'name': 'l0_CollisionGeom_1',
    'type': 'capsule',
    'fromto': '0 0 0.02 0 0 0.25',
    'size': '0.08',
}, {
    'name': 'l0_CollisionGeom_2',
    'type': 'capsule',
    'fromto': '0.08 -0.01 0.23 0.08 0.04 0.23',
    'size': '0.09',
}, {
    'name': 'l0_CollisionGeom_3',
    'type': 'capsule',
    'fromto': '0.03 -0.02 0.13 0.03 -0.02 0.18',
    'size': '0.07',
}]

_HEAD_COLLISION_KWARGS = [{
    'name': 'head_CollisionGeom_1',
    'type': 'capsule',
    'fromto': '0 0 0.02 0 0 0.21',
    'size': '0.05',
}, {
    'name': 'head_CollisionGeom_2',
    'type': 'box',
    'pos': '0.02 0 0.11',
    'size': '0.02 0.13 0.09',
}]

_L1_COLLISION_KWARGS = [{
    'name': 'l1_CollisionGeom_1',
    'type': 'capsule',
    'fromto': '0 0 0.03 0 0 0.13',
    'size': '0.074',
}, {
    'name': 'l1_CollisionGeom_2',
    'type': 'capsule',
    'fromto': '0 -0.1 0.13 0 -0.065 0.13',
    'size': '0.075',
}]

_L2_COLLISION_KWARGS = [{
    'name': 'l2_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 0 0.02 0 0 .26',
    'size': '0.07',
}]

_L3_COLLISION_KWARGS = [{
    'name': 'l3_CollisionGeom_1',
    'type': 'capsule',
    'fromto': '0 0 -0.14 0 0 -0.02',
    'size': '0.06',
}, {
    'name': 'l3_CollisionGeom_2',
    'type': 'capsule',
    'fromto': '0 -0.1 -0.12 0 -0.05 -0.12',
    'size': '0.06',
}]

_L4_COLLISION_KWARGS = [{
    'name': 'l4_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 0 0.03 0 0 .28',
    'size': '0.06',
}]

_L5_COLLISION_KWARGS = [{
    'name': 'l5_CollisionGeom_1',
    'type': 'capsule',
    'fromto': '0 0 0.04 0 0 0.08',
    'size': '0.07',
}, {
    'name': 'l5_CollisionGeom_2',
    'type': 'capsule',
    'fromto': '0 0 0.1 0 -0.05 0.1',
    'size': '0.07',
}]

_L6_COLLISION_KWARGS = [{
    'name': 'l6_CollisionGeom',
    'type': 'capsule',
    'fromto': '0 -0.005 -0.002 0 0.035 -0.002',
    'size': '0.05',
}]

# Dictionary mapping body names to a list of their collision geoms
_COLLISION_GEOMS_DICT = {
    'arm_base_link': _ARM_BASE_LINK_COLLISION_KWARGS,
    'l0': _L0_COLLISION_KWARGS,
    'head': _HEAD_COLLISION_KWARGS,
    'l1': _L1_COLLISION_KWARGS,
    'l2': _L2_COLLISION_KWARGS,
    'l3': _L3_COLLISION_KWARGS,
    'l4': _L4_COLLISION_KWARGS,
    'l5': _L5_COLLISION_KWARGS,
    'l6': _L6_COLLISION_KWARGS,
}


class Sawyer(robot_arm.RobotArm):
  """A class representing a Sawyer robot arm."""

  # Define member variables that are created in the _build function. This is to
  # comply with pytype correctly.
  _joints: List[types.MjcfElement]
  _actuators: List[types.MjcfElement]
  _mjcf_root: mjcf.RootElement
  _actuation: consts.Actuation
  _joint_torque_sensors: List[types.MjcfElement]
  _sawyer_root: mjcf.RootElement

  def _build(
      self,
      name: str = 'sawyer',
      actuation: consts.Actuation = consts.Actuation.INTEGRATED_VELOCITY,
      with_pedestal: bool = False,
      use_rotated_gripper: bool = True,
  ) -> None:
    """Initializes Sawyer.

    Args:
      name: The name of this robot. Used as a prefix in the MJCF name
        attributes.
      actuation: Instance of `sawyer_constants.Actuation` specifying which
        actuation mode to use.
      with_pedestal: If true, mount the sawyer robot on its pedestal.
      use_rotated_gripper: If True, mounts the gripper in a rotated position to
        match the real placement of the gripper on the physical Sawyer. Only set
        to False for backwards compatibility.
    """
    self._sawyer_root = mjcf.from_path(consts.SAWYER_XML)
    if with_pedestal:
      pedestal_root = mjcf.from_path(consts.SAWYER_PEDESTAL_XML)
      pedestal_root.find('site', 'sawyer_attachment').attach(self._sawyer_root)
      self._mjcf_root = pedestal_root
    else:
      self._mjcf_root = self._sawyer_root

    self._actuation = actuation
    self._mjcf_root.model = name

    self._add_mjcf_elements(use_rotated_gripper)
    self._add_actuators()
    self._add_collision_geoms()

  @property
  def collision_geom_group(self):
    collision_geom_group = [
        geom.full_identifier for geom in self._collision_geoms
    ]
    return collision_geom_group

  @property
  def collision_geom_group_basket_v4(self):
    """Collision geom group to use with the v4 RGB basket.

    The v4 basket is higher than the previous ones. This resulted in the
    collisions geoms of the robot being in collision with the basket
    collision geoms before the robot started moving. This made moving
    the robot impossible as the QP velocity controller was trying to avoid
    collisions that were already there. This geom group solves this problem.
    This group should be used with the collision geoms of the basket struct.
    Note that the default `collision_geom_group` should still be used with the
    collision group of the cameras of the basket.
    """
    # We ignore the collision geoms located at the base of the robot
    ignored_geoms = [geom['name'] for geom in _L0_COLLISION_KWARGS]
    ignored_geoms.append(_L1_COLLISION_KWARGS[0]['name'])
    collision_geom_group = []
    for geom in self._collision_geoms:
      if geom.name not in ignored_geoms:
        collision_geom_group.append(geom.full_identifier)
    return collision_geom_group

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
    """List of joint elements belonging to the arm."""
    if not self._joints:
      raise AttributeError('Robot joints is None.')
    return self._joints

  @property
  def actuators(self) -> List[types.MjcfElement]:
    """List of actuator elements belonging to the arm."""
    if not self._actuators:
      raise AttributeError('Robot actuators is None.')
    return self._actuators

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to this robot arm."""
    if not self._mjcf_root:
      raise AttributeError('Robot mjcf_root is None.')
    return self._mjcf_root

  @property
  def name(self) -> str:
    """Name of the robot arm."""
    return self.mjcf_model.model

  @property
  def wrist_site(self) -> types.MjcfElement:
    """Get the MuJoCo site of the wrist.

    Returns:
      MuJoCo site
    """
    return self._wrist_site

  @property
  def joint_torque_sensors(self) -> List[types.MjcfElement]:
    """Get MuJoCo sensor of the joint torques."""
    return self._joint_torque_sensors

  @property
  def attachment_site(self):
    """Override wrist site for attachment, but NOT the one for observations."""
    return self._attachment_site

  def set_joint_angles(self, physics: mjcf.Physics,
                       joint_angles: np.ndarray) -> None:
    """Sets the joints of the robot to a given configuration.

    This function allows to change the joint configuration of the sawyer arm
    and sets the controller to prevent the impedance controller from moving back
    to the previous configuration.

    Args:
      physics: A `mujoco.Physics` instance.
      joint_angles: The desired joints configuration for the robot arm.
    """
    physics_joints = models_utils.binding(physics, self._joints)
    physics_actuators = models_utils.binding(physics, self._actuators)

    physics_joints.qpos[:] = joint_angles
    if self._actuation == consts.Actuation.INTEGRATED_VELOCITY:
      physics_actuators.act[:] = physics_joints.qpos[:]

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
    if self._actuation == consts.Actuation.INTEGRATED_VELOCITY:
      physics_actuators = models_utils.binding(physics, self._actuators)
      physics_actuators.act[:] = np.clip(
          physics_actuators.act[:],
          a_min=consts.JOINT_LIMITS['min'],
          a_max=consts.JOINT_LIMITS['max'])

  def _add_mjcf_elements(self, use_rotated_gripper: bool):
    """Defines the arms MJCF joints and sensors."""
    self._joints = [
        self._sawyer_root.find('joint', j) for j in consts.JOINT_NAMES
    ]
    self._joint_torque_sensors = [
        sensor for sensor in self._sawyer_root.find_all('sensor')
        if sensor.tag == 'torque' and re.match(r'^j\d_site$', sensor.site.name)
    ]

    self._wrist_site = self._sawyer_root.find('site', consts.WRIST_SITE_NAME)

    if use_rotated_gripper:
      # Change the attachment site so it is aligned with the real sawyer. This
      # will allow having the gripper oriented in the same way in both sim and
      # real.
      hand_body = self._sawyer_root.find('body', 'hand')
      hand_body.add(
          'site',
          type='sphere',
          name='real_aligned_tcp',
          pos=(0, 0, 0),
          quat=consts.ROTATION_QUATERNION_MINUS_90DEG_AROUND_Z)
      self._attachment_site = self._sawyer_root.find(
          'site', 'real_aligned_tcp')
    else:
      self._attachment_site = self._wrist_site

  def _add_collision_geoms(self):
    """Add collision geoms."""
    # Note that the MJCF model being passed is sawyer_root.
    self._collision_geoms = models_utils.attach_collision_geoms(
        self._sawyer_root, _COLLISION_GEOMS_DICT)

  def _add_actuators(self):
    """Adds the Mujoco actuators to the robot arm."""
    if self._actuation not in consts.Actuation:
      raise ValueError((f'Actuation {self._actuation} is not a valid actuation.'
                        'Please specify one of '
                        f'{list(consts.Actuation.__members__.values())}'))

    if self._actuation == consts.Actuation.INTEGRATED_VELOCITY:
      self._add_integrated_velocity_actuators()

  def _add_integrated_velocity_actuators(self) -> None:
    """Adds integrated velocity actuators to the mjcf model.

    This function adds integrated velocity actuators and default class
    attributes to the mjcf model according to the values in `sawyer_constants`,
    `_SAWYER_ACTUATOR_PARAMS` and `_INTEGRATED_VELOCITY_DEFAULT_DCLASS`.
    `self._actuators` is created to contain the list of actuators created.
    """
    # Add default class attributes.
    for name, defaults in _INTEGRATED_VELOCITY_DEFAULT_DCLASS.items():
      default_dclass = self._sawyer_root.default.add('default', dclass=name)
      for tag, attributes in defaults.items():
        element = getattr(default_dclass, tag)
        for attr_name, attr_val in attributes.items():
          setattr(element, attr_name, attr_val)

    # Construct list of ctrlrange tuples from act limits and actuation mode.
    ctrl_ranges = list(
        zip(consts.ACTUATION_LIMITS[self._actuation]['min'],
            consts.ACTUATION_LIMITS[self._actuation]['max']))

    # Construct list of forcerange tuples from effort limits.
    force_ranges = list(zip(consts.EFFORT_LIMITS['min'],
                            consts.EFFORT_LIMITS['max']))

    def add_actuator(i: int) -> types.MjcfElement:
      """Add an actuator."""
      params = _SAWYER_ACTUATOR_PARAMS[self._actuation][_JOINT_SIZES[i]]
      actuator = self._sawyer_root.actuator.add(
          'general',
          name=f'j{i}',
          ctrllimited=True,
          forcelimited=True,
          ctrlrange=ctrl_ranges[i],
          forcerange=force_ranges[i],
          dyntype='integrator',
          biastype='affine',
          gainprm=params.gainprm,
          biasprm=params.biasprm)
      actuator.joint = self._joints[i]
      return actuator

    self._actuators = [add_actuator(i) for i in range(consts.NUM_DOFS)]
