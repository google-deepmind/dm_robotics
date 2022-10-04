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

"""MuJoCo utility functions."""

from typing import Tuple

from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.transformations import transformations as tr
import numpy as np

mjlib = mjbindings.mjlib


def actuator_spec(physics, actuators) -> Tuple[np.ndarray, np.ndarray]:
  """Returns the spec of the actuators."""
  num_actions = len(actuators)

  control_range = physics.bind(actuators).ctrlrange
  is_limited = physics.bind(actuators).ctrllimited.astype(bool)

  minima = np.full(num_actions, fill_value=-np.inf, dtype=np.float32)
  maxima = np.full(num_actions, fill_value=np.inf, dtype=np.float32)
  minima[is_limited], maxima[is_limited] = control_range[is_limited].T

  return minima, maxima


def get_site_pose(physics, site_entity):
  """Returns world pose of prop as homogeneous transform.

  Args:
    physics: A mujoco.Physics
    site_entity: The entity of a prop site (with prefix)

  Returns:
    A 4x4 numpy array containing the world site pose
  """
  binding = physics.bind(site_entity)
  xpos = binding.xpos.reshape(3, 1)
  xmat = binding.xmat.reshape(3, 3)

  transform_world_site = np.vstack(
      [np.hstack([xmat, xpos]), np.array([0, 0, 0, 1])])
  return transform_world_site


def get_site_relative_pose(physics, site_a, site_b):
  """Computes pose of site_a in site_b frame.

  Args:
    physics: The physics object.
    site_a: The site whose pose to calculate.
    site_b: The site whose frame of reference to use.

  Returns:
    transform_siteB_siteA: A 4x4 transform representing the pose
      of siteA in the siteB frame
  """
  transform_world_a = get_site_pose(physics, site_a)
  transform_world_b = get_site_pose(physics, site_b)
  transform_b_world = tr.hmat_inv(transform_world_b)
  return transform_b_world.dot(transform_world_a)


def get_site_vel(physics, site_entity, world_frame=False, reorder=True):
  """Returns the 6-dim [rotational, translational] velocity of the named site.

  This 6-vector represents the instantaneous velocity of coordinate system
  attached to the site.  If flg_local==0, this vector is rotated to
  world coordinates.

  Args:
    physics: A mujoco.Physics
    site_entity: The entity of a prop site (with prefix)
    world_frame: If True return vel in world frame, else local to site.
    reorder: (bool) If True, swaps the order of the return velocity
      from [rotational, translational] to [translational, rotational]
  """
  flg_local = 0 if world_frame else 1
  idx = physics.model.name2id(site_entity.full_identifier,
                              enums.mjtObj.mjOBJ_SITE)
  site_vel = np.zeros(6)
  mjlib.mj_objectVelocity(physics.model.ptr, physics.data.ptr,
                          enums.mjtObj.mjOBJ_SITE, idx, site_vel, flg_local)

  if reorder:
    return np.hstack([site_vel[3:], site_vel[0:3]])
  else:
    return site_vel


def get_site_relative_vel(physics, site_a, site_b, frame='world'):
  """Returns the relative velocity of the named sites.

  This 6-vector represents the instantaneous velocity of coordinate system
  attached to the site.

  Args:
    physics: The physics object.
    site_a: The site whose pose to calculate.
    site_b: The site whose frame of reference to use.
    frame: (str) A string indicating the frame in which the relative vel
      is reported.  Options: "world", "a", "b" (Default: "world")

  Raises:
    ValueError: if invalid frame argument
  """

  v_world_a = get_site_vel(physics, site_a, world_frame=True, reorder=True)
  v_world_b = get_site_vel(physics, site_b, world_frame=True, reorder=True)
  vrel_world = v_world_a - v_world_b

  if frame == 'world':
    return vrel_world
  elif frame == 'a':
    rot_world_a = get_site_pose(physics, site_a)
    rot_world_a[0:3, 3] = 0
    rot_a_world = np.linalg.inv(rot_world_a)
    vrel_a = tr.velocity_transform(rot_a_world, vrel_world)
    return vrel_a
  elif frame == 'b':
    rot_world_b = get_site_pose(physics, site_b)
    rot_world_b[0:3, 3] = 0
    rot_b_world = np.linalg.inv(rot_world_b)
    vrel_b = tr.velocity_transform(rot_b_world, vrel_world)
    return vrel_b
  else:
    raise ValueError('Invalid frame spec \'{}\''.format(frame))
