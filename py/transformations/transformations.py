# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rigid-body transformations including velocities and static forces."""

# pylint: disable=unused-import
from dm_robotics.transformations._transformations import axisangle_to_euler
from dm_robotics.transformations._transformations import axisangle_to_quat
from dm_robotics.transformations._transformations import axisangle_to_rmat
from dm_robotics.transformations._transformations import cross_2d
from dm_robotics.transformations._transformations import cross_mat_from_vec3
from dm_robotics.transformations._transformations import euler_to_axisangle
from dm_robotics.transformations._transformations import euler_to_quat
from dm_robotics.transformations._transformations import euler_to_rmat
from dm_robotics.transformations._transformations import force_transform
from dm_robotics.transformations._transformations import force_transform_2d
from dm_robotics.transformations._transformations import hmat_inv
from dm_robotics.transformations._transformations import hmat_to_pos_quat
from dm_robotics.transformations._transformations import hmat_to_poseuler
from dm_robotics.transformations._transformations import hmat_to_twist
from dm_robotics.transformations._transformations import integrate_hmat
from dm_robotics.transformations._transformations import integrate_quat
from dm_robotics.transformations._transformations import mat_to_quat
from dm_robotics.transformations._transformations import matrix_to_postheta_2d
from dm_robotics.transformations._transformations import pos_quat_to_hmat
from dm_robotics.transformations._transformations import pos_to_hmat
from dm_robotics.transformations._transformations import poseuler_to_hmat
from dm_robotics.transformations._transformations import positive_leading_quat
from dm_robotics.transformations._transformations import postheta_to_matrix_2d
from dm_robotics.transformations._transformations import quat_angle
from dm_robotics.transformations._transformations import quat_axis
from dm_robotics.transformations._transformations import quat_between_vectors
from dm_robotics.transformations._transformations import quat_conj
from dm_robotics.transformations._transformations import quat_diff_active
from dm_robotics.transformations._transformations import quat_diff_passive
from dm_robotics.transformations._transformations import quat_dist
from dm_robotics.transformations._transformations import quat_exp
from dm_robotics.transformations._transformations import quat_inv
from dm_robotics.transformations._transformations import quat_log
from dm_robotics.transformations._transformations import quat_mul
from dm_robotics.transformations._transformations import quat_rotate
from dm_robotics.transformations._transformations import quat_slerp
from dm_robotics.transformations._transformations import quat_to_axisangle
from dm_robotics.transformations._transformations import quat_to_euler
from dm_robotics.transformations._transformations import quat_to_mat
from dm_robotics.transformations._transformations import rmat_to_axisangle
from dm_robotics.transformations._transformations import rmat_to_euler
from dm_robotics.transformations._transformations import rmat_to_hmat
from dm_robotics.transformations._transformations import rmat_to_rot6
from dm_robotics.transformations._transformations import rot6_to_rmat
from dm_robotics.transformations._transformations import rotate_vec6
from dm_robotics.transformations._transformations import rotation_matrix_2d
from dm_robotics.transformations._transformations import rotation_x_axis
from dm_robotics.transformations._transformations import rotation_y_axis
from dm_robotics.transformations._transformations import rotation_z_axis
from dm_robotics.transformations._transformations import twist_to_hmat
from dm_robotics.transformations._transformations import velocity_transform
from dm_robotics.transformations._transformations import velocity_transform_2d

# pytype: disable=import-error
# pylint: disable=g-import-not-at-top,reimported
try:
  # Use faster C extension versions if _transformations_quat is available.
  from dm_robotics.transformations._transformations_quat import axisangle_to_quat
  from dm_robotics.transformations._transformations_quat import hmat_to_pos_quat
  from dm_robotics.transformations._transformations_quat import integrate_quat
  from dm_robotics.transformations._transformations_quat import mat_to_quat
  from dm_robotics.transformations._transformations_quat import pos_quat_to_hmat
  from dm_robotics.transformations._transformations_quat import quat_angle
  from dm_robotics.transformations._transformations_quat import quat_conj
  from dm_robotics.transformations._transformations_quat import quat_dist
  from dm_robotics.transformations._transformations_quat import quat_exp
  from dm_robotics.transformations._transformations_quat import quat_inv
  from dm_robotics.transformations._transformations_quat import quat_log
  from dm_robotics.transformations._transformations_quat import quat_mul
  from dm_robotics.transformations._transformations_quat import quat_rotate
  from dm_robotics.transformations._transformations_quat import quat_slerp
  from dm_robotics.transformations._transformations_quat import quat_to_mat

  # TODO(benmoran) Consider quaternion implementations of other functions:
  # from dm_robotics.transformations._transformations import quat_axis
  # from dm_robotics.transformations._transformations \
  #  import quat_between_vectors
  # from dm_robotics.transformations._transformations import quat_diff_active
  # from dm_robotics.transformations._transformations import quat_diff_passive
  # from dm_robotics.transformations._transformations import quat_to_axisangle
  # from dm_robotics.transformations._transformations import quat_to_euler
  HAVE_NUMPY_QUATERNION = True
except ImportError:
  HAVE_NUMPY_QUATERNION = False
# pytype: enable=import-error
# pylint: enable=g-import-not-at-top,reimported
