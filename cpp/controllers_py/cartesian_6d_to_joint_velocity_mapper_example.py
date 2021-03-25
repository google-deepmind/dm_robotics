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
"""Example usage for cartesian_6d_to_joint_velocity_mapper module."""

import threading
import time
from typing import Sequence

from absl import app
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.suite import humanoid
from dm_control.viewer import gui
from dm_robotics.controllers import cartesian_6d_to_joint_velocity_mapper
import numpy as np

mjlib = mujoco.wrapper.mjbindings.mjlib

_WINDOW_HEIGHT = 480
_WINDOW_WIDTH = 640


def _create_mapper_params(
    physics: mujoco.Physics
) -> cartesian_6d_to_joint_velocity_mapper.Parameters:
  """Creates the parameters for the Cartesian 6D to joint velocity mapper."""
  # Set params for controlling the left hand, and enabling active collision
  # avoidance between the left arm and the floor.
  params = cartesian_6d_to_joint_velocity_mapper.Parameters()
  params.model = physics.model
  params.joint_ids = [19, 20, 21]
  params.object_type = enums.mjtObj.mjOBJ_SITE
  params.object_name = "left_hand"
  params.integration_timestep = 0.005  # 5ms

  params.enable_joint_position_limits = False
  params.joint_position_limit_velocity_scale = 0.95
  params.minimum_distance_from_joint_position_limit = 0.01  # ~0.5deg.

  params.enable_joint_velocity_limits = True
  params.joint_velocity_magnitude_limits = [0.5, 0.5, 0.5]

  params.enable_joint_acceleration_limits = False
  params.remove_joint_acceleration_limits_if_in_conflict = True
  params.joint_acceleration_magnitude_limits = [1.0, 1.0, 1.0]

  params.enable_collision_avoidance = True
  params.collision_avoidance_normal_velocity_scale = 0.01
  params.minimum_distance_from_collisions = 0.005
  params.collision_detection_distance = 10.0
  params.collision_pairs = [(["left_upper_arm", "left_lower_arm",
                              "left_hand"], ["floor"])]

  print(params.cartesian_velocity_task_weighting_matrix)
  params.cartesian_velocity_task_weighting_matrix = [  #
      1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ]  #
  print(params.cartesian_velocity_task_weighting_matrix)

  params.check_solution_validity = True
  params.solution_tolerance = 1e-3
  params.regularization_weight = 1e-3
  params.enable_nullspace_control = True
  params.return_error_on_nullspace_failure = False
  params.nullspace_projection_slack = 1e-7

  return params


def _control_humanoid(physics: mujoco.Physics, physics_lock: threading.Lock):
  """Controls the humanoid's left arm to move towards the floor."""

  params = _create_mapper_params(physics)
  mapper = cartesian_6d_to_joint_velocity_mapper.Mapper(params)

  # Move the left arm down towards the floor.
  nullspace_bias = [-1.0, 0.0, 1.0]
  target_velocity = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]

  while True:
    with physics_lock:
      # Compute joint velocities.
      solution = mapper.compute_joint_velocities(physics.data, target_velocity,
                                                 nullspace_bias)

      # Set joint velocities. Note that `solution` is already sorted in
      # ascending order of `params.joint_ids`.
      physics.data.qvel[:] = [0.0] * physics.model.nv
      for joint_id, velocity in zip(sorted(params.joint_ids), solution):
        dof_adr = physics.model.jnt_dofadr[joint_id]
        physics.data.qvel[dof_adr] = velocity

      # Integrate, run MuJoCo kinematics, and render.
      mjlib.mj_integratePos(physics.model.ptr, physics.data.qpos,
                            physics.data.qvel,
                            params.integration_timestep.total_seconds())
      mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)
    time.sleep(params.integration_timestep.total_seconds())


def _render_image(physics: mujoco.Physics,
                  physics_lock: threading.Lock) -> np.ndarray:
  """Returns a view of the scene as a numpy array."""
  with physics_lock:
    return physics.render(height=_WINDOW_HEIGHT, width=_WINDOW_WIDTH)


def main(argv: Sequence[str]):
  del argv

  physics = humanoid.Physics.from_xml_string(*humanoid.get_model_and_assets())
  physics_lock = threading.Lock()

  # Place the humanoid in a position where the left hand can collide with the
  # floor if it moves down.
  physics.data.qpos[2] = 0.3
  mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

  # Start control thread to compute velocities and integrate.
  control_thread = threading.Thread(
      target=lambda: _control_humanoid(physics, physics_lock))
  control_thread.start()

  # Show the rendered image. Note how the left arm avoids collisions with the
  # floor.
  while True:
    window = gui.RenderWindow(
        width=_WINDOW_WIDTH,
        height=_WINDOW_HEIGHT,
        title="Cartesian6dVelocityToJointVelocityMapperExample")
    window.update(lambda: _render_image(physics, physics_lock))


if __name__ == "__main__":
  app.run(main)
