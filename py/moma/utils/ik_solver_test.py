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

"""Tests for ik_solver."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.models.robots.robot_arms import sawyer
from dm_robotics.moma.utils import ik_solver
from dm_robotics.transformations import transformations as tr
import numpy as np

# Linear and angular tolerance when comparing the end pose and the target pose.
_LINEAR_TOL = 1e-4
_ANGULAR_TOL = 1e-4


class IkSolverTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('sawyer_with_gripper', True), ('sawyer_without_gripper', False),)
  def test_ik_solver_with_pose(self, with_gripper):
    # Seed for randomness to prevent flaky tests.
    np.random.seed(42)
    rng = np.random.RandomState(42)

    # Change the ik site depending if the gripper is attached or not.
    arm = sawyer.Sawyer()
    if with_gripper:
      gripper = robotiq_2f85.Robotiq2F85()
      arm.attach(gripper)
      ik_site = gripper.tool_center_point
    else:
      ik_site = arm.wrist_site

    solver = ik_solver.IkSolver(
        arm.mjcf_model, arm.joints, ik_site)

    # Create the distibution from which the ref_poses will be sampled.
    pos_dist = pose_distribution.UniformPoseDistribution(
        min_pose_bounds=[
            0.30, -0.15, 0.10, 2 * np.pi / 3, -np.pi / 5, -np.pi / 4],
        max_pose_bounds=[
            0.55, 0.15, 0.40, 4 * np.pi / 3, np.pi / 5, np.pi / 4])

    # Each iteration samples a new target position, solves the IK and checks
    # that the solution is correct.
    for _ in range(500):
      # Sample a new ref_pose
      position, quaternion = pos_dist.sample_pose(rng)
      ref_pose = geometry.Pose(position, quaternion)

      # Check that we can solve the problem and that the solution is within
      # the joint ranges.
      qpos_sol = solver.solve(
          ref_pose, linear_tol=_LINEAR_TOL, angular_tol=_ANGULAR_TOL,
          early_stop=True, stop_on_first_successful_attempt=True)
      self.assertIsNotNone(qpos_sol)
      min_range = solver._joints_binding.range[:, 0]
      max_range = solver._joints_binding.range[:, 1]
      np.testing.assert_array_compare(np.less_equal, qpos_sol, max_range)
      np.testing.assert_array_compare(np.greater_equal, qpos_sol, min_range)

      # Check the max linear and angular errors are satisfied.
      geometry_physics = mujoco_physics.wrap(solver._physics)
      solver._joints_binding.qpos[:] = qpos_sol
      end_pose = geometry_physics.world_pose(ik_site)
      linear_error = np.linalg.norm(end_pose.position - ref_pose.position)
      angular_error = np.linalg.norm(_get_orientation_error(
          end_pose.quaternion, ref_pose.quaternion))
      self.assertLessEqual(linear_error, _LINEAR_TOL)
      self.assertLessEqual(angular_error, _ANGULAR_TOL)

  def test_ik_solver_when_solution_found_but_last_attempt_failed(self):
    """Test correct qpos is returned when last attempt failed.

    This test is used to ensure that if the solver finds a solution on one
    of the attempts and that the final attempt the solver does fails, the
    correct qpos is returned.
    """
    # Seed for randomness to prevent flaky tests.
    np.random.seed(42)
    rng = np.random.RandomState(42)

    arm = sawyer.Sawyer()
    ik_site = arm.wrist_site
    solver = ik_solver.IkSolver(
        arm.mjcf_model, arm.joints, ik_site)

    # Create the distibution from which the ref_poses will be sampled.
    pos_dist = pose_distribution.UniformPoseDistribution(
        min_pose_bounds=[
            0.30, -0.15, 0.10, 2 * np.pi / 3, -np.pi / 5, -np.pi / 4],
        max_pose_bounds=[
            0.55, 0.15, 0.40, 4 * np.pi / 3, np.pi / 5, np.pi / 4])

    # We continue until we find a solve where we can test the behaviour.
    found_solution_and_final_attempt_failed = False
    while not found_solution_and_final_attempt_failed:
      # Sample a new ref_pose
      position, quaternion = pos_dist.sample_pose(rng)
      ref_pose = geometry.Pose(position, quaternion)

      # Check that a solution has been found
      qpos_sol = solver.solve(ref_pose)
      self.assertIsNotNone(qpos_sol)

      # Check if the final attempt joint configuration is a solution.
      geometry_physics = mujoco_physics.wrap(solver._physics)
      last_attempt_end_pose = geometry_physics.world_pose(ik_site)
      last_attempt_linear_error = np.linalg.norm(
          last_attempt_end_pose.position - ref_pose.position)
      last_attempt_angular_error = np.linalg.norm(_get_orientation_error(
          last_attempt_end_pose.quaternion, ref_pose.quaternion))

      # If it is not a solution check that the returned qpos is a solution.
      if (last_attempt_linear_error > _LINEAR_TOL or
          last_attempt_angular_error > _ANGULAR_TOL):
        found_solution_and_final_attempt_failed = True
        solver._joints_binding.qpos[:] = qpos_sol
        solution_pose = geometry_physics.world_pose(ik_site)
        linear_error = np.linalg.norm(
            solution_pose.position - ref_pose.position)
        angular_error = np.linalg.norm(_get_orientation_error(
            solution_pose.quaternion, ref_pose.quaternion))
        self.assertLessEqual(linear_error, _LINEAR_TOL)
        self.assertLessEqual(angular_error, _ANGULAR_TOL)

  def test_raises_when_nullspace_reference_wrong_length(self):
    # Change the ik site depending if the gripper is attached or not.
    arm = sawyer.Sawyer()
    solver = ik_solver.IkSolver(
        arm.mjcf_model, arm.joints, arm.wrist_site)
    ref_pose = geometry.Pose([0., 0., 0.], [1., 0., 0., 0.])
    wrong_nullspace_ref = np.asarray([0., 0., 0.])
    with self.assertRaises(ValueError):
      solver.solve(ref_pose, nullspace_reference=wrong_nullspace_ref)

  def test_raises_when_initial_joint_confiugration_wrong_length(self):
    # Change the ik site depending if the gripper is attached or not.
    arm = sawyer.Sawyer()
    solver = ik_solver.IkSolver(
        arm.mjcf_model, arm.joints, arm.wrist_site)
    ref_pose = geometry.Pose([0., 0., 0.], [1., 0., 0., 0.])
    wrong_initial_joint_configuration = np.asarray([0., 0., 0.])
    with self.assertRaises(ValueError):
      solver.solve(
          ref_pose,
          inital_joint_configuration=wrong_initial_joint_configuration)

  def test_return_none_when_passing_impossible_target(self):
    # Change the ik site depending if the gripper is attached or not.
    arm = sawyer.Sawyer()
    solver = ik_solver.IkSolver(
        arm.mjcf_model, arm.joints, arm.wrist_site)
    ref_pose = geometry.Pose([3., 3., 3.], [1., 0., 0., 0.])
    qpos_sol = solver.solve(ref_pose)
    self.assertIsNone(qpos_sol)


def _get_orientation_error(
    to_quat: np.ndarray, from_quat: np.ndarray) -> np.ndarray:
  """Returns error between the two quaternions."""
  err_quat = tr.quat_diff_active(from_quat, to_quat)
  return tr.quat_to_axisangle(err_quat)

if __name__ == '__main__':
  absltest.main()
