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

# python3
"""A subtask with customizable timestep preprocessors and termination functions."""

from typing import Optional, Sequence, Union

from absl import logging
from dm_robotics.agentflow import core
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import util as af_util
from dm_robotics.agentflow.decorators import overrides
from dm_robotics.agentflow.preprocessors import timestep_preprocessor as preprocessors
from dm_robotics.geometry import geometry
import numpy as np
import tree


class MaxStepsTermination(preprocessors.TimestepPreprocessor):
  """Terminate when the maximum number of steps is reached."""

  def __init__(self, max_steps: int, terminal_discount: float = 1.):
    """Initialize MaxStepsTermination.

    Args:
      max_steps: The maximum number of steps in the episode.
      terminal_discount: A scalar discount to set when the step threshold is
        exceeded.
    """
    super().__init__()
    self._max_steps = max_steps
    self._terminal_discount = terminal_discount
    self._steps = 0
    self._episode_reward_sum = None

  @overrides(preprocessors.TimestepPreprocessor)
  def _process_impl(
      self, timestep: preprocessors.PreprocessorTimestep
  ) -> preprocessors.PreprocessorTimestep:

    if timestep.first():
      self._steps = 0
      self._episode_reward_sum = None

    self._steps += 1
    if timestep.reward is not None:
      if self._episode_reward_sum is None:
        # Initialize (possibly nested) reward sum.
        self._episode_reward_sum = tree.map_structure(
            lambda _: 0.0, timestep.reward)
      # Update (possibly nested) reward sum.
      self._episode_reward_sum = tree.map_structure(
          lambda x, y: x + y, self._episode_reward_sum, timestep.reward)

    if self._steps >= self._max_steps:
      ts = timestep._replace(pterm=1.0, discount=self._terminal_discount,
                             result=core.OptionResult.failure_result())
      af_util.log_info('Episode timeout', 'red')
      logging.info(
          'Terminating with discount %s because maximum steps reached '
          '(%s)', ts.discount, self._max_steps)
      logging.info(
          'Reward Sum: %r, Last Reward: %r',
          self._episode_reward_sum, ts.reward)
      return ts
    return timestep

  @overrides(preprocessors.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    self._terminal_discount = input_spec.discount_spec.dtype.type(
        self._terminal_discount)
    return input_spec


class RewardThresholdTermination(preprocessors.TimestepPreprocessor):
  """Terminate when the agent receives more than threshold reward on a step."""

  def __init__(self, reward_threshold: Union[float, np.ndarray],
               terminal_discount: float = 0., sparse_mode: bool = False):
    """Initialize RewardThresholdTermination.

    Args:
      reward_threshold: Scalar or array threshold on reward current found in
        timestep. If reward exceeds threshold(s), triggers termination according
        to `sparse_mode`.
      terminal_discount: The discount to set when the threshold is exceeded.
      sparse_mode: If False (default), simply adds the discount and a `SUCCESS`
        result for any timesteps with reward exceeding threshold. Note this
        typically results in the post-threshold reward being seen twice, because
        agentflow always LAST steps an option after pterm. If `sparse_mode` is
        True, the first post-threshold step only sets `pterm=1` and a SUCCESS
        result, but uses the last (cached) sub-threshold reward until a LAST
        timestep is received.
    """
    super().__init__()
    self._reward_threshold = reward_threshold
    self._terminal_discount = terminal_discount
    self._sparse_mode = sparse_mode
    self._reset()

  def _reset(self) -> None:
    self._terminate_next_step = False
    self._last_subthresh_reward = None

  def _log_success(self, timestep: preprocessors.PreprocessorTimestep):
    af_util.log_info('Episode successful!', 'yellow')
    logging.info(
        'Terminating with discount (%s) because reward (%s) crossed '
        'threshold (%s)', timestep.discount, timestep.reward,
        self._reward_threshold)

  @overrides(preprocessors.TimestepPreprocessor)
  def _process_impl(
      self, timestep: preprocessors.PreprocessorTimestep
  ) -> preprocessors.PreprocessorTimestep:
    if timestep.first():
      self._reset()

    if np.all(timestep.reward >= self._reward_threshold):
      if self._sparse_mode:
        # Only provide super-threshold reward on LAST timestep.
        if self._terminate_next_step and timestep.last():
          # Reward still > threshold and received LAST timestep; terminate.
          timestep = timestep._replace(
              pterm=1.0,
              discount=self._terminal_discount,
              result=core.OptionResult.success_result())
          self._log_success(timestep)
          self._reset()
          return timestep
        else:
          # Signal termination and cache reward, but set reward back to last
          # sub-threshold value until we receive a LAST step.
          timestep = timestep._replace(
              pterm=1.0,
              reward=self._last_subthresh_reward,
              result=core.OptionResult.success_result())
          self._terminate_next_step = True
      else:
        # Default mode: simply request termination and set terminal discount.
        timestep = timestep._replace(
            pterm=1.0,
            discount=self._terminal_discount,
            result=core.OptionResult.success_result())
        self._log_success(timestep)

      return timestep

    else:
      self._last_subthresh_reward = timestep.reward
      logging.log_every_n(
          logging.INFO, 'Not terminating; reward (%s) less '
          'than threshold (%s)', 100, timestep.reward, self._reward_threshold)
    return timestep

  @overrides(preprocessors.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    self._terminal_discount = input_spec.discount_spec.dtype.type(
        self._terminal_discount)
    return input_spec


class LeavingWorkspaceTermination(preprocessors.TimestepPreprocessor):
  """Terminate if the robot's tool center point leaves the workspace."""

  def __init__(self,
               tcp_pos_obs: str,
               workspace_center: Sequence[float],
               workspace_radius: float,
               terminal_discount: float = 0.):
    """Initialize LeavingWorkspaceTermination.

    Args:
      tcp_pos_obs: A string key into the observation from the timestep in which
        to find a 3-dim array representing the tool center-point position in
        world-coords.
      workspace_center: A 3-dim array representing the position of the center of
        the workspace in world coords.
      workspace_radius: A float representing the radius of the workspace sphere.
      terminal_discount: A scalar discount to set when the workspace is violated
    """
    super().__init__()
    self._tcp_pos_obs = tcp_pos_obs
    self._workspace_centre = np.array(workspace_center, dtype=np.float32)
    self._workspace_radius = workspace_radius
    self._terminal_discount = terminal_discount

  def _process_impl(
      self, timestep: preprocessors.PreprocessorTimestep
  ) -> preprocessors.PreprocessorTimestep:
    try:
      tcp_site_pos = timestep.observation[self._tcp_pos_obs]
    except KeyError as key_error:
      raise KeyError(
          ('{} not a valid observation name. Valid names are '
           '{}').format(self._tcp_pos_obs,
                        list(timestep.observation.keys()))) from key_error

    dist = np.linalg.norm(tcp_site_pos[:3] - self._workspace_centre)
    if dist > self._workspace_radius:
      ts = timestep._replace(pterm=1.0, discount=self._terminal_discount,
                             result=core.OptionResult.failure_result())
      logging.info(
          'Terminating with discount %s because out of bounds. \n'
          'tcp_site_pos (xyz): %s\n'
          'workspace_centre: %s\n'
          'dist: %s,'
          'workspace_radius: %s\n', ts.discount, tcp_site_pos,
          self._workspace_centre, dist, self._workspace_radius)
      return ts
    else:
      return timestep

  @overrides(preprocessors.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    self._terminal_discount = input_spec.discount_spec.dtype.type(
        self._terminal_discount)
    return input_spec


class LeavingWorkspaceBoxTermination(preprocessors.TimestepPreprocessor):
  """Terminate if the robot's tool center point leaves the 6D workspace box."""

  def __init__(self,
               tcp_pos_obs: str,
               tcp_quat_obs: str,
               workspace_centre: geometry.PoseStamped,
               workspace_limits: np.ndarray,
               tcp_offset: Optional[geometry.Pose] = None,
               terminal_discount: float = 0.):
    """Initialize LeavingWorkspaceBoxTermination.

    Args:
      tcp_pos_obs: A string key into the observation from the timestep in which
        to find a 3-dim array representing the tool center-point position in
        world-coords.
      tcp_quat_obs: A tring key into the observation from the timestep in which
        to find a 4-dim array representing the tool center-point orientation
        quaternion in world-coords.
      workspace_centre: A PoseStamped object representing the pose of the center
        of the workspace in world coords.
      workspace_limits: A 6D pos-euler array (XYZ ordering) that defines the
        maximal limits of the workspace relative to the workspace centre. It is
        important to note that these limits are defined in the coordinate space
        of the workspace centre, NOT in world coordinates.
      tcp_offset: An optional offset from the TCP pose of the point on the arm
        that is checked against the workspace bounds.
      terminal_discount: A scalar discount to set when the workspace is violated
    """

    super().__init__()
    self._tcp_pos_obs = tcp_pos_obs
    self._tcp_quat_obs = tcp_quat_obs
    self._workspace_centre = workspace_centre
    self._workspace_limits = workspace_limits
    self._tcp_offset = tcp_offset
    self._terminal_discount = terminal_discount

    if len(workspace_limits) != 6:
      raise ValueError('Workspace Limits should be 6-dim pos-euler vector')

  @overrides(preprocessors.TimestepPreprocessor)
  def _process_impl(
      self, timestep: preprocessors.PreprocessorTimestep
  ) -> preprocessors.PreprocessorTimestep:
    tcp_pos = timestep.observation[self._tcp_pos_obs]
    tcp_quat = timestep.observation[self._tcp_quat_obs]

    tcp_pose = geometry.Pose(tcp_pos, tcp_quat)

    if self._tcp_offset is not None:
      offset_tcp_pose = tcp_pose.mul(self._tcp_offset)
    else:
      offset_tcp_pose = tcp_pose

    offset_tcp_pose = geometry.PoseStamped(offset_tcp_pose, None)
    rel_pose = offset_tcp_pose.get_relative_pose(self._workspace_centre)

    dist = np.abs(rel_pose.to_poseuler())

    if np.all(dist < self._workspace_limits):
      return timestep
    else:
      ts = timestep._replace(pterm=1.0, discount=self._terminal_discount,
                             result=core.OptionResult.failure_result())
      logging.info(
          'Terminating with discount %s because out of bounds. \n'
          'tcp_site_pos (xyz): %s\n'
          'tcp_site_quat (wxyz): %s\n'
          'workspace_centre: %s\n'
          'workspace_limits: %s\n'
          'dist: %s\n', ts.discount, tcp_pos, tcp_quat,
          self._workspace_centre, self._workspace_limits, dist)
      return ts

  @overrides(preprocessors.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:

    self._check_inputs(input_spec)

    self._terminal_discount = input_spec.discount_spec.dtype.type(
        self._terminal_discount)
    return input_spec

  def _check_inputs(self, input_spec: spec_utils.TimeStepSpec) -> None:
    """Check that we have the correct keys in the observations."""
    for key in [self._tcp_pos_obs, self._tcp_quat_obs]:
      if key not in input_spec.observation_spec:
        raise KeyError(('{} not a valid observation name. Valid names are '
                        '{}').format(self._tcp_pos_obs,
                                     list(input_spec.observation_spec.keys())))


class ObservationThresholdTermination(preprocessors.TimestepPreprocessor):
  """Terminate if the observation is in the vicinity of a specified value."""

  def __init__(
      self,
      observation: str,
      desired_obs_values: Sequence[float],
      norm_threshold: float,
      terminal_discount: float = 0.,
  ):
    """Initialize ObservationThresholdTermination.

    Args:
      observation: A string key into the observation from the timestep in which
        to find a N-dim array representing the observation for which to set a
        threshold.
      desired_obs_values: A N-dim array representing the desired observation
        values.
      norm_threshold: A float representing the error norm below which it should
        terminate.
      terminal_discount: A scalar discount to set when terminating.

    Raises:
      KeyError: if `observation` is not a valid observation name.
    """
    super().__init__()
    self._obs = observation
    self._desired_obs_values = np.array(desired_obs_values, dtype=np.float32)
    self._norm_threshold = norm_threshold
    self._terminal_discount = terminal_discount

  def _process_impl(
      self, timestep: preprocessors.PreprocessorTimestep
  ) -> preprocessors.PreprocessorTimestep:
    try:
      obs_values = timestep.observation[self._obs]
    except KeyError as key_error:
      raise KeyError(
          ('{} not a valid observation name. Valid names are '
           '{}').format(self._obs,
                        list(timestep.observation.keys()))) from key_error

    error_norm = np.linalg.norm(obs_values[:] - self._desired_obs_values)
    if error_norm <= self._norm_threshold:
      ts = timestep._replace(
          pterm=1.0,
          discount=self._terminal_discount,
          result=core.OptionResult.failure_result())
      logging.info(
          'Terminating with discount %s because %s is within the threshold.\n'
          'observation values: %s\n'
          'desired observation values: %s\n'
          'error norm: %s,'
          'threshold: %s\n', self._terminal_discount, self._obs, obs_values,
          self._desired_obs_values, error_norm, self._norm_threshold)
      return ts
    else:
      return timestep

  @overrides(preprocessors.TimestepPreprocessor)
  def _output_spec(
      self, input_spec: spec_utils.TimeStepSpec) -> spec_utils.TimeStepSpec:
    self._terminal_discount = input_spec.discount_spec.dtype.type(
        self._terminal_discount)
    return input_spec
