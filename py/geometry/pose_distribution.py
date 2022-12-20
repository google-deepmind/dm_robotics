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
"""A module for sampling prop and robot pose distributions."""

import abc
from typing import Callable, Optional, Sequence, Tuple, Union
from dm_robotics.geometry import geometry
from dm_robotics.transformations import transformations as tr
import numpy as np


SamplePoseFn = Callable[[np.random.RandomState, Optional[geometry.Physics]],
                        Tuple[np.ndarray, np.ndarray]]


class Distribution(abc.ABC):
  """A basic interface for probability distributions."""

  @abc.abstractmethod
  def sample(self, random_state: np.random.RandomState) -> np.ndarray:
    """Returns a sample from the distribution."""
    pass

  @abc.abstractmethod
  def mean(self) -> np.ndarray:
    """Returns the mean of the distribution."""
    pass


class PoseDistribution(abc.ABC):
  """An interface for pose distributions."""

  @abc.abstractmethod
  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a (pos, quat) pose tuple sampled from some distribution.

    Args:
      random_state: Numpy random state for sampling.
      physics: Required if the frame for the distribution has a Grounding.
    """

  @abc.abstractmethod
  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the  mean (pos, quat) pose tuple of the distribution.

    Args:
      physics: Required if the frame for the distribution has a Grounding.
        parent.
    """


class PoseStampedDistribution(PoseDistribution):
  """A PoseDistribution allowing parameterization relative to other frames."""

  def __init__(self, pose_dist: 'PoseDistribution', frame: geometry.Frame):
    super().__init__()
    self._pose_dist = pose_dist
    self._frame = frame

  @property
  def pose_dist(self):
    return self._pose_dist

  @property
  def frame(self):
    return self._frame

  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a (pos, quat) pose tuple sampled from some distribution.

    Args:
      random_state: Numpy random state for sampling.
      physics: Required if the frame for the distribution has a Grounding.
    """
    sampled_local_pose = geometry.Pose(
        *self._pose_dist.sample_pose(random_state))
    sampled_world_pose = geometry.PoseStamped(
        pose=sampled_local_pose, frame=self._frame).get_world_pose(physics)
    return sampled_world_pose.position, sampled_world_pose.quaternion

  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the  mean (pos, quat) pose tuple of the distribution."""
    mean_local_pose = geometry.Pose(*self._pose_dist.mean_pose())
    mean_world_pose = geometry.PoseStamped(
        pose=mean_local_pose, frame=self._frame).get_world_pose(physics)
    return mean_world_pose.position, mean_world_pose.quaternion


class CompositePoseDistribution(PoseDistribution):
  """A PoseDistribution composed of a pose and a quaternion distribution."""

  def __init__(self, pos_dist: Distribution, quat_dist: Distribution):
    super().__init__()
    self._pos_dist = pos_dist
    self._quat_dist = quat_dist

  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    return (self._pos_dist.sample(random_state),
            self._quat_dist.sample(random_state))

  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    return (self._pos_dist.mean(), self._quat_dist.mean())


def truncated_normal_pose_distribution(mean_pose: Union[Sequence[float],
                                                        geometry.Pose],
                                       pos_sd: Sequence[float],
                                       rot_sd: Sequence[float],
                                       pos_clip_sd: float = 2.,
                                       rot_clip_sd: float = 2.):
  """Convenience-method for generating a TruncatedNormal PoseDistribution.

  Args:
    mean_pose: (list or Pose) A mean-pose represented as a 6D list composed of
      3D pose and 3D euler angle, or a `Pose`.
    pos_sd: (3d array) Standard deviation of the position (in meters), relative
      to `mean_pose`.
    rot_sd: (3d array) Standard deviation represented as axis-angle, relative to
      `mean_pose`.
    pos_clip_sd: (float) Scalar threshold on position standard-deviation.
    rot_clip_sd: (float) Scalar threshold on standard-deviation.

  Returns:
    A CompositePoseDistribution with the provided parameters.
  """
  if isinstance(mean_pose, list) or isinstance(mean_pose, np.ndarray):
    mean_pose = geometry.Pose.from_poseuler(mean_pose)
  elif not isinstance(mean_pose, geometry.Pose):
    raise ValueError('Invalid mean_pose argument ({}).  Expected a list or '
                     'numpy array, or a `Pose`'.format(mean_pose))

  pos_dist = TruncatedNormal(mean_pose.position, pos_sd, pos_clip_sd)
  quat_dist = TruncatedNormalQuaternion(mean_pose.quaternion, rot_sd,
                                        rot_clip_sd)
  return CompositePoseDistribution(pos_dist, quat_dist)


class ConstantPoseDistribution(PoseDistribution):
  """A distribution with only a single pose with probability 1."""

  def __init__(self, pose):
    """Constructor.

    Args:
      pose: a 6D list composed of 3D pose and 3D euler angle.
    """
    super().__init__()
    self._pos = pose[:3]
    self._quat = tr.euler_to_quat(pose[3:], ordering='XYZ')

  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del random_state
    del physics
    return (self._pos, self._quat)

  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    return (self._pos, self._quat)


class LambdaPoseDistribution(PoseDistribution):
  """A distribution the samples using given lambdas."""

  def __init__(self, sample_pose_fn: SamplePoseFn,
               mean_pose_fn: Callable[[Optional[geometry.Physics]],
                                      Tuple[np.ndarray, np.ndarray]]):
    """Constructor.

    Args:
      sample_pose_fn: a callable for obtaining a sample pose.
      mean_pose_fn: a callable for obtaining the mean of sampled poses.
    """
    super().__init__()

    self._sample_pose_fn = sample_pose_fn
    self._mean_pose_fn = mean_pose_fn

  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    return self._sample_pose_fn(random_state, physics)

  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    return self._mean_pose_fn(physics)


class WeightedDiscretePoseDistribution(PoseDistribution):
  """A distribution of a fixed number of poses each with a relative probability."""

  def __init__(self, weighted_poses: Sequence[Tuple[float, np.ndarray]]):
    """Constructor.

    Args:
      weighted_poses: a list of tuples of (probability, pose). The probability
        is relative (i.e. does not need to be normalized), and the pose 6D array
        composed of 3D pose and 3D euler angle.
    """
    super().__init__()
    self._poses = [pose for _, pose in weighted_poses]
    self._weights = np.array([weight for weight, _ in weighted_poses])
    self._weights /= np.sum(self._weights)

  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    chosen = random_state.choice(self._poses, p=self._weights)
    pos = chosen[:3]
    quat = tr.euler_to_quat(chosen[3:], ordering='XYZ')
    return pos, quat

  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    # Note: this returns the mode, not the mean.
    ml_pose_idx = np.argmax(self._weights)
    ml_pose = self._poses[ml_pose_idx]
    pos = ml_pose[:3]
    quat = tr.euler_to_quat(ml_pose[3:], ordering='XYZ')

    return pos, quat


class UniformPoseDistribution(PoseDistribution):
  """Distribution of uniformly distributed poses in a given range."""

  def __init__(self, min_pose_bounds: Sequence[float],
               max_pose_bounds: Sequence[float]):
    """Constructor.

    Args:
      min_pose_bounds: a 6D list composed of 3D pose and 3D euler angle.
      max_pose_bounds: a 6D list composed of 3D pose and 3D euler angle.
    """
    super().__init__()
    self._min_pose_bounds = np.array(min_pose_bounds)
    self._max_pose_bounds = np.array(max_pose_bounds)

  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    pose = random_state.uniform(self._min_pose_bounds, self._max_pose_bounds)
    pos = pose[:3]
    quat = tr.euler_to_quat(pose[3:], ordering='XYZ')
    return pos, quat

  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    min_pose = geometry.Pose.from_poseuler(self._min_pose_bounds)
    max_pose = geometry.Pose.from_poseuler(self._max_pose_bounds)
    mean_pos = min_pose.position + (max_pose.position - min_pose.position) / 2
    mean_quat = tr.quat_slerp(min_pose.quaternion, max_pose.quaternion, 0.5)

    return mean_pos, mean_quat


def _points_to_pose(
    to_pos: np.ndarray,
    from_pos: np.ndarray,
    y_hint: Optional[np.ndarray] = None,
    extra_quat: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
  """Computes pose at `from_pos` s.t. z-axis is pointing towards `to_pos`."""
  if y_hint is None:
    y_hint = np.array([0., 1., 0.])
  else:
    y_hint = y_hint / np.linalg.norm(y_hint)

  view_dir = to_pos - from_pos
  view_dir /= np.linalg.norm(view_dir)

  # Build right-handed coordinate system with z-axis towards target, and x-axis
  # orthogonal to y-axis hint.
  z = view_dir
  x = np.cross(y_hint, z)
  y = np.cross(z, x)
  rmat = np.stack([x, y, z], axis=1)
  rmat = rmat / np.linalg.norm(rmat, axis=0)
  quat = tr.axisangle_to_quat(tr.rmat_to_axisangle(rmat))

  if extra_quat is not None:
    quat = tr.quat_mul(quat, extra_quat)

  return from_pos, quat


class LookAtPoseDistribution(PoseDistribution):
  """Distribution looking from a view-point to a target-point."""

  def __init__(self,
               look_at: Distribution,
               look_from: Distribution,
               y_hint: Optional[Union[np.ndarray, Callable[[],
                                                           np.ndarray]]] = None,
               extra_quat: Optional[np.ndarray] = None):
    """Initialize LookAtPoseDistribution.

    This distribution returns poses centered at `look_from` and with the +z-axis
    pointing towards `look_at`. It is parameterized by two distributions over
    points, and accepts a user-provided constraint for the remaining degree-of-
    freedom around the z-axis.

    Args:
      look_at: A `Distribution` over the 3D point to look at.
      look_from: A `Distribution` over the 3D point to look from.
      y_hint: Optional array or callable returning 3-vector to cross with the
        looking direction to produce the x-axis of the sampled pose. This is
        required because the full pose is under-constrained given only
        `from` and `to` points, so rather than baking in a solution we expose
        this to the user as a "hint". This can be anything, but the motivating
        examples are:
          1) Maintaining a fixed pose as the object moves -- pass the y-axis of
             object's current pose.
          2) Minimizing the difference w.r.t. the current (wrist-mounted) camera
             pose -- pass the y-axis of TCP's current pose.
        Failure to do either of these will result in a "rolling" behavior along
        the z-axis as the object or robot moves.
      extra_quat: Optional quaternion [w, i, j, k] to apply as a final rotation
        after solving for the viewing direction. If omitted, the z-axis will
        point towards `look_at`, and the x-axis will be orthogonal to `y_hint`.
    """
    super().__init__()
    self._look_at = look_at
    self._look_from = look_from
    self._extra_quat = extra_quat

    if y_hint is None:
      self._y_hint = np.array([0., 1., 0.])
    elif isinstance(y_hint, np.ndarray):
      self._y_hint = y_hint / np.linalg.norm(y_hint)
    else:
      self._y_hint = y_hint

  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    look_at = self._look_at.sample(random_state)
    look_from = self._look_from.sample(random_state)
    y_hint = self._y_hint() if callable(self._y_hint) else self._y_hint
    return _points_to_pose(look_at, look_from, y_hint, self._extra_quat)

  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    look_at = self._look_at.mean()
    look_from = self._look_from.mean()
    return _points_to_pose(look_at, look_from, self._y_hint, self._extra_quat)


class DomePoseDistribution(PoseDistribution):
  """Distribution within a dome (half sphere with a thickness).

  Dome sits on the x-y plane and the probe is initialized looking down.
  Radius and angles are uniformly sampled, hence points are not
  uniform in the volume.
  """

  def __init__(self, center, r_min, r_max, theta_max):
    """Constructor.

    Args:
      center: 3D list for the position of the dome center.
      r_min: Minimum radius.
      r_max: Maximum radius.
      theta_max: Maximum polar angle
    """
    super().__init__()
    self._center = center
    self._r_min = r_min
    self._r_max = r_max
    self._theta_max = theta_max

  def sample_pose(
      self,
      random_state: np.random.RandomState,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    del physics
    r = random_state.uniform(self._r_min, self._r_max)
    theta = random_state.uniform(0., self._theta_max)
    phi = random_state.uniform(-np.pi, np.pi)

    x = self._center[0] + r * np.sin(theta) * np.cos(phi)
    y = self._center[1] + r * np.sin(theta) * np.sin(phi)
    z = self._center[2] + r * np.cos(theta)

    pos = np.asarray([x, y, z])
    quat = tr.euler_to_quat([0, np.pi, 0.], ordering='XYZ')
    return pos, quat

  def mean_pose(
      self,
      physics: Optional[geometry.Physics] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def _sample_with_limits(random_state: np.random.RandomState,
                        sd: np.ndarray,
                        clip_sd: float,
                        max_steps: int = 500) -> np.ndarray:
  """Rejection-samples from a zero-mean truncated-normal distribution.

  Same as normal distribution except that values exceeding minimum or maximum
  limits are resampled.  See also tf.truncated_normal.

  Args:
    random_state: Numpy `RandomState` object.
    sd: A list or array of standard deviations.  Must be greater or equal to
      zero.
    clip_sd: (float) Scalar threshold on standard-deviation. Values larger than
      this will be re-sampled.
    max_steps: (int) Maximum number of times to resample

  Returns:
    An array filled with random truncated normal values.

  Raises:
    ValueError: If invalid sd provided.
    RuntimeError: If max_steps exceeded before a valid sample is obtained.
  """
  if np.any(sd < 0):
    raise ValueError('Invalid sd {}'.format(sd))
  samp = random_state.normal(scale=sd)
  i = 0
  while i < max_steps:
    bad_idxs = np.logical_or(samp < -(sd * clip_sd), samp > (sd * clip_sd))
    if np.any(bad_idxs):
      samp[bad_idxs] = random_state.normal(scale=sd[bad_idxs])
      i += 1
    else:
      break
  if np.any(bad_idxs):
    raise ValueError('Failed to sample within limits {} (clip_sd: {})'.format(
        samp, clip_sd))

  return samp


class UniformDistribution(Distribution):
  """Generic Uniform Distribution wrapping `numpy.random.uniform`."""

  def __init__(self,
               low: Union[float, Sequence[float]] = 0.,
               high: Union[float, Sequence[float]] = 1.):
    """Constructor.

    Args:
      low: Lower boundary of the output interval. All values generated will be
        greater than or equal to low. The default value is 0.
      high: Upper boundary of the output interval. All values generated will be
        less than or equal to high. The default value is 1.0.
    """
    super().__init__()
    self._low = np.array(low)
    self._high = np.array(high)

  def sample(self, random_state: np.random.RandomState) -> np.ndarray:
    return random_state.uniform(self._low, self._high)

  def mean(self) -> np.ndarray:
    return self._low + (self._high - self._low) / 2.


class TruncatedNormal(Distribution):
  """Generic Truncated Normal Distribution."""

  def __init__(self, mean, sd, clip_sd=2.):
    """Constructor.

    Args:
      mean: (array-like) Mean.
      sd: (array-like) Standard deviation.
      clip_sd: (float) Scalar threshold on standard-deviation. Values larger
        than this will be re-sampled.
    """
    super().__init__()
    self._mean = np.array(mean, dtype=np.float32)
    self._sd = np.array(sd, dtype=np.float32)
    self._clip_sd = clip_sd

  def sample(self, random_state: np.random.RandomState) -> np.ndarray:
    return self._mean + _sample_with_limits(random_state, self._sd,
                                            self._clip_sd)

  def mean(self) -> np.ndarray:
    return self._mean


class TruncatedNormalQuaternion(TruncatedNormal):
  """Truncated Normal Distribution over Quaternions.

  The deviation of this distribution is parameterized by axis-angle to allow
  control of each cartesian DOF independently.

  E.g. The following will generate a distribution that only varies about the
  x-axis with maximum deviation of 2-radians:
  >>> TruncatedNormalQuaternion([1., 0., 0., 0.], [1., 0., 0.], 2.)
  And the following will generate a distribution that varies over the
  y & z axes of the frame relative to the rotation described by an axis-angle:
  >>> TruncatedNormalQuaternion([0.2, 0.3, 0.], [0., 0.5, 0.5], 2.)
  """

  def __init__(self, mean, sd, clip_sd=2.):
    """Constructor.

    Args:
      mean: (3d or 4d array) Mean quaternion repesented either as a 4-dim array
        [w, i, j, k] or a 3-dim axis-angle (with angle encoded in length).
      sd: (3d array) Standard deviation represented as axis-angle.
      clip_sd: (float) Scalar threshold on standard-deviation. Values larger
        than this will be re-sampled.
    """
    super().__init__(mean, sd, clip_sd)
    if len(mean) == 3:
      self._mean = tr.axisangle_to_quat(mean)

  def sample(self, random_state: np.random.RandomState) -> np.ndarray:
    axisangle = _sample_with_limits(random_state, self._sd, self._clip_sd)
    offset_quat = tr.axisangle_to_quat(axisangle)
    return tr.quat_mul(self._mean, offset_quat)
