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
"""Classes for representing frames, twists, wrenches and accelerations.

This mimics the ROS geometry_msgs interface, but using arbitrary frame
identifiers coupled with a physics that can interpret those frames.

The primary use case is to decouple the parameterization of coordinate frames
and related quantities in user-code from an underlying world model, which can be
a physics engine, a kinematics library like KDL, a real-robot frame system, e.g.
ROS tf.

Pose, Twist, Wrench, Accel and Vec6 instances are immutable, so too are their
Stamped counterparts.

All geometry types can be compared and are hashable.

Arithmetic:
This applies to `Accel`, `Twist`, `Wrench`, and `Vec6`.

The arithmetic operators, + - * /, perform piece-wise operations, they are
intended to work with Vec6 instances.

Accel, Wrench, and Twist cannot have arithmetic operations applied to
instances of each-other.

For example:
```python
twist = Twist()
wrench = Wrench()
accel = Accel()

# Arithmetic does not apply between twist, wrench and accel:
twist + twist  # raises TypeError
wrench / accel  # raises TypeError

# Instead, it works with Vec6 instances:
vec6 = Vec6()
twist + vec6  # returns a Twist
vec6 * wrench  # returns a Wrench
```
"""

import abc
from typing import Any, Optional, Sequence, Text, Union

from dm_robotics.transformations import transformations as tr
import numpy as np

Grounding = Any  # The world pose of a Grounding is given by a Physics.
Frame = Union[Grounding, "PoseStamped"]

_IDENTITY_QUATERNION = np.array([1, 0, 0, 0], dtype=np.float64)
_ZERO_POSITION = np.zeros(3, dtype=np.float64)
_DEFAULT = "default_constant_string"


class Physics(abc.ABC):
  """Interface for 'Physics' as needed by this library.

  Unlike control.Physics, we only require the ability to get the world pose
  of scene elements using some identifier (aka Grounding).
  """

  @abc.abstractmethod
  def world_pose(self,
                 frame: Grounding,
                 get_pos: bool = True,
                 get_rot: bool = True) -> "Pose":
    """Return world pose of the provided frame.

    Args:
      frame: A frame identifier.
      get_pos: If False, zero out position entries.
      get_rot: If False, make the rotation an identity quaternion.

    Returns:
      A `geometry.Pose` containing the requested pose.
    """
    raise NotImplementedError


def frame_world_pose(frame: Optional[Frame],
                     physics: Optional[Physics] = None) -> "Pose":
  """Traverses the pose hierarchy to compute the world pose.

  Args:
    frame: A frame identifier.  Can be a Grounding, a `PoseStamped`, or None.
      None is interpreted as world frame.
    physics: Required if the frame of the root pose is a Grounding, in which
      case we need a physics to get its world pose.  If the root pose is None,
      we assume the highest-level frame is world.

  Returns:
    A `Pose` containing the world pose

  Raises:
    ValueError: If `frame` is a Grounding but no Physics was provided.
  """
  if frame is None:
    return Pose()
  elif isinstance(frame, PoseStamped):
    return frame.get_world_pose(physics)
  else:
    if physics is None:
      raise ValueError(
          "A `geometry.Physics` object is required to compute frame poses")
    return physics.world_pose(frame)


def frame_relative_pose(frame1: Optional[Frame],
                        frame2: Optional[Frame],
                        physics: Optional[Physics] = None) -> "Pose":
  """Computes the pose of `frame1` with respect to `frame2`.

  Args:
    frame1: A frame.  Can be an Grounding, a `PoseStamped`, or None.
    frame2: A frame.  Can be an Grounding, a `PoseStamped`, or None.
    physics: Required if the frame of the root pose is a Grounding, in which
      case we need a physics to get its world pose.  If the root pose is None,
      we assume the highest-level frame is world.

  Returns:
     A `Pose` containing the pose of frame1 w.r.t. `frame2`
  """
  pose_world_frame1 = frame_world_pose(frame1, physics=physics)
  pose_world_frame2 = frame_world_pose(frame2, physics=physics)
  return pose_world_frame2.inv().mul(pose_world_frame1)


class Pose:
  """A class for representing a pose.

  Internally this class uses position and quaternion, but exposes a matrix
  interface.  Equivalent to ROS geometry_msgs/Pose, except for the
  quaternion order. Here, [w, x, y, z] is used, whereas
  geometry_msgs/Quaternion uses [x, y, z, w].

  Instances of this class are immutable.
  Instances can be copied with copy.copy().
  """
  __slots__ = ("_position", "_quaternion", "_name")

  def __init__(self, position=None, quaternion=None, name=""):
    if position is None:
      self._position = _ZERO_POSITION
    else:
      self._position = np.asarray(position, dtype=np.float64)
      if self._position is position:  # Copy only if required.
        self._position = np.copy(self._position)

    if quaternion is None:
      self._quaternion = _IDENTITY_QUATERNION
    else:
      self._quaternion = np.asarray(quaternion, dtype=np.float64)
      if self._quaternion is quaternion:  # Copy only if required.
        self._quaternion = np.copy(self._quaternion)

    if name is None:
      raise ValueError("Name should be a string not None")
    self._name = name

    # Prevent mutation through read-only properties.
    self._position.flags.writeable = False
    self._quaternion.flags.writeable = False

  def __repr__(self) -> Text:
    if self.name:
      name = f"name={self.name}, "
    else:
      name = ""
    return f"Pose({name}position={self.position}, quaternion={self.quaternion})"

  def __eq__(self, other):
    if isinstance(other, Pose):
      return np.allclose(self.position, other.position) and (
          np.allclose(self.quaternion, other.quaternion) or
          np.allclose(self.quaternion,
                      -1 * other.quaternion)) and (self.name == other.name)
    else:
      return NotImplemented

  def __hash__(self):
    return hash(
        tuple([self._name] + self._position.tolist() +
              self._quaternion.tolist()))

  def mul(self, other: "Pose", name: Text = "") -> "Pose":
    """Multiplies other pose by this pose.

    Args:
      other: The other Pose to multiply by.
      name: An optional name to set in the resulting Pose.

    Returns:
      Resulting pose.
    """
    new_pos = self.position + tr.quat_rotate(self.quaternion, other.position)
    new_quat = tr.quat_mul(self.quaternion, other.quaternion)
    return Pose(new_pos, new_quat, name=name)

  def inv(self) -> "Pose":
    inv_quat = tr.quat_inv(self.quaternion)
    return Pose(
        position=tr.quat_rotate(inv_quat, -1 * self.position),
        quaternion=inv_quat)

  @property
  def hmat(self) -> np.ndarray:
    hmat = tr.quat_to_mat(self.quaternion)
    hmat[0:3, 3] = self.position
    return hmat

  @classmethod
  def from_hmat(cls, hmat: Union[np.ndarray, Sequence[float]]) -> "Pose":
    position = hmat[0:3, 3]
    quaternion = tr.mat_to_quat(hmat)
    return cls(position, quaternion)

  @classmethod
  def from_poseuler(cls,
                    poseuler: Union[np.ndarray, Sequence[float]],
                    ordering: Text = "XYZ") -> "Pose":
    position = poseuler[0:3]
    quaternion = tr.euler_to_quat(poseuler[3:6], ordering=ordering)
    return cls(position, quaternion)

  def to_poseuler(self, ordering="XYZ"):
    return np.hstack(
        [self.position,
         tr.quat_to_euler(self.quaternion, ordering=ordering)])

  def to_posquat(self):
    return np.hstack([self.position, self.quaternion])

  @property
  def position(self):
    return self._position

  @property
  def quaternion(self):
    return self._quaternion

  @property
  def name(self):
    return self._name

  def replace(self, position=_DEFAULT, quaternion=_DEFAULT, name=_DEFAULT):
    if position is _DEFAULT:
      position = self.position
    if quaternion is _DEFAULT:
      quaternion = self.quaternion
    if name is _DEFAULT:
      name = self.name
    return Pose(position=position, quaternion=quaternion, name=name)

  def with_quaternion(self, quaternion):
    return self.replace(quaternion=quaternion)

  def with_position(self, position):
    return self.replace(position=position)


class PoseStamped(object):
  """A class for representing a pose relative to a parent frame.

  The purpose of this class is to simplify the process of computing relative
  transformations between scene elements.  Every `PoseStamped` has a parent
  frame, which can be `None`, a `Grounding`, or another `PoseStamped`.

  The semantics of these possible parents are as follows:
    None: The pose is interpreted as a world pose.
    Grounding: The pose is interpreted in the frame of the Grounding, and
      a `Physics` object is used to resolve its world pose.
    PoseStamped: The pose is interpreted as a child of another `PoseStamped`,
      and the world pose is resolved recursively until a `Grounding` or
      `None` is found.

  Every `PoseStamped` can therefore be grounded in a common `world` frame, and
  by extension, any relative pose can be computed.

  Equivalent to ROS geometry_msgs/PoseStamped, supporting an arbitrary
  `Grounding` or another `PoseStamped` (or None) instead of a just string
  frame_id for the frame identifier.

  PoseStamped is immutable.
  """
  __slots__ = ("_pose", "_frame", "_name")
  base_type = Pose

  def __init__(self,
               pose: Union[Pose, np.ndarray, None],
               frame: Optional[Frame] = None,
               name: Text = ""):
    """Initialize PoseStamped.

    Args:
      pose: A `Pose` object, if None is give a default Pose is used.
      frame: A frame identifier.  Can be an Grounding, a `PoseStamped`, or None.
        If None, users should assume world frame.
      name: Optional name of the frame.

    Raises:
      ValueError: if `frame` or `name` arguments is invalid
    """
    if isinstance(pose, np.ndarray):
      pose = Pose.from_hmat(pose)
    elif pose is None:
      pose = Pose()

    if isinstance(frame, Pose):
      raise ValueError(
          "A Pose is not a frame, did you mean `PoseStamped(pose, None)`?")
    if name is None:
      raise ValueError("Name should be a string not None")

    self._pose = pose
    self._frame = frame  # type: Union[Grounding, PoseStamped, None]
    self._name = name

  def __repr__(self) -> Text:
    if self.name:
      name_str = "name:{}, ".format(self.name)
    else:
      name_str = ""

    return "{}({}pose={}, frame={})".format(self.__class__.__name__, name_str,
                                            self.pose, self.frame)

  def __eq__(self, other):
    if isinstance(other, PoseStamped):
      return (self.pose == other.pose) and (self.frame == other.frame) and (
          self.name == other.name)
    else:
      return NotImplemented

  def __hash__(self):
    return hash((self._pose, self._name))

  def to_world(self, physics: Optional[Physics] = None) -> "PoseStamped":
    """Returns this pose in the world frame - flattens the frame hierarchy.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to get its world pose.  If the root pose is None,
        we assume the highest-level frame is world.

    Returns:
      A new PoseStamped, in the world frame.
    """
    return PoseStamped(pose=self.get_world_pose(physics), frame=None)

  def to_frame(self,
               frame: Optional[Frame],
               physics: Optional[Physics] = None) -> "PoseStamped":
    """Returns this pose in the given frame.

    Args:
      frame: A frame identifier.  Can be an Grounding, a `PoseStamped`, or None.
        If None, world frame is assumed.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to get its world pose.  If the root pose is None,
        we assume the highest-level frame is world.

    Returns:
       A new PoseStamped, in the given frame.
    """
    return PoseStamped(
        pose=self.get_relative_pose(frame, physics=physics), frame=frame)

  def get_relative_pose(self,
                        other: Optional[Frame],
                        physics: Optional[Physics] = None) -> Pose:
    """Computes the pose of this frame with respect to `other`.

    Args:
      other: A frame.  Can be an Grounding, a `PoseStamped`, or None.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
       A `Pose` containing the pose of self in frame `other`
    """
    return frame_relative_pose(self, other, physics=physics)

  def get_world_pose(self, physics: Optional[Physics] = None) -> Pose:
    """Recursively computes the world pose given the current frame.

    Args:
      physics: Required if the frame of the root pose is a reference that only a
        physics instance can resolve, If the root pose is None, we assume the
        highest-level frame is world.

    Returns:
      A `Pose` containing the world pose

    Raises:
      ValueError: If the root frame is a Grounding and no physics was provided.
    """

    # recurse back through poses until we have the transform wrt a grounding
    frame = self.frame
    pose = self.pose if self.pose is not None else Pose()

    if isinstance(frame, PoseStamped):
      pose_world_element = frame.get_world_pose(physics)
    elif frame is None:
      # if no grounding at root, assume pose contains a world-frame transform
      pose_world_element = Pose()
    else:
      # Get the world pose of the frame from physics.
      if physics is None:
        raise ValueError(
            "A Physics object is required for frames with a grounding")
      else:
        pose_world_element = physics.world_pose(frame)

    return pose_world_element.mul(pose)

  @property
  def data(self) -> Pose:
    """Returns pose.  Provides a common data accessor across stamped types."""
    return self._pose

  @property
  def pose(self) -> Pose:
    return self._pose

  @property
  def frame(self):
    return self._frame

  @property
  def name(self):
    return self._name

  def replace(self, pose=_DEFAULT, frame=_DEFAULT, name=_DEFAULT):
    if pose is _DEFAULT:
      pose = self.pose
    if frame is _DEFAULT:
      frame = self.frame
    if name is _DEFAULT:
      name = self.name
    return PoseStamped(pose=pose, frame=frame, name=name)

  def with_pose(self, pose):
    return self.replace(pose=pose)

  def with_frame(self, frame):
    return self.replace(frame=frame)


class HybridPoseStamped(PoseStamped):
  """A PoseStamped with a dynamically-overridable position or orientation.

  HybridPoseStamped is a convenience class to represent a PoseStamped whose
  world position or orientation can be overridden using user-provided values.

  This is useful when we want to define a frame which has a fixed position or
  orientation relative to a dynamic frame, i.e. one with a grounding as a
  parent.

  For example, we often want to define a coordinate frame for joystick actions
  as having the position of some gripper-site, but the orientation of the robot
  or world frame.

  In this case we can't simply compute a fixed Pose to transform the gripper's
  PoseStamped to the desired frame at init-time because of the lazy-evaluation
  of Grounding frames.
  """

  def __init__(
      self,
      pose: Union[Pose, np.ndarray, None],
      frame: Optional[Frame] = None,
      name: Text = "",
      position_override: Optional[Frame] = None,
      quaternion_override: Optional[Frame] = None,
  ):
    """Initialize PoseStamped.

    The override parameters are mutually exclusive, only one may be supplied.

    Args:
      pose: A `Pose` object.
      frame: A frame identifier.  Can be a Grounding, a `PoseStamped`, or None.
        If None, users should assume world frame.
      name: Frame name.
      position_override: A position override for the final world-pose. the
        frame's world-pose is evaluated and the position is used to override the
        position of `frame`.
      quaternion_override: A quaternion override for the final world-pose. The
        frame's world-pose is evaluated and the quaternion is used to override
        the rotation of `frame`.

    Raises:
      ValueError: If both position_override and quaternion_override are given.
    """
    super().__init__(pose, frame, name)
    if position_override is not None and quaternion_override is not None:
      raise ValueError("Attempting to create a HybridPoseStamped with "
                       "multiple position / quaternion overrides. "
                       "Just create a child frame.")

    self._position_override = position_override
    self._quaternion_override = quaternion_override

  def __eq__(self, other):
    if isinstance(other, HybridPoseStamped):
      return (super().__eq__(other) and
              (self.position_override == other.position_override) and
              (self.quaternion_override == other.quaternion_override))
    else:
      return NotImplemented

  def __hash__(self):
    return hash((HybridPoseStamped, super().__hash__()))

  def __repr__(self) -> Text:
    return ("{}(pose={}, frame={}, position_override={}, "
            "quaternion_override={})".format(self.__class__.__name__, self.pose,
                                             self.frame, self.position_override,
                                             self.quaternion_override))

  @property
  def position_override(self):
    return self._position_override

  @property
  def quaternion_override(self):
    return self._quaternion_override

  def get_world_pose(self, physics: Optional[Physics] = None) -> Pose:
    """Recursively computes the world pose given the current frame.

    For HybridPoseStamped the Pose members override the final transform
    rather than post-multiplying it.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      A `Pose` containing the world pose

    Raises:
      ValueError: If a Grounding is the root frame and no Physics was provided.
    """
    world_pose = super().get_world_pose(physics)

    if self._position_override is not None:
      position_override = frame_world_pose(self._position_override,
                                           physics).position
      return world_pose.with_position(position_override)

    elif self._quaternion_override is not None:
      quaternion_override = frame_world_pose(self._quaternion_override,
                                             physics).quaternion
      return world_pose.with_quaternion(quaternion_override)

    else:
      return world_pose

  def replace(self,
              pose=_DEFAULT,
              frame=_DEFAULT,
              name=_DEFAULT,
              position_override=_DEFAULT,
              quaternion_override=_DEFAULT):
    if pose is _DEFAULT:
      pose = self.pose
    if frame is _DEFAULT:
      frame = self.frame
    if name is _DEFAULT:
      name = self.name
    if position_override is _DEFAULT:
      position_override = self.position_override
    if quaternion_override is _DEFAULT:
      quaternion_override = self.quaternion_override
    return HybridPoseStamped(
        pose=pose,
        frame=frame,
        name=name,
        position_override=position_override,
        quaternion_override=quaternion_override)


class Vec6(object):
  """A helper base-class with operators for 6-vector types.

  Immutable.
  """
  __slots__ = ("_data",)

  def __init__(self, vec=None):
    if vec is None:
      self._data = np.zeros(6)
    else:
      self._data = np.asarray(vec)
      assert self._data.shape == (6,)
      # Defensive copy only if required.
      if self._data is vec:
        self._data = np.copy(self._data)
    self._data.flags.writeable = False

  def __getitem__(self, idx: int):
    return self._data[idx]

  def __repr__(self):
    return "{}({})".format(self.__class__.__name__, repr(self._data))

  def __add__(self, other):
    rhs = other.data if isinstance(other, Vec6) else other
    return type(self)(self.data.__add__(rhs))

  def __radd__(self, other_rhs):
    rhs = other_rhs.data if isinstance(other_rhs, Vec6) else other_rhs
    return type(self)(self.data.__add__(rhs))

  def __sub__(self, other):
    rhs = other.data if isinstance(other, Vec6) else other
    return type(self)(self.data.__sub__(rhs))

  def __mul__(self, other):
    rhs = other.data if isinstance(other, Vec6) else other
    return type(self)(self.data.__mul__(rhs))

  def __rmul__(self, other_rhs):
    rhs = other_rhs.data if isinstance(other_rhs, Vec6) else other_rhs
    return type(self)(self.data.__mul__(rhs))

  def __truediv__(self, other):
    rhs = other.data if isinstance(other, Vec6) else other
    return type(self)(self.data.__truediv__(rhs))

  def __eq__(self, other):
    if isinstance(other, Vec6):
      return np.allclose(self.data, other.data)
    else:
      return NotImplemented

  def __hash__(self):
    return hash(tuple([type(self)] + self._data.tolist()))

  def _with_data(self, data_slice, value):
    """Creates a new object with `_data[data_slice]` set to `value`."""
    new_data = np.copy(self._data)
    new_data[data_slice] = value
    return type(self)(new_data)

  @property
  def data(self):
    return self._data

  def with_data(self, value):
    return self._with_data(slice(0, 6), value)


class VectorStamped(object):
  """A generic class for representing vectors relative to a Frame.

  This class can be helpful for defining frame-dependent quantities which aren't
  specifically position or force related quantities, e.g. a control gain.

  Frame transformations for generic vectors only perform rotation, and omit the
  cross-product terms in velocity_transform and force_transform.  I.e. they
  only support changing the view on the vector.

  * Currently only supports 6-vectors.

  VectorStamped is immutable.
  """
  __slots__ = ("_vector", "_frame")
  base_type = Vec6

  def __init__(self, vector: Optional[Union[Sequence[float], np.ndarray, Vec6]],
               frame: Optional[Frame]):
    """Initialize VectorStamped.

    Args:
      vector: A `Vec6` or simply a 6-dim numpy array or list.  If None, a
        default `Vec6` is used, with all components being zero.
      frame: A frame identifier.  Can be an Grounding, a `PoseStamped`, or None.
        If None, users should assume world frame.
    """
    if isinstance(vector, self.base_type):
      self._vector = vector
    else:
      self._vector = self.base_type(vector)
    self._frame = frame

  def __repr__(self):
    return "VectorStamped(vector={}, frame={})".format(self.vector, self.frame)

  def __eq__(self, other):
    if isinstance(other, VectorStamped):
      return self.data == other.data and self.frame == other.frame
    else:
      return NotImplemented

  def __hash__(self):
    return hash((VectorStamped, self.vector))

  def to_frame(self,
               frame: Optional[Frame],
               physics: Optional[Physics] = None) -> "VectorStamped":
    """Sets the frame and updates the vector accordingly.

    This function will not change the implied world vector, but it will result
    in the vector being expressed with respect to a new frame.

    Args:
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      A new VectorStamped with the given frame.
    """
    return VectorStamped(
        vector=self.get_relative_vector(frame, physics=physics), frame=frame)

  def to_world(self, physics: Optional[Physics] = None) -> "VectorStamped":
    """Converts vector to the world frame - flattens the frame hierarchy.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      A new VectorStamped in the world frame.
    """
    return VectorStamped(vector=self.get_world_vector(physics), frame=None)

  def get_relative_vector(self,
                          frame: Optional[Frame],
                          physics: Optional[Physics] = None) -> Vec6:
    """Returns this vector in frame `frame`.

    Args:
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.
    """
    pose_frame_self = frame_relative_pose(self.frame, frame, physics=physics)
    vector_frame = tr.rotate_vec6(pose_frame_self.hmat, self.vector.data)
    return self.base_type(vector_frame)

  def get_world_vector(self, physics: Optional[Physics] = None) -> Vec6:
    """Computes equivalent vector in the world frame.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      A `Vec6` containing this vector in the world frame.

    Raises:
      ValueError: If a Grounding is the root frame no Physics
        was provided.
    """
    pose_world_frame = frame_world_pose(self.frame, physics)
    vector_world = tr.rotate_vec6(pose_world_frame.hmat, self.vector.data)
    return self.base_type(vector_world)

  @property
  def data(self):
    """Returns vector.  Provides a common data accessor across stamped types."""
    return self._vector

  @property
  def vector(self):
    return self._vector

  @property
  def frame(self):
    return self._frame

  def with_vector(self, vector):
    return VectorStamped(vector=vector, frame=self.frame)

  def with_frame(self, frame):
    return VectorStamped(vector=self.vector, frame=frame)


class Twist(Vec6):
  """A class for representing a cartesian velocity.

  Equivalent to ROS geometry_msgs/Twist, except represented as a single numpy
  6-dim array as [linear, angular]

  This class is immutable.
  """
  __slots__ = ("_data",)

  def _linear_slice(self):
    return slice(0, 3)

  def _angular_slice(self):
    return slice(3, 6)

  def __add__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__add__(other)

  def __sub__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot subtract these types, consider Vec6")
    else:
      return super().__sub__(other)

  def __mul__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot multiply these types, consider Vec6")
    else:
      return super().__mul__(other)

  def __truediv__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot divide these types, consider Vec6")
    else:
      return super().__truediv__(other)

  def __radd__(self, other_rhs):  # pylint: disable=useless-super-delegation
    return super().__radd__(other_rhs)

  def __rmul__(self, other_rhs):  # pylint: disable=useless-super-delegation
    return super().__rmul__(other_rhs)

  @property
  def linear(self):
    """Cartesian linear velocity."""
    return self._data[self._linear_slice()]

  def with_linear(self, value):
    return self._with_data(self._linear_slice(), value)

  @property
  def angular(self):
    """Cartesian angular velocity."""
    return self._data[self._angular_slice()]

  def with_angular(self, value):
    return self._with_data(self._angular_slice(), value)

  @property
  def full(self):
    return self.data

  def with_full(self, value):
    return self.with_data(value)

  def __repr__(self):
    return "Twist(linear={}, angular={})".format(self.linear, self.angular)


class TwistStamped(object):
  """A class for representing a twist relative to a Frame.

  The purpose of this class is to simplify the process of converting cartesian
  velocities between frames.  For example it is often necessary to define a
  desired twist with respect to an interest point on a grasped object, and then
  convert it to a wrist or pinch-site for manipulation control.

  Equivalent to ROS geometry_msgs/TwistStamped, but supports a Grounding
  or a `PoseStamped` (or None) instead of a just string frame_id for the frame
  identifier.

  TwistStamped is immutable.
  """
  __slots__ = ("_twist", "_frame")
  base_type = Twist

  def __init__(self, twist: Optional[Union[Twist, np.ndarray, Sequence[float]]],
               frame: Optional[Frame]):
    """Initialize TwistStamped.

    Args:
      twist: A `Twist` or simply a 6-dim numpy array or list,   If None, a
        default `Twist` is used, with all components being zero.
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
    """
    if isinstance(twist, self.base_type):
      self._twist = twist
    else:
      self._twist = self.base_type(twist)
    self._frame = frame

  def __repr__(self):
    return "TwistStamped(twist={}, frame={})".format(self.twist, self.frame)

  def __eq__(self, other):
    if isinstance(other, TwistStamped):
      return self.data == other.data and self.frame == other.frame
    else:
      return NotImplemented

  def __hash__(self):
    return hash((TwistStamped, self._twist))

  def to_frame(self,
               frame: Optional[Frame],
               physics: Optional[Physics] = None) -> "TwistStamped":
    """Returns a new TwistStamped with the given frame and updated twist.

    This function will not change the implied world twist, but it will result in
    the twist being expressed with respect to a new frame.

    Args:
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      A new `TwistStamped`, with the given frame.
    """
    return TwistStamped(
        twist=self.get_relative_twist(frame, physics=physics), frame=frame)

  def to_world(self, physics: Optional[Physics] = None) -> "TwistStamped":
    """Converts twist to the world frame - flattens the frame hierarchy.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      A new `TwistStamped`, in the world frame.
    """
    return TwistStamped(twist=self.get_world_twist(physics), frame=None)

  def get_relative_twist(self,
                         frame: Optional[Frame],
                         physics: Optional[Physics] = None) -> Twist:
    """Returns this twist in frame `frame`.

    Args:
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.
    """
    pose_frame_self = frame_relative_pose(self.frame, frame, physics=physics)
    twist_frame = tr.velocity_transform(pose_frame_self.hmat, self.twist.full)
    return self.base_type(twist_frame)

  def get_world_twist(self,
                      physics: Optional[Physics] = None,
                      rot_only: bool = False) -> Twist:
    """Computes equivalent twist in the world frame.

    Note that by default this is NOT simply the twist of this frame rotated to
    the world frame (unless rot_only is True).  Rather, it is the instantaneous
    velocity of the world origin when rigidly attached to this twist's frame.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.
      rot_only: (optional) If True, drops the translation to the world origin.
        Use this as a shortcut to obtaining the twist of this frame as viewed in
        world coords, without creating a new frame and explicitly calling
        `to_frame`.

    Returns:
      A new `Twist` representing this twist at the world frame origin.

    Raises:
      ValueError: If a Grounding is the root frame no Physics was provided.
    """
    pose_world_frame = frame_world_pose(self.frame, physics)
    if rot_only:
      pose_world_frame = pose_world_frame.with_position(_ZERO_POSITION)
    twist_world = tr.velocity_transform(pose_world_frame.hmat, self.twist.full)
    return self.base_type(twist_world)

  @property
  def data(self) -> Twist:
    """Returns twist.  Provides a common data accessor across stamped types."""
    return self._twist

  @property
  def twist(self) -> Twist:
    return self._twist

  @property
  def frame(self):
    return self._frame

  def with_twist(self, twist):
    return TwistStamped(twist=twist, frame=self.frame)

  def with_frame(self, frame):
    return TwistStamped(twist=self.twist, frame=frame)


class Wrench(Vec6):
  """A class for representing a cartesian wrench.

  Equivalent to ROS geometry_msgs/Wrench, except represented as a single numpy
  6-dim array as [force, torque]

  This class is immutable.
  """
  __slots__ = ("_data",)

  def _force_slice(self):
    return slice(0, 3)

  def _torque_slice(self):
    return slice(3, 6)

  def __add__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__add__(other)

  def __sub__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__sub__(other)

  def __mul__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__mul__(other)

  def __truediv__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__truediv__(other)

  def __radd__(self, other_rhs):  # pylint: disable=useless-super-delegation
    return super().__radd__(other_rhs)

  def __rmul__(self, other_rhs):  # pylint: disable=useless-super-delegation
    return super().__rmul__(other_rhs)

  @property
  def force(self):
    return self._data[self._force_slice()]

  def with_force(self, value):
    return self._with_data(self._force_slice(), value)

  @property
  def torque(self):
    return self._data[self._torque_slice()]

  def with_torque(self, value):
    return self._with_data(self._torque_slice(), value)

  @property
  def full(self):
    return self.data

  def with_full(self, value):
    return self.with_data(value)

  def __repr__(self):
    return "Wrench(force={}, torque={})".format(self.force, self.torque)


class WrenchStamped(object):
  """A class for representing a wrench relative to a Frame.

  The purpose of this class is to simplify the process of converting cartesian
  wrenches between frames.  For example it is often necessary to define a
  desired wrench with respect to an interest point on a grasped object, and then
  convert it to a wrist or pinch-site for manipulation control.

  Equivalent to ROS geometry_msgs/WrenchStamped, but supports a Grounding
  or a `PoseStamped` (or None) instead of a just string frame_id for the frame
  identifier.

  WrenchStamped is immutable.
  """
  __slots__ = ("_wrench", "_frame")
  base_type = Wrench

  def __init__(self, wrench: Optional[Union[Wrench, np.ndarray,
                                            Sequence[float]]],
               frame: Optional[Frame]):
    """Initialize WrenchStamped.

    Args:
      wrench: A `Wrench` or simply a 6-dim numpy array or list.  If None, a
        default `Wrench` is used, with all components being zero.
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
    """
    if isinstance(wrench, self.base_type):
      self._wrench = wrench
    else:
      self._wrench = self.base_type(wrench)
    self._frame = frame

  def __eq__(self, other):
    if isinstance(other, WrenchStamped):
      return self.data == other.data and self.frame == other.frame
    else:
      return NotImplemented

  def __hash__(self):
    return hash((WrenchStamped, self._wrench))

  def to_frame(self,
               frame: Optional[Frame],
               physics: Optional[Physics] = None) -> "WrenchStamped":
    """Sets the frame and updates the wrench accordingly.

    This function will not change the implied world wrench, but it will result
    in the wrench being expressed with respect to a new frame.

    Args:
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      A `WrenchStamped`, in the given frame.
    """
    return WrenchStamped(
        wrench=self.get_relative_wrench(frame, physics=physics), frame=frame)

  def to_world(self, physics: Optional[Physics] = None) -> "WrenchStamped":
    """Converts wrench to the target frame - flattens the frame hierarchy.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      A `WrenchStamped`, in the world frame.
    """
    return WrenchStamped(wrench=self.get_world_wrench(physics), frame=None)

  def get_relative_wrench(self,
                          frame: Optional[Frame],
                          physics: Optional[Physics] = None) -> Wrench:
    """Returns this wrench in frame `frame`.

    Args:
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.
    """
    pose_frame_self = frame_relative_pose(self.frame, frame, physics=physics)
    wrench_frame = tr.force_transform(pose_frame_self.hmat, self.wrench.full)
    return self.base_type(wrench_frame)

  def get_world_wrench(self,
                       physics: Optional[Physics] = None,
                       rot_only: bool = False) -> Wrench:
    """Computes equivalent wrench in the world frame.

    Note that by default this is NOT simply the wrench of this frame rotated to
    the world frame (unless rot_only is True).  Rather, it is the instantaneous
    wrench of a point rigidly attached to this wrench's frame that is currently
    at the world origin.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.
      rot_only: (optional) If True, drops the translation to the world origin.
        Use this as a shortcut to obtaining the wrench of this frame as viewed
        in world coords, without creating a new frame and explicitly calling
        `to_frame`.

    Returns:
      A new `Wrench` representing this wrench at the world frame origin.

    Raises:
      ValueError: If a Grounding is the root frame no Physics was provided.
    """
    pose_world_frame = frame_world_pose(self.frame, physics)
    if rot_only:
      pose_world_frame = pose_world_frame.with_position(_ZERO_POSITION)
    wrench_world = tr.force_transform(pose_world_frame.hmat, self.wrench.full)
    return self.base_type(wrench_world)

  @property
  def data(self):
    """Returns wrench.  Provides a common data accessor across stamped types."""
    return self._wrench

  @property
  def wrench(self):
    return self._wrench

  @property
  def frame(self):
    return self._frame

  def with_wrench(self, wrench):
    return WrenchStamped(wrench=wrench, frame=self.frame)

  def with_frame(self, frame):
    return WrenchStamped(wrench=self.wrench, frame=frame)

  def __repr__(self):
    return "WrenchStamped(wrench={}, frame={})".format(self.wrench, self.frame)


class Accel(Vec6):
  """A class for representing a cartesian acceleration.

  Equivalent to ROS geometry_msgs/Accel, except represented as a single numpy
  6-dim array as [linear, angular].

  This class is immutable.
  """
  __slots__ = ("_data",)

  def _linear_slice(self):
    return slice(0, 3)

  def _angular_slice(self):
    return slice(3, 6)

  def __add__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__add__(other)

  def __sub__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__sub__(other)

  def __mul__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__mul__(other)

  def __truediv__(self, other):
    if isinstance(other, (Accel, Twist, Wrench)):
      raise TypeError("Cannot add these types, consider Vec6")
    else:
      return super().__truediv__(other)

  def __radd__(self, other_rhs):  # pylint: disable=useless-super-delegation
    return super().__radd__(other_rhs)

  def __rmul__(self, other_rhs):  # pylint: disable=useless-super-delegation
    return super().__rmul__(other_rhs)

  @property
  def linear(self):
    """Cartesian linear acceleration."""
    return self._data[self._linear_slice()]

  def with_linear(self, value):
    return self._with_data(self._linear_slice(), value)

  @property
  def angular(self):
    """Cartesian angular acceleration."""
    return self._data[self._angular_slice()]

  def with_angular(self, value):
    return self._with_data(self._angular_slice(), value)

  @property
  def full(self):
    return self.data

  def with_full(self, value):
    return self.with_data(value)


class AccelStamped(object):
  """A class for representing an acceleration relative to a Frame.

  The purpose of this class is to simplify the process of converting cartesian
  accelerations between frames.  For example it is often necessary to define a
  desired accel with respect to an interest point on a grasped object, and then
  convert it to a wrist or pinch-site for manipulation control.

  Equivalent to ROS geometry_msgs/AccelStamped, but supports a Grounding
  or a `PoseStamped` (or None) instead of a just string frame_id for the frame
  identifier.

  AccelStamped is immutable.
  """
  __slots__ = ("_accel", "_frame")
  base_type = Accel

  def __init__(self, accel: Optional[Union[Accel, np.ndarray, Sequence[float]]],
               frame: Optional[Frame]):
    """Initialize AccelStamped.

    Args:
      accel: A `Accel` or simply a 6-dim numpy array or list.  If None, a
        default `Accel` is used, with all components being zero.
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
    """
    if isinstance(accel, self.base_type):
      self._accel = accel
    else:
      self._accel = self.base_type(accel)
    self._frame = frame

  def __eq__(self, other):
    if isinstance(other, AccelStamped):
      return self.data == other.data and self.frame == other.frame
    else:
      return NotImplemented

  def __hash__(self):
    return hash((AccelStamped, self._accel))

  def to_frame(self,
               frame: Optional[Frame],
               physics: Optional[Physics] = None) -> "AccelStamped":
    """Sets the frame and updates the accel accordingly.

    This function will not change the implied world accel, but it will result
    in the accel being expressed with respect to a new frame.

    Args:
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      An AccelStamped in the given frame.
    """
    return AccelStamped(
        accel=self.get_relative_accel(frame, physics=physics), frame=frame)

  def to_world(self, physics: Optional[Physics] = None) -> "AccelStamped":
    """Converts accel to the world frame - flattens the frame hierarchy.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
      An AccelStamped in the world frame.
    """
    return AccelStamped(accel=self.get_world_accel(physics), frame=None)

  def get_relative_accel(self,
                         frame: Optional[Frame],
                         physics: Optional[Physics] = None) -> Accel:
    """Converts accel to the target frame.

    Args:
      frame: A frame identifier.  Can be a Grounding, a PoseStamped, or None. If
        None, users should assume world frame.
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.

    Returns:
       accel_frame: A `Accel` containing the accel in frame `frame`
    """
    pose_frame_self = frame_relative_pose(self.frame, frame, physics=physics)
    accel_frame = tr.force_transform(pose_frame_self.hmat, self.accel.full)
    return self.base_type(accel_frame)

  def get_world_accel(self,
                      physics: Optional[Physics] = None,
                      rot_only: bool = False) -> Accel:
    """Computes equivalent acceleration in the world frame.

    Note that by default this is NOT simply the accel of this frame rotated to
    the world frame (unless rot_only is True).  Rather, it is the instantaneous
    accel of a point rigidly attached to this accel's frame that is currently
    at the world origin.

    Args:
      physics: Required if the frame of the root pose is a Grounding, in which
        case we need a physics to look up its world pose.  If the root pose is
        None, we assume the highest-level frame is world.
      rot_only: (optional) If True, drops the translation to the world origin.
        Use this as a shortcut to obtaining the accel of this frame as viewed in
        world coords, without creating a new frame and explicitly calling
        `to_frame`.

    Returns:
      A new `Accel` representing this acceleration at the world frame origin.

    Raises:
      ValueError: If a Grounding is the root frame no Physics was provided.
    """
    pose_world_frame = frame_world_pose(self.frame, physics)
    if rot_only:
      pose_world_frame = pose_world_frame.with_position(_ZERO_POSITION)
    accel_world = tr.force_transform(pose_world_frame.hmat, self.accel.full)
    return self.base_type(accel_world)

  @property
  def data(self):
    """Returns accel.  Provides a common data accessor across stamped types."""
    return self._accel

  @property
  def accel(self):
    return self._accel

  @property
  def frame(self):
    return self._frame

  def with_accel(self, accel):
    return AccelStamped(accel=accel, frame=self.frame)

  def with_frame(self, frame):
    return AccelStamped(accel=self.accel, frame=frame)

  def __repr__(self):
    return "AccelStamped(accel={}, frame={})".format(self.accel, self.frame)
