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
"""Action spaces for manipulation."""

from typing import Callable, Generator, Iterable, List, Optional, Text

from absl import logging
from dm_control import mjcf  # type: ignore
from dm_env import specs
from dm_robotics import agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
import numpy as np

# Internal profiling


def action_limits(
    cartesian_translation_action_limit: List[float],
    cartesian_rotation_action_limit: List[float]) -> List[float]:
  """Returns the action limits for the robot in this subtask."""
  if len(cartesian_translation_action_limit) == 3:
    cartesian_vals = cartesian_translation_action_limit
  else:
    assert len(cartesian_translation_action_limit) == 1
    cartesian_vals = cartesian_translation_action_limit * 3

  if len(cartesian_rotation_action_limit) == 3:
    euler_vals = cartesian_rotation_action_limit
  else:
    assert len(cartesian_rotation_action_limit) == 1
    euler_vals = cartesian_rotation_action_limit * 3

  return cartesian_vals + euler_vals


class _DelegateActionSpace(af.ActionSpace[specs.BoundedArray]):
  """Delegate action space.

  Base class for delegate action spaces that exist to just give a specific type
  for an action space object (vs using prefix slicer directly).
  """

  def __init__(self, action_space: af.ActionSpace[specs.BoundedArray]):
    self._action_space = action_space

  @property
  def name(self):
    return self._action_space.name

  def spec(self) -> specs.BoundedArray:
    return self._action_space.spec()

  def project(self, action: np.ndarray) -> np.ndarray:
    assert len(action) == self._action_space.spec().shape[0]
    return self._action_space.project(action)


class ArmJointActionSpace(_DelegateActionSpace):
  """Arm joint action space.

  This is just giving a type name to a particular action space.
  I.e. while it just delegates to the underlying action space, the name
  tells us that this projects from an arm joint space to some other space.
  """
  pass


class GripperActionSpace(_DelegateActionSpace):
  """Gripper action space.

  This is just giving a type name to a particular action space.
  I.e. while it just delegates to the underlying action space, the name
  tells us that this projects from a gripper joint space to some other space.
  """
  pass


class CartesianTwistActionSpace(_DelegateActionSpace):
  """Cartesian Twist action space.

  This is just giving a type name to a particular action space.
  I.e. while it just delegates to the underlying action space, the name
  tells us that this projects from a cartesian twist space to some other space.
  """
  pass


class RobotArmActionSpace(_DelegateActionSpace):
  """Action space for a robot arm.

  This is just giving a type name to a particular action space.
  I.e. while it just delegates to the underlying action space, the name
  tells us that this projects an action for the arm. This should be used when
  users don't care about the nature of the underlying action space (for example,
  joint action space or cartesian action space).
  """
  pass


class CompositeActionSpace(af.ActionSpace[specs.BoundedArray]):
  """Composite Action Space consisting of other action spaces.

  Assumptions (which are verified):
  1. The input action spaces all `project` to the same shape space. I.e. the
     output of project for each space has the same shape.
  2. The input action spaces are all one dimensional.
  3. The input action spaces are all ActionSpace[BoundedArray] instances.

  Behaviour:
  The outputs of the composed action spaces are merged not by concatenation -
  they all output arrays of the same shape - but by using NaN as a sentinel, as
  as missing value.  An example merging:
    input1: [1.0, 2.0, NaN, NaN, Nan]
    input2: [NaN, NaN, NaN, 4.0, 5.0]
    merged: [1.0, 2.0, NaN, 4.0, 5.0]

  It is invalid if there is a position where both inputs have a non-NaN value.
  """

  def __init__(self,
               action_spaces: Iterable[af.ActionSpace[specs.BoundedArray]],
               name: Optional[Text] = None):
    if not action_spaces:
      logging.warning('CompositeActionSpace created with no action_spaces.')
    sub_specs = [space.spec() for space in action_spaces]
    for spec in sub_specs:
      if not isinstance(spec, specs.BoundedArray):
        raise ValueError('spec {} is not a BoundedArray, (type: {})'.format(
            spec, type(spec)))

    sizes = [spec.shape[0] for spec in sub_specs]
    minimums = [spec.minimum for spec in sub_specs]
    maximums = [spec.maximum for spec in sub_specs]

    if sub_specs:
      spec_dtype = np.find_common_type([spec.dtype for spec in sub_specs], [])
      min_dtype = np.find_common_type([lim.dtype for lim in minimums], [])
      max_dtype = np.find_common_type([lim.dtype for lim in maximums], [])
      minimum = np.concatenate(minimums).astype(min_dtype)
      maximum = np.concatenate(maximums).astype(max_dtype)
    else:
      # No input spaces; we have to default the data type.
      spec_dtype = np.float32
      minimum = np.asarray([], dtype=spec_dtype)
      maximum = np.asarray([], dtype=spec_dtype)

    self._component_action_spaces = action_spaces
    self._composite_action_spec = specs.BoundedArray(
        shape=(sum(sizes),),
        dtype=spec_dtype,
        minimum=minimum,
        maximum=maximum,
        name='\t'.join([spec.name for spec in sub_specs if spec.name]))

    self._name = name or '_'.join(
        [space.name for space in action_spaces if space.name])

  @property
  def name(self) -> Text:
    return self._name

  def spec(self) -> specs.BoundedArray:
    return self._composite_action_spec

  # Profiling for .wrap('CompositeActionSpace.project')
  def project(self, action: np.ndarray) -> np.ndarray:
    if not self._component_action_spaces:
      return np.asarray([], dtype=self.spec().dtype)

    # Check input value has correct shape (and legal values).
    spec_utils.validate(self._composite_action_spec, action, ignore_nan=True)

    cur_action = None  # type: np.ndarray
    for action_space, action_component in zip(self._component_action_spaces,
                                              self._action_components(action)):
      projection = action_space.project(action_component)
      if cur_action is None:
        cur_action = np.full(
            projection.shape, fill_value=np.nan, dtype=projection.dtype)
      elif not np.all(cur_action.shape == projection.shape):
        raise ValueError(f'Projected actions differ in shape! cur_action: '
                         f'{cur_action.shape}, projection: {projection.shape}')
      cur_empty_indices = np.isnan(cur_action)
      proj_empty_indices = np.isnan(projection)
      assert np.all(
          np.logical_or(proj_empty_indices, cur_empty_indices)
      ), 'The projection and current action empty indices do not align'
      proj_valid_indices = np.logical_not(proj_empty_indices)
      cur_action[proj_valid_indices] = projection[proj_valid_indices]

    assert cur_action is not None, 'Program error, no action created!'

    return cur_action

  def _action_components(
      self, action: np.ndarray) -> Generator[np.ndarray, None, None]:
    start_index = 0
    for action_space in self._component_action_spaces:
      input_length = action_space.spec().shape[0]
      end_index = start_index + input_length

      assert end_index <= self._composite_action_spec.shape[0]
      action_component = action[start_index:end_index]
      start_index = end_index

      yield action_component
    assert start_index == self._composite_action_spec.shape[0]


class ReframeVelocityActionSpace(af.ActionSpace):
  """Transforms a twist from one frame to another."""

  def __init__(self,
               spec: specs.BoundedArray,
               physics_getter: Callable[[], mjcf.Physics],
               input_frame: geometry.Frame,
               output_frame: geometry.Frame,
               name: Text = 'ReframeVelocity'):
    self._spec = spec
    self._physics_getter = physics_getter
    self._physics = mujoco_physics.from_getter(physics_getter)
    self._input_frame = input_frame
    self._output_frame = output_frame
    self._name = name

  @property
  def name(self) -> Text:
    return self._name

  def spec(self) -> specs.BoundedArray:
    return self._spec

  def project(self, action: np.ndarray) -> np.ndarray:
    input_twist = geometry.TwistStamped(action, self._input_frame)
    output_twist = input_twist.to_frame(
        self._output_frame, physics=self._physics)
    output_action = output_twist.twist.full
    return spec_utils.shrink_to_fit(value=output_action, spec=self._spec)
