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

"""A Moma Base Task."""

import collections
from typing import Callable, Dict, Optional, Sequence, List

from absl import logging
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.moma import effector
from dm_robotics.moma import prop
from dm_robotics.moma import robot
from dm_robotics.moma import sensor as moma_sensor
import numpy as np

# Internal profiling

_REWARD_TYPE = np.float32
_DISCOUNT_TYPE = np.float32

SceneInitializer = Callable[[np.random.RandomState], None]


class BaseTask(composer.Task):
  """Base class for MoMa tasks.

  This class is parameterized by the required components of a MoMa task. This
  includes:
   - the arena (cell, basket, floor, and other world objects)
   - the robot (an encapsulation of the arm, gripper, and optionally things like
     bracelet camera)
   - props (free and fixed elements whose placement can be controlled by the
     initializer)
   - the scene initializer (composes the scene and positions objects/robots for
     each episode)
  """

  def __init__(self, task_name: str, arena: composer.Arena,
               robots: Sequence[robot.Robot], props: Sequence[prop.Prop],
               extra_sensors: Sequence[moma_sensor.Sensor],
               extra_effectors: Sequence[effector.Effector],
               scene_initializer: SceneInitializer,
               episode_initializer: composer.Initializer,
               control_timestep: float):
    """Base Task Constructor.

    The initializers are run before every episode, in a two step process.
    1. The scene initializer is run which can modify the scene (MJCF), for
       example by moving bodies that have no joints.
    2. The episode initializer is run after the MJCF is compiled, this can
       modify any degree of freedom in the scene, like props that have a
       free-joint, robot arm joints, etc.

    Args:
      task_name: The name of the task.
      arena: The arena Entity to use.
      robots: List of robots that make up the task.
      props: List of props that are in the scene.
      extra_sensors: A list of sensors that aren't tied to a specific robot (eg:
        a scene camera sensors, or a sensor that detects some prop's pose).
      extra_effectors: A list of effectors that aren't tied to a specific robot.
      scene_initializer: An initializer, called before every episode, before
        scene compilation.
      episode_initializer: An initializer, called before every episode, after
        scene compilation.
      control_timestep: The control timestep of the task.
    """

    self._task_name = task_name
    self._arena = arena
    self._robots = robots
    self._props = props
    self._extra_sensors = extra_sensors
    self._extra_effectors = extra_effectors
    self._scene_initializer = scene_initializer
    self._episode_initializer = episode_initializer
    self.control_timestep = control_timestep
    self._initialize_sensors()
    self._teardown_callables: List[Callable[[], None]] = []

  def _initialize_sensors(self):
    for s in self.sensors:
      s.initialize_for_task(
          self.control_timestep,
          self.physics_timestep,
          self.physics_steps_per_control_step,
      )

  def name(self):
    return self._task_name

  @property
  def episode_initializer(self):
    return self._episode_initializer

  @property
  def sensors(self) -> Sequence[moma_sensor.Sensor]:
    """Returns all of the sensors for this task."""
    sensors = []
    for rbt in self._robots:
      sensors += rbt.sensors

    sensors += self._extra_sensors
    return sensors

  @property
  def effectors(self) -> Sequence[effector.Effector]:
    """Returns all of the effectors for this task."""
    effectors = []
    for rbt in self._robots:
      effectors += rbt.effectors

    effectors += self._extra_effectors
    return effectors

  def effectors_action_spec(
      self,
      physics: mjcf.Physics,
      effectors: Optional[Sequence[effector.Effector]] = None
  ) -> specs.BoundedArray:
    """Returns the action spec for a sequence of effectors.

    Args:
      physics: The environment physics.
      effectors: Optional subset of effectors for which to get the action spec.
        If this is None or empty, then all of the tasks effectors are used to
        compose the action spec.
    """
    a_specs = [a.action_spec(physics) for a in (effectors or self.effectors)]
    return spec_utils.merge_specs(a_specs)

  @property
  def root_entity(self):
    return self._arena

  @property
  def arena(self):
    return self._arena

  @property
  def robots(self) -> Sequence[robot.Robot]:
    """Returns the task robots."""
    return self._robots

  @property
  def props(self) -> Sequence[prop.Prop]:
    """Returns the Props used in the task."""
    return self._props

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    # No entity observables, only explicitly defined task observables.
    base_obs = self.task_observables
    return collections.OrderedDict(base_obs)

  @property
  def task_observables(self) -> Dict[str, observable.Observable]:
    all_observables = {}

    for sensor in self.sensors:
      common = sensor.observables.keys() & all_observables.keys()
      if common:
        logging.error('Sensors have conflicting observables: %s', common)

      all_observables.update(sensor.observables)

    for k in all_observables:
      all_observables[k] = self._restrict_type(all_observables[k])

    return all_observables

  # Profiling for .wrap()
  def before_step(self, physics, actions, random_state):
    """Function called before every environment step."""
    # Moma base task is actuated directly using individual effectors. This is
    # done by calling each target effector's set_control method prior to calling
    # the environment step method. This can be done either manually or by
    # interfacing with a convenient wrapper such as SubTaskEnvironment or
    # MomaOption.
    pass

  def after_compile(self, physics: mjcf.Physics,
                    random_state: np.random.RandomState):
    """Initialization requiring access to physics or completed mjcf model."""
    for ef in self.effectors:
      ef.after_compile(self.root_entity.mjcf_model.root, physics)
    for se in self.sensors:
      se.after_compile(self.root_entity.mjcf_model.root, physics)

  def add_teardown_callable(self, teardown_fn: Callable[[], None]):
    """Adds function to be called when the task is closed."""
    self._teardown_callables.append(teardown_fn)

  def close(self):
    """Closes all the effectors and  sensors of the tasks.

    We might need to do some clean up after the task is over. This is mainly
    the case when using a real environment when we need to close connections
    that are made to the real robot.
    """
    for eff in self.effectors:
      eff.close()
    for sen in self.sensors:
      sen.close()
    for teardown_fn in self._teardown_callables:
      teardown_fn()

  def get_reward(self, physics):
    return _REWARD_TYPE(0.0)

  def get_discount(self, physics):
    return _DISCOUNT_TYPE(1.0)

  def get_reward_spec(self):
    return specs.Array(shape=(), dtype=_REWARD_TYPE, name='reward')

  def get_discount_spec(self):
    return specs.BoundedArray(
        shape=(), dtype=_DISCOUNT_TYPE, minimum=0., maximum=1., name='discount')

  def initialize_episode_mjcf(self, random_state):
    self._scene_initializer(random_state)

  # Profiling for .wrap()
  def initialize_episode(self, physics, random_state):
    """Function called at the beginning of every episode in sim (just once otw).

    Args:
      physics: An `mjcf.Physics`
      random_state: an `np.random.RandomState`
    """
    self._episode_initializer(physics, random_state)

    for e in self.effectors:
      e.initialize_episode(physics, random_state)

    for s in self.sensors:
      s.initialize_episode(physics, random_state)

  def action_spec(self, physics):
    # Moma Base Task has empty action spec as it is actuated directly through
    # Effectors using their set_control method. This is done by calling each
    # target effector's set_control method prior to calling the environment step
    # method. This can be done either manually or by interfacing with a
    # convenient wrapper such as SubTaskEnvironment or MomaOption.
    return specs.BoundedArray(
        shape=(0,),
        dtype=np.float32,
        minimum=np.array([], dtype=np.float32),
        maximum=np.array([], dtype=np.float32))

  def set_episode_initializer(self, episode_initializer: composer.Initializer):
    self._episode_initializer = episode_initializer

  def set_scene_initializer(self, scene_initializer: SceneInitializer):
    self._scene_initializer = scene_initializer

  def _restrict_type(self, obs: observable.Observable) -> observable.Observable:
    # When saving experiences as tf.Example protos there is some type coercion
    # involved. By simplifying the observation data types here we can ensure
    # that the data from loaded experience will match the data types of the
    # environment spec.
    casted = CastObservable(obs)
    casted.enabled = obs.enabled
    return casted


class CastObservable(observable.Observable):
  """Casts an observable while retaining its other attributes.

  This ensures that the Updater works correctly with observables that have their
  data type cast.  I.e. the aggregator, update_interval etc continue to work.
  """

  def __init__(self, delegate: observable.Observable):
    super().__init__(delegate.update_interval,
                     delegate.buffer_size,
                     delegate.delay,
                     delegate.aggregator,
                     delegate.corruptor)
    self._delegate = delegate

  @property
  def array_spec(self) -> Optional[specs.Array]:
    delegate_spec = self._delegate.array_spec
    if delegate_spec is None:
      return delegate_spec

    new_type = self._cast_type(delegate_spec.dtype)
    if new_type == delegate_spec.dtype:
      return delegate_spec
    else:
      return delegate_spec.replace(dtype=new_type)

  def observation_callable(
      self,
      physics: mjcf.Physics,
      random_state: Optional[np.random.RandomState] = None
  ) -> Callable[[], np.ndarray]:
    delegate_callable = self._delegate.observation_callable(
        physics, random_state)

    def cast():
      source = delegate_callable()
      new_type = self._cast_type(source.dtype)
      if new_type == source.dtype:
        return source
      else:
        return source.astype(new_type)
    return cast

  def _callable(self, physics: mjcf.Physics):
    # Overridden, but not used.
    del physics
    raise AssertionError('Should not be called')

  def _cast_type(self, dtype):
    """Get the type that the given dtype should be cast to."""
    if dtype in (np.uint8, np.float32, np.int64):
      return dtype
    if np.issubdtype(dtype, np.floating):
      return np.float32
    elif np.issubdtype(dtype, np.integer):
      return np.int64
    else:
      return dtype
