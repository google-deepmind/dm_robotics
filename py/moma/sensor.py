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
"""Abstract sensor interface definition ."""

import abc
import enum
from typing import Dict

from dm_control import mjcf
from dm_control.composer.observation import observable
import numpy as np


class Sensor(abc.ABC):
  """Abstract sensor interface, sensors generate observations.

  `Sensor`s. have `observables`, it is these objects that supply sensed values.
  At its simplest, an Observable is a callable that takes a physics and returns
  a sensed value (which could be a `np.ndarray`).

  The instances returned by observables are stored and used by the composer
  environment to create the environment's observations (state).
  """

  def initialize_for_task(self, control_timestep_seconds: float,
                          physics_timestep_seconds: float,
                          physics_steps_per_control_step: int):
    """Setup the sensor for this task.

    Called before the base environment is setup.
    A sensor may need to know the control or physics frequencies to function
    correctly.  This method is a place to determine these things.

    Args:
      control_timestep_seconds: How many seconds there are between control
        timesteps, where the actuators can change control signals.
      physics_timestep_seconds: How many seconds there are between simulation
        step.
      physics_steps_per_control_step: Number of physics steps for every control
        timestep.  (`control_timestep_seconds` / `physics_timestep_seconds`)
    """

  def close(self):
    """Clean up after we are done using the sensor.

    Called to clean up when we are done using the sensor. This is
    mainly used for real sensors that might need to close all the connections
    to the robot.
    """

  def after_compile(self, mjcf_model: mjcf.RootElement,
                    physics: mjcf.Physics) -> None:
    """Method called after the MJCF model has been compiled and finalized.

    Args:
      mjcf_model: The root element of the scene MJCF model.
      physics: Compiled physics.
    """

  @abc.abstractmethod
  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    """Called on a new episode, after the environment has been reset.

    This is called before the agent has got a timestep in the episode that
    is about to start.  Sensors can reset any state they may have.

    Args:
      physics: The MuJoCo physics the environment uses.
      random_state: A PRNG seed.
    """

  @property
  @abc.abstractmethod
  def observables(self) -> Dict[str, observable.Observable]:
    """Get the observables for this Sensor.

    This will be called after `initialize_for_task`.

    It's expected that the keys in this dict are values from
    `self.get_obs_key(SOME_ENUM_VALUE)`, the values are Observables.
    See the class docstring for more information about Observable.

    subclassing `dm_control.composer.observation.observable.Generic` is a simple
    way to create an `Observable`.  Observables have many properties that are
    used by composer to alter how the values it produces are processed.

    See the code `dm_control.composer.observation` for more information.
    """

  @property
  @abc.abstractmethod
  def name(self) -> str:
    pass

  @abc.abstractmethod
  def get_obs_key(self, obs: enum.Enum) -> str:
    """Returns the key to an observable provided by this sensor."""
