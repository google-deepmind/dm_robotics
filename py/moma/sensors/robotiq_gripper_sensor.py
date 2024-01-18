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

"""Sensor for a sim Robotiq gripper."""

from typing import Dict, Optional

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.moma.sensors import robotiq_gripper_observations
import numpy as np
import typing_extensions


@typing_extensions.runtime
class Gripper(typing_extensions.Protocol):

  def convert_position(self, position, **unused_kwargs):
    pass

  def grasp_sensor_callable(self, physics) -> int:
    pass


class RobotiqGripperSensor(robotiq_gripper_observations.RobotiqGripperSensor):
  """Robotiq gripper sensor for pos, vel, and grasp related observations."""

  def __init__(self, gripper: Gripper, name: str):
    self._gripper = gripper
    self._name = name
    self._observables = None

  def initialize_for_task(
      self,
      control_timestep_seconds: float,
      physics_timestep_seconds: float,
      physics_steps_per_control_step: int,
  ):
    def velocity_from_positions(cur_prev_positions):
      return cur_prev_positions[1] - cur_prev_positions[0]

    observations = robotiq_gripper_observations.Observations
    self._observables = {
        self.get_obs_key(observations.POS): observable.Generic(
            self._pos,
            # Convert the raw joint pos to a sensor output.
            corruptor=self._gripper.convert_position,
        ),
        self.get_obs_key(observations.VEL): observable.Generic(
            self._pos,
            buffer_size=2,
            update_interval=physics_steps_per_control_step,
            corruptor=self._gripper.convert_position,
            aggregator=velocity_from_positions,
        ),
        self.get_obs_key(observations.GRASP): observable.Generic(self._grasp),
        self.get_obs_key(observations.HEALTH_STATUS): observable.Generic(
            self.health_status
        ),
    }

    for obs in self._observables.values():
      obs.enabled = True

  def initialize_episode(
      self, physics: mjcf.Physics, random_state: np.random.RandomState
  ) -> None:
    pass

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    if self._observables is None:
      raise ValueError(
          'Observables are not initialized. Call initialize_for_task first.'
      )
    return self._observables

  @property
  def name(self) -> str:
    return self._name

  def get_obs_key(self, obs: robotiq_gripper_observations.Observations) -> str:
    return obs.get_obs_key(self._name)

  def _pos(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self._gripper.joint_sensor).sensordata  # pytype: disable=attribute-error

  def _grasp(self, physics: mjcf.Physics) -> np.ndarray:
    return np.array([self._gripper.grasp_sensor_callable(physics)],
                    dtype=np.uint8)

  def health_status(self, physics: Optional[mjcf.Physics] = None) -> np.ndarray:
    del physics
    # Always report a "ready" status for the sim grippers.
    return np.array([robotiq_gripper_observations.HealthStatus.READY.value],
                    dtype=np.uint8)
