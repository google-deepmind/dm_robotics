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

"""Sample task runner that creates a task environments and runs it."""

from absl import app
from dm_env import specs
from dm_robotics.moma.tasks import run_loop
from dm_robotics.moma.tasks.example_task import example_task
from dm_robotics.moma.utils import mujoco_rendering
import numpy as np


def main(argv):
  del argv
  env = example_task.build_task_environment()
  agent = RandomAgent(env.action_spec())

  rendering_obs = mujoco_rendering.Observer.build(env)
  rendering_obs.camera_config = {
      'distance': 2.5,
      'lookat': [0., 0., 0.],
      'elevation': -45.0,
      'azimuth': 90.0,
  }
  run_loop.run(env, agent, observers=[rendering_obs], max_steps=100)
  # We need to ensure that we close the environment


class RandomAgent:
  """An agent that emits uniform random actions."""

  def __init__(self, action_spec: specs.BoundedArray):
    self._spec = action_spec
    self._shape = action_spec.shape
    self._range = action_spec.maximum - action_spec.minimum

  def step(self, unused_timestep):
    action = (np.random.rand(*self._shape) - 0.5) * self._range
    np.clip(action, self._spec.minimum, self._spec.maximum, action)
    return action.astype(self._spec.dtype)


if __name__ == '__main__':
  app.run(main)
