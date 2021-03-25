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

"""Simple run loop for an agent and environment."""

import numpy as np


def run(environment, agent, observers, max_steps):
  """Runs the agent 'in' the environment.

  The loop is:
  1. The `environment` is reset, producing a state.
  2. The `agent` is given that state and produces an action.
  3. That action is given to the environment.
  4. The `environment` produces a new state.
  5. GOTO 2

  At most `max_steps` are demanded from the agent.

  The `environment` cam produce three types of step:
  * FIRST: The first step in an episode.
    The next step will be MID or LAST.
  * MID: A step that is neither the first nor last.
  * LAST: The last step in this episode.
    The next step will be FIRST.

  Depending on the type of step emitted by the environment, the `observers`
  have different methods called:
  * FIRST: `observer.begin_episode(0)`
  * MID: `observer.step(0, env_timestep, agent_action)`
  * LAST: `observer.end_episode(0, 0, env_timestep)`

  The `agent_action` passed to `observer.step` is the action the agent emitted
  given `env_timestep`, at the time the observer is called, the action has not
  yet been given to the environment.

  When a LAST timestep is received, the agent is given that timestep, but the
  action it emits is discarded.

  Args:
    environment: The environment to run the agent "in".
    agent: The agent that produced actions.
    observers: A sequence of observers, see the docstring.
    max_steps: The maximum number of time to step the agent.
  """

  step_count = 0

  while step_count < max_steps:
    # Start a new episode:
    timestep = _start_new_episode(environment)
    _observe_begin(observers)

    # Step until the end of episode (or max_steps hit):
    while not timestep.last() and step_count < max_steps:
      # Get an action from the agent:
      action = agent.step(timestep)
      step_count += 1
      _ensure_no_nans(action)
      _observe_step(observers, timestep, action)

      # Get a new state (timestep) from the environment:
      timestep = environment.step(action)
      timestep = _fix_timestep(timestep, environment)

    # Tell the observers the episode has ended.
    if step_count < max_steps:
      agent.step(timestep)
      step_count += 1
      _observe_end(observers, timestep)


def _start_new_episode(env):
  timestep = _fix_timestep(env.reset(), env)
  if not timestep.first():
    raise ValueError('Expected first timestep, but received {}.'.format(
        str(timestep.step_type)))
  return timestep


def _fix_timestep(timestep, env):
  """Ensures the output timestep has a reward and discount."""
  if timestep.reward is None:
    reward_spec = env.reward_spec()
    if reward_spec.shape:
      reward = np.zeros(shape=reward_spec.shape, dtype=reward_spec.dtype)
    else:
      reward = reward_spec.dtype.type(0.0)
    timestep = timestep._replace(reward=reward)
  if timestep.discount is None:
    timestep = timestep._replace(discount=env.discount_spec().dtype.type(1.0))
  return timestep


def _observe_begin(observers):
  for obs in observers:
    obs.begin_episode(0)


def _observe_step(observers, timestep, action):
  for obs in observers:
    obs.step(0, timestep, action)


def _observe_end(observers, timestep):
  for obs in observers:
    obs.end_episode(0, 0, timestep)


def _ensure_no_nans(action):
  if any(np.isnan(action)):
    raise ValueError('NaN in agent actions')
