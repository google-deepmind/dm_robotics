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
"""A minimal workflow for combining a scripted and RL agent using AgentFlow.

This example demonstrates a way to build an AgentFlow experiment in which all
I/O is handled by the `Environment`, which is backed by arbitrary python
callables.

The primary advantage of this workflow is that all I/O with the robot is
centralized in the `ProxyEnvironment`.  This can simplify logging and debugging,
and also allow for system designs in which agents can operate on the output of
other agents.

For the alternative "direct-dispatch" model see `direct_dispatch_workflow.py`

High level steps:
1) Implement stubs for receiving state and sending actions.
2) Provide stubs to a proxy env that manages I/O.
3) Create an AgentFlow policy that generates valid actions, e.g. from an n.n.
4) Create a no-op subtask to hold the agent so we can attach a logging observer.
5) Create an logging observer and attach to agent.
6) Create an AgentFlow `Option` implementing the desired reset behaviour.
7) Create a run loop and go.

Notes:
  * This workflow is blocking iff the state or action stubs block.  For an RPC-
    style interface consider dm_env_rpc (useful if env and agent live in
    different processes) or a custom-RPC service.
  * The `ActionCallback` currently lives in the environment, but it could easily
    be moved closer to the agent, e.g. in the SubTask (as an ActionSpace) or the
    Policy itself.
"""

import abc
from typing import Mapping, Optional, Text

from absl import app
import dm_env
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import subtask
from dm_robotics.agentflow.loggers import print_logger
from dm_robotics.agentflow.loggers import subtask_logger
from dm_robotics.agentflow.meta_options import control_flow
import numpy as np


class ObservationCallback(abc.ABC):
  """Base class for state-callbacks."""

  @abc.abstractmethod
  def __call__(self) -> Mapping[Text, np.ndarray]:
    pass

  @abc.abstractmethod
  def observation_spec(self) -> Mapping[Text, specs.Array]:
    pass


class ActionCallback(abc.ABC):
  """Base class for action-callbacks."""

  @abc.abstractmethod
  def __call__(self, action: np.ndarray) -> None:
    pass

  @abc.abstractmethod
  def action_spec(self) -> specs.BoundedArray:
    pass


def observation_update_stub() -> np.ndarray:
  observation = np.random.rand(4)
  print(f"observation_update_stub called! Returning observation {observation}")
  return observation


def send_action_stub(action: np.ndarray) -> None:
  print(f"send_action_stub called with {action}!")


class ExampleObservationUpdater(ObservationCallback):
  """Example State-Update callback."""

  def __call__(self) -> Mapping[Text, np.ndarray]:
    return {"stub_observation": observation_update_stub()}

  def observation_spec(self) -> Mapping[Text, specs.Array]:
    return {
        "stub_observation":
            specs.Array((4,), dtype=np.float64, name="stub_observation")
    }


class ExampleActionSender(ActionCallback):
  """Example SendAction callback."""

  def __call__(self, action: np.ndarray) -> None:
    send_action_stub(action)

  def action_spec(self) -> specs.BoundedArray:
    return specs.BoundedArray((2,),
                              dtype=np.float64,
                              minimum=-np.ones(2),
                              maximum=np.ones(2),
                              name="stub action")


class ExampleSubTask(subtask.SubTask):
  """A No-op subtask that allows us to attach a logging observer."""

  def __init__(self, observation_spec: Mapping[Text, specs.Array],
               action_spec: specs.BoundedArray,
               max_steps: int):
    super().__init__()
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._max_steps = max_steps
    self._step_idx = 0.

  def observation_spec(self) -> Mapping[Text, specs.Array]:
    """Defines the observation seen by agents for trained on this subtask."""
    return self._observation_spec

  def reward_spec(self) -> specs.Array:
    return specs.Array(shape=(), dtype=np.float64, name="reward")

  def discount_spec(self) -> specs.Array:
    return specs.BoundedArray(
        shape=(), dtype=np.float64, minimum=0., maximum=1., name="discount")

  def arg_spec(self) -> Optional[specs.Array]:
    """Defines the arg to be passed by the parent task during each step."""
    return  # This example doesn't use parameterized-options.

  def action_spec(self) -> specs.BoundedArray:
    """Defines the action spec seen by agents that run on this subtask."""
    return self._action_spec

  def agent_to_parent_action(self, agent_action: np.ndarray) -> np.ndarray:
    """Convert an action from the agent to the parent task."""
    return agent_action

  def parent_to_agent_timestep(self, parent_timestep: dm_env.TimeStep,
                               own_arg_key: Text) -> dm_env.TimeStep:
    if parent_timestep.first():
      self._step_idx = 0.
    self._step_idx += 1
    return parent_timestep

  def pterm(self, parent_timestep: dm_env.TimeStep,
            own_arg_key: Text) -> float:
    if self._step_idx >= self._max_steps:
      print(f"Terminating subtask because max_steps reached {self._step_idx}.")
      return 1.
    return 0.


class ExamplePolicy(core.Policy):
  """Stub policy for running learning machinery."""

  def __init__(self, action_spec: specs.BoundedArray, name: str):
    super().__init__(name)
    self._action_spec = action_spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    return np.random.rand(2).astype(self._action_spec.dtype)  # Bounds...


class ExampleScriptedOption(core.Option):
  """Stub option for running scripted controller."""

  def __init__(self, action_spec: specs.BoundedArray,
               name: str,
               max_steps: int):
    super().__init__(name)
    self._action_spec = action_spec
    self._max_steps = max_steps
    self._step_idx = 0

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    if timestep.first():
      self._step_idx = 0
    self._step_idx += 1
    return np.random.rand(2).astype(self._action_spec.dtype)  # Run controller

  def pterm(self, timestep: dm_env.TimeStep) -> float:
    del timestep
    if self._step_idx >= self._max_steps:
      print(f"Terminating option because max_steps reached {self._step_idx}.")
      return 1.
    return 0.

  def result(self, unused_timestep: dm_env.TimeStep) -> core.OptionResult:
    return core.OptionResult(termination_reason=core.TerminationType.SUCCESS)


class ProxyEnvironment(dm_env.Environment):
  """An environment that receives observation from a callback."""

  def __init__(self,
               observation_cb: ObservationCallback,
               action_cb: Optional[ActionCallback] = None):
    """Initializes ProxyEnvironment.

    Args:
      observation_cb: An object for retrieving observation from somewhere.
      action_cb: An object for dispatching agent actions. If `None` actions are
        discarded
    """
    self._observation_cb = observation_cb
    self._action_cb = action_cb

  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    return dm_env.restart(observation=self._observation_cb())

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    if self._action_cb:
      self._action_cb(action)
    return dm_env.transition(reward=0.,
                             discount=1.,
                             observation=self._observation_cb())

  def observation_spec(self) -> Mapping[Text, specs.Array]:
    """Returns the observation spec."""
    return self._observation_cb.observation_spec()

  def action_spec(self) -> specs.BoundedArray:
    """Returns the action spec."""
    if self._action_cb:
      return self._action_cb.action_spec()
    return specs.BoundedArray((),
                              minimum=[],
                              maximum=[],
                              dtype=np.float64,
                              name="dummy_action")


def main(_):
  # Stubs for pulling observation and sending action to some external system.
  observation_cb = ExampleObservationUpdater()
  action_cb = ExampleActionSender()

  # Create an environment that forwards the observation and action calls.
  env = ProxyEnvironment(observation_cb, action_cb)

  # Stub policy that runs the desired agent.
  policy = ExamplePolicy(action_cb.action_spec(), "agent")

  # Wrap policy into an agent that logs to the terminal.
  task = ExampleSubTask(env.observation_spec(), action_cb.action_spec(), 10)
  logger = print_logger.PrintLogger()
  aggregator = subtask_logger.EpisodeReturnAggregator()
  logging_observer = subtask_logger.SubTaskLogger(logger, aggregator)
  agent = subtask.SubTaskOption(task, policy, [logging_observer])

  reset_op = ExampleScriptedOption(action_cb.action_spec(), "reset", 3)
  main_loop = control_flow.Repeat(5, control_flow.Sequence([reset_op, agent]))

  # Run the episode.
  timestep = env.reset()
  while True:
    action = main_loop.step(timestep)
    timestep = env.step(action)

    # Terminate if the environment or main_loop requests it.
    if timestep.last() or (main_loop.pterm(timestep) > np.random.rand()):
      if not timestep.last():
        termination_timestep = timestep._replace(step_type=dm_env.StepType.LAST)
        main_loop.step(termination_timestep)
      break


if __name__ == "__main__":
  app.run(main)
