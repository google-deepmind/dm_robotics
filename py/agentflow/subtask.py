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
"""SubTask and its observer, see class doc-strings."""

import abc
from typing import Any, Mapping, Optional, Sequence, Text, Tuple

import dm_env
from dm_env import specs
from dm_robotics.agentflow import core
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import util  # pylint: disable=unused-import
from dm_robotics.agentflow.decorators import overrides
import numpy as np

# Internal profiling


class SubTask(abc.ABC):
  """A SubTask defines a task for subtask policies, given a parent task.

  SubTask is used to define learning problems for agents used inside a larger
  policy.  SubTask defines everything required by `Option` except for the
  policy itself, and is combined with a subtask policy in `SubTaskOption`.

  Its primary use-case is to build Options which can internally train an RL
  agent. From the agent's perspective it lives in a regular RL environment, but
  from AgentFlow's perspective it's just another option node that can be plugged
  into a graph.

  In addition to training internally, a SubTaskOption can provide an `arg_spec`
  to define an abstract action-space to a parent `MetaOption` (any `Option` can
  do this).  If a SubTask does this, it should expect an observation with the
  key and spec, and do something appropriate (presumably condition the policy).

  In summary, SubTask can be used in two modes:
  1) As a stand-alone task-definition which trains up a sub-policy to be used
     by the parent as a black-box option.  E.g. an insertion task, which will
     be invoked by the parent at the appropriate time.
  2) As an interface between a parent and child policy for learning a
     parameterized policy, e.g. a reaching task parameterized by some
     representation of goal-pose.

  In either case, SubTask's job is to pack an observation during each timestep
  that contains any relevant features or sensor information, as with any task.
  It must also implement a reward function, and the standard life-cycle methods
  for an `Option` controlling initiation and termination.

  For case (2), `SubTask` must also define `arg_spec`, which is passed to both
  the parent agent for initializing its arg-generating module, and to
  the child agent for initializing its arg-consuming module.  The parent arg
  is then parsed out of the observation at each step, and passed along to the
  child.

  All methods on SubTask that take a timestep, expect that timestep to be
  from the environment aka 'parent'.
  All Option-related methods also take `arg_key`, selected by SubTaskOption, in
  order to allow a given SubTask to train multiple Policies.
  """

  def __init__(self, name: Optional[Text] = None) -> None:
    """Initialize SubTask.

    Args:
      name: (optional) A name for the subtask.
    """
    self._name = name or self.__class__.__name__
    self._uid = core.uid_generator()
    self._key_prefix = "{}_{}".format(self._name, self._uid)
    self._default_arg_key = "{}_arg".format(self._key_prefix)

  @property
  def name(self) -> Text:
    return self._name

  def get_arg_key(self, policy: Optional[core.Policy]) -> Text:
    """The key under which the SubTask can find args from the parent.

    Note: if controlling an `Option` this key should match `Option.arg_key`.

    Args:
      policy: The policy that this subtask is wrapping.  By default SubTask uses
        the arg_key of this policy, if available.

    Returns:
      A string key into timestep.observation under which args can be found.
    """
    if hasattr(policy, "arg_key"):
      return policy.arg_key
    elif not hasattr(self, "_default_arg_key"):
      raise AttributeError("arg_key not defined.  Did you forget to call "
                           "`super` on your SubTask's __init__()?")
    return self._default_arg_key

  @abc.abstractmethod
  def observation_spec(self) -> Mapping[Text, specs.Array]:
    """Defines the observation seen by agents for trained on this subtask."""
    pass

  @abc.abstractmethod
  def arg_spec(self) -> Optional[specs.Array]:
    """Defines the arg to be passed by the parent task during each step."""
    pass

  @abc.abstractmethod
  def action_spec(self) -> specs.BoundedArray:
    """Defines the action spec seen by agents that run on this subtask."""
    pass

  def reward_spec(self) -> specs.Array:
    """Describes the reward returned by the environment.

    By default this is assumed to be a single float.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return specs.Array(shape=(), dtype=np.float32, name="reward")

  def discount_spec(self) -> specs.Array:
    """Describes the discount returned by the environment.

    By default this is assumed to be a single float between 0 and 1.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0., maximum=1., name="discount")

  @abc.abstractmethod
  def agent_to_parent_action(self, agent_action: np.ndarray) -> np.ndarray:
    """Convert an action from the agent to the parent task."""
    pass

  def reset(self, parent_timestep: dm_env.TimeStep):
    """Called when a new episode is begun."""
    pass

  @abc.abstractmethod
  def parent_to_agent_timestep(self, parent_timestep: dm_env.TimeStep,
                               own_arg_key: Text) -> dm_env.TimeStep:
    """Converts a timestep from the parent to one consumable by the agent.

    This method should not modify parent_timestep. This is the right place to
    compute pterm.

    Args:
      parent_timestep: A TimeStep from the parent task or environment.
      own_arg_key: A string key into parent_timestep.observation in which the
        arg for the current Option can be found.

    Returns:
      A timestep for the agent that this subtask encloses.
    """
    pass

  @abc.abstractmethod
  def pterm(self, parent_timestep: dm_env.TimeStep, own_arg_key: Text) -> float:
    """Returns the termination probability for the current state.

    Args:
      parent_timestep: A TimeStep from the parent task or environment.
      own_arg_key: A string key into parent_timestep.observation in which the
        arg for the current Option can be found.
    """
    pass

  def subtask_result(self, parent_timestep: dm_env.TimeStep,
                     own_arg_key: Text) -> core.OptionResult:
    """Return an OptionResult conditional on the timestep and arg."""
    del parent_timestep
    del own_arg_key
    return core.OptionResult(core.TerminationType.SUCCESS, data=None)

  def render_frame(self, canvas) -> None:
    pass


class SubTaskObserver(abc.ABC):
  """An observer that can be attached to SubTaskOption.

  The observer is called on every step with both the environment-side and
  agent-side timesteps and actions, this allows us to (for example) log the
  environment side observations and the agent-side actions.  This is useful to
  log the raw observations and the actions that the agent took (rather than the
  projection of that action to the environment's action space).
  """

  @abc.abstractmethod
  def step(self, parent_timestep: dm_env.TimeStep, parent_action: np.ndarray,
           agent_timestep: dm_env.TimeStep, agent_action: np.ndarray) -> None:
    pass


class SubTaskOption(core.Option):
  """An option composed of a `SubTask` and a `Policy`.

  The SubTask is responsible for:
    Defining option-specific lifecycle methods (pterm),
    Defining the environment the agent operates in.

  The Agent is responsible for:
    Defining the policy (through step)
    Defining agent lifecycle methods (begin_episode, step, end_episode,
      tear_down).
  """

  def __init__(self,
               sub_task: SubTask,
               agent: Any,
               observers: Optional[Sequence[SubTaskObserver]] = None,
               name: Text = "SubTaskOption"):
    """Builds a SubTaskOption.

    Args:
      sub_task: A SubTask object defining life-cycle and agent-interface.
      agent: Any object with a `step(environment_timestep)` method.
      observers: Observers to invoke from step after actions are determined.
      name: An arbitrary name for the resulting Option.
    """
    super().__init__(name=name)
    self._task = sub_task
    self._agent = agent
    self._observers = []
    if observers:
      self._observers.extend(observers)

    # Spec expected by agents running in the subtask.
    self._task_timestep_spec = spec_utils.TimeStepSpec(
        observation_spec=self._task.observation_spec(),
        reward_spec=self._task.reward_spec(),
        discount_spec=self._task.discount_spec())

  @property
  def subtask(self) -> SubTask:
    """Returns the underlying subtask for this SubTaskOption."""
    return self._task

  @property
  def agent(self) -> Any:  # Should we just assume these are always `Policy`?
    """Returns the underlying policy for this SubTaskOption."""
    return self._agent

  @overrides(core.Option)
  def arg_spec(self):
    return self._task.arg_spec()

  @property
  @overrides(core.Option)
  def arg_key(self) -> Text:
    """The key under which the SubTask can find args from the parent.

    Returns:
      A string key into timestep.observation under which args can be found.
    """
    return self._task.get_arg_key(self._agent)

  @overrides(core.Option)
  # Profiling for .wrap("SubTaskOption.on_selected")
  def on_selected(self,
                  parent_timestep: dm_env.TimeStep,
                  prev_option_result=None) -> None:
    """Process first timestep and delegate to agent."""
    if parent_timestep.first():
      self._task.reset(parent_timestep)
    if isinstance(self._agent, core.Option):
      agent_timestep = self._task.parent_to_agent_timestep(
          parent_timestep, self.arg_key)
      self._agent.on_selected(agent_timestep, prev_option_result)

  @overrides(core.Option)
  # Profiling for .wrap("SubTaskOption.step")
  def step(self, parent_timestep: dm_env.TimeStep) -> np.ndarray:
    """Delegate step to policy, and run subtask hooks."""
    if parent_timestep.first():
      self._task.reset(parent_timestep)

    # pyformat: disable
    # pylint: disable=line-too-long
    # pyformat: disable
    # Create timestep for agent, cache SubTask termination signal.
    with util.nullcontext():  # create agent timestep
      agent_timestep = self._create_agent_timestep(parent_timestep)

    # Get action from the agent, and validate it.
    with util.nullcontext():  # step agent
      agent_action = self._agent.step(agent_timestep)

    spec_utils.validate(self._task.action_spec(), agent_action, ignore_nan=True)

    with util.nullcontext():  # agent_to_parent_action
      # Subtask converts the agent action to an action for the environment.
      parent_action = self._task.agent_to_parent_action(agent_action)

    with util.nullcontext():  # run observers
      for obs in self._observers:
        obs.step(
            parent_timestep=parent_timestep,
            parent_action=parent_action,
            agent_timestep=agent_timestep,
            agent_action=agent_action)
    # pyformat: enable
    # pylint: enable=line-too-long
    # pyformat: enable

    return parent_action

  def _create_agent_timestep(
      self, parent_timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Generates a timestep for the agent."""
    agent_timestep = self._task.parent_to_agent_timestep(
        parent_timestep, self.arg_key)

    # Check that the timestep we pass the agent matches the task spec, which
    # tells the agent what to expect in the timestep.
    spec_utils.validate_timestep(self._task_timestep_spec, agent_timestep)
    return agent_timestep

  @overrides(core.Option)
  # Profiling for .wrap("SubTaskOption.pterm")
  def pterm(self, parent_timestep: dm_env.TimeStep) -> float:
    """Delegate pterm to subtask."""
    return self._task.pterm(parent_timestep, self.arg_key)

  @overrides(core.Option)
  # Profiling for .wrap("SubTaskOption.result")
  def result(self, parent_timestep: dm_env.TimeStep) -> core.OptionResult:
    return self._task.subtask_result(parent_timestep, self.arg_key)

  @overrides(core.Option)
  # Profiling for .wrap("SubTaskOption.render_frame")
  def render_frame(self, canvas) -> None:
    if hasattr(self._task, "render_frame"):
      self._task.render_frame(canvas)
    if hasattr(self._agent, "render_frame"):
      self._agent.render_frame(canvas)

  def child_policies(self) -> Tuple[core.Policy]:
    return (self._agent,)

  def add_observer(self, observer: SubTaskObserver) -> None:
    self._observers.append(observer)
