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
"""Core AgentFlow types."""

import abc
import enum
import functools
from typing import Any, Generic, Iterable, Optional, Text, TypeVar

import dm_env
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.decorators import register_class_properties
from dm_robotics.agentflow.decorators import register_property
import numpy as np


Arg = np.ndarray  # pylint: disable=invalid-name
ArgSpec = specs.Array


class TerminationType(enum.IntEnum):
  # The episode ended in an ordinary (internal) terminal state.
  SUCCESS = 0

  # The episode ended in failure.
  FAILURE = 1

  # The episode was preempted by an upstream signal.
  PREEMPTED = 2


class UidGenerator(object):
  """Generates hashable unique identifiers for options."""

  def __init__(self):
    self._next_id = 1

  def __call__(self):
    return_value = self._next_id
    self._next_id += 1
    return return_value


# A shared module-level UidGenerator. We don't enforce a strict singleton
# pattern to allow UidGenerator to be used in local contexts as well.
uid_generator = UidGenerator()


class OptionResult(object):
  """The result of an Option, encapsulating the termination reason and data."""

  def __init__(self,
               termination_reason: TerminationType,
               data: Optional[Any] = None,
               termination_text: Optional[Text] = None,
               termination_color: Optional[Text] = None):
    assert termination_reason is not None
    self._termination_reason = termination_reason
    self._data = data
    self.termination_text = termination_text or ''
    self.termination_color = termination_color or ''

  @property
  def termination_reason(self) -> TerminationType:
    return self._termination_reason

  @termination_reason.setter
  def termination_reason(self, reason: TerminationType) -> None:
    self._termination_reason = reason

  @property
  def data(self):
    return self._data

  @classmethod
  def success_result(cls):
    return cls(termination_reason=TerminationType.SUCCESS)

  @classmethod
  def failure_result(cls):
    return cls(termination_reason=TerminationType.FAILURE)

  def __str__(self):
    return 'OptionResult({}, {}, {})'.format(self.termination_reason,
                                             self.termination_text, self.data)

  def __eq__(self, other: 'OptionResult'):
    return ((self._termination_reason == other.termination_reason) and
            (self._data == other.data))

  def __hash__(self):
    return hash((self._termination_reason, self._data))


Spec = TypeVar('Spec', bound=specs.Array)


class ActionSpace(Generic[Spec], abc.ABC):
  """A mapping between actions; for example from cartesian to joint space.

  An action space defines a `spec` which actions 'in' the space must adhere to,
  and a `project` method that converts actions from that to another space.
  """

  @property
  @abc.abstractmethod
  def name(self) -> Text:
    """Returns the name of this action space."""
    raise NotImplementedError()

  @abc.abstractmethod
  def spec(self) -> Spec:
    """Spec of values that can be passed to `project`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def project(self, action: np.ndarray) -> np.ndarray:
    """Project input action (which adheres to `spec()`) to another action."""
    raise NotImplementedError()


class IdentityActionSpace(ActionSpace[Spec]):
  """Identity action space."""

  def __init__(self, spec: Spec, name: Text = 'identity'):
    self._spec = spec
    self._name = name

  @property
  def name(self) -> Text:
    return self._name

  def spec(self) -> Spec:
    return self._spec

  def project(self, action: np.ndarray) -> np.ndarray:
    spec_utils.validate(self._spec, action, ignore_nan=True)
    return action


@register_class_properties
class Policy(abc.ABC):
  """Base class for agents."""

  def __init__(self, name: Optional[Text] = None) -> None:
    """Initialize Policy.

    Args:
      name: (optional) A name for the policy.
    """

    self._name = name or self.__class__.__name__

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Determine what action to send to the environment.

    Args:
      timestep: A timestep containing observations, reward etc.

    Returns:
      A action that the environment (or SubTask) understands.
    """
    raise NotImplementedError

  def render_frame(self, canvas) -> None:
    """Optional method for rendering to a "canvas".

    Args:
      canvas: An object that instances can draw on. AgentFlow does not
        assume any particular interface for the Canvas, but it does forward
        calls down the graph from the top-level `Option` in order to allow
        users to implement arbitrary drawing logic.

        I.e.:
        canvas = MyCanvas()
        agent = BigAgentFlowGraph()
        agent.render_frame(canvas)  # All nodes should see `canvas`.
    """
    pass

  @property
  @register_property
  def name(self) -> Text:
    return self._name if hasattr(self, '_name') else self.__class__.__name__

  def child_policies(self) -> Iterable['Policy']:
    return []

  def setup(self) -> None:
    """Called once, before the run loop starts."""
    for child in self.child_policies():
      child.setup()

  def tear_down(self) -> None:
    """Called once, after the run loop ends."""
    for child in self.child_policies():
      child.tear_down()


@functools.total_ordering  # for __gt__, __le__, __ge__
@register_class_properties
class Option(Policy):  # pytype: disable=ignored-metaclass
  """Abstract option class.

  Option lifecycle:
    If the framework decides to select an option, `on_selected` is called.
    Next, the option enters the standard agent lifecycle methods:
      `step` is called repeatedly, the step_type of the timesteps that are
      passed must follow these rules:
      *  The first timestep must have step_type of `FIRST`
      *  The last timestep (before another FIRST) must have step_type of `LAST`
      *  All other timesteps must have a step type of `MID`.
    while an option is being `step`ped, AgentFlow will call `pterm` (which
    returns a float [0,1]).  This is an advisory signal - returning a pterm of
    1.0 does not guarantee that the next step will be a `LAST` step.
    When an option is terminated (a `LAST` step is given), an `OptionResult`
    is obtained from `result`. AgentFlow will use that result in subsequent
    calls to `on_selected`.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, name: Optional[Text] = None) -> None:
    """Initialize Option.

    Args:
      name: (optional) A name for the option.
    """
    super(Option, self).__init__(name)
    self._uid = uid_generator()
    self._key_prefix = '{}_{}'.format(self._name, self._uid)
    self._arg_key = '{}_arg'.format(self._key_prefix)

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    raise NotImplementedError

  def arg_spec(self) -> Optional[ArgSpec]:
    """Returns an arg_spec for the option.

    A arg_spec should be an `Array` (following v2 convention), allowing a
    parent `MetaOption` to communicate runtime-arguments to the option.
    """
    return

  @property
  @register_property
  def name(self) -> Text:
    return self._name

  @name.setter
  @register_property
  def name(self, value: Text):
    self._name = value

  @property
  @register_property
  def key_prefix(self) -> Text:
    """Auto-assigned prefix to ensure no collisions on arg_keys."""
    return self._key_prefix

  @property
  @register_property
  def arg_key(self) -> Text:
    return self._arg_key

  @property
  @register_property
  def uid(self) -> int:
    """Returns the auto-generated UID.

    It's not expected that this property is overridden.
    """
    return self._uid

  def on_selected(self,
                  timestep: dm_env.TimeStep,
                  prev_option_result: Optional[OptionResult] = None) -> None:
    pass

  def pterm(self, timestep: dm_env.TimeStep) -> float:
    """Returns the termination probability for the current state.

    Args:
      timestep: an AgentTimeStep namedtuple.
    """
    del timestep
    return 0.0

  def result(self, unused_timestep: dm_env.TimeStep) -> OptionResult:
    return OptionResult(termination_reason=TerminationType.SUCCESS)

  def __eq__(self, other):
    return isinstance(other, Option) and self.uid == other.uid

  def __ne__(self, other):
    return not self.__eq__(other)

  def __lt__(self, other):
    return self.name < other.name

  def __hash__(self):
    return self.uid

  def __str__(self):
    return 'uid={}, name=\"{}\"'.format(self.uid, self.name)

  def __repr__(self):
    # not technically a repr but useful for shell debugging
    return '{}({})'.format(self.__class__.__name__, str(self))


class MetaOption(Option):
  """Base class for Options that can use other Options.

  This class exists only to define a base interface for meta-Options, such that
  users can mix-and-match different mechanisms to drive options in a coherent
  pytype-checkable way.
  """

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    raise NotImplementedError
