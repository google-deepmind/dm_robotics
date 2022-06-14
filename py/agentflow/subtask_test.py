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
"""Tests for subtask."""

from typing import List, NamedTuple, Optional, Text
from typing import Union
from unittest import mock

from absl.testing import absltest
import dm_env
from dm_env import specs
import dm_robotics.agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import testing_functions
import numpy as np

valid_value = testing_functions.valid_value
random_timestep = testing_functions.random_timestep


def random_step_type():
  return np.random.choice(list(dm_env.StepType))


def _random_timestep(obs_spec: Union[None, specs.Array,
                                     spec_utils.ObservationSpec] = None):
  if obs_spec is None:
    key = testing_functions.random_string(3)
    spec = testing_functions.random_array_spec()
    obs_val = valid_value(spec)
    observation = {key: obs_val}
  else:
    observation = valid_value(obs_spec)
  return dm_env.TimeStep(
      step_type=random_step_type(),
      reward=np.float32(np.random.random()),
      discount=np.float32(np.random.random()),
      observation=observation)


def _random_result():
  termination_reason = np.random.choice(
      [af.TerminationType.SUCCESS, af.TerminationType.FAILURE])
  data = np.random.random(size=(5,)).astype(np.float32)
  return af.OptionResult(termination_reason, data)


# A custom comparison function because nan != nan and our timesteps contain nan
# when there is no reward.
def _timestep_equals(lhs, rhs):
  # If not iterable do normal comparison.
  try:
    iter(lhs)
  except TypeError:
    return lhs == rhs

  for field_a, field_b in zip(lhs, rhs):
    if field_a == field_b:
      return True
    if np.isnan(field_a) and np.isnan(field_b):
      return True
    return False


class StubSubTask(af.SubTask):

  def __init__(self, observation_spec: spec_utils.ObservationSpec,
               action_spec: specs.Array):
    self._observation_spec = observation_spec
    self._action_spec = action_spec

    # A list of the agent actions passed to agent_to_parent_action.
    self.actual_agent_actions = []  # type: List[np.ndarray]

    # A list of parent actions to return from agent_to_parent_action.
    # This list is popped from as agent_to_parent_action is called.
    self.parent_actions = []  # type: List[np.ndarray]

    # Timesteps received by parent_to_agent_timestep
    self.actual_parent_timesteps = []  # type: List[dm_env.TimeStep]

    # Timesteps to return from parent_to_agent_timestep.
    # This list is popped from as parent_to_agent_timestep is called.
    self.agent_timesteps = []  # type: List[dm_env.TimeStep]

  def observation_spec(self) -> spec_utils.ObservationSpec:
    return self._observation_spec

  def arg_spec(self) -> Optional[specs.Array]:
    return None

  def action_spec(self) -> specs.BoundedArray:
    return self._action_spec

  def agent_to_parent_action(self, agent_action: np.ndarray) -> np.ndarray:
    self.actual_agent_actions.append(np.copy(agent_action))
    if not self.parent_actions:
      raise ValueError("No more actions to return.")
    return self.parent_actions.pop(0)

  def parent_to_agent_timestep(self, parent_timestep: dm_env.TimeStep,
                               arg_key: Text) -> dm_env.TimeStep:
    self.actual_parent_timesteps.append(parent_timestep)
    if not self.agent_timesteps:
      raise ValueError("no more agent timesteps")
    return self.agent_timesteps.pop(0)

  def pterm(self, parent_timestep: dm_env.TimeStep,
            own_arg_key: Text) -> float:
    return 0.


ObserverStep = NamedTuple("ObserverStep", [("parent_timestep", dm_env.TimeStep),
                                           ("parent_action", np.ndarray),
                                           ("agent_timestep", dm_env.TimeStep),
                                           ("agent_action", np.ndarray)])


class SpySubTaskObserver(af.SubTaskObserver):

  def __init__(self):
    self.steps = []  # type: List[ObserverStep]

  def step(self, parent_timestep: dm_env.TimeStep, parent_action: np.ndarray,
           agent_timestep: dm_env.TimeStep, agent_action: np.ndarray) -> None:

    self.steps.append(
        ObserverStep(parent_timestep, parent_action, agent_timestep,
                     agent_action))


class SubTaskOptionTest(absltest.TestCase):

  def testTaskDefinesOptionArgSpec(self):
    agent = mock.MagicMock(spec=af.Policy)
    task = mock.MagicMock(spec=af.SubTask)

    spec = testing_functions.random_array_spec()
    task.arg_spec.return_value = spec

    option = af.SubTaskOption(task, agent)
    actual_arg_spec = option.arg_spec()

    self.assertEqual(actual_arg_spec, spec)

  def testTaskDelegatesArgKeyToOptionIfPossible(self):
    policy = mock.MagicMock(spec=af.Policy)
    option = mock.MagicMock(spec=af.Option)

    random_action_spec = testing_functions.random_array_spec(shape=(5,))
    random_observation_spec = testing_functions.random_array_spec(shape=(10,))
    task = testing_functions.IdentitySubtask(
        observation_spec=random_observation_spec,
        action_spec=random_action_spec,
        steps=100)

    task_arg_key = testing_functions.random_string()
    option_arg_key = testing_functions.random_string()
    type(option).arg_key = mock.PropertyMock(return_value=option_arg_key)
    task._default_arg_key = task_arg_key

    sto_wrapping_policy = af.SubTaskOption(task, policy)
    sto_wrapping_option = af.SubTaskOption(task, option)

    self.assertEqual(sto_wrapping_option.arg_key, option_arg_key)
    self.assertEqual(sto_wrapping_policy.arg_key, task_arg_key)

  def testPtermTakenFromAgentTimestep(self):
    # pterm of the SubTaskOption should delegate to the SubTask.

    # 1. Arrange:
    task_action_spec = testing_functions.random_array_spec(shape=(5,))
    task_obs_spec = testing_functions.random_observation_spec()
    agent_action = valid_value(task_action_spec)
    parent_action = np.random.random(size=(5,)).astype(np.float32)
    subtask_timestep = _random_timestep(task_obs_spec)

    task = mock.MagicMock(spec=af.SubTask)
    task.parent_to_agent_timestep.return_value = subtask_timestep
    task.pterm.return_value = 0.2
    task.action_spec.return_value = task_action_spec
    task.agent_to_parent_action.return_value = parent_action

    timestep = random_timestep(observation={})
    task.observation_spec.return_value = task_obs_spec
    task.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward")
    task.discount_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="discount")

    agent = mock.MagicMock(spec=af.Policy)
    agent.step.return_value = agent_action

    option = af.SubTaskOption(task, agent)

    # 2. Act:
    option.step(timestep)

    # 3. Assert:
    self.assertEqual(option.pterm(timestep), 0.2)

  def testStepTimestepFromSubtask(self):
    # The timestep the agent sees in begin_episode should come from the task.

    # 1. Arrange:
    task_action_spec = testing_functions.random_array_spec(shape=(5,))
    task_obs_spec = testing_functions.random_observation_spec(shape=(4,))
    agent_action = valid_value(task_action_spec)
    parent_action = np.random.random(size=(5,)).astype(np.float32)
    parent_timestep = _random_timestep()
    parent_timestep_without_reward = parent_timestep._replace(
        reward=np.float("nan"))
    subtask_timestep = _random_timestep(task_obs_spec)
    pterm = 0.7

    task = mock.MagicMock(spec=af.SubTask)
    task.parent_to_agent_timestep.return_value = subtask_timestep
    task.pterm.return_value = pterm
    task.action_spec.return_value = task_action_spec
    task.agent_to_parent_action.return_value = parent_action

    task.observation_spec.return_value = task_obs_spec
    task.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward")
    task.discount_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="discount")

    agent = mock.MagicMock(spec=af.Policy)
    agent.step.return_value = agent_action

    option = af.SubTaskOption(task, agent)

    # 2. Act:
    actual_option_action = option.step(parent_timestep)

    # 3. Assert:
    # Check that the task was given the correct timestep to pack.
    testing_functions.assert_calls(
        task.parent_to_agent_timestep,
        [(parent_timestep_without_reward, option.arg_key)],
        equals_fn=_timestep_equals)

    # Check that the agent was given the timestep from the task.
    testing_functions.assert_calls(agent.step, [(subtask_timestep,)])

    # Check that the task was given the agent aciton to convert to an action
    # for the parent environment.
    testing_functions.assert_calls(task.agent_to_parent_action,
                                   [(agent_action,)])

    # Check that this parent environment action is the one that's returned.
    np.testing.assert_equal(actual_option_action, parent_action)

  def testObservable(self):
    # Arrange:
    env_def = testing_functions.EnvironmentSpec.random()

    subtask = StubSubTask(
        observation_spec=testing_functions.random_observation_spec(),
        action_spec=testing_functions.random_array_spec())
    subtask_def = testing_functions.EnvironmentSpec.for_subtask(subtask)

    # Agent definition (agent operates 'in' the SubTask):
    agent_action1 = subtask_def.create_action()
    agent_action2 = subtask_def.create_action()
    agent = af.Sequence([af.FixedOp(agent_action1), af.FixedOp(agent_action2)])

    # Observer - this is the class under test (CUT / SUT).
    observer = SpySubTaskObserver()
    # This is the option that the observer is observing.
    subtask_option = af.SubTaskOption(subtask, agent, [observer])

    # Define how the subtask will behave (two parts):
    # Part 1 - The timesteps it will pass to the agent
    agent_timestep1 = subtask_def.create_timestep(
        step_type=dm_env.StepType.FIRST)
    agent_timestep2 = subtask_def.create_timestep(step_type=dm_env.StepType.MID)
    subtask.agent_timesteps.append(agent_timestep1)
    subtask.agent_timesteps.append(agent_timestep2)

    # Part 2 - The actions it will return to the parent.
    env_action1 = env_def.create_action()
    env_action2 = env_def.create_action()
    subtask.parent_actions.append(env_action1)
    subtask.parent_actions.append(env_action2)

    # Act:
    # Drive the subtask_option.  This should result in our listener being
    # invoked twice (once per step).  Each invocation should contain the
    # env-side timestep and action and the subtask-side timestep and action.
    env_timestep1 = env_def.create_timestep(step_type=dm_env.StepType.FIRST)
    env_timestep2 = env_def.create_timestep(step_type=dm_env.StepType.MID)

    actual_parent_action1 = subtask_option.step(env_timestep1)
    actual_parent_action2 = subtask_option.step(env_timestep2)

    # Assert:
    # Check that the observer was passed the expected values.
    np.testing.assert_almost_equal(env_action1, actual_parent_action1)
    np.testing.assert_almost_equal(env_action2, actual_parent_action2)

    # Check the timesteps and actions that were given to the listener.
    self.assertLen(observer.steps, 2)

    step = observer.steps[0]
    testing_functions.assert_timestep(env_timestep1, step.parent_timestep)
    testing_functions.assert_value(env_action1, step.parent_action)
    testing_functions.assert_timestep(agent_timestep1, step.agent_timestep)
    testing_functions.assert_value(agent_action1, step.agent_action)

    step = observer.steps[1]
    testing_functions.assert_timestep(env_timestep2, step.parent_timestep)
    testing_functions.assert_value(env_action2, step.parent_action)
    testing_functions.assert_timestep(agent_timestep2, step.agent_timestep)
    testing_functions.assert_value(agent_action2, step.agent_action)


if __name__ == "__main__":
  absltest.main()
