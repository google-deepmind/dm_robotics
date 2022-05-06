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
"""Tests subtask -> environment adaptor."""
import enum
import sys
from typing import Dict, List, Optional, Text, Tuple

from absl.testing import absltest
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
import dm_env
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow import subtask
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.decorators import overrides
from dm_robotics.agentflow.options import basic_options
from dm_robotics.moma import base_task
from dm_robotics.moma import effector
from dm_robotics.moma import entity_initializer
from dm_robotics.moma import moma_option
from dm_robotics.moma import scene_initializer
from dm_robotics.moma import sensor as moma_sensor
from dm_robotics.moma import subtask_env
from dm_robotics.moma import subtask_env_builder
from dm_robotics.moma.models.arenas import empty
import numpy as np


def action(spec, value):
  return np.full(shape=spec.shape, fill_value=value, dtype=spec.dtype)


def effectors_action_spec(effectors):
  a_specs = [a.action_spec(None) for a in effectors]
  return spec_utils.merge_specs(a_specs)


class FakeEffector(effector.Effector):

  def __init__(self, prefix: str, dof: int):
    self._prefix = prefix
    self._dof = dof

    self.received_commands = []

  def initialize_episode(self, physics, random_state):
    pass

  def action_spec(self, physics):
    actuator_names = [(self.prefix + str(i)) for i in range(self._dof)]

    return specs.BoundedArray(
        shape=(self._dof,),
        dtype=np.float32,
        minimum=[-100.0] * self._dof,
        maximum=[100.0] * self._dof,
        name='\t'.join(actuator_names))

  def set_control(self, physics, command):
    del physics
    self.received_commands.append(command)

  @property
  def prefix(self):
    return self._prefix


class FakeSubtask(subtask.SubTask):
  """Adds a 'fake' observation and an action, which is ignored."""

  def __init__(self,
               base_env: dm_env.Environment,
               effectors: List[FakeEffector],
               max_steps: int,
               name: Optional[Text] = None) -> None:
    super().__init__(name)
    self._steps_taken = 0
    self._max_steps = max_steps
    self._observation_spec = dict(base_env.observation_spec())
    self._observation_spec['fake'] = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=[0], maximum=[sys.maxsize])

    effectors_spec = effectors_action_spec(effectors)
    action_spec_shape = (effectors_spec.shape[0] + 1,)
    action_spec_minimum = list(effectors_spec.minimum) + [-1000]
    action_spec_maximum = list(effectors_spec.maximum) + [1000]

    self._action_spec = specs.BoundedArray(
        shape=action_spec_shape,
        dtype=effectors_spec.dtype,
        minimum=action_spec_minimum,
        maximum=action_spec_maximum)
    self._reward_spec = base_env.reward_spec()
    self._discount_spec = base_env.discount_spec()

  @overrides(subtask.SubTask)
  def observation_spec(self):
    return self._observation_spec

  @overrides(subtask.SubTask)
  def action_spec(self):
    return self._action_spec

  @overrides(subtask.SubTask)
  def reward_spec(self) -> specs.Array:
    return self._reward_spec

  @overrides(subtask.SubTask)
  def discount_spec(self) -> specs.Array:
    return self._discount_spec

  @overrides(subtask.SubTask)
  def arg_spec(self):
    return None

  @overrides(subtask.SubTask)
  def agent_to_parent_action(self, agent_action: np.ndarray) -> np.ndarray:
    # Throw away the last value in the array
    return agent_action[:len(agent_action) - 1]

  @overrides(subtask.SubTask)
  def reset(self, parent_timestep: dm_env.TimeStep):
    self._steps_taken = 0
    return parent_timestep

  @overrides(subtask.SubTask)
  def parent_to_agent_timestep(
      self,
      parent_timestep: dm_env.TimeStep,
      own_arg_key: Optional[Text] = None) -> Tuple[dm_env.TimeStep, float]:
    self._steps_taken += 1
    child_observation = dict(parent_timestep.observation)
    child_observation['fake'] = np.asarray([self._steps_taken],
                                           dtype=np.float32)
    timestep = parent_timestep._replace(observation=child_observation)
    return timestep

  def pterm(self, parent_timestep: dm_env.TimeStep, arg_key: Text) -> float:
    return 1.0 if self._steps_taken >= self._max_steps else 0.0

  def assert_timesteps_from_subtask(self, *timesteps: dm_env.TimeStep):
    for timestep in timesteps:
      if 'fake' not in timestep.observation:
        return False
    return True


class FakeSubTaskObserver(subtask.SubTaskObserver):

  def __init__(self):
    self.last_parent_timestep = None
    self.last_parent_action = None

    self.last_agent_timestep = None
    self.last_agent_action = None

  def step(self, parent_timestep: dm_env.TimeStep,
           parent_action: Optional[np.ndarray], agent_timestep: dm_env.TimeStep,
           agent_action: Optional[np.ndarray]) -> None:

    self.last_parent_timestep = parent_timestep
    self.last_parent_action = parent_action

    self.last_agent_timestep = agent_timestep
    self.last_agent_action = agent_action


# Test method:
# Create a SubTask and Reset option to drive the environment.
# Create an EnvironmentAdaptor from them.
# Ensure that both systems drive the underlying Environment the same way.
class SubTaskEnvironmentTest(absltest.TestCase):

  def testEnvironmentDriver(self):
    base_env = testing_functions.SpyEnvironment()
    effectors = [
        FakeEffector('fake_effector_1', 3),
        FakeEffector('faky_mcfakeface', 2)
    ]
    effectors_spec = effectors_action_spec(effectors)

    sub_task = FakeSubtask(base_env, effectors, max_steps=3, name='FakeSubTask')
    # Action that we send to the Environment-from-SubTask adaptor.
    agent_action = action(sub_task.action_spec(), 22)
    # Action that this adaptor should send to the base environment.
    base_agent_action = action(effectors_spec, 22)

    # Action that the adaptor sends to the base environment for reset().
    reset_action = action(effectors_spec, 11)

    reset = moma_option.MomaOption(
        physics_getter=lambda: base_env.physics,
        effectors=effectors,
        delegate=basic_options.FixedOp(reset_action, num_steps=2, name='Reset'))

    with subtask_env.SubTaskEnvironment(base_env, effectors, sub_task,
                                        reset) as env:
      timestep1 = env.reset()
      # Step the env 3 times (that's the limit of the subtask)
      # Check the corresponding actions sent to the base environment.
      timestep2 = env.step(agent_action)
      timestep3 = env.step(agent_action)
      timestep4 = env.step(agent_action)

      # Check the timesteps are as expected:
      self.assertEqual(timestep1.step_type, dm_env.StepType.FIRST)
      self.assertEqual(timestep2.step_type, dm_env.StepType.MID)
      self.assertEqual(timestep3.step_type, dm_env.StepType.MID)
      self.assertEqual(timestep4.step_type, dm_env.StepType.LAST)
      sub_task.assert_timesteps_from_subtask(timestep1, timestep2, timestep3,
                                             timestep4)

      # Check the actions that were received by the base environment
      # It should get 2 reset steps + 3 subtask steps.

      self.assertLen(effectors[0].received_commands, 5)
      self.assertLen(effectors[1].received_commands, 5)

      self._assert_actions(reset_action[:3],
                           effectors[0].received_commands[0:2])
      self._assert_actions(reset_action[3:],
                           effectors[1].received_commands[0:2])

      self._assert_actions(base_agent_action[:3],
                           effectors[0].received_commands[2:5])
      self._assert_actions(base_agent_action[3:],
                           effectors[1].received_commands[2:5])

      effectors[0].received_commands.clear()
      effectors[1].received_commands.clear()

      # Now, step the env once more.  This should provoke a reset.
      timestep5 = env.step(agent_action)
      self.assertEqual(timestep5.step_type, dm_env.StepType.FIRST)
      sub_task.assert_timesteps_from_subtask(timestep5)

      self.assertLen(effectors[0].received_commands, 2)
      self.assertLen(effectors[1].received_commands, 2)

      self._assert_actions(reset_action[:3],
                           effectors[0].received_commands[0:2])
      self._assert_actions(reset_action[3:],
                           effectors[1].received_commands[0:2])

      effectors[0].received_commands.clear()
      effectors[1].received_commands.clear()

      # Continuing to step the environment should single-step the base env.
      timestep6 = env.step(agent_action)
      timestep7 = env.step(agent_action)
      timestep8 = env.step(agent_action)

      self.assertEqual(timestep6.step_type, dm_env.StepType.MID)
      self.assertEqual(timestep7.step_type, dm_env.StepType.MID)
      self.assertEqual(timestep8.step_type, dm_env.StepType.LAST)

      sub_task.assert_timesteps_from_subtask(timestep6, timestep7, timestep8)

      self.assertLen(effectors[0].received_commands, 3)
      self.assertLen(effectors[1].received_commands, 3)

      self._assert_actions(base_agent_action[:3],
                           effectors[0].received_commands[0:3])
      self._assert_actions(base_agent_action[3:],
                           effectors[1].received_commands[0:3])

      effectors[0].received_commands.clear()
      effectors[1].received_commands.clear()

  def testAdaptorResetLifecycle(self):
    # When the environment adaptor uses the reset option, it should LAST step
    # the reset option - this is a test of that behaviour.
    base_env = testing_functions.SpyEnvironment()
    effectors = [
        FakeEffector('fake_effector_1', 3),
        FakeEffector('fake_effector_2', 2)
    ]
    effectors_spec = effectors_action_spec(effectors)

    sub_task = FakeSubtask(base_env, effectors, max_steps=1, name='FakeSubTask')

    # Action that we send to the Environment-from-SubTask adaptor.
    agent_action = action(sub_task.action_spec(), 22)
    # Action that the adaptor sends to the base environment for reset().
    reset_action = action(effectors_spec, 11)

    # Two steps of reset, FIRST, MID.
    # Thereafter, reset will return pterm() 1.0 and should receive another
    # step, with step_type LAST.
    reset = testing_functions.SpyOp(reset_action, num_steps=2, name='Reset')
    wrapped_reset = moma_option.MomaOption(
        physics_getter=lambda: base_env.physics,
        effectors=effectors,
        delegate=reset)
    with subtask_env.SubTaskEnvironment(base_env, effectors, sub_task,
                                        wrapped_reset) as env:
      env.reset()
      # Step the env once (that is the subtask limit)
      env.step(agent_action)
      # step again, provoking reset.
      env.step(agent_action)

      # We should have reset twice, therefore stepping the reset option:
      # Reset 1: FIRST, MID, LAST
      # Reset 2: FIRST, MID, LAST
      reset_step_timesteps = [ts.step for ts in reset.timesteps if ts.step]
      step_types = [ts.step_type for ts in reset_step_timesteps]
      self.assertEqual(step_types, [
          dm_env.StepType.FIRST, dm_env.StepType.MID, dm_env.StepType.LAST,
          dm_env.StepType.FIRST, dm_env.StepType.MID, dm_env.StepType.LAST
      ])

  def testNoneFirstRewardDiscount(self):
    # The dm_env interface specifies that the first timestep must have None
    # for the reward and discount. This test checks subtask env complies.
    base_env = testing_functions.SpyEnvironment()
    effectors = [
        FakeEffector('fake_effector_1', 3),
        FakeEffector('fake_effector_2', 2)
    ]
    effectors_spec = effectors_action_spec(effectors)

    sub_task = FakeSubtask(
        base_env, effectors, max_steps=10, name='FakeSubTask')

    # Action that we send to the Environment-from-SubTask adaptor.
    agent_action = action(sub_task.action_spec(), 22)
    # Action that the adaptor sends to the base environment for reset().
    reset_action = action(effectors_spec, 11)

    # Two steps of reset, FIRST, MID.
    # Thereafter, reset will return pterm() 1.0 and should receive another
    # step, with step_type LAST.
    reset = testing_functions.SpyOp(reset_action, num_steps=2, name='Reset')
    wrapped_reset = moma_option.MomaOption(
        physics_getter=lambda: base_env.physics,
        effectors=effectors,
        delegate=reset)
    with subtask_env.SubTaskEnvironment(base_env, effectors, sub_task,
                                        wrapped_reset) as env:
      ts = env.step(agent_action)

      self.assertIsNone(ts.reward)
      self.assertIsNone(ts.discount)

      reset_ts = env.reset()
      self.assertIsNone(reset_ts.reward)
      self.assertIsNone(reset_ts.discount)

  def testObserver(self):
    base_env = testing_functions.SpyEnvironment()
    effectors = [
        FakeEffector('fake_effector_1', 3),
        FakeEffector('faky_mcfakeface', 2)
    ]
    effectors_spec = effectors_action_spec(effectors)

    sub_task = FakeSubtask(base_env, effectors, max_steps=3, name='FakeSubTask')
    # Action that we send to the Environment-from-SubTask adaptor.
    agent_action = action(sub_task.action_spec(), 22)

    # Action that the adaptor sends to the base environment for reset().
    reset_action = action(effectors_spec, 11)

    reset = moma_option.MomaOption(
        physics_getter=lambda: base_env.physics,
        effectors=effectors,
        delegate=basic_options.FixedOp(reset_action, num_steps=2, name='Reset'))

    observer = FakeSubTaskObserver()

    with subtask_env.SubTaskEnvironment(base_env, effectors, sub_task,
                                        reset) as env:
      env.add_observer(observer)

      timestep1 = env.reset()
      self.assertIsNone(observer.last_parent_timestep)
      self.assertIsNone(observer.last_parent_action)
      self.assertIsNone(observer.last_agent_timestep)
      self.assertIsNone(observer.last_agent_action)

      # Step the env 3 times (that's the limit of the subtask)
      # Check the corresponding actions sent to the base environment.
      timestep2 = env.step(agent_action)
      self.assertEqual(observer.last_agent_timestep.step_type,
                       dm_env.StepType.FIRST)

      # The fake observation is added by the subtask, and should not be present
      # in the parent timestep.
      self.assertNotIn('fake', observer.last_parent_timestep.observation)
      self.assertIn('fake', observer.last_agent_timestep.observation)
      self.assertTrue(
          np.array_equal(observer.last_agent_action, [22, 22, 22, 22, 22, 22]))
      self.assertTrue(
          np.array_equal(observer.last_parent_action, [22, 22, 22, 22, 22]))

      timestep3 = env.step(agent_action)
      self.assertEqual(observer.last_parent_timestep.step_type,
                       dm_env.StepType.MID)
      self.assertEqual(observer.last_agent_timestep.step_type,
                       dm_env.StepType.MID)
      self.assertNotIn('fake', observer.last_parent_timestep.observation)
      self.assertIn('fake', observer.last_agent_timestep.observation)
      self.assertTrue(
          np.array_equal(observer.last_agent_action, [22, 22, 22, 22, 22, 22]))
      self.assertTrue(
          np.array_equal(observer.last_parent_action, [22, 22, 22, 22, 22]))

      timestep4 = env.step(agent_action)
      self.assertEqual(observer.last_agent_timestep.step_type,
                       dm_env.StepType.LAST)
      self.assertIsNone(observer.last_parent_action)
      self.assertIsNone(observer.last_agent_action)

      # Check the timesteps are as expected:
      self.assertEqual(timestep1.step_type, dm_env.StepType.FIRST)
      self.assertEqual(timestep2.step_type, dm_env.StepType.MID)
      self.assertEqual(timestep3.step_type, dm_env.StepType.MID)
      self.assertEqual(timestep4.step_type, dm_env.StepType.LAST)

      # Run a second episode.
      timestep1 = env.reset()

      # Observer should not have been stepped and wil still have none actions.
      self.assertIsNone(observer.last_parent_action)
      self.assertIsNone(observer.last_agent_action)

      # Step the env 3 times (that's the limit of the subtask)
      # Check the corresponding actions sent to the base environment.
      timestep2 = env.step(agent_action)
      self.assertEqual(observer.last_agent_timestep.step_type,
                       dm_env.StepType.FIRST)

      # The fake observation is added by the subtask, and should not be present
      # in the parent timestep.
      self.assertNotIn('fake', observer.last_parent_timestep.observation)
      self.assertIn('fake', observer.last_agent_timestep.observation)
      self.assertTrue(
          np.array_equal(observer.last_agent_action, [22, 22, 22, 22, 22, 22]))
      self.assertTrue(
          np.array_equal(observer.last_parent_action, [22, 22, 22, 22, 22]))

      timestep3 = env.step(agent_action)
      self.assertEqual(observer.last_parent_timestep.step_type,
                       dm_env.StepType.MID)
      self.assertEqual(observer.last_agent_timestep.step_type,
                       dm_env.StepType.MID)
      self.assertNotIn('fake', observer.last_parent_timestep.observation)
      self.assertIn('fake', observer.last_agent_timestep.observation)
      self.assertTrue(
          np.array_equal(observer.last_agent_action, [22, 22, 22, 22, 22, 22]))
      self.assertTrue(
          np.array_equal(observer.last_parent_action, [22, 22, 22, 22, 22]))

      timestep4 = env.step(agent_action)
      self.assertEqual(observer.last_agent_timestep.step_type,
                       dm_env.StepType.LAST)
      self.assertIsNone(observer.last_parent_action)
      self.assertIsNone(observer.last_agent_action)

      # Check the timesteps are as expected:
      self.assertEqual(timestep1.step_type, dm_env.StepType.FIRST)
      self.assertEqual(timestep2.step_type, dm_env.StepType.MID)
      self.assertEqual(timestep3.step_type, dm_env.StepType.MID)
      self.assertEqual(timestep4.step_type, dm_env.StepType.LAST)

  def _assert_actions(self, expected_action: np.ndarray,
                      actual_actions: List[np.ndarray]):
    for actual_action in actual_actions:
      np.testing.assert_array_almost_equal(expected_action, actual_action)


class SpySensor(moma_sensor.Sensor):

  def __init__(self):
    self.control_timestep_seconds = None
    self.physics_timestep_seconds = None
    self.physics_steps_per_control_step = None
    self.episode_count = 0

  def initialize_for_task(self, control_timestep_seconds: float,
                          physics_timestep_seconds: float,
                          physics_steps_per_control_step: int):
    self.control_timestep_seconds = control_timestep_seconds
    self.physics_timestep_seconds = physics_timestep_seconds
    self.physics_steps_per_control_step = physics_steps_per_control_step

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    self.episode_count += 1

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    return {}

  @property
  def name(self) -> str:
    return 'SpySensor'

  def get_obs_key(self, obs: enum.Enum) -> str:
    return ''


class SubTaskEnvBuilderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._arena = _build_arena(name='empty')
    self._static_initializer = scene_initializer.CompositeSceneInitializer([])
    self._dynamic_initializer = entity_initializer.CallableInitializer(
        lambda unused_physics, unused_state: None)

  def testSensorSetup(self):
    spy_sensor = SpySensor()
    task = base_task.BaseTask(
        task_name='empty',
        arena=self._arena,
        robots=[],
        props=[],
        extra_sensors=[spy_sensor],
        extra_effectors=[],
        scene_initializer=self._static_initializer,
        episode_initializer=self._dynamic_initializer,
        control_timestep=0.1)

    builder = subtask_env_builder.SubtaskEnvBuilder()
    self.assertEqual(spy_sensor.physics_steps_per_control_step, 100)
    builder.set_task(task)
    builder.build_base_env()


def _build_arena(name: str) -> composer.Arena:
  """Builds an arena Entity."""
  arena = empty.Arena(name)
  arena.ground.size = (2.0, 2.0, 2.0)
  arena.mjcf_model.option.timestep = 0.001
  arena.mjcf_model.option.gravity = (0., 0., -1.0)
  arena.mjcf_model.size.nconmax = 1000
  arena.mjcf_model.size.njmax = 2000
  arena.mjcf_model.visual.__getattr__('global').offheight = 480
  arena.mjcf_model.visual.__getattr__('global').offwidth = 640
  arena.mjcf_model.visual.map.znear = 0.0005
  return arena


if __name__ == '__main__':
  absltest.main()
