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
"""Test for Intermediate."""

from typing import Mapping, Optional, Text, Tuple

from absl.testing import absltest
import dm_env
from dm_env import specs
from dm_robotics import agentflow as af
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.rendering import intermediate
from dm_robotics.agentflow.rendering import subgraph
import numpy as np

ENTRY = intermediate.ENTRY_NODE
EXIT = intermediate.EXIT_NODE


class TestSubTask(af.SubTask):
  """A test subtask."""

  def __init__(self,
               base_obs_spec: Mapping[Text, specs.Array],
               name: Optional[Text] = None):
    super(TestSubTask, self).__init__(name)
    self._base_obs_spec = base_obs_spec

  def observation_spec(self) -> Mapping[Text, specs.Array]:
    return self._base_obs_spec

  def arg_spec(self) -> Optional[specs.Array]:
    return

  def action_spec(self) -> specs.Array:
    return specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0., maximum=1., name='dummy_act')

  def agent_to_parent_action(self, agent_action: np.ndarray) -> np.ndarray:
    return np.hstack((agent_action, np.zeros(2)))  # Return full action.

  def parent_to_agent_timestep(self, parent_timestep: dm_env.TimeStep,
                               arg_key: Text) -> Tuple[dm_env.TimeStep, float]:
    return (parent_timestep, 1.0)

  def pterm(self, parent_timestep: dm_env.TimeStep, own_arg_key: Text) -> float:
    del parent_timestep, own_arg_key
    return 0.0


class TestPolicy(af.Policy):
  """A test policy."""

  def __init__(self,
               action_spec: specs.Array,
               unused_observation_spec: Mapping[Text, specs.Array],
               name: Optional[Text] = None):
    super(TestPolicy, self).__init__(name)
    self._action_spec = action_spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    return np.random.rand(self._action_spec.shape[0])


def test_subtask(name: Text):
  return TestSubTask(
      base_obs_spec={'foo': specs.Array(dtype=np.float32, shape=(2,))},
      name=name)


def test_policy(name: Text):
  return TestPolicy(
      specs.Array(dtype=np.float32, shape=(2,)),
      {'foo': specs.Array(dtype=np.float32, shape=(2,))},
      name=name)


class TestDelegateOption(af.DelegateOption):

  def step(self, timestep):
    return np.arange(2)


class IntermediateTest(absltest.TestCase):

  def test_atomic(self):
    """Simple atomic options should become nodes."""

    option = testing_functions.atomic_option_with_name('The Label')
    graph = intermediate.render(option)

    self.assertIsInstance(graph, subgraph.Node)
    self.assertNotIsInstance(graph, subgraph.Graph)
    self.assertEqual(graph, subgraph.Node(label='The Label', type='option'))

  def test_while(self):
    """A while loop should become a subgraph with a condition 'option'."""

    inner_option = testing_functions.atomic_option_with_name('Inner')
    loop = af.While(lambda x: True, inner_option, name='Loop')
    graph = intermediate.render(loop)

    self.assertIsInstance(graph, subgraph.Graph)

    cond_node = subgraph.Node(label='Condition', type='condition')
    inner_node = subgraph.Node(label='Inner', type='option')

    self.assertEqual(
        graph,
        subgraph.Graph(
            label='Loop',
            nodes=frozenset({ENTRY, cond_node, inner_node, EXIT}),
            edges=frozenset({
                subgraph.Edge(ENTRY, cond_node),
                subgraph.Edge(cond_node, inner_node, label='True'),
                subgraph.Edge(cond_node, EXIT, label='False'),
                subgraph.Edge(inner_node, cond_node, label='While'),
            })))

  def test_repeat(self):
    """Repeat is a special while loop."""
    # The condition is the number of iterations, so we can avoid two nodes in
    # the subgraph by labelling the return arc.

    num_iter = 3
    inner_option = testing_functions.atomic_option_with_name('Inner')
    loop = af.Repeat(num_iter, inner_option, name='Loop')
    graph = intermediate.render(loop)

    inner_node = subgraph.Node(label='Inner', type='option')

    expected_graph = subgraph.Graph(
        label='Loop',
        nodes=frozenset({ENTRY, inner_node, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, inner_node, label=''),
            subgraph.Edge(inner_node, inner_node, label='3 times'),
            subgraph.Edge(inner_node, EXIT, label=''),
        }))
    self.assertEqual(graph, expected_graph)

  def test_sequence(self):
    o1 = testing_functions.atomic_option_with_name('o1')
    o2 = testing_functions.atomic_option_with_name('o2')
    o3 = testing_functions.atomic_option_with_name('o3')
    seq = af.Sequence([o1, o2, o3],
                      terminate_on_option_failure=False,
                      name='Sequence Name')

    node1 = subgraph.Node(label='o1', type='option')
    node2 = subgraph.Node(label='o2', type='option')
    node3 = subgraph.Node(label='o3', type='option')

    edges = {
        subgraph.Edge(ENTRY, node1),
        subgraph.Edge(node1, node2),
        subgraph.Edge(node2, node3),
        subgraph.Edge(node3, EXIT)
    }

    nodes = {ENTRY, node1, node2, node3, EXIT}

    expected_graph = subgraph.Graph(
        'Sequence Name', nodes=frozenset(nodes), edges=frozenset(edges))
    actual_graph = intermediate.render(seq)

    self.assertEqual(actual_graph, expected_graph)

  def test_sequence_early_exit(self):
    # When a sequence has an early exit (terminate_on_option_failure=True),
    # then a failure of any option in the sequence results in failure (exit)
    # of the whole sequence, therefore there are failure edges from every
    # option to the exit option (except for the last option which results in
    # the option ending whether or not it fails).
    #
    o1 = testing_functions.atomic_option_with_name('o1')
    o2 = testing_functions.atomic_option_with_name('o2')
    o3 = testing_functions.atomic_option_with_name('o3')
    seq = af.Sequence([o1, o2, o3],
                      terminate_on_option_failure=True,
                      name='Sequence Name')

    node1 = subgraph.Node(label='o1', type='option')
    node2 = subgraph.Node(label='o2', type='option')
    node3 = subgraph.Node(label='o3', type='option')

    edges = {
        subgraph.Edge(ENTRY, node1),
        subgraph.Edge(node1, node2, intermediate.SUCCESS_STR),
        subgraph.Edge(node1, EXIT, intermediate.FAILURE_STR),
        subgraph.Edge(node1, EXIT, intermediate.PREEMPTED_STR),
        subgraph.Edge(node2, node3, intermediate.SUCCESS_STR),
        subgraph.Edge(node2, EXIT, intermediate.FAILURE_STR),
        subgraph.Edge(node2, EXIT, intermediate.PREEMPTED_STR),
        subgraph.Edge(node3, EXIT)
    }

    nodes = {ENTRY, node1, node2, node3, EXIT}

    expected_graph = subgraph.Graph(
        'Sequence Name', nodes=frozenset(nodes), edges=frozenset(edges))
    actual_graph = intermediate.render(seq)

    self.assertEqual(actual_graph, expected_graph)

  def test_concurrent(self):
    action_size = 2
    o1 = testing_functions.atomic_option_with_name('o1', action_size)
    o2 = testing_functions.atomic_option_with_name('o2', action_size)
    o3 = testing_functions.atomic_option_with_name('o3', action_size)
    seq = af.ConcurrentOption([o1, o2, o3],
                              action_spec=specs.Array(
                                  shape=(action_size,), dtype=np.float64),
                              name='Concurrent Name')

    node1 = subgraph.Node(label='o1', type='option')
    node2 = subgraph.Node(label='o2', type='option')
    node3 = subgraph.Node(label='o3', type='option')

    edges = {
        subgraph.Edge(ENTRY, node1),
        subgraph.Edge(ENTRY, node2),
        subgraph.Edge(ENTRY, node3),
        subgraph.Edge(node1, EXIT),
        subgraph.Edge(node2, EXIT),
        subgraph.Edge(node3, EXIT)
    }

    nodes = {ENTRY, node1, node2, node3, EXIT}

    expected_graph = subgraph.Graph(
        'Concurrent Name', nodes=frozenset(nodes), edges=frozenset(edges))
    actual_graph = intermediate.render(seq)

    print(expected_graph)
    print(actual_graph)

    self.assertEqual(actual_graph, expected_graph)

  def test_subtask_option(self):
    # Nodes of type 'sub_task_option' should have a label that is a comma-
    # separated list of `agent_name`, `subtask_name`, and `policy_name`.

    subtask_name = 'test_subtask'
    policy_name = 'test_policy'
    agent_name = 'Test SubTask Option'
    subtask = test_subtask(subtask_name)
    policy = test_policy(policy_name)
    op = af.SubTaskOption(subtask, policy, name=agent_name)

    actual_graph = intermediate.render(op)
    expected_graph = subgraph.Node(
        label=','.join([agent_name, subtask_name, policy_name]),
        type='sub_task_option')

    print(expected_graph)
    print(actual_graph)

    self.assertEqual(actual_graph, expected_graph)

  def test_cond(self):
    # Conditions have a true and false branch.
    # The subgraph is represented as:
    # entry --> condition
    #               +----- TRUE ----> [ true option ] -----+
    #               |                                      |
    #               +----- FALSE ---> [ false option ] ----+
    #                                                      |
    #                                                     exit
    # I.e. 5 nodes and 5 edges:

    cond = lambda ts, res: True
    true_option = testing_functions.atomic_option_with_name('TrueOption')
    false_option = testing_functions.atomic_option_with_name('FalseOption')
    option = af.Cond(cond, true_option, false_option, name='Conditional')

    cond_node = subgraph.Node(label='Condition', type='condition')
    true_node = subgraph.Node(label='TrueOption', type='option')
    false_node = subgraph.Node(label='FalseOption', type='option')

    expected_graph = subgraph.Graph(
        label='Conditional',
        nodes=frozenset([ENTRY, cond_node, true_node, false_node, EXIT]),
        edges=frozenset([
            subgraph.Edge(ENTRY, cond_node),
            subgraph.Edge(cond_node, true_node, label='True'),
            subgraph.Edge(cond_node, false_node, label='False'),
            subgraph.Edge(true_node, EXIT),
            subgraph.Edge(false_node, EXIT),
        ]))

    actual_graph = intermediate.render(option)
    self.assertEqual(actual_graph, expected_graph)

  def test_delegate(self):
    """Test DelegateOption subclasses."""

    # We have a number of DelegateOptions, this test uses a test-only subclass
    # to ensure that even if the common subclasses are handled specially, that
    # all DelegateOption instances render.

    delegate_option = testing_functions.atomic_option_with_name('Bottom')
    option = TestDelegateOption(delegate_option, name='Delegator')
    actual_graph = intermediate.render(option)
    self.assertIsInstance(actual_graph, subgraph.Graph)

    bottom_node = subgraph.Node(label='Bottom', type='option')
    expected_graph = subgraph.Graph(
        label='Delegator',
        nodes=frozenset({ENTRY, bottom_node, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, bottom_node),
            subgraph.Edge(bottom_node, EXIT)
        }))
    self.assertEqual(actual_graph, expected_graph)

  def test_subgraph_in_graph(self):
    """Test that rendering is recursive for options."""
    o1 = testing_functions.atomic_option_with_name('seq_o1')
    o2 = testing_functions.atomic_option_with_name('seq_o2')
    seq = af.Sequence([o1, o2],
                      terminate_on_option_failure=False,
                      name='Sequence')
    option = TestDelegateOption(delegate=seq, name='Delegator')

    # option delegates to a sequence.
    # option is rendered as a graph, and so is sequence.
    # This means we should have a graph in a graph.

    # Construct the expected graph (inner graph first).
    inner_node1 = subgraph.Node(label='seq_o1', type='option')
    inner_node2 = subgraph.Node(label='seq_o2', type='option')

    inner_graph = subgraph.Graph(
        'Sequence',
        nodes=frozenset({ENTRY, inner_node1, inner_node2, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, inner_node1),
            subgraph.Edge(inner_node1, inner_node2),
            subgraph.Edge(inner_node2, EXIT),
        }))

    expected_graph = subgraph.Graph(
        'Delegator',
        nodes=frozenset({ENTRY, inner_graph, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, inner_graph),
            subgraph.Edge(inner_graph, EXIT)
        }))

    actual_graph = intermediate.render(option)

    self.assertEqual(actual_graph, expected_graph)


if __name__ == '__main__':
  absltest.main()
