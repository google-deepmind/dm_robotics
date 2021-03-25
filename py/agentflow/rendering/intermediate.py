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
"""Module to convert AgentFlow policies into a Graph.

The job of this module is to translate (represent) each component of
AgentFlow (E.g. a Conditional option, a Sequence) as an *abstract* graph.

This means the rendering doesn't have to know about AgentFlow.
"""

from typing import Any, Callable, Type

from dm_robotics import agentflow as af
from dm_robotics.agentflow.rendering import subgraph

# Special nodes.
ENTRY_NODE = subgraph.Node('_entry_', 'entry')
EXIT_NODE = subgraph.Node('_exit_', 'exit')

# Edge types
SUCCESS_STR = af.TerminationType.SUCCESS.name
FAILURE_STR = af.TerminationType.FAILURE.name
PREEMPTED_STR = af.TerminationType.PREEMPTED.name


def render(agent: af.Policy) -> subgraph.Node:
  """Render a policy as a subgraph.Node."""
  # Look for most-specific match for agent type, falling back to generic node.
  for cls in agent.__class__.mro():
    try:
      return _RENDER_FUNCS[cls](agent)
    except KeyError as ex:
      if ex.args[0] == cls:
        continue
      raise ex
  return subgraph.Node(agent.name, type='option')


def add_renderer(node_type: Type[af.Policy],
                 render_func: Callable[[Any], subgraph.Graph]):
  """Adds a custom renderer for the provided AgentFlow node type.

  This callable should generate the intermediate `subgraph.Graph`
  representation, not a final output e.g. pydot, png, networkx, etc.

  Args:
    node_type: The type of node to associate with `render_func`.
    render_func: A callable taking an instance of type `node_type` and returning
      an intermediate `subgraph.Graph` representation to be passed to the final
      renderer front-end.
  """
  _RENDER_FUNCS.update({node_type: render_func})


def _repeat_to_graph(agent: af.Repeat) -> subgraph.Graph:
  node = render(_get_only_child(agent))

  return subgraph.Graph(
      label=agent.name,
      nodes=frozenset([ENTRY_NODE, node, EXIT_NODE]),
      edges=frozenset([
          subgraph.Edge(ENTRY_NODE, node),
          subgraph.Edge(node, node, label='{} times'.format(agent.num_iters)),
          subgraph.Edge(node, EXIT_NODE)
      ]))


def _get_only_child(agent: af.Policy) -> af.Policy:
  children = list(agent.child_policies())
  if len(children) != 1:
    raise ValueError('{} should have one child, actual: {}'.format(
        agent.name, [child.name for child in children]))
  return children[0]


def _while_to_graph(agent: af.While) -> subgraph.Graph:
  condition_node = subgraph.Node(label='Condition', type='condition')
  node = render(_get_only_child(agent))

  return subgraph.Graph(
      label=agent.name,
      nodes=frozenset([ENTRY_NODE, condition_node, node, EXIT_NODE]),
      edges=frozenset([
          subgraph.Edge(ENTRY_NODE, condition_node),
          subgraph.Edge(condition_node, node, label='True'),
          subgraph.Edge(condition_node, EXIT_NODE, label='False'),
          subgraph.Edge(node, condition_node, label='While')
      ]))


def _sequence_to_graph(agent: af.Sequence) -> subgraph.Graph:
  """Convert a `Sequence` to a `Graph`."""
  early_exit = agent.terminate_on_option_failure
  child_nodes = [render(option) for option in agent.child_policies()]

  edges = {
      subgraph.Edge(ENTRY_NODE, child_nodes[0]),
      subgraph.Edge(child_nodes[-1], EXIT_NODE),
  }

  for from_node, to_node in zip(child_nodes[:-1], child_nodes[1:]):
    if early_exit:
      edges.add(subgraph.Edge(from_node, to_node, SUCCESS_STR))
      edges.add(subgraph.Edge(from_node, EXIT_NODE, FAILURE_STR))
      edges.add(subgraph.Edge(from_node, EXIT_NODE, PREEMPTED_STR))
    else:
      edges.add(subgraph.Edge(from_node, to_node))

  nodes = {ENTRY_NODE, EXIT_NODE}
  nodes.update(child_nodes)

  return subgraph.Graph(
      agent.name, nodes=frozenset(nodes), edges=frozenset(edges))


def _concurrent_to_graph(agent: af.ConcurrentOption) -> subgraph.Graph:
  """Convert a `ConcurrentOption` to a `Graph`."""
  child_nodes = [render(option) for option in agent.child_policies()]

  edges = set()

  for child_node in child_nodes:
    edges.add(subgraph.Edge(ENTRY_NODE, child_node))
    edges.add(subgraph.Edge(child_node, EXIT_NODE))

  nodes = {ENTRY_NODE, EXIT_NODE}
  nodes.update(child_nodes)

  return subgraph.Graph(
      agent.name, nodes=frozenset(nodes), edges=frozenset(edges))


def _sto_to_graph(agent: af.SubTaskOption) -> subgraph.Node:
  """Convert a `SubTaskOption` to a `Graph`."""
  node_label = '{},{},{}'.format(agent.name or 'SubTask Option',
                                 agent.subtask.name or 'SubTask',
                                 agent.agent.name or 'Policy')
  return subgraph.Node(label=node_label, type='sub_task_option')


def _cond_to_graph(agent: af.Cond) -> subgraph.Graph:
  condition_node = subgraph.Node(label='Condition', type='condition')
  true_node = render(agent.true_branch)
  false_node = render(agent.false_branch)

  return subgraph.Graph(
      agent.name,
      nodes=frozenset(
          [ENTRY_NODE, condition_node, true_node, false_node, EXIT_NODE]),
      edges=frozenset([
          subgraph.Edge(ENTRY_NODE, condition_node),
          subgraph.Edge(condition_node, true_node, label='True'),
          subgraph.Edge(condition_node, false_node, label='False'),
          subgraph.Edge(true_node, EXIT_NODE),
          subgraph.Edge(false_node, EXIT_NODE)
      ]))


def _delegate_to_graph(agent: af.DelegateOption) -> subgraph.Graph:
  delegate_node = render(agent.delegate)

  return subgraph.Graph(
      label=agent.name,
      nodes=frozenset({ENTRY_NODE, delegate_node, EXIT_NODE}),
      edges=frozenset({
          subgraph.Edge(ENTRY_NODE, delegate_node),
          subgraph.Edge(delegate_node, EXIT_NODE)
      }))


# Rendering methods for transforming agentflow nodes to graph-intermediates.
_RENDER_FUNCS = {
    af.Repeat: _repeat_to_graph,
    af.While: _while_to_graph,
    af.Sequence: _sequence_to_graph,
    af.ConcurrentOption: _concurrent_to_graph,
    af.SubTaskOption: _sto_to_graph,
    af.Cond: _cond_to_graph,
    af.DelegateOption: _delegate_to_graph,
}
