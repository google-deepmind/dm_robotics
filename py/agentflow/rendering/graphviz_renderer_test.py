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
"""Test for GraphvizRenderer."""

import random
import re
from typing import Iterable, Sequence, Text

from absl import logging
from absl.testing import absltest
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.rendering import graphviz_renderer
from dm_robotics.agentflow.rendering import intermediate
from dm_robotics.agentflow.rendering import subgraph
import pydot

_TEST_EDGE_HIDING = False
ENTRY = intermediate.ENTRY_NODE
EXIT = intermediate.EXIT_NODE


class GraphvizRendererTest(absltest.TestCase):

  def test_render_flat_graph(self):
    # render a very simple graph, check for all nodes and edges.
    node1 = subgraph.Node(label='node1', type='option')

    graph = subgraph.Graph(
        label='graph1',
        nodes=frozenset({ENTRY, node1, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, node1),
            subgraph.Edge(node1, EXIT),
        }))

    graphviz_graph = graphviz_renderer.render(graph)
    logging.info('graphviz source: %s', graphviz_graph.to_string())

    nodes = graphviz_graph.get_node_list()
    self.assertNames(nodes, ['_ENTRY', '_node1', '_EXIT'])

    edges = graphviz_graph.get_edge_list()
    self.assertLen(edges, 2)
    self.assertEdge(edges, '_ENTRY', '_node1')
    self.assertEdge(edges, '_node1', '_EXIT')

  def test_render_subtask_option(self):
    # render a subtask option, check shape and HTML-label.
    agent_name = 'agent'
    subtask_name = 'subtask'
    policy_name = 'policy'
    node1 = subgraph.Node(
        label=','.join([agent_name, subtask_name, policy_name]),
        type='sub_task_option')

    graph = subgraph.Graph(
        label='graph1',
        nodes=frozenset({ENTRY, node1, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, node1),
            subgraph.Edge(node1, EXIT),
        }))

    graphviz_graph = graphviz_renderer.render(graph)
    logging.info('graphviz source: %s', graphviz_graph.to_string())

    nodes = graphviz_graph.get_node_list()
    expected_name = ('_' +  # for namespace.
                     ','.join([agent_name, subtask_name, policy_name]))
    self.assertNames(nodes, ['_ENTRY', expected_name, '_EXIT'])
    self.assertNode(
        nodes,
        expected_name,
        shape='plaintext',
        label=re.compile('<.*.>')  # tests for an HTML-style label.
        )

  def test_top_graph_attributes(self):
    graph = subgraph.Graph(
        label='graph1',
        nodes=frozenset({ENTRY, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, EXIT),
        }))

    graphviz_graph = graphviz_renderer.render(graph)
    self.assertTrue(graphviz_graph.get('compound'))
    self.assertTrue(graphviz_graph.get('newrank'))

  def test_entry_and_exit_nodes(self):
    graph = subgraph.Graph(
        label='graph1',
        nodes=frozenset({ENTRY, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, EXIT),
        }))

    graphviz_graph = graphviz_renderer.render(graph)

    entry_node = self._get_node(graphviz_graph, ENTRY)
    self.assertEqual(entry_node.get('shape'), 'circle')
    self.assertEqual(entry_node.get('style'), 'filled')
    self.assertEqual(entry_node.get('fillcolor'), 'black')

    exit_node = self._get_node(graphviz_graph, EXIT)
    self.assertEqual(exit_node.get('shape'), 'doublecircle')
    self.assertEqual(exit_node.get('style'), 'filled')
    self.assertEqual(exit_node.get('fillcolor'), 'black')

  def test_failure_edges(self):
    node1 = subgraph.Node(label=testing_functions.random_string(5))
    graph = subgraph.Graph(
        label='graph1',
        nodes=frozenset({ENTRY, node1, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, node1, type=intermediate.FAILURE_STR),
            subgraph.Edge(node1, EXIT),
        }))

    graphviz_graph = graphviz_renderer.render(graph)

    entry_node1_edge = self._get_edge(graphviz_graph, ENTRY, node1)
    self.assertIn('constraint', entry_node1_edge.get_attributes())
    self.assertFalse(entry_node1_edge.get('constraint'))

  def test_topologically_sorted_nodes(self):
    # Dot attempts to put nodes in the order they appear in the source file,
    # emitting them in a topologically sorted order means that the rendering
    # is better and more consistent.

    # This is tested by generating some sequences with randomly named nodes
    # and checking that the order in the file is not random but is ordered.

    for _ in range(10):
      node1 = subgraph.Node(label=testing_functions.random_string(5))
      node2 = subgraph.Node(label=testing_functions.random_string(5))
      node3 = subgraph.Node(label=testing_functions.random_string(5))

      nodes = [ENTRY, node1, node2, node3, EXIT]
      edges = [
          subgraph.Edge(nodes[0], nodes[1]),
          subgraph.Edge(nodes[1], nodes[2]),
          subgraph.Edge(nodes[2], nodes[3]),
          subgraph.Edge(nodes[3], nodes[4])
      ]

      random.shuffle(nodes)
      random.shuffle(edges)
      graph = subgraph.Graph(
          label='graph', nodes=frozenset(nodes), edges=frozenset(edges))

      graphviz_graph = graphviz_renderer.render(graph)
      logging.info('graphviz_graph\n%s\n', graphviz_graph.to_string())

      entry_num = self._get_node(graphviz_graph, ENTRY).get_sequence()
      node1_num = self._get_node(graphviz_graph, node1).get_sequence()
      node2_num = self._get_node(graphviz_graph, node2).get_sequence()
      node3_num = self._get_node(graphviz_graph, node3).get_sequence()
      exit_num = self._get_node(graphviz_graph, EXIT).get_sequence()

      self.assertGreater(node1_num, entry_num)
      self.assertGreater(node2_num, node1_num)
      self.assertGreater(node3_num, node2_num)
      self.assertGreater(exit_num, node3_num)

  def test_graph_with_no_exit(self):
    # Dot attempts to put nodes in the order they appear in the source file,
    # emitting them in a topologically sorted order means that the rendering
    # is better and more consistent.

    # This is tested by generating some sequences with randomly named nodes
    # and checking that the order in the file is not random but is ordered.

    node1 = subgraph.Node(label='node1')
    node2 = subgraph.Node(label='node2')

    nodes = [ENTRY, node1, node2]
    edges = [
        subgraph.Edge(nodes[0], nodes[1]),
        subgraph.Edge(nodes[1], nodes[2])
    ]

    graph = subgraph.Graph(
        label='graph', nodes=frozenset(nodes), edges=frozenset(edges))

    graphviz_graph = graphviz_renderer.render(graph)

    nodes = graphviz_graph.get_node_list()
    self.assertNames(nodes, ['_ENTRY', '_node1', '_node2'])

    edges = graphviz_graph.get_edge_list()
    self.assertLen(edges, 2)
    self.assertEdge(edges, '_ENTRY', '_node1')
    self.assertEdge(edges, '_node1', '_node2')

  def test_subgraph(self):
    inner_graph = subgraph.Graph(
        label='inner_graph',
        nodes=frozenset({ENTRY, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, EXIT),
        }))

    top_graph = subgraph.Graph(
        label='top_graph',
        nodes=frozenset({ENTRY, inner_graph, EXIT}),
        edges=frozenset({
            subgraph.Edge(ENTRY, inner_graph),
            subgraph.Edge(inner_graph, EXIT),
        }))

    graphviz_graph = graphviz_renderer.render(top_graph)
    # compound must be set for subgraphs to work.
    self.assertTrue(graphviz_graph.get('compound'))

    nodes = graphviz_graph.get_nodes()
    subgraphs = graphviz_graph.get_subgraph_list()
    self.assertNames(nodes, ['_ENTRY', '_EXIT'])

    # The name is significant, it must start with cluster_
    self.assertNames(subgraphs, ['cluster__inner_graph'])

    # Edges to/from a subgraph should have have lhead/ltail set respectively.
    # This makes the edge end/start on the subgraph rather than node in it.
    # Moreover, the node in the subgraph that's used for these edges is
    # always the entry node.  I.e. edges to and from the entry node.
    if _TEST_EDGE_HIDING:
      edges = graphviz_graph.get_edge_list()
      self.assertEdge(
          edges, '_ENTRY', '_inner_graph_ENTRY', lhead='cluster__inner_graph')
      self.assertEdge(
          edges, '_inner_graph_ENTRY', '_EXIT', ltail='cluster__inner_graph')

  def assertEdge(self, edges: Iterable[pydot.Edge], source: Text,
                 destination: Text, **attributes):
    found = False
    matched = False

    for edge in edges:
      if edge.get_source() == source and edge.get_destination() == destination:
        found = True
        # check attributes.
        for key, value in attributes.items():
          if edge.get(key) != value:
            self.fail(
                'Edge from {} to {} has attributes: {}, but expected {} = {}'
                .format(source, destination, attributes, key, value))

        matched = True

    if not found:
      self.fail('No edge from {} to {} found.  All edges: {}'.format(
          source, destination,
          [(e.get_source, e.get_destination()) for e in edges]))
    if not matched:
      self.fail(
          'Edge from {} to {} found, but attributes did not match: {}'.format(
              source, destination, attributes))

  def assertNode(self, nodes: Iterable[pydot.Node], name: Text, **attributes):
    found = False
    matched = False

    for node in nodes:
      if node.get_name() == name:
        found = True
        # check attributes.
        for key, value in attributes.items():
          try:
            is_regex = isinstance(value, re._pattern_type)
          except AttributeError:
            is_regex = isinstance(value, re.Pattern)
          if is_regex:
            matched = re.search(value, node.get(key)) is not None
          else:
            matched = node.get(key) == value
          if not matched:
            self.fail('Node {} has attributes: {}, but expected {} = {}'.format(
                name, attributes, key, value))

        matched = True

    if not found:
      self.fail('No node {} found.  All nodes: {}'.format(
          name, [(n.get_name()) for n in nodes]))
    if not matched:
      self.fail('Node {} found, but attributes did not match: {}'.format(
          name, attributes))

  def assertNames(self, nodes: Iterable[pydot.Common],
                  expected: Sequence[Text]):
    actual = set((node.get_name() for node in nodes))
    expected = set(expected)
    self.assertEqual(actual, expected)

  def _get_node(self, graphviz_graph: pydot.Graph,
                data_node: subgraph.Node) -> pydot.Node:
    node_name = self._graphviz_node_name(data_node)
    matches = graphviz_graph.get_node(node_name)
    if not matches:
      raise AssertionError('No node named: {}.  Names: {}'.format(
          node_name,
          [node.get_name() for node in graphviz_graph.get_node_list()]))
    if len(matches) > 1:
      raise AssertionError('Multiple nodes named: {}.  Names: {}'.format(
          node_name,
          [node.get_name() for node in graphviz_graph.get_node_list()]))
    return matches[0]

  def _graphviz_node_name(self, data_node: subgraph.Node):
    if data_node == ENTRY:
      return '_ENTRY'
    elif data_node == EXIT:
      return '_EXIT'
    else:
      return '_{}'.format(data_node.label)

  def _get_edge(self, graphviz_graph: pydot.Graph, begin: subgraph.Node,
                end: subgraph.Node) -> pydot.Edge:
    begin_name = self._graphviz_node_name(begin)
    end_name = self._graphviz_node_name(end)
    edges = graphviz_graph.get_edge(begin_name, end_name)
    if len(edges) == 1:
      return edges[0]
    else:
      raise AssertionError(
          'Expected to find one edge from {} to {}, but got: {}'.format(
              begin_name, end_name, len(edges)))


class GraphvizValidationTest(absltest.TestCase):

  def test_invalid_edge_nodes(self):
    node1 = subgraph.Node(testing_functions.random_string())
    node2 = subgraph.Node(testing_functions.random_string())
    node3 = subgraph.Node(testing_functions.random_string())

    edges = {
        subgraph.Edge(node1, node2),
        subgraph.Edge(node2, node3),  # node3 not in graph.
        subgraph.Edge(ENTRY, node1),
        subgraph.Edge(node2, EXIT)
    }

    # Because edge2 is node2 -> node3 and node3 is not in the graph, this
    # graph is invalid and should not be instantiable.
    try:
      graph = subgraph.Graph(
          testing_functions.random_string(),
          nodes={ENTRY, node1, node2, node3, EXIT},
          edges=edges)
      graphviz_renderer.render(graph)
    except ValueError as expected:
      del expected

  def test_invalid_no_edge_from_entry(self):
    node1 = subgraph.Node(testing_functions.random_string())
    node2 = subgraph.Node(testing_functions.random_string())
    node3 = subgraph.Node(testing_functions.random_string())

    edges = {subgraph.Edge(node1, node2), subgraph.Edge(node2, EXIT)}

    try:
      graph = subgraph.Graph(
          testing_functions.random_string(),
          nodes={ENTRY, node1, node2, node3, EXIT},
          edges=edges)
      graphviz_renderer.render(graph)
    except ValueError as expected:
      del expected

  def test_valid_no_edge_to_exit(self):
    node1 = subgraph.Node(testing_functions.random_string())
    node2 = subgraph.Node(testing_functions.random_string())
    node3 = subgraph.Node(testing_functions.random_string())

    edges = {
        subgraph.Edge(ENTRY, node1),
        subgraph.Edge(node1, node2),
    }

    # An option that never terminates is valid.
    graph = subgraph.Graph(
        testing_functions.random_string(),
        nodes={ENTRY, node1, node2, node3, EXIT},
        edges=edges)
    graphviz_renderer.render(graph)


if __name__ == '__main__':
  absltest.main()
