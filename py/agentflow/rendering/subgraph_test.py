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
"""Test for Subgraph."""

import random

from absl import logging
from absl.testing import absltest
from dm_robotics.agentflow import testing_functions
from dm_robotics.agentflow.rendering import subgraph


class SubgraphTest(absltest.TestCase):

  def test_construction(self):
    node1 = subgraph.Node(testing_functions.random_string())
    node2 = subgraph.Node(testing_functions.random_string())

    nodes = {node1, node2}

    edges = {
        subgraph.Edge(node1, node2),
        subgraph.Edge(node2, node1),
    }

    graph = subgraph.Graph(
        testing_functions.random_string(), nodes=nodes, edges=edges)

    # That's the test, this constructor should work
    self.assertIsNotNone(graph)
    self.assertLen(graph.nodes, 2)
    self.assertLen(graph.edges, 2)

  def test_hashable(self):
    # This is important, we don't want hashing to break.

    def create_graph(graph_name, node1_name, node2_name):
      node1 = subgraph.Node(node1_name)
      node2 = subgraph.Node(node2_name)

      nodes = {node1, node2}

      edges = {
          subgraph.Edge(node1, node2),
          subgraph.Edge(node2, node1),
      }

      return subgraph.Graph(
          graph_name,
          nodes=frozenset(nodes),
          edges=frozenset(edges))

    graph_name = testing_functions.random_string()
    name1 = testing_functions.random_string()
    name2 = testing_functions.random_string()

    graph1 = create_graph(graph_name, name1, name2)
    graph2 = create_graph(graph_name, name1, name2)

    self.assertIsNot(graph1, graph2)
    self.assertEqual(graph1, graph2)
    self.assertEqual(hash(graph1), hash(graph2))

  def test_topological_sort_no_loop(self):
    node1 = subgraph.Node(testing_functions.random_string())
    node2 = subgraph.Node(testing_functions.random_string())
    node3 = subgraph.Node(testing_functions.random_string())

    # Edges, unsorted, graph is: ENTRY -> node1 -> node2 -> node3 -> EXIT
    edges = [
        subgraph.Edge(node1, node2),
        subgraph.Edge(node2, node3),
    ]

    sorted_nodes = list(subgraph.topologically_sorted_nodes(edges))
    self.assertEqual(sorted_nodes, [node1, node2, node3])

  def test_topological_sort_with_loop(self):
    node1 = subgraph.Node(testing_functions.random_string())
    node2 = subgraph.Node(testing_functions.random_string())
    node3 = subgraph.Node(testing_functions.random_string())

    # ENTRY node1 -> node2 <--> node3 -> EXIT  I.e. Loop between node 2 and 3

    edges = [
        subgraph.Edge(node1, node2),
        subgraph.Edge(node2, node3),
        subgraph.Edge(node3, node2),
    ]
    random.shuffle(edges)

    sorted_nodes = list(subgraph.topologically_sorted_nodes(edges))

    order1 = [node1, node2, node3]
    order2 = [node1, node3, node2]

    if sorted_nodes != order1 and sorted_nodes != order2:
      logging.info('order1: %s', order1)
      logging.info('order2: %s', order2)
      logging.info('sorted_nodes: %s', sorted_nodes)
      self.fail('bad sort order, see log.')


if __name__ == '__main__':
  absltest.main()
