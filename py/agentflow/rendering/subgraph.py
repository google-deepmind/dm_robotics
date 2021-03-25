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
"""Module for rendering AgentFlow agents."""

import collections
import sys
from typing import FrozenSet, Iterable, Optional, Text

import attr


@attr.s(frozen=True)
class Node(object):
  label = attr.ib(type=Text)
  type = attr.ib(type=Text, default='')


@attr.s(frozen=True)
class Edge(object):
  begin = attr.ib(type=Node)
  end = attr.ib(type=Node)
  label = attr.ib(type=Text, default='')
  type = attr.ib(type=Text, default='')


@attr.s(frozen=True)
class Graph(Node):
  """An graph (vertices and edges) which may be a vertex in another graph."""
  nodes = attr.ib(type=FrozenSet[Node], default=attr.Factory(frozenset))
  edges = attr.ib(type=FrozenSet[Edge], default=attr.Factory(frozenset))


def topologically_sorted_nodes(edges: Iterable[Edge]) -> Iterable[Node]:
  """Yields the nodes in topologically sorted order.

  The edges may contain cycles, the order of nodes in a cycle is chosen
  arbitrarily.

  Args:
    edges: Edges between nodes to sort.
  """

  # Map of Node -> incoming (non-negative) edge count
  incoming_count = collections.defaultdict(int)  # Dict[Node, int]

  for edge in edges:
    incoming_count[edge.begin] += 0  # ensure all nodes are included.
    incoming_count[edge.end] += 1

  while incoming_count:
    low_count = sys.maxsize
    lowest = None  # type: Optional[Node]

    # Find next node with lowest incoming node count (can be > 0 if there are
    # loops)
    for node, in_count in incoming_count.items():
      if in_count < low_count:
        lowest = node
        low_count = in_count
      elif in_count == low_count:
        # in the case of loops or multiple entry points, tie-break by label.
        assert lowest
        if node.label < lowest.label:
          lowest = node

    assert lowest
    yield lowest

    # Remove this node from the incoming count dict
    del incoming_count[lowest]

    # Reduce in-count of remaining nodes that the `lowest` node points to.
    for edge in edges:
      if edge.begin == lowest and edge.end in incoming_count:
        incoming_count[edge.end] -= 1
