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
"""Module to convert subgraph Graphs into graphviz graphs for rendering."""

import itertools
import os
import tempfile
from typing import Any, Callable, Dict, Iterable, Text, Type, Union

from absl import logging
from dm_robotics import agentflow as af
from dm_robotics.agentflow.rendering import intermediate
from dm_robotics.agentflow.rendering import subgraph
import lxml
from lxml.html import builder as E
import pydot

# Color names for HTML nodes.
_BACKGROUND_COLOR = '#00000015'
_SUBTASK_COLOR = '#00999925'
_POLICY_COLOR = '#ffff0025'
_SUBTASK_OPTION_COLOR = '#0000d015'


def add_itermediate_renderer(
    node_type: Type[af.Policy], render_func: Callable[[Any],
                                                      subgraph.Graph]) -> None:
  """Adds a custom renderer for the provided AgentFlow node type."""
  intermediate.add_renderer(node_type, render_func)


def render(graph: subgraph.Graph) -> Union[pydot.Graph]:
  """Render the subgraph as a graphviz graph."""
  try:
    _validate_state_machine_graph(graph)
  except ValueError:
    logging.error('Invalid Graph: %s', graph)
    raise

  renderer = _RenderedGraph(graph, depth=0, namespace='')
  return renderer.graph


def to_png(graph: Union[pydot.Dot, pydot.Subgraph]) -> Union[Text, bytes]:
  """Render a graphviz to a PNG."""
  return graph.create_png(prog='dot')  # pytype: disable=attribute-error


def open_viewer(option: af.Policy) -> None:
  """Render a graphviz to a PNG and xdg-open it (Linux assumed).

  This is a simple entry point if you just want to get a graph on the screen.

  Args:
    option: The option (typed as policy) to render.
  """
  graph = intermediate.render(option)
  if not isinstance(graph, subgraph.Graph):
    raise ValueError('policy is not composite.')

  graphviz_graph = render(graph)

  temp_dir = tempfile.mkdtemp()

  raw_filename = os.path.join(temp_dir, 'graph.dot')
  png_filename = os.path.join(temp_dir, 'graph.png')

  graphviz_graph.write_raw(raw_filename, prog='dot')  # pytype: disable=attribute-error
  graphviz_graph.write_png(png_filename, prog='dot')  # pytype: disable=attribute-error

  print('Rendered option {} to {} and {}'.format(option.name, raw_filename,
                                                 png_filename))
  os.system('xdg-open {}'.format(png_filename))


def _validate_state_machine_graph(graph: subgraph.Graph):
  """Validates assumptions about how subgraph.Graph is used."""

  # Assumes the graph is acyclic. Otherwise this will overflow the stack.
  for node in graph.nodes:
    if isinstance(node, subgraph.Graph):
      _validate_state_machine_graph(node)

  # Every edge in the graph must start and end on a node in the graph.
  for edge in graph.edges:
    if edge.begin not in graph.nodes or edge.end not in graph.nodes:
      raise ValueError('Nodes of edge {} not in graph, nodes are: {}'.format(
          edge, [n.label for n in graph.nodes]))

  # There must be an entry node.
  if not any(edge.begin == intermediate.ENTRY_NODE for edge in graph.edges):
    raise ValueError('No edge from ENTRY_NODE')


class _RenderedGraph(object):
  """Renderer for subgraph.Graph objects.

  This creates the graph and subgraphs recursively.

  Each node in the graph that is itself a subgraph.Graph is rendered with a
   _RenderedGraph and stored in self._subgraphs.
  """

  def __init__(self, graph: subgraph.Graph, depth: int, namespace: Text):
    """Render the graph to a graphviz graph.

    Args:
      graph: A graph to convert.
      depth: The depth in the hierarchy, with 0 being top.
      namespace: A namespace to ensure name uniqueness between graphs.
    """

    self._depth = depth
    self._namespace = namespace
    self._data_graph = graph
    self._graphviz_graph = self._create_graphviz_container(
        graph.label, depth, namespace)
    self._graphviz_nodes = {}  # type: Dict[Text, pydot.Node]
    self._subgraphs = {}  # type: Dict[Text, '_RenderedGraph']

    # Add each node to the graphviz graph.
    for node in self._sorted_nodes():
      if isinstance(node, subgraph.Graph):
        child = _RenderedGraph(
            node, depth=depth+1, namespace=self._namespaced(node.label))
        self._subgraphs[node.label] = child
        self._graphviz_graph.add_subgraph(child.graph)
      else:
        if node == intermediate.ENTRY_NODE:
          graphviz_node = self._create_entry_viz_node()
        elif node == intermediate.EXIT_NODE:
          graphviz_node = self._create_exit_viz_node()
        elif node.type == 'sub_task_option':
          graphviz_node = self._create_sto_viz_node(node)
        else:
          graphviz_node = self._create_graphviz_node(node)

        self._graphviz_nodes[node.label] = graphviz_node
        self._graphviz_graph.add_node(graphviz_node)

    # Add edges between nodes, this is tricker because of subgraphs.
    for edge in self._data_graph.edges:
      self._graphviz_graph.add_edge(self._create_graphviz_edge(edge))

  @property
  def graph(self) -> Union[pydot.Graph]:
    return self._graphviz_graph

  @property
  def entry_node(self) -> pydot.Node:
    return self._graphviz_nodes[intermediate.ENTRY_NODE.label]

  @property
  def exit_node(self) -> pydot.Node:
    return self._graphviz_nodes[intermediate.EXIT_NODE.label]

  def _sorted_nodes(self) -> Iterable[subgraph.Node]:
    """Returns all graph nodes, in a topologically sorted order."""
    # Notes:
    # 1. The primary ordering is of nodes according to success transitions,
    #   I.e. we will emit the nodes reachable from a success before those
    #   reachable by failure.
    # 2. All nodes will be emitted, and it's possible for a node to be
    #   detached from all other nodes.

    success = list(
        subgraph.topologically_sorted_nodes([
            edge for edge in self._data_graph.edges
            if edge.type != intermediate.FAILURE_STR
        ]))
    failure = list(
        subgraph.topologically_sorted_nodes([
            edge for edge in self._data_graph.edges
            if edge.type == intermediate.FAILURE_STR
        ]))
    failure = [node for node in failure if node not in success]
    floating = [
        node for node in self._data_graph.nodes
        if node not in success and node not in failure
    ]

    return itertools.chain(success, failure, floating)

  def _create_graphviz_container(
      self, label: Text, depth: int,
      namespace: Text) -> Union[pydot.Dot, pydot.Subgraph]:
    """A 'container' is a top-level Dot object or a subgraph."""
    # nodesep size is in inches
    if depth == 0:
      return pydot.Dot(compound=True, labeljust='l', nodesep=0.2, newrank=True)
    else:
      return pydot.Subgraph(
          'cluster_{}'.format(namespace),
          bgcolor=_BACKGROUND_COLOR,
          color='black',
          label=label,
          shape='box',
          style='rounded')

  def _create_graphviz_node(self, node: subgraph.Node) -> pydot.Node:
    if node == intermediate.ENTRY_NODE:
      assert False, 'use _create_entry_viz_node'
    elif node == intermediate.EXIT_NODE:
      assert False, 'use _create_exit_viz_node'
    else:
      return pydot.Node(
          name=self._namespaced(node.label),
          label=node.label,
          shape='box',
          style='rounded' if node.type == 'option' else '')

  def _create_graphviz_edge(self, edge: subgraph.Edge) -> pydot.Edge:
    """Create an graphviz edge from a graph edge."""

    begin_node, end_node = edge.begin, edge.end

    attrs = {'xlabel': edge.label, 'color': self._edge_color(edge)}

    # Edges to or from a subgraph are really edges to nodes in different
    # subgraphs. However, we want them to render as though they come from the
    # subgraph itself, not a node in that subgraph.
    # This is achieved by clipping to the cluster with lhead and ltail.
    #
    # Edges to and from the subgraph go via the ENTRY node rather
    # than to the ENTRY and from the EXIT. This allows graphs to be laid-out
    # more compactly.

    from_graph = isinstance(begin_node, subgraph.Graph)
    to_graph = isinstance(end_node, subgraph.Graph)

    if from_graph:
      child = self._subgraphs[begin_node.label]
      out_viz_node = child.exit_node
      # attrs['ltail'] = child.graph.get_name()  # To hide edge under subgraph.
    else:
      out_viz_node = self._graphviz_nodes[begin_node.label]

    if to_graph:
      child = self._subgraphs[end_node.label]
      in_viz_node = child.entry_node
      # attrs['lhead'] = child.graph.get_name()  # To hide edge under subgraph.
    else:
      in_viz_node = self._graphviz_nodes[end_node.label]

    # If an edge goes from one subgraph to another, then don't use that
    # edge to imply ranking.  This results in more compact graphs.
    if from_graph and to_graph:
      attrs['constraint'] = False
      attrs['minlen'] = 2.0

    # If this is a failure, don't use the edge to imply ranking.
    if edge.type == intermediate.FAILURE_STR:
      attrs['constraint'] = False

    return pydot.Edge(out_viz_node, in_viz_node, **attrs)

  def _create_entry_viz_node(self) -> pydot.Node:
    attrs = {
        'label': '',
        'shape': 'circle',
        'style': 'filled',
        'fillcolor': 'black',
        'fixedsize': True,
        'width': 0.2
    }
    return pydot.Node(name=self._namespaced('ENTRY'), **attrs)

  def _create_exit_viz_node(self) -> pydot.Node:
    attrs = {
        'label': '',
        'shape': 'doublecircle',
        'style': 'filled',
        'fillcolor': 'black',
        'fixedsize': True,
        'width': 0.2
    }
    return pydot.Node(name=self._namespaced('EXIT'), **attrs)

  def _create_sto_viz_node(self, node: subgraph.Node) -> pydot.Node:
    """Makes an HTML-style node for a SubTaskOption."""
    agent_name, subtask_name, policy_name = node.label.split(',')

    # HTML styles below interpreted by graphviz-special, not vanilla HTML, see:
    # https://graphviz.org/doc/info/shapes.html#html
    table_element = E.TABLE(
        E.TR(E.TD(agent_name, BORDER='0', COLSPAN='2')),
        E.TR(
            E.TD(
                subtask_name,
                BORDER='1',
                STYLE='RADIAL',
                BGCOLOR=_SUBTASK_COLOR),
            E.TD(
                policy_name,
                BORDER='1',
                STYLE='RADIAL',
                BGCOLOR=_POLICY_COLOR),
        ),
        BORDER='1',
        STYLE='ROUNDED',
        BGCOLOR=_SUBTASK_OPTION_COLOR,
        CELLBORDER='1',
        CELLPADDING='3',
        CELLSPACING='5')
    table_str = lxml.etree.tostring(table_element)
    if isinstance(table_str, bytes):
      table_str = table_str.decode('utf-8')  # lxml generates bytes in python3.
    html_str = '<{}>'.format(table_str)  # Wrap with '<>' to designate HTML tag.
    attrs = {
        'label': html_str,
        'fontcolor': 'black',
        'shape': 'plaintext',
    }
    return pydot.Node(name=self._namespaced(node.label), **attrs)

  def _namespaced(self, name: Text) -> Text:
    return '{}_{}'.format(self._namespace, name)

  def _edge_color(self, edge: subgraph.Edge):
    """Returns edge color string according to edge type."""
    if not edge.type:
      return 'black'
    elif edge.type == intermediate.SUCCESS_STR:
      return 'green'
    elif edge.type == intermediate.FAILURE_STR:
      return 'red'
    elif edge.type == intermediate.PREEMPTED_STR:
      return 'yellow'
    else:
      logging.warning('Unknown edge type: %s', edge.type)
      return 'purple'
