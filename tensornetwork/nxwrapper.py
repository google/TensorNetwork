# Copyright 2019 The TensorNetwork Authors
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

"""NetworkX wrapper for plotting simple tree-like Tensor Networks.

A simple wrapper that creates a `networkx` graph from the nodes and edges
of a tensor network.

Restrictions:
  Supports only tree tensor networks (with nodes arranged to levels and only
    consecutive levels share edges).
  Supports dangling edges only at the top or bottom level (not in the middle).
  Does not support trace edges or multiple edges between nodes.
"""
# TODO: General wrapper with TensorNetwork object as input.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import matplotlib
import networkx as nx
import numpy as np
from typing import List, Optional, Union, Text, Dict
from tensornetwork import tensornetwork


def draw_tree(root: Union[tensornetwork.Node, List[tensornetwork.Node],
                          List[List[tensornetwork.Node]]],
              axes: matplotlib.axes.Axes,
              colormap: Union[Text, List[Optional[Text]]] = "red",
              options: Dict[Text, Union[float, int, List[int], bool]] = dict(),
              ) -> None:
  """Plots the tree network graph.

  Args:
    root: Root node or nodes of the tree to plot.
      User can specify order of nodes at each level by giving a list of lists.
    axes: Axes object to use for plotting the tensor network.
    colormap: Color of nodes. If a list is given a different color is applied
      to each tree level.
    options: Dictionary with plotting options.
      Available options:
        x_margin: Minimum horizontal distance between nodes in the plot.
        y_margin: Vertical distance between consecutive tree levels.
        dangling_size: Size (length) of dangling edges in the plot.
        dangling_angle: When a node has multiple dangling edges, this defines
          the angle between them in the plot.
        sizes: Size of nodes. If a list is given a different size is applied
          to each tree level.
        top_to_bottom: If true the root of the tree is at the top.
        with_labels: If true the names of each node are shown in the plot.
  """
  if "with_labels" in options:
    with_labels = options.pop("with_labels")
  else:
    with_labels = True
  plotter = TreePlotWrapper(root, colormap=colormap, options=options)
  node_color = [plotter.colors[n] for n in plotter.graph.node]
  node_size = [plotter.sizes[n] for n in plotter.graph.node]
  axes.axis("off")
  nx.draw_networkx(plotter.graph, plotter.pos, with_labels=with_labels,
                   labels=plotter.labels, ax=axes,
                   node_color=node_color, node_size=node_size)


class TreePlotWrapper(object):
  """Wraps tree-like TensorNetwork to NetworkX for plotting."""
  # TODO: Consider case with multiple edges between nodes.
  # Problem with the way networkx handles multiple edges between nodes.
  # Should also change nx.Graph() to nx.MultiGraph().

  def __init__(self,
               root: Union[tensornetwork.Node, List[tensornetwork.Node],
                                         List[List[tensornetwork.Node]]],
               colormap: Union[Text, List[Text]] = "red",
               options: Dict[Text, Union[float, int, List[int], bool]] = dict()
               ) -> None:
    """Creates NetworkX graph.

    Args:
      root: Root node.
      colormap: Color of nodes.
      options: Dictionary with plotting options.
        (see `draw_tree` for more details)
    """
    self.graph = nx.Graph()
    self.set_options(options)
    x_margin, y_margin = self.options["x_margin"], self.options["y_margin"]
    if isinstance(root, list) and isinstance(root[0], list):
      self.levels = root
    else:
      self.levels = self._levels_from_root(root)

    self.x_max = (max(len(level) for level in self.levels) + 1) * x_margin
    self.colormap = self._list_by_repetition(colormap, len(self.levels))
    self.sizemap = self._list_by_repetition(self.options["sizes"],
                                            len(self.levels))
    self.pos, self.labels = {}, {}
    self.colors, self.sizes = {}, {}

    # Ghost is a node connected to the open end of a dangling edge.
    # It is used for plotting dangling edges, as networkx edges should
    # be between two existing nodes. It is plotted in white color
    # and zero size so that it remains invisible.
    # `ghost_counter` = number of ghost nodes = number of dangling edges
    self.ghost_counter = 0
    self.node_set, self.edge_set = set(), set()

    y = 0
    for i, line in enumerate(self.levels):
      self._add_level_of_nodes(line, i, y)
      y += (1 - 2 * int(self.options["top_to_bottom"])) * y_margin

  def set_options(self, options: Dict[Text, Union[float, int,
                                      List[Optional[int]], bool]]) -> None:
    """Sets unspecified options to their default values.

    Args:
      options: Dictionary with plotting options.
    """
    self.options = {"x_margin": 1.0, "y_margin": 1.0,
                    "dangling_size": 0.8, "dangling_angle": 1.2,
                    "sizes": 1000, "top_to_bottom": True}
    for k in options.keys():
      if k in self.options:
        self.options[k] = options[k]
      else:
        raise KeyError("Unknown option.")

  @staticmethod
  def _list_by_repetition(input: Union[int, Text, List[Union[int, Text]]],
                          length: int) -> List[Union[int, Text]]:
    """Creates a list of specific length from an element or  another list."""
    if isinstance(input, list):
      return [input[i % len(input)] for i in range(length)]
    return length * [input]

  @staticmethod
  def _levels_from_root(root: Union[List, tensornetwork.Node]
                        ) -> List[List[tensornetwork.Node]]:
    """Finds tree levels via BFS.

    Args:
      root: Root or roots of the tree.

    Returns:
      levels: Lists with the nodes at each tree level from root to leaves.
        The order of
    """
    if isinstance(root, list):
      node_lists = [root[:]]
    elif isinstance(root, tensornetwork.Node):
      node_lists = [[root]]
    else:
      raise ValueError("Root must be either a Node or a list of Nodes.")
    queue = collections.deque(node_lists[0])
    marked = set(node_lists[0])
    while len(queue):
      node_lists.append([])
      for parent in node_lists[-2]:
        assert parent is queue.popleft()
        for edge in parent.edges:
          child = None
          if edge.node1 is not parent:
            child = edge.node1
          else:
            child = edge.node2
          if child is not None and child not in marked:
            marked.add(child)
            node_lists[-1].append(child)
            queue.append(child)
    return node_lists[:-1]

  def _add_node(self, node: Union[tensornetwork.Node, Text],
                x: float, y: float, name: Text,
                color: Text = "red", size: int = 1000) -> None:
    """Adds a node in the nx graph.

    Args:
      node: TensorNetwork node to add (or string for ghost nodes).
      x: Horizontal position of the node in the plot.
      y: Vertical position of the node in the plot.
      name: Label name. If None the name from TensorNetwork node is used.
      color: Node's color in the plot.
      size: Node's size in the plot.
    """
    self.graph.add_node(node)
    self.node_set.add(node)
    self.pos[node] = (x, y)
    self.labels[node] = name
    self.colors[node] = color
    self.sizes[node] = size

  def _add_dangling_edges(self, node: tensornetwork.Node, top: bool = True) -> None:
    """Adds dangling edges in the nx graph by creating ghost nodes.

    Args:
      node: TensorNetwork node that has dangling edges.
      top: If true the dangling edges are directed above the node in the plot.
        Otherwise they are directed below.
    """
    n_dangling = len(node.edges) - len(node.get_all_nondangling())
    sign = 2 * int(top) - 1
    size, angle = self.options["dangling_size"], self.options["dangling_angle"]
    phi = np.linspace(angle, np.pi - angle, n_dangling + 2)[1:-1]
    for i in range(n_dangling):
      x = self.pos[node][0] + sign * size * np.cos(phi[i])
      y = self.pos[node][1] + sign * size * np.sin(phi[i])
      ghost = "ghost{}".format(self.ghost_counter)
      self._add_node(ghost, x, y, name="", color="white", size=0)
      self.ghost_counter += 1
      self.graph.add_edge(node, ghost)

  def _add_level_of_nodes(self, level: List[tensornetwork.Node], level_ind: int,
                          y_coordinate: float) -> None:
    """Adds a complete level of nodes in the nx graph.

    Args:
      level: List with the nodes in the level to be added.
      level_ind: The level index (depth).
      y_coordinate: Vertical position of the level in the plot.
    """
    x_coords = np.linspace(0.0, self.x_max, len(level) + 2)[1:-1]
    for i, node in enumerate(level):
      self._add_node(node, x_coords[i], y_coordinate, name=node.name,
                     color=self.colormap[level_ind], size=self.sizemap[level_ind])
      flag = True
      for edge in node.edges:
        if edge.is_dangling() and flag:
          # this should be called once per node,
          # no matter how many dangling edges this node has
          flag = False
          if level_ind == 0:
            self._add_dangling_edges(node, top=self.options["top_to_bottom"])
          elif level_ind == len(self.levels) - 1:
            self._add_dangling_edges(node,
                                     top=not self.options["top_to_bottom"])
          else:
            raise NotImplementedError(
                "Only top or bottom nodes can have dangling edges.")
        elif edge not in self.edge_set:
          if edge.node1 in self.node_set and edge.node2 in self.node_set:
            self.graph.add_edge(edge.node1, edge.node2)
            self.edge_set.add(edge)
