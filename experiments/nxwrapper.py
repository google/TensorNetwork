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
of a tensor network. Works only for tensor networks that have a tree-like
structure: the nodes are arranged to levels and only consecutive levels share
edges. It currently has various restrictions, for example it supports dangling
edges only at the top or the bottom level (not in the middle).
"""
# TODO: More general wrapper with TensorNetwork object as input.

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import matplotlib
import networkx as nx
import numpy as np
import seaborn as sns
from typing import List, Optional, Union, Text
from tensornetwork import tensornetwork


def count_dangling(node: tensornetwork.Node) -> int:
  """Calculates the number of dangling edges of a node.

  Args:
    node: TensorNetwork node.

  Returns:
    n: Number of dangling edges this node has.
  """
  n = 0
  for edge in node.edges:
    if edge.is_dangling():
      n += 1
  return n


class PlotWrapper(object):
  """Wraps tree-like TensorNetwork to NetworkX for plotting."""
  # TODO: Consider case with multiple edges between nodes.
  # Problem with the way networkx handles multiple edges between nodes.
  # Should also change nx.Graph() to nx.MultiGraph().

  def __init__(self,
               node_lists: List[List[tensornetwork.Node]],
               x_margin: float = 1.0,
               y_margin: float = 1.0,
               dangling_size: float = 1.0,
               dangling_angle: float = 1.2,
               top_to_bottom: bool = True) -> None:
    """Creates NetworkX graph.

    Args:
      node_lists: List with TensorNetwork nodes groupped by level/depth.
      x_margin: Minimum horizontal distance between nodes in the plot.
      y_margin: Vertical distance between consecutive tree levels.
      dangling_size: Size (length) of dangling edges in the plot.
      dangling_angle: When a node has multiple dangling edges, this defines
        the angle between them in the plot.
      top_to_bottom: If true the plot is created from top to bottom from the
        given levels in node_lists.
    """
    self.graph = nx.Graph()
    self.levels = node_lists

    self.x_max = (max(len(level) for level in self.levels) + 1) * x_margin

    self.dangling_size, self.dangling_angle = dangling_size, dangling_angle
    # Default colormap
    self.colormap = sns.color_palette("deep")

    self.pos, self.labels = {}, {}
    self.colors, self.sizes = {}, {}

    # Ghost is a node connected to the open end of a dangling edge.
    # It is used for plotting dangling edges, as networkx edges should
    # be between two existing nodes. It is plotted in white color
    # and zero size so that it remains invisible.
    # `ghost_counter` = number of ghost nodes = number of dangling edges
    self.ghost_counter = 0
    self.node_set, self.edge_set = set(), set()

    if top_to_bottom:
      y = (len(node_lists) - 1) * y_margin
    else:
      y = 0
    for i, line in enumerate(node_lists):
      self._add_level_of_nodes(line, i, y)
      y += (1 - 2 * int(top_to_bottom)) * y_margin

  def draw(self, axes: matplotlib.axes.Axes,
           with_labels: bool = True) -> None:
    """Plots the network graph.

    Args:
      axes: Axes object to use for plotting the tensor network.
      with_labels: If True, plot node names on nodes.
    """
    node_color, node_size = [], []
    for n in self.graph.node:
      node_color.append(self.colors[n])
      node_size.append(self.sizes[n])
    axes.axis("off")
    nx.draw_networkx(self.graph, self.pos, with_labels=with_labels,
                     labels=self.labels, ax=axes,
                     node_color=node_color, node_size=node_size)

  def set_level_colors(self, color_list: List[Optional[Text]]) -> None:
    """Sets node color at each level.

    Args:
      color_list: List with color of each layer.
        If the list contains None, the default value is used for this level.
    """
    for i, c in enumerate(color_list):
      if c is not None:
        for n in self.levels[i]:
          self.colors[n] = c

  def set_level_sizes(self, size_list: List[Optional[int]]) -> None:
    """Sets node size at each level.

    Args:
      size_list: List with size of each layer.
        If the list contains None, the default value is used for this level.
    """
    for i, c in enumerate(size_list):
      if c is not None:
        for n in self.levels[i]:
          self.sizes[n] = c

  def _add_node(self, node: Union[tensornetwork.Node, Text],
                x: float, y: float, name: Text,
                color: Text = "red", size: int = 1000) -> None:
    """Adds a node in the nx graph.

    Args:
      node: TensorNetwork node to add (or string for ghost nodes).
      x: Horizontal position of the node in the plot.
      y: Vertical position of the node in the plot.
      name: Label name. If None the name from TN node is used.
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
    n_dangling = count_dangling(node)

    sign = 2 * int(top) - 1
    phi = np.linspace(self.dangling_angle,
                      np.pi - self.dangling_angle,
                      n_dangling + 2)[1:-1]
    for i in range(n_dangling):
      x = self.pos[node][0] + sign * self.dangling_size * np.cos(phi[i])
      y = self.pos[node][1] + sign * self.dangling_size * np.sin(phi[i])
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
                     color=self.colormap[level_ind])
      flag = True
      for edge in node.edges:
        if edge.is_dangling() and flag:
          # this should be called once per node,
          # no matter how many dangling edges this node has
          flag = False
          if level_ind == 0:
            self._add_dangling_edges(node)
          elif level_ind == len(self.levels) - 1:
            self._add_dangling_edges(node, top=False)
          else:
            raise NotImplementedError(
                "Only top or bottom nodes can have dangling edges.")
        elif edge not in self.edge_set:
          if edge.node1 in self.node_set and edge.node2 in self.node_set:
            self.graph.add_edge(edge.node1, edge.node2)
            self.edge_set.add(edge)
