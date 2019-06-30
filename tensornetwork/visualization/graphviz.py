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
"""Implementation of TensorNetwork Graphviz visualization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import graphviz
from typing import Optional, Text
from tensornetwork import network


def to_graphviz(
    net: network.TensorNetwork, 
    graph: Optional[graphviz.Graph] = None,
    include_all_names: bool = False,
    engine: Text = "neato"
  ) -> graphviz.Graph:
  """Create a graphviz Graph that is isomorphic to the given TensorNetwork.

  Args:
    net: A `TensorNetwork`.
    graph: An optional `graphviz.Graph` object to write to. Use this only
      if you wish to set custom attributes for the graph.
    include_all_names: Whether to include all of the names in the graph.
      If False, all names starting with '__' (which are almost always just
      the default generated names) will be dropped to reduce clutter.
    engine: The graphviz engine to use. Only applicable if `graph` is None.

  Returns:
    The `graphviz.Graph` object.
  """
  if graph is None:
    graph = graphviz.Graph('G', engine=engine)
  for node in net.nodes_set:
    if not node.name.startswith("__") or include_all_names:
      label = node.name
    else:
      label = ""
    graph.node(str(node.signature), label=label)
  seen_edges = set()
  for node in net.nodes_set:
    for i, edge in enumerate(node.edges):
      if edge in seen_edges:
        continue
      seen_edges.add(edge)
      if not edge.name.startswith("__") or include_all_names:
        edge_label = edge.name
      else:
        edge_label = ""
      if edge.is_dangling():
        # We need to create an invisible node for the dangling edge 
        # to connect to.
        graph.node(
            "{}_{}".format(node.signature, i), 
            label="",
             _attributes={"style":"invis"})
        graph.edge(
            "{}_{}".format(node.signature, i), 
            str(node.signature),
            label=edge_label)
      else:
        graph.edge(
          str(edge.node1.signature),
          str(edge.node2.signature),
          label=edge_label)
  return graph
