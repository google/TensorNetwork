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

import graphviz
from typing import Optional, Text, Iterable
from tensornetwork.network_components import AbstractNode


#pylint: disable=no-member
def to_graphviz(nodes: Iterable[AbstractNode],
                graph: Optional[graphviz.Graph] = None,
                include_all_names: bool = False,
                engine: Text = "neato") -> graphviz.Graph:
  """Create a graphviz Graph that is isomorphic to the given TensorNetwork.

  Args:
    nodes: a collection of nodes
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
    #pylint: disable=no-member
    graph = graphviz.Graph('G', engine=engine)
  for node in nodes:
    if not node.name.startswith("__") or include_all_names:
      label = node.name
    else:
      label = ""
    graph.node(str(id(node)), label=label)
  seen_edges = set()
  for node in nodes:
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
            "{}_{}".format(id(node), i),
            label="",
            _attributes={"style": "invis"})
        graph.edge("{}_{}".format(id(node), i), str(id(node)), label=edge_label)
      else:
        graph.edge(str(id(edge.node1)), str(id(edge.node2)), label=edge_label)
  return graph
