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
"""Implementation of TensorNetwork structure."""

import collections
from typing import (Any, Dict, List, Optional, Set, Text, Tuple, Union,
                    Sequence, Iterable, Type)
import numpy as np
import json

#pylint: disable=useless-import-alias
from tensornetwork.network_components import (AbstractNode, Node, CopyNode,
                                              Edge, disconnect,
                                              outer_product_final_nodes)
from tensornetwork.backends import backend_factory
from tensornetwork.backends.abstract_backend import AbstractBackend
from tensornetwork.network_components import connect, contract_parallel
Tensor = Any


def copy(nodes: Iterable[AbstractNode],
         conjugate: bool = False) -> Tuple[dict, dict]:
  """Copy the given nodes and their edges.

  This will return a tuple linking original nodes/edges to their copies.
  If nodes A and B are connected but only A is passed in to be
  copied, the edge between them will become a dangling edge.

  Args:
    nodes: An Iterable (Usually a `list` or `set`) of `nodes`.
    conjugate: Boolean. Whether to conjugate all of the nodes
      (useful for calculating norms and reduced density matrices).

  Returns:
    A tuple containing:
      node_dict:
        A dictionary mapping the nodes to their copies.
      edge_dict:
        A dictionary mapping the edges to their copies.
  """
  node_dict = {}
  for node in nodes:
    node_dict[node] = node.copy(conjugate)
  edge_dict = {}
  for edge in get_all_edges(nodes):
    node1 = edge.node1
    axis1 = edge.node1.get_axis_number(edge.axis1)
    # edge dangling or node2 does not need to be copied
    if edge.is_dangling() or edge.node2 not in node_dict:
      new_edge = Edge(node_dict[node1], axis1, edge.name)
      node_dict[node1].add_edge(new_edge, axis1)
      edge_dict[edge] = new_edge
      continue

    node2 = edge.node2
    axis2 = edge.node2.get_axis_number(edge.axis2)
    # copy node2 but not node1
    if node1 not in node_dict:
      new_edge = Edge(node_dict[node2], axis2, edge.name)
      node_dict[node2].add_edge(new_edge, axis2)
      edge_dict[edge] = new_edge
      continue

    # both nodes should be copied
    new_edge = Edge(node_dict[node1], axis1, edge.name, node_dict[node2], axis2)
    if not edge.is_trace():
      node_dict[node2].add_edge(new_edge, axis2)
      node_dict[node1].add_edge(new_edge, axis1)

    edge_dict[edge] = new_edge

  return node_dict, edge_dict


def replicate_nodes(nodes: Iterable[AbstractNode],
                    conjugate: bool = False) -> List[AbstractNode]:
  """Copy the given nodes and their edges.

  If nodes A and B are connected but only A is passed in to be
  copied, the edge between them will become a dangling edge.

  Args:
    nodes: An `Iterable` (Usually a `List` or `Set`) of `Nodes`.
    conjugate: Boolean. Whether to conjugate all of the nodes
        (useful for calculating norms and reduced density
        matrices).

  Returns:
    A list containing the copies of the nodes.
  """
  new_nodes, _ = copy(nodes, conjugate=conjugate)
  return [new_nodes[node] for node in nodes]


def remove_node(node: AbstractNode) -> Tuple[Dict[Text, Edge], Dict[int, Edge]]:
  """Remove a node from the network.

  Args:
    node: The node to be removed.

  Returns:
    A tuple of:
      disconnected_edges_by_name:
        A Dictionary mapping `node`'s axis names to the newly broken edges.
      disconnected_edges_by_axis:
        A Dictionary mapping `node`'s axis numbers to the newly broken edges.
  """
  disconnected_edges_by_name = {}
  disconnected_edges_by_axis = {}
  for i, name in enumerate(node.axis_names):
    if not node[i].is_dangling() and not node[i].is_trace():
      edge1, edge2 = disconnect(node[i])
      new_disconnected_edge = edge1 if edge1.node1 is not node else edge2
      disconnected_edges_by_axis[i] = new_disconnected_edge
      disconnected_edges_by_name[name] = new_disconnected_edge
  return disconnected_edges_by_name, disconnected_edges_by_axis


def split_node(
    node: AbstractNode,
    left_edges: List[Edge],
    right_edges: List[Edge],
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    relative: Optional[bool] = False,
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[AbstractNode, AbstractNode, Tensor]:
  """Split a `node` using Singular Value Decomposition.

  Let :math:`M` be the matrix created by flattening `left_edges` and 
  `right_edges` into 2 axes. 
  Let :math:`U S V^* = M` be the SVD of :math:`M`. 
  This will split the network into 2 nodes. 
  The left node's tensor will be :math:`U \\sqrt{S}` 
  and the right node's tensor will be
  :math:`\\sqrt{S} V^*` where :math:`V^*` is the adjoint of :math:`V`.

  The singular value decomposition is truncated if `max_singular_values` or
  `max_truncation_err` is not `None`.

  The truncation error is the 2-norm of the vector of truncated singular
  values. If only `max_truncation_err` is set, as many singular values will
  be truncated as possible while maintaining:
  `norm(truncated_singular_values) <= max_truncation_err`.
  If `relative` is set `True` then `max_truncation_err` is understood
  relative to the largest singular value.

  If only `max_singular_values` is set, the number of singular values kept
  will be `min(max_singular_values, number_of_singular_values)`, so that
  `max(0, number_of_singular_values - max_singular_values)` are truncated.

  If both `max_truncation_err` and `max_singular_values` are set,
  `max_singular_values` takes priority: The truncation error may be larger
  than `max_truncation_err` if required to satisfy `max_singular_values`.

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    max_singular_values: The maximum number of singular values to keep.
    max_truncation_err: The maximum allowed truncation error.
    relative: Multiply `max_truncation_err` with the largest singular value.
    left_name: The name of the new left node. If `None`, a name will be 
      generated automatically.
    right_name: The name of the new right node. If `None`, a name will be
      generated automatically.
    edge_name: The name of the new `Edge` connecting the new left and
      right node. If `None`, a name will be generated automatically.
      The new axis will get the same name as the edge.

  Returns:
    A tuple containing:
      left_node:
        A new node created that connects to all of the `left_edges`.
        Its underlying tensor is :math:`U \\sqrt{S}`
      right_node:
        A new node created that connects to all of the `right_edges`.
        Its underlying tensor is :math:`\\sqrt{S} V^*`
      truncated_singular_values:
        The vector of truncated singular values.
  Raises:
    AttributeError: If `node` has no backend attribute
  """

  if not hasattr(node, 'backend'):
    raise AttributeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  if node.axis_names and edge_name:
    left_axis_names = []
    right_axis_names = [edge_name]
    for edge in left_edges:
      left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                             else node.axis_names[edge.axis2])
    for edge in right_edges:
      right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                              else node.axis_names[edge.axis2])
    left_axis_names.append(edge_name)
  else:
    left_axis_names = None
    right_axis_names = None

  backend = node.backend
  transp_tensor = node.tensor_from_edge_order(left_edges + right_edges)

  u, s, vh, trun_vals = backend.svd(transp_tensor,
                                    len(left_edges),
                                    max_singular_values,
                                    max_truncation_err,
                                    relative=relative)
  sqrt_s = backend.sqrt(s)
  u_s = backend.broadcast_right_multiplication(u, sqrt_s)
  vh_s = backend.broadcast_left_multiplication(sqrt_s, vh)

  left_node = Node(u_s,
                   name=left_name,
                   axis_names=left_axis_names,
                   backend=backend)

  left_axes_order = [
      edge.axis1 if edge.node1 is node else edge.axis2 for edge in left_edges
  ]
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(left_axes_order[i], node, i, left_node)

  right_node = Node(vh_s,
                    name=right_name,
                    axis_names=right_axis_names,
                    backend=backend)

  right_axes_order = [
      edge.axis1 if edge.node1 is node else edge.axis2 for edge in right_edges
  ]
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(right_axes_order[i], node, i + 1, right_node)

  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  node.fresh_edges(node.axis_names)
  return left_node, right_node, trun_vals


def split_node_qr(
    node: AbstractNode,
    left_edges: List[Edge],
    right_edges: List[Edge],
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[AbstractNode, AbstractNode]:
  """Split a `node` using QR decomposition.

  Let :math:`M` be the matrix created by 
  flattening `left_edges` and `right_edges` into 2 axes. 
  Let :math:`QR = M` be the QR Decomposition of :math:`M`.
  This will split the network into 2 nodes.
  The `left node`'s tensor will be :math:`Q` (an orthonormal matrix)
  and the `right node`'s tensor will be :math:`R` (an upper triangular matrix)

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    left_name: The name of the new left node. If `None`, a name will be
      generated automatically.
    right_name: The name of the new right node. If `None`, a name will be
      generated automatically.
    edge_name: The name of the new `Edge` connecting the new left and right
      node. If `None`, a name will be generated automatically.

  Returns:
    A tuple containing:
      left_node:
        A new node created that connects to all of the `left_edges`.
        Its underlying tensor is :math:`Q`
      right_node:
        A new node created that connects to all of the `right_edges`.
        Its underlying tensor is :math:`R`
  Raises:
    AttributeError: If `node` has no backend attribute
  """

  if not hasattr(node, 'backend'):
    raise AttributeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  if node.axis_names and edge_name:
    left_axis_names = []
    right_axis_names = [edge_name]
    for edge in left_edges:
      left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                             else node.axis_names[edge.axis2])
    for edge in right_edges:
      right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                              else node.axis_names[edge.axis2])
    left_axis_names.append(edge_name)
  else:
    left_axis_names = None
    right_axis_names = None

  backend = node.backend
  transp_tensor = node.tensor_from_edge_order(left_edges + right_edges)

  q, r = backend.qr(transp_tensor, len(left_edges))
  left_node = Node(q,
                   name=left_name,
                   axis_names=left_axis_names,
                   backend=backend)

  left_axes_order = [
      edge.axis1 if edge.node1 is node else edge.axis2 for edge in left_edges
  ]
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(left_axes_order[i], node, i, left_node)

  right_node = Node(r,
                    name=right_name,
                    axis_names=right_axis_names,
                    backend=backend)

  right_axes_order = [
      edge.axis1 if edge.node1 is node else edge.axis2 for edge in right_edges
  ]
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(right_axes_order[i], node, i + 1, right_node)

  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  node.fresh_edges(node.axis_names)

  return left_node, right_node


def split_node_rq(
    node: AbstractNode,
    left_edges: List[Edge],
    right_edges: List[Edge],
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[AbstractNode, AbstractNode]:
  """Split a `node` using RQ (reversed QR) decomposition.

  Let :math:`M` be the matrix created by 
  flattening `left_edges` and `right_edges` into 2 axes. 

  Let :math:`QR = M^*` be the QR Decomposition of :math:`M^*`. 
  This will split the network into 2 nodes. 

  The left node's tensor will be :math:`R^*` (a lower triangular matrix) 
  and the right node's tensor will be :math:`Q^*` (an orthonormal matrix)

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    left_name: The name of the new left node. If `None`, a name will be
      generated automatically.
    right_name: The name of the new right node. If `None`, a name will be
      generated automatically.
    edge_name: The name of the new `Edge` connecting the new left and
      right node. If `None`, a name will be generated automatically.

  Returns:
    A tuple containing:
      left_node:
        A new node that connects to all of the `left_edges`.
        Its underlying tensor is :math:`R^*`
      right_node:
        A new node that connects to all of the `right_edges`.
        Its underlying tensor is :math:`Q^*`

  Raises:
    AttributeError: If `node` has no backend attribute
  """

  if not hasattr(node, 'backend'):
    raise AttributeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  if node.axis_names and edge_name:
    left_axis_names = []
    right_axis_names = [edge_name]
    for edge in left_edges:
      left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                             else node.axis_names[edge.axis2])
    for edge in right_edges:
      right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                              else node.axis_names[edge.axis2])
    left_axis_names.append(edge_name)
  else:
    left_axis_names = None
    right_axis_names = None
  backend = node.backend
  transp_tensor = node.tensor_from_edge_order(left_edges + right_edges)

  r, q = backend.rq(transp_tensor, len(left_edges))
  left_node = Node(r,
                   name=left_name,
                   axis_names=left_axis_names,
                   backend=backend)

  left_axes_order = [
      edge.axis1 if edge.node1 is node else edge.axis2 for edge in left_edges
  ]
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(left_axes_order[i], node, i, left_node)

  right_node = Node(q,
                    name=right_name,
                    axis_names=right_axis_names,
                    backend=backend)

  right_axes_order = [
      edge.axis1 if edge.node1 is node else edge.axis2 for edge in right_edges
  ]

  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(right_axes_order[i], node, i + 1, right_node)

  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  node.fresh_edges(node.axis_names)
  return left_node, right_node


def split_node_full_svd(
    node: AbstractNode,
    left_edges: List[Edge],
    right_edges: List[Edge],
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    relative: Optional[bool] = False,
    left_name: Optional[Text] = None,
    middle_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    left_edge_name: Optional[Text] = None,
    right_edge_name: Optional[Text] = None,
) -> Tuple[AbstractNode, AbstractNode, AbstractNode, Tensor]:
  """Split a node by doing a full singular value decomposition.

  Let :math:`M` be the matrix created by 
  flattening `left_edges` and `right_edges` into 2 axes. 
  Let :math:`U S V^* = M` be the Singular Value Decomposition of :math:`M`.

  The left most node will be :math:`U` tensor of the SVD, the middle node is
  the diagonal matrix of the singular values, ordered largest to smallest,
  and the right most node will be the :math:`V*` tensor of the SVD.

  The singular value decomposition is truncated if `max_singular_values` or
  `max_truncation_err` is not `None`.

  The truncation error is the 2-norm of the vector of truncated singular
  values. If only `max_truncation_err` is set, as many singular values will
  be truncated as possible while maintaining:
  `norm(truncated_singular_values) <= max_truncation_err`.
  If `relative` is set `True` then `max_truncation_err` is understood
  relative to the largest singular value.

  If only `max_singular_values` is set, the number of singular values kept
  will be `min(max_singular_values, number_of_singular_values)`, so that
  `max(0, number_of_singular_values - max_singular_values)` are truncated.

  If both `max_truncation_err` and `max_singular_values` are set,
  `max_singular_values` takes priority: The truncation error may be larger
  than `max_truncation_err` if required to satisfy `max_singular_values`.

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    max_singular_values: The maximum number of singular values to keep.
    max_truncation_err: The maximum allowed truncation error.
    relative: Multiply `max_truncation_err` with the largest singular value.
    left_name: The name of the new left node. If None, a name will be 
      generated automatically.
    middle_name: The name of the new center node. If `None`, a name will be
      generated automatically.
    right_name: The name of the new right node. If `None`, a name will be
      generated automatically.
    left_edge_name: The name of the new left `Edge` connecting
      the new left node (:math:`U`) and the new central node (:math:`S`).
      If `None`, a name will be generated automatically.
    right_edge_name: The name of the new right `Edge` connecting
      the new central node (:math:`S`) and the new right node (:math:`V*`).
      If `None`, a name will be generated automatically.

  Returns:
    A tuple containing:
      left_node:
        A new node created that connects to all of the `left_edges`.
        Its underlying tensor is :math:`U`
      singular_values_node:
        A new node that has 2 edges connecting `left_node` and `right_node`.
        Its underlying tensor is :math:`S`
      right_node:
        A new node created that connects to all of the `right_edges`.
        Its underlying tensor is :math:`V^*`
      truncated_singular_values:
        The vector of truncated singular values.

  Raises:
    AttributeError: If `node` has no backend attribute
  """

  if not hasattr(node, 'backend'):
    raise AttributeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  if node.axis_names and left_edge_name and right_edge_name:
    left_axis_names = []
    right_axis_names = [right_edge_name]
    for edge in left_edges:
      left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                             else node.axis_names[edge.axis2])
    for edge in right_edges:
      right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                              else node.axis_names[edge.axis2])
    left_axis_names.append(left_edge_name)
    center_axis_names = [left_edge_name, right_edge_name]
  else:
    left_axis_names = None
    center_axis_names = None
    right_axis_names = None

  backend = node.backend
  transp_tensor = node.tensor_from_edge_order(left_edges + right_edges)

  u, s, vh, trun_vals = backend.svd(transp_tensor,
                                    len(left_edges),
                                    max_singular_values,
                                    max_truncation_err,
                                    relative=relative)
  left_node = Node(u,
                   name=left_name,
                   axis_names=left_axis_names,
                   backend=backend)
  singular_values_node = Node(backend.diagflat(s),
                              name=middle_name,
                              axis_names=center_axis_names,
                              backend=backend)

  right_node = Node(vh,
                    name=right_name,
                    axis_names=right_axis_names,
                    backend=backend)

  left_axes_order = [
      edge.axis1 if edge.node1 is node else edge.axis2 for edge in left_edges
  ]
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(left_axes_order[i], node, i, left_node)

  right_axes_order = [
      edge.axis1 if edge.node1 is node else edge.axis2 for edge in right_edges
  ]
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(right_axes_order[i], node, i + 1, right_node)
  connect(left_node.edges[-1],
          singular_values_node.edges[0],
          name=left_edge_name)
  connect(singular_values_node.edges[1],
          right_node.edges[0],
          name=right_edge_name)
  node.fresh_edges(node.axis_names)
  return left_node, singular_values_node, right_node, trun_vals


def _reachable(nodes: Set[AbstractNode]) -> Set[AbstractNode]:
  if not nodes:
    raise ValueError("Reachable requires at least 1 node.")
  node_que = collections.deque(nodes)
  seen_nodes = set()
  while node_que:
    node = node_que.popleft()
    if node not in seen_nodes:
      seen_nodes.add(node)
    for e in node.edges:
      for n in e.get_nodes():
        if n is not None and n not in seen_nodes:
          node_que.append(n)
          seen_nodes.add(n)
  return seen_nodes


def reachable(
    inputs: Union[AbstractNode, Iterable[AbstractNode], Edge, Iterable[Edge]]
) -> Set[AbstractNode]:
  """Computes all nodes reachable from `node` or `edge.node1` by connected
  edges.

  Args:
    inputs: A `AbstractNode`/`Edge` or collection of `AbstractNodes`/`Edges`
  Returns:
    A set of `AbstractNode` objects that can be reached from `node`
    via connected edges.
  Raises:
    ValueError: If an unknown value for `strategy` is passed.
  """

  if isinstance(inputs, AbstractNode):
    inputs = {inputs}
  elif isinstance(inputs, Edge):
    inputs = {inputs.node1}  # pytype: disable=attribute-error
  elif isinstance(inputs, list) and all(isinstance(x, Edge) for x in inputs):
    inputs = {x.node1 for x in inputs}
  return _reachable(set(inputs))


def check_correct(nodes: Iterable[AbstractNode],
                  check_connections: Optional[bool] = True) -> None:
  """Check if the network defined by `nodes` fulfills necessary consistency
  relations.

  Args:
    nodes: A list of `AbstractNode` objects.
    check_connections: Check if the network is connected.

  Returns:
    `None`

  Raises:
    ValueError: If the network defined by `nodes` is not
      correctly structured.
  """
  for node in nodes:
    for i, edge in enumerate(node.edges):
      if edge.node1 is not node and edge.node2 is not node:
        raise ValueError("Edge '{}' does not connect to node '{}'."
                         "Edge's nodes: '{}', '{}'.".format(
                             edge, node, edge.node1, edge.node2))

      is_edge_node_consistent = False
      if edge.node1 is node:
        if edge.axis1 == i:
          is_edge_node_consistent = True
      if edge.node2 is node:
        if edge.axis2 == i:
          is_edge_node_consistent = True
      if not is_edge_node_consistent:
        raise ValueError(
            "Edge '{}' does not point to '{}' on the correct axis. "
            "Edge axes: {}, {}. Node axis: {}.".format(edge, node, edge.axis1,
                                                       edge.axis2, i))
  if check_connections:
    check_connected(nodes)


def check_connected(nodes: Iterable[AbstractNode]) -> None:
  """Check if all nodes in `nodes` are connected.

  Args:
    nodes: A list of `nodes`.
    
  Returns:
    `None`

  Raises:
    ValueError: If not all nodes in `nodes` are connected.
  """
  nodes = list(nodes)
  if not set(nodes) <= reachable([nodes[0]]):
    raise ValueError("Non-connected graph")


def get_all_nodes(edges: Iterable[Edge]) -> Set[AbstractNode]:
  """Return the set of nodes connected to edges."""
  nodes = set()
  for edge in edges:
    if edge.node1 is not None:
      nodes |= {edge.node1}
    if edge.node2 is not None:
      nodes |= {edge.node2}

  return nodes


def get_all_edges(nodes: Iterable[AbstractNode]) -> Set[Edge]:
  """Return the set of edges of all nodes."""
  edges = set()
  for node in nodes:
    edges |= set(node.edges)
  return edges


def get_subgraph_dangling(nodes: Iterable[AbstractNode]) -> Set[Edge]:
  """Get all of the edges that are "relatively dangling" to the given nodes.

  A "relatively dangling" edge is an edge that is either actually dangling
  or is connected to another node that is outside of the given collection
  of `nodes`.

  Args:
    nodes: A set of nodes.

  Returns:
    The set of "relatively dangling" edges.
  """
  output = set()
  for edge in get_all_edges(nodes):
    if edge.is_dangling() or not set(edge.get_nodes()) <= set(nodes):
      output.add(edge)
  return output


def contract_trace_edges(node: AbstractNode) -> AbstractNode:
  """contract all trace edges of `node`.

  Args:
    node: A `AbstractNode` object.

  Returns:
    A new `AbstractNode` obtained from contracting all trace edges.

  """
  res = node
  for edge in res.edges:
    if edge.is_trace():
      res = contract_parallel(edge)
      break
  return res

def reduced_density(traced_out_edges: Iterable[Edge]) -> Tuple[dict, dict]:
  """Constructs the tensor network for a reduced density matrix, if it is pure.

  The tensor network connected to `traced_out_edges` is assumed to be a pure
  quantum state (a state vector). This modifies the network so that it
  describes the reduced density matrix obtained by "tracing out" the specified
  edges.

  This is done by making a conjugate copy of the original network and
  connecting each edge in `traced_out_edges` with its conjugate counterpart.

  The edges in `edge_dict` corresponding to `traced_out_edges` will be the
  new non-dangling edges connecting the state with its conjugate.

  Args:
    traced_out_edges: A list of dangling edges.

  Returns:
    A tuple containing:
      node_dict: A dictionary mapping the nodes in the original network to
        their conjugate copies.
      edge_dict: A dictionary mapping edges in the original network to their
        conjugate copies.
  """

  if list(filter(lambda x: not x.is_dangling(), traced_out_edges)):
    raise ValueError("traced_out_edges must only include dangling edges!")

  # Get all reachable nodes.
  old_nodes = reachable(get_all_nodes(traced_out_edges))

  # Copy and conjugate all reachable nodes.
  node_dict, edge_dict = copy(old_nodes, True)
  for t_edge in traced_out_edges:
    # Add each edge to the copied nodes as new edge.
    edge_dict[t_edge] = edge_dict[t_edge] ^ t_edge

  return node_dict, edge_dict


def switch_backend(nodes: Iterable[AbstractNode], new_backend: Text) -> None:
  """Change the backend of the nodes.

  This will convert all `node`'s tensors to the `new_backend`'s Tensor type.

  Args:
    nodes: iterable of nodes
    new_backend (str): The new backend.
    dtype (datatype): The dtype of the backend. If `None`,
      a defautl dtype according to config.py will be chosen.

  Returns:
    None
  """
  if new_backend == 'symmetric':
    if np.all([n.backend.name == 'symmetric' for n in nodes]):
      return
    raise ValueError("switching to `symmetric` backend not possible")

  backend = backend_factory.get_backend(new_backend)
  for node in nodes:
    if node.backend.name != "numpy":
      raise NotImplementedError("Can only switch backends when the current "
                                "backend is 'numpy'. Current backend "
                                "is '{}'".format(node.backend))
    node.tensor = backend.convert_to_tensor(node.tensor)
    node.backend = backend


def get_neighbors(node: AbstractNode) -> List[AbstractNode]:
  """Get all of the neighbors that are directly connected to the given node.

  Note: `node` will never be in the returned list, even if `node` has a
  trace edge.

  Args:
    node: A node.

  Returns:
    All of the neighboring edges that share an `Edge` with `node`.
  """
  neighbors = []
  neighbors_set = set()
  for edge in node.edges:
    if not edge.is_dangling() and not edge.is_trace():
      if edge.node1 is node:
        if edge.node2 not in neighbors_set:
          neighbors.append(edge.node2)
          neighbors_set.add(edge.node2)
      elif edge.node1 not in neighbors_set:
        neighbors.append(edge.node1)
        neighbors_set.add(edge.node1)
  return neighbors


def _build_serial_binding(
    edge_binding: Dict[str, Union[Edge, Iterable[Edge]]],
    edge_id_dict: Dict[Edge, int]) -> Dict[str, Iterable[int]]:
  if not edge_binding or not edge_id_dict:
    return {}
  serial_edge_binding = {}
  for k, v in edge_binding.items():
    if not isinstance(k, str):
      raise TypeError(f'Binding keys must be of type string, not {type(k)}')

    binding_list = []
    # pylint: disable=isinstance-second-argument-not-valid-type
    if isinstance(v, Iterable):
      for e in v:
        if not isinstance(e, Edge):
          raise TypeError(
              'Binding elements must be Edges or iterables of Edges')
        e_id = edge_id_dict.get(e)
        if e_id is not None:
          binding_list.append(e_id)
    else:
      if not isinstance(v, Edge):
        raise TypeError('Binding elements must be Edges or iterables of Edges')
      e_id = edge_id_dict.get(v)
      if e_id is not None:
        binding_list.append(e_id)
    if binding_list:
      serial_edge_binding[k] = tuple(binding_list)
  return serial_edge_binding


def nodes_to_json(nodes: List[AbstractNode],
                  edge_binding: Optional[Dict[str, Union[Edge, Iterable[Edge]]]]
                  = None) -> str:
  """
  Create a JSON string representing the Tensor Network made up of the given 
  nodes. Nodes and their attributes, edges and their attributes and tensor
  values are included.
  
  Tensors are serialized according the the format used by each tensors backend.
  
  For edges spanning included nodes and excluded nodes the edge attributes are 
  preserved in the serialization but the connection to the excluded node is 
  dropped. The original edge is not modified.
  
  Args:
    nodes: A list of nodes making up a tensor network.
    edge_binding: A dictionary containing {str->edge} bindings. Edges that are 
      not included in the serialized network are ommited from the dictionary.
    
  Returns:
    A string representing the JSON serialized tensor network.
    
  Raises:
    TypeError: If an edge_binding dict is passed with non string keys, or non 
      Edge values.
  """
  network_dict = {
      'nodes': [],
      'edges': [],
  }
  node_id_dict = {}
  edge_id_dict = {}

  # Build serialized Nodes
  for i, node in enumerate(nodes):
    node_id_dict[node] = i
    network_dict['nodes'].append({
        'id': i,
        'attributes': node.to_serial_dict(),
    })
  edges = get_all_edges(nodes)

  # Build serialized edges
  for i, edge in enumerate(edges):
    edge_id_dict[edge] = i
    node_ids = [node_id_dict.get(n) for n in edge.get_nodes()]
    attributes = edge.to_serial_dict()
    attributes['axes'] = [
        a if node_ids[j] is not None else None
        for j, a in enumerate(attributes['axes'])
    ]
    edge_dict = {
        'id': i,
        'node_ids': node_ids,
        'attributes': attributes,
    }
    network_dict['edges'].append(edge_dict)

  serial_edge_binding = _build_serial_binding(edge_binding, edge_id_dict)
  if serial_edge_binding:
    network_dict['edge_binding'] = serial_edge_binding
  return json.dumps(network_dict)


def nodes_from_json(json_str: str) -> Tuple[List[AbstractNode],
                                            Dict[str, Tuple[Edge]]]:
  """
  Create a tensor network from a JSON string representation of a tensor network.
  
  Args:
    json_str: A string representing a JSON serialized tensor network.
    
  Returns:
    A list of nodes making up the tensor network.
    A dictionary of {str -> (edge,)} bindings. All dictionary values are tuples
      of Edges.
    
  """
  network_dict = json.loads(json_str)
  nodes = []
  node_ids = {}
  edge_lookup = {}
  edge_binding = {}
  for n in network_dict['nodes']:
    node = Node.from_serial_dict(n['attributes'])
    nodes.append(node)
    node_ids[n['id']] = node
  for e in network_dict['edges']:
    e_nodes = [node_ids.get(n_id) for n_id in e['node_ids']]
    axes = e['attributes']['axes']
    edge = Edge(node1=e_nodes[0],
                axis1=axes[0],
                node2=e_nodes[1],
                axis2=axes[1],
                name=e['attributes']['name'])
    edge_lookup[e['id']] = edge
    for node, axis in zip(e_nodes, axes):
      if node is not None:
        node.add_edge(edge, axis, override=True)
  for k, v in network_dict.get('edge_binding', {}).items():
    for e_id in v:
      edge_binding[k] = edge_binding.get(k, ()) + (edge_lookup[e_id],)

  return nodes, edge_binding


def redirect_edge(edge: Edge, new_node: AbstractNode,
                  old_node: AbstractNode) -> None:
  """
  Redirect `edge` from `old_node` to `new_node`.
  Routine updates `new_node` and `old_node`.
  `edge` is added to `new_node`, `old_node` gets a
  new Edge instead of `edge`.

  Args:
    edge: An Edge.
    new_node: The new `Node` object.
    old_node: The old `Node` object.

  Returns:
    None

  Raises:
    ValueError: if `edge` does not point to `old_node`.
  """
  if not edge.is_trace():
    if edge.is_dangling():
      if edge.node1 is not old_node:
        raise ValueError(f"edge {edge} is not pointing "
                         f"to old_node {old_node}")
      edge.node1 = new_node
      axis = edge.axis1
    else:
      if edge.node1 is old_node:
        edge.node1 = new_node
        axis = edge.axis1
      elif edge.node2 is old_node:
        edge.node2 = new_node
        axis = edge.axis2
      else:
        raise ValueError(f"edge {edge} is not pointing "
                         f"to old_node {old_node}")
    new_node.add_edge(edge, axis, True)
    new_edge = Edge(old_node, axis)
    old_node.add_edge(new_edge, axis, True)
  else:
    if edge.node1 is not old_node:
      raise ValueError(f"edge {edge} is not pointing "
                       f"to old_node {old_node}")
    edge.node1 = new_node
    edge.node2 = new_node
    axis1 = edge.axis1
    axis2 = edge.axis2
    new_node.add_edge(edge, axis1, True)
    new_node.add_edge(edge, axis2, True)
    new_edge = Edge(old_node, axis1, None, old_node, axis2)
    old_node.add_edge(new_edge, axis1, True)
    old_node.add_edge(new_edge, axis2, True)
