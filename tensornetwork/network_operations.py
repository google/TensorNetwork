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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union, \
    Sequence
import numpy as np

#pylint: disable=useless-import-alias
import tensornetwork.config as config
from tensornetwork.network_components import BaseNode, Node, CopyNode, Edge
from tensornetwork.backends import backend_factory
from tensornetwork.backends.base_backend import BaseBackend
from tensornetwork.network_components import connect
Tensor = Any


def conj(node: BaseNode,
         name: Optional[Text] = None,
         axis_names: Optional[List[Text]] = None) -> BaseNode:
  """Conjugate `node`
  Args:
    node: A `BaseNode`. 
    name: Optional name to give the new node.
    axis_names: Optional list of names for the axis.
  Returns:
    A new node. The complex conjugate of `node`.
  Raises:
    TypeError: If `node` has no `backend` attribute.
  """
  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))
  backend = node.backend
  if not axis_names:
    axis_names = node.axis_names

  return Node(
      backend.conj(node.tensor),
      name=name,
      axis_names=axis_names,
      backend=backend.name)


def transpose(node: BaseNode,
              permutation: Sequence[Union[Text, int]],
              name: Optional[Text] = None,
              axis_names: Optional[List[Text]] = None) -> BaseNode:
  """Transpose `node`
  Args:
    node: A `BaseNode`. 
    permutation: A list of int ro str. The permutation of the axis
    name: Optional name to give the new node.
    axis_names: Optional list of names for the axis.
  Returns:
    A new node. The transpose of `node`.
  Raises:
    TypeError: If `node` has no `backend` attribute.
    ValueError: If either `permutation` is not the same as expected or
      if you try to permute with a trace edge.
    AttributeError: If `node` has no tensor.
  """

  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  perm = [node.get_axis_number(p) for p in permutation]
  if not axis_names:
    axis_names = node.axis_names

  new_node = Node(
      node.tensor,
      name=name,
      axis_names=node.axis_names,
      backend=node.backend.name)
  return new_node.reorder_axes(perm)


def split_node(
    node: BaseNode,
    left_edges: List[Edge],
    right_edges: List[Edge],
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[BaseNode, BaseNode, Tensor]:
  """Split a `Node` using Singular Value Decomposition.

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`U S V^* = M` be the Singular Value Decomposition of 
  :math:`M`. This will split the network into 2 nodes. The left node's 
  tensor will be :math:`U \\sqrt{S}` and the right node's tensor will be 
  :math:`\\sqrt{S} V^*` where :math:`V^*` is
  the adjoint of :math:`V`.

  The singular value decomposition is truncated if `max_singular_values` or
  `max_truncation_err` is not `None`.

  The truncation error is the 2-norm of the vector of truncated singular
  values. If only `max_truncation_err` is set, as many singular values will
  be truncated as possible while maintaining:
  `norm(truncated_singular_values) <= max_truncation_err`.

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
    left_name: The name of the new left node. If `None`, a name will be 
      generated automatically.
    right_name: The name of the new right node. If `None`, a name will be 
      genenerated automatically.
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
  """

  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
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
  node.reorder_edges(left_edges + right_edges)

  u, s, vh, trun_vals = backend.svd_decomposition(
      node.tensor, len(left_edges), max_singular_values, max_truncation_err)
  sqrt_s = backend.sqrt(s)
  u_s = u * sqrt_s
  # We have to do this since we are doing element-wise multiplication against
  # the first axis of vh. If we don't, it's possible one of the other axes of
  # vh will be the same size as sqrt_s and would multiply across that axis
  # instead, which is bad.
  sqrt_s_broadcast_shape = backend.concat(
      [backend.shape(sqrt_s), [1] * (len(vh.shape) - 1)], axis=-1)
  vh_s = vh * backend.reshape(sqrt_s, sqrt_s_broadcast_shape)
  left_node = Node(
      u_s, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = Node(
      vh_s, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  node.fresh_edges(node.axis_names)
  return left_node, right_node, trun_vals


def split_node_qr(
    node: BaseNode,
    left_edges: List[Edge],
    right_edges: List[Edge],
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[BaseNode, BaseNode]:
  """Split a `Node` using QR decomposition

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`QR = M` be the QR Decomposition of
  :math:`M`. This will split the network into 2 nodes. The left node's
  tensor will be :math:`Q` (an orthonormal matrix) and the right node's 
  tensor will be :math:`R` (an upper triangular matrix)

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
  """
  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
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
  node.reorder_edges(left_edges + right_edges)
  q, r = backend.qr_decomposition(node.tensor, len(left_edges))
  left_node = Node(
      q, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = Node(
      r, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  return left_node, right_node


def split_node_rq(
    node: BaseNode,
    left_edges: List[Edge],
    right_edges: List[Edge],
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[BaseNode, BaseNode]:
  """Split a `Node` using RQ (reversed QR) decomposition

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`QR = M^*` be the QR Decomposition of
  :math:`M^*`. This will split the network into 2 nodes. The left node's
  tensor will be :math:`R^*` (a lower triangular matrix) and the right 
    node's tensor will be :math:`Q^*` (an orthonormal matrix)

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
        A new node created that connects to all of the `left_edges`.
        Its underlying tensor is :math:`Q`
      right_node:
        A new node created that connects to all of the `right_edges`.
        Its underlying tensor is :math:`R`
  """
  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
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
  node.reorder_edges(left_edges + right_edges)
  r, q = backend.rq_decomposition(node.tensor, len(left_edges))
  left_node = Node(
      r, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = Node(
      q, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  return left_node, right_node


def split_node_full_svd(
    node: BaseNode,
    left_edges: List[Edge],
    right_edges: List[Edge],
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    left_name: Optional[Text] = None,
    middle_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    left_edge_name: Optional[Text] = None,
    right_edge_name: Optional[Text] = None,
) -> Tuple[BaseNode, BaseNode, BaseNode, Tensor]:
  """Split a node by doing a full singular value decomposition.

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`U S V^* = M` be the Singular Value Decomposition of
  :math:`M`.

  The left most node will be :math:`U` tensor of the SVD, the middle node is
  the diagonal matrix of the singular values, ordered largest to smallest,
  and the right most node will be the :math:`V*` tensor of the SVD.

  The singular value decomposition is truncated if `max_singular_values` or
  `max_truncation_err` is not `None`.

  The truncation error is the 2-norm of the vector of truncated singular
  values. If only `max_truncation_err` is set, as many singular values will
  be truncated as possible while maintaining:
  `norm(truncated_singular_values) <= max_truncation_err`.

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
    left_name: The name of the new left node. If None, a name will be 
      generated automatically.
    middle_name: The name of the new center node. If None, a name will be 
      generated automatically.
    right_name: The name of the new right node. If None, a name will be 
      generated automatically.
    left_edge_name: The name of the new left `Edge` connecting
      the new left node (`U`) and the new central node (`S`).
      If `None`, a name will be generated automatically.
    right_edge_name: The name of the new right `Edge` connecting
      the new central node (`S`) and the new right node (`V*`).
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
  """
  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
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

  node.reorder_edges(left_edges + right_edges)
  u, s, vh, trun_vals = backend.svd_decomposition(
      node.tensor, len(left_edges), max_singular_values, max_truncation_err)
  left_node = Node(
      u, name=left_name, axis_names=left_axis_names, backend=backend.name)
  singular_values_node = Node(
      backend.diag(s),
      name=middle_name,
      axis_names=center_axis_names,
      backend=backend.name)

  right_node = Node(
      vh, name=right_name, axis_names=right_axis_names, backend=backend.name)

  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(
      left_node.edges[-1], singular_values_node.edges[0], name=left_edge_name)
  connect(
      singular_values_node.edges[1], right_node.edges[0], name=right_edge_name)
  return left_node, singular_values_node, right_node, trun_vals


def reachable(nodes: Union[BaseNode, List[BaseNode]]) -> Set[BaseNode]:
  """
  Computes all nodes reachable from `node` by connected edges.
  Args:
    node: A `BaseNode`
  Returns:
    A list of `BaseNode` objects that can be reached from `node`
    via connected edges.
  Raises:
    ValueError: If an unknown value for `strategy` is passed.
  """

  def _reachable(node):
    # Fastest way to get a single item from a set.
    node_que = collections.deque()
    seen_nodes = {node}
    node_que.append(node)
    while node_que:
      node = node_que.popleft()
      for e in node.edges:
        for n in e.get_nodes():
          if n is not None and n not in seen_nodes:
            node_que.append(n)
            seen_nodes.add(n)
    return seen_nodes

  if isinstance(nodes, BaseNode):
    nodes = [nodes]
  reachable_nodes = {nodes[0]}
  for node in nodes:
    reachable_nodes |= _reachable(node)
  return reachable_nodes


def check_correct(nodes: Union[List[BaseNode], Set[BaseNode]],
                  check_connections: Optional[bool] = True) -> None:
  """
  Check if the network defined by `nodes` fulfills necessary
  consistency relations.

  Args:
    nodes: A list of `BaseNode` objects.
    check_connections: Check if the network is connected.
  Returns:
    None
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


def check_connected(nodes: Union[List[BaseNode], Set[BaseNode]]) -> None:
  """
  Check if all nodes in `nodes` are connected.
  Args:
    nodes: A list of nodes.
  Returns:
    None:
  Raises:
    ValueError: If not all nodes in `nodes` are connected.
  """
  if not set(nodes) <= reachable([list(nodes)[0]]):
    raise ValueError("Non-connected graph")


def get_all_nodes(edges: Union[List[Edge], Set[Edge]]) -> Set[BaseNode]:
  """Return the set of nodes connected to edges."""
  nodes = set()
  for edge in edges:
    if edge.node1 is not None:
      nodes |= {edge.node1}
    if edge.node2 is not None:
      nodes |= {edge.node2}

  return nodes


def get_all_edges(nodes: Union[List[BaseNode], Set[BaseNode]]) -> Set[Edge]:
  """Return the set of edges of all nodes."""
  edges = set()
  for node in nodes:
    edges |= set(node.edges)
  return edges
