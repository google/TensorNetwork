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
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Type, Union
import typing
import numpy as np
from tensornetwork.backends import base_backend
import weakref

Tensor = Any
# This is required because of the circular dependancy between
# network_components.py and network.py types.
TensorNetwork = Any

class Node:
  """Node for the TensorNetwork graph.

  A Node represents a concrete tensor in a tensor network. The number of edges
  for a node represents the rank of that tensor.

  For example:

  * A node with no edges means this node represents a scalar value.
  * A node with a single edge means this node is a vector.
  * A node with two edges represents a matrix.
  * A node with three edges is a tensor of rank 3, etc.

  Each node can have an arbitrary rank/number of edges, each of which can have
  an arbitrary dimension.
  """

  def __init__(
    self, tensor: Tensor, name: Text, axis_names: List[Text],
    network: TensorNetwork) -> None:
    """Create a node for the TensorNetwork.

    Args:
      tensor: The concrete tensor that is represented by this node. Can be
        either a numpy array or a tensorflow tensor.
      name: Name of the node. Used primarily for debugging.
      axis_names: List of names for each of the tensor's axes.
      network: The TensorNetwork this Node belongs to.

    Raises:
      ValueError: If there is a repeated name in `axis_names` or if the length
        doesn't match the shape of the tensor.
    """
    self._tensor = tensor
    self.name = name
    self.network = network
    self.edges = [
        Edge(edge_name, self, i) for i, edge_name in enumerate(axis_names)
    ]
    if axis_names is not None:
      self.add_axis_names(axis_names)
    else:
      self.axis_names = None
    self.signature = -1

  def get_rank(self) -> int:
    """Return rank of tensor represented by self."""
    return len(self.tensor.shape)

  def set_signature(self, signature: int) -> None:
    """Set the signature for the node.

    Signatures are numbers that uniquely identify a node inside of a
    TensorNetwork.
    """
    self.signature = signature

  def add_axis_names(self, axis_names: List[Text]) -> None:
    """Add axis names to a Node.

    Args:
      axis_names: List of names for each of the tensor's axes.

    Raises:
      ValueError: If there is a repeated name in `axis_names` or if the length
        doesn't match the shape of the tensor.
    """
    if len(axis_names) != len(set(axis_names)):
      raise ValueError("Not all axis names are unique.")
    if len(axis_names) != len(self.tensor.shape):
      raise ValueError("axis_names is not the same length as the tensor shape."
                       "axis_names length: {}, tensor.shape length: {}".format(
                           len(axis_names), len(self.tensor.shape)))
    self.axis_names = axis_names[:]

  def add_edge(self,
               edge: "Edge",
               axis: Union[int, Text],
               override: bool = False) -> None:
    """Add an edge to the node on the given axis.

    Args:
      edge: The edge to add.
      axis: The axis the edge points to.
      override: If true, replace the existing edge with the new one.

    Raises:
      ValueError: If the edge on axis is not dangling.
    """
    axis_num = self.get_axis_number(axis)
    if axis_num < 0 or axis_num >= len(self.tensor.shape):
      raise ValueError("Axis must be positive and less than rank of the tensor")
    if not self.edges[axis_num].is_dangling() and not override:
      raise ValueError(
          "Node '{}' already has a non-dangling edge for axis {}".format(
              self, axis))
    self.edges[axis_num] = edge

  def get_tensor(self):
    return self.tensor

  def set_tensor(self, tensor):
    self.tensor = tensor

  @property
  def shape(self):
    return self.network.backend.shape_tuple(self._tensor)
  
  @property
  def tensor(self) -> Tensor:
    return self._tensor

  @tensor.setter
  def tensor(self, tensor: Tensor) -> Tensor:
    self._tensor = tensor

  def reorder_edges(self, edge_order: List["Edge"]) -> "Node":
    """Reorder the edges for this given Node.

    This will reorder the node's edges and transpose the underlying tensor
    accordingly.

    Args:
      edge_order: List of edges. The order in the list determins the new edge
        ordering.

    Returns:
      This node post reordering.

    Raises:
      ValueError: If either the list of edges is not the same as expected or if
        you try to reorder with a trace edge.
    """
    if set(edge_order) != set(self.edges):
      raise ValueError("Given edge order does not match expected edges. "
                       "Found: {}, Expected: {}".format(edge_order, self.edges))
    for edge in edge_order:
      if edge.node1 == edge.node2:
        raise ValueError("Edge reordering does not support trace edges. "
                         "Found trace edge: '{}'".format(edge))

    permutation = []
    for i, edge in enumerate(edge_order):
      # This is O(n^2), but the number of edges will likely never be >100
      # so this should be fine for now.
      old_position = self.edges.index(edge)
      permutation.append(old_position)
      edge.update_axis(old_position, self, i, self)
    self.edges = edge_order[:]
    self.tensor = self.network.backend.transpose(
        self.tensor, perm=permutation)
    if self.axis_names is not None:
      # Update axis_names:
      tmp_axis_names = []
      for i in permutation:
        tmp_axis_names.append(self.axis_names[i])
      self.axis_names = tmp_axis_names
    return self

  def reorder_axes(self, perm: List[int]) -> "Node":
    """Reorder axes of the node's tensor.

    This will also update all of the node's edges.

    Args:
      perm: Permutation of the dimensions of the node's tensor.

    Returns:
      This node post reordering.
    """
    if set(perm) != set(range(len(self.edges))):
      raise ValueError("A full permutation was not passed. "
                       "Permutation passed: {}".format(perm))
    self.tensor = self.network.backend.transpose(self.tensor, perm=perm)
    tmp_edges = []
    for i, position in enumerate(perm):
      edge = self.edges[position]
      edge.update_axis(position, self, i, self)
      tmp_edges.append(edge)
    self.edges = tmp_edges
    return self

  def get_axis_number(self, axis: Union[Text, int]) -> int:
    """Get the axis number for a given axis name or value."""
    if isinstance(axis, int):
      return axis
    try:
      return self.axis_names.index(axis)
    except ValueError:
      raise ValueError("Axis name '{}' not found for node '{}'".format(
          axis, self))

  def get_dimension(self, axis: Union[Text, int]) -> Optional[int]:
    """Get the dimension on the given axis.

    Args:
      axis: The axis of the underlying tensor.

    Returns:
      The dimension of the given axis.

    Raises:
      ValueError: if axis isn't an int or if axis is too large or small.
    """
    axis_num = self.get_axis_number(axis)
    if axis_num < 0 or axis_num >= len(self.tensor.shape):
      raise ValueError("Axis must be positive and less than rank of the tensor")
    return self.network.backend.shape_tuple(self.tensor)[axis_num]

  def get_edge(self, axis: Union[int, Text]) -> "Edge":
    axis_num = self.get_axis_number(axis)
    return self.edges[axis_num]

  def get_all_edges(self):
    # Copy to prevent overwriting.
    return self.edges[:]

  def get_all_nondangling(self):
    """Return the set of nondangling edges connected to this node."""
    return {edge for edge in self.edges if not edge.is_dangling()}

  def set_name(self, name):
    self.name = name

  def has_nondangling_edge(self):
    for e in self.edges:
      if not e.is_dangling():
        return True
    return False

  def __getitem__(self, key: Union[int, Text]) -> "Edge":
    return self.get_edge(key)

  def __str__(self) -> Text:
    return self.name

  def __lt__(self, other):
    if not isinstance(other, Node):
      raise ValueError("Object {} is not a Node type.".format(other))
    return id(self) < id(other)

  def __matmul__(self, other: "Node") -> "Node":
    if not isinstance(other, Node):
      raise TypeError("Cannot use '@' with type '{}'".format(type(other)))
    if other.network is not self.network:
      raise ValueError("Cannot use '@' on nodes in different networks.")
    return self.network.contract_between(self, other)

class CopyNode(Node):

  def __init__(self,
               rank: int,
               dimension: int,
               name: Text,
               axis_names: List[Text],
               network: TensorNetwork,
               dtype: Type[np.number] = np.float64) -> None:
    # TODO: Make this computation lazy, once Node doesn't require tensor
    # at instatiation.
    copy_tensor = self.make_copy_tensor(rank, dimension, dtype)
    copy_tensor = network.backend.convert_to_tensor(copy_tensor)
    super().__init__(copy_tensor, name, axis_names, network)

  @staticmethod
  def make_copy_tensor(rank: int, dimension: int,
                       dtype: Type[np.number]) -> Tensor:
    shape = (dimension,) * rank
    copy_tensor = np.zeros(shape, dtype=dtype)
    i = np.arange(dimension)
    copy_tensor[(i,) * rank] = 1
    return copy_tensor

  def _is_my_trace(self, edge: "Edge") -> bool:
    return edge.node1 is self and edge.node2 is self

  def _get_partner(self, edge: "Edge") -> Tuple[Node, int]:
    if edge.node1 is self:
      assert edge.axis2 is not None
      return edge.node2, edge.axis2
    assert edge.node2 is self
    return edge.node1, edge.axis1

  def get_partners(self) -> Dict[Node, Set[int]]:
    partners = {}  # type: Dict[Node, Set[int]]
    for edge in self.edges:
      if edge.is_dangling():
        raise ValueError('Cannot contract copy tensor with dangling edges')
      if self._is_my_trace(edge):
        continue
      partner_node, shared_axis = self._get_partner(edge)
      if partner_node not in partners:
        partners[partner_node] = set()
      partners[partner_node].add(shared_axis)
    return partners

  _VALID_SUBSCRIPTS = list(
      'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

  def _make_einsum_input_term(self, node: Node, shared_axes: Set[int],
                              next_index: int) -> Tuple[str, int]:
    indices = []
    for axis in range(node.get_rank()):
      if axis in shared_axes:
        indices.append(0)
      else:
        indices.append(next_index)
        next_index += 1
    term = "".join(self._VALID_SUBSCRIPTS[i] for i in indices)
    return term, next_index

  def _make_einsum_output_term(self, next_index: int) -> str:
    return "".join(self._VALID_SUBSCRIPTS[i] for i in range(1, next_index))

  def _make_einsum_expression(self, partners: Dict[Node, Set[int]]) -> str:
    next_index = 1  # zero is reserved for the shared index
    einsum_input_terms = []
    for partner_node, shared_axes in partners.items():
      einsum_input_term, next_index = self._make_einsum_input_term(
          partner_node, shared_axes, next_index)
      einsum_input_terms.append(einsum_input_term)
    einsum_output_term = self._make_einsum_output_term(next_index)
    einsum_expression = ",".join(einsum_input_terms) + "->" + einsum_output_term
    return einsum_expression

  def compute_contracted_tensor(self) -> Tensor:
    """Compute tensor corresponding to contraction of self with neighbors."""
    partners = self.get_partners()
    einsum_expression = self._make_einsum_expression(partners)
    tensors = [partner.get_tensor() for partner in partners]
    return self.network.backend.einsum(einsum_expression, *tensors)


class Edge:
  """Edge for the TensorNetwork graph.

  Each edge represents a vector space common to the tensors it connects and over
  which a contraction may be performed. In numpy terms, each edge represents a
  `tensordot` operation over the given axes.
  There are 3 main types of edges:

  Standard Edge:
    A standard edge is like any other edge you would find in a normal
    undirected graph as they connect two different nodes. This edge represents
    a tensor contraction of the underlying tensors along their given axes.
    The two axes must be the same dimension.

  Dangling Edge:
    A dangling edge is an edge that only connects to a single node and only one
    part of the edge connects to the node. The other end is left "dangling".
    These types of edges can not be contrated and represent additional
    dimensions on the underlying tensor. After all other edges are contracted,
    the final result will have the same rank as the number of dangling edges. If
    there are no dangling edges, then the final value will be a scalar.

  Trace Edges:
    Trace edges are edges that connects a node to itself. These edges represent
    a trace along the given axis. Once again, the axes must be the same
    dimension.
  """

  def __init__(self,
               name: Text,
               node1: Node,
               axis1: int,
               node2: Optional[Node] = None,
               axis2: Optional[int] = None) -> None:
    """Create an Edge.

    Args:
      name: Name of the edge. Used primarily for debugging.
      node1: One of the nodes edge connects.
      axis1: The axis of node1 that represents this edge.
      node2: The other node that this edge connects. Can be `None` if edge is
        dangling.
      axis2: The axis of node2 that represents this edge. Must be `None` if
        node2 is `None`.

    Raises:
      ValueError: If node2 and axis2 are not either both `None` or both
        not be `None`.
    """
    if (node2 is None) != (axis2 is None):
      raise ValueError(
          "node2 and axis2 must either be both None or both not be None")
    self.name = name
    self.node1 = node1
    self.axis1 = axis1
    self.node2 = node2
    self.axis2 = axis2
    self._is_dangling = node2 is None
    self.signature = -1

  def set_signature(self, signature: int) -> None:
    if self.is_dangling():
      raise ValueError(
        "Do not set a signature for dangling edge '{}'.".format(self))
    self.signature = signature

  def get_nodes(self) -> List[Optional[Node]]:
    """Get the nodes of the edge."""
    return [self.node1, self.node2]

  def update_axis(self, old_axis: int, old_node: Node, new_axis: int,
                  new_node: Node) -> None:
    """Update the node that Edge is connected to.

    Args:
      old_axis: The old axis that the edge pointed to.
      old_node: The old node that the edge pointed to.
      new_axis: The new axis that the edge should point to.
      new_node: The new node that replaces the old_node.

    Raises:
      AssertionError: Whether the edge actually contained `old_node`.
    """
    if self.axis1 == old_axis and self.node1 is old_node:
      self.axis1 = new_axis
      self.node1 = new_node
    elif self.axis2 == old_axis and self.node2 is old_node:
      self.axis2 = new_axis
      self.node2 = new_node
    else:
      raise ValueError("Edge '{}' did not contain node '{}' on axis {}. "
                       "node1: '{}', axis1: {}, node2: '{}', axis2: {}".format(
                           self, old_node, old_axis, self.node1, self.axis1,
                           self.node2, self.axis2))

  @property
  def node1(self) -> Node:
    val = self._node1()
    if val is None:
      raise ValueError("node1 for edge '{}' no longer exists.".format(self))
    return val

  @property
  def node2(self) -> Optional[Node]:
    if self._is_dangling:
      return None
    if self._node2() is None:
      raise ValueError("node2 for edge '{}' no longer exists.".format(self))
    return self._node2()

  @node1.setter
  def node1(self, node: Node) -> None:
    # pylint: disable=attribute-defined-outside-init
    self._node1 = weakref.ref(node)

  @node2.setter
  def node2(self, node: Optional[Node]) -> None:
    # pylint: disable=attribute-defined-outside-init
    self._node2 = weakref.ref(node) if node else None
    if node is None:
      self._is_dangling = True

  @property
  def dimension(self):
    return self.node1.shape[self.axis1]

  def is_dangling(self) -> bool:
    """Whether this edge is a dangling edge."""
    return self._is_dangling

  def is_trace(self) -> bool:
    return self.node1 is self.node2

  def is_being_used(self):
    """Whether the nodes this edge points to also use this edge.

    During edge flattening, nodes can change their edges. Since
    deleting objects in python isn't possible, we use this to ensure that the
    edge is actually being used by the given nodes.

    Returns:
      Whether this edge is actually being used.
    """
    result = self is self.node1[self.axis1]
    if self.node2 is not None:
      result = result and self is self.node2[self.axis2]
    return result

  def set_name(self, name: Text) -> None:
    self.name = name

  def __xor__(self, other: "Edge") -> "Edge":
    return self.node1.network.connect(self, other)

  def __lt__(self, other):
    if not isinstance(other, Edge):
      raise TypeError("Cannot compare 'Edge' with type {}".format(type(Edge)))
    return self.signature < other.signature

  def __str__(self) -> Text:
    return self.name
