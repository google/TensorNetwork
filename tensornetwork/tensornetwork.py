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
from typing import List, Optional, Union, Text, Tuple
import numpy as np
import tensorflow as tf
import weakref
from tensornetwork import decompositions
import tensordot2

class Node:
  """Node for the TensorNetwork graph.

  A Node represents a concrete tensor in a tensor network. The number of edges
  for a node represents the rank of that tensor.

  For example:
  O : No edges means this node represents a scalar value.
  -O : A single edge means this node is a vector.
  -O-: Two edges represents a matrix.
  -O- : Three edges are tensor of rank 3, etc.
   |

  Each node can have an arbitrary rank/number of edges, each of which can have
  an arbitrary dimension.
  """

  def __init__(self,
               tensor: Union[np.ndarray, tf.Tensor],
               name: Text,
               axis_names: List[Text]) -> None:
    """Create a node for the TensorNetwork.

    Args:
      tensor: The concrete tensor that is represented by this node. Can be
        either a numpy array or a tensorflow tensor.
      name: Name of the node. Used primarily for debugging.
      axis_names: List of names for each of the tensor's axes.

    Raises:
      ValueError: If there is a repeated name in `axis_names` or if the length
        doesn't match the shape of the tensor.
    """
    self.tensor = tf.convert_to_tensor(tensor)
    self.name = name
    self.edges = [Edge(edge_name, self, i)
                  for i, edge_name in enumerate(axis_names)]
    if axis_names is not None:
      self.add_axis_names(axis_names)
    else:
      self.axis_names = None

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

  def reorder_edges(self, edge_order: List["Edge"]) -> "Node":
    """Reorder the edges for this given Node.

    This will reorder the node's edges and transpose the underlying tensor
    accordingly.

    Args:
      edge_order: List of edges. The order in the list determins the new edge
        ordering.

    Returns:
      self: This node post reordering.

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
    self.tensor = tf.transpose(self.tensor, perm=permutation)
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
      self: This node post reordering.
    """
    if set(perm) != set(range(len(self.edges))):
      raise ValueError("A full permutation was not passed. "
                       "Permutation passed: {}".format(perm))
    self.tensor = tf.transpose(self.tensor, perm=perm)
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

  def get_dimension(self, axis: Union[Text, int]) -> int:
    """Get the dimension on the given axis.

    Args:
      axis: The axis of the underlying tensor.

    Returns:
      dimension: The dimension of the given axis.

    Raises:
      ValueError: if axis isn't an int or if axis is too large or small.
    """
    axis_num = self.get_axis_number(axis)
    if axis_num < 0 or axis_num >= len(self.tensor.shape):
      raise ValueError("Axis must be positive and less than rank of the tensor")
    return tf.shape(self.tensor)[axis_num]

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
      tf.errors.InvalidArgumentError: If the dimensions of axis1 and axis2 are
        not equal.
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
    self._node1 = weakref.ref(node)

  @node2.setter
  def node2(self, node: Optional[Node]) -> None:
    self._node2 = weakref.ref(node) if node else None
    if node is None:
      self._is_dangling = True

  def is_dangling(self) -> bool:
    """Whether this edge is a dangling edge."""
    return self._is_dangling

  def is_being_used(self):
    """Whether the nodes this edge points to also use this edge.

    During edge flattening, nodes can change their edges. Since
    deleting objects in python isn't possible, we use this to ensure that the
    edge is actually being used by the given nodes.

    Returns:
      bool: Whether this edge is actually being used.
    """
    result = self is self.node1[self.axis1]
    if self.node2 is not None:
      result = result and self is self.node2[self.axis2]
    return result

  def set_name(self, name):
    self.name = name

  def __str__(self) -> Text:
    return self.name


class TensorNetwork:
  """Implementation of a TensorNetwork."""

  def __init__(self) -> None:
    self.nodes_set = set()
    self.edge_order = []
    # These increments are only used for generating names.
    self.node_increment = 0
    self.edge_increment = 0

  def _new_edge_name(self, name: Optional[Text]) -> Text:
    if name is None:
      name = "__Edge_{}".format(self.edge_increment)
    self.edge_increment += 1
    return name

  def _new_node_name(self, name: Optional[Text]) -> Text:
    if name is None:
      name = "__Node_{}".format(self.node_increment)
    self.node_increment += 1
    return name

  # TODO: Add pytypes once we figure out why it crashes.
  def add_subnetwork(self, subnetwork) -> None:
    """Add a subnetwork to an existing network.

    Args:
      subnetwork: A TensorNetwork object. The nodes and edges of this network
        will be merged into the original network.
    """
    self.nodes_set |= subnetwork.nodes_set
    # Add increment for namings.
    self.node_increment += subnetwork.node_increment
    self.edge_increment += subnetwork.edge_increment
    self.edge_order += subnetwork.edge_order

  # TODO: Add pytypes once we figure out why it crashes.
  @classmethod
  def merge_networks(cls, networks):
    """Merge several networks into a single network.

    Args:
      networks: An iterable of TensorNetworks.

    Returns:
      new_network: A new network created by the merging of all of the given
        networks.
    """
    new_network = cls()
    for network in networks:
      new_network.add_subnetwork(network)
    return new_network

  def add_node(self,
               tensor: Union[np.ndarray, tf.Tensor],
               name: Optional[Text] = None,
               axis_names: Optional[List[Text]] = None) -> Node:
    """Create a new node in the network.

    Args:
      tensor: The concrete tensor for the node.
      name: The name of the new node. If None, a name will be generated
        automatically.
      axis_names: Optional list of strings to name each of the axes.

    Returns:
      new_node: The new node object.
    Raises:
      ValueError: If `name` already exists in the network.
    """
    tensor = tf.convert_to_tensor(tensor)
    name = self._new_node_name(name)
    if axis_names is None:
      axis_names = [self._new_edge_name(None) for _ in range(len(tensor.shape))]
    new_node = Node(tensor, name, axis_names)
    self.nodes_set.add(new_node)
    return new_node

  def connect(self, edge1: Edge, edge2: Edge,
              name: Optional[Text] = None) -> Edge:
    """Join two dangling edges into a new edge.

    Args:
      edge1: The first dangling edge.
      edge2: The second dangling edge.
      name: Optional name to give the new edge.

    Returns:
      new_edge: A new edge created by joining the two dangling edges together.
    Raises:
      ValueError: If either edge1 or edge2 is not a dangling edge.
    """
    for edge in [edge1, edge2]:
      if not edge.is_dangling():
        raise ValueError("Edge '{}' is not a dangling edge. "
                         "This edge points to nodes: '{}' and '{}'".format(
                             edge, edge.node1, edge.node2))
    node1 = edge1.node1
    node2 = edge2.node1
    axis1_num = node1.get_axis_number(edge1.axis1)
    axis2_num = node2.get_axis_number(edge2.axis1)
    name = self._new_edge_name(name)
    new_edge = Edge(name, node1, axis1_num, node2, axis2_num)
    node1.add_edge(new_edge, axis1_num)
    node2.add_edge(new_edge, axis2_num)
    self.edge_order.append(new_edge)
    return new_edge

  def disconnect(self, edge: Edge, dangling_edge_name_1: Optional[Text] = None,
                 dangling_edge_name_2: Optional[Text] = None) -> List[Edge]:
    """Break a edge into two dangling edges.

    Args:
      edge: An edge to break.
      dangling_edge_name_1: Optional name to give the new dangling edge 1.
      dangling_edge_name_2: Optional name to give the new dangling edge 2.

    Returns:
      dangling_edge_1: A new dangling edge.
      dangling_edge_2: A new dangling edge.

    Raises:
      ValueError: If input edge is a dangling one.
    """
    if edge.is_dangling():
      raise ValueError("Attempted to break a dangling edge '{}'.".format(edge))
    node1 = edge.node1
    node2 = edge.node2
    dangling_edge_name_1 = self._new_edge_name(dangling_edge_name_1)
    dangling_edge_name_2 = self._new_edge_name(dangling_edge_name_2)
    dangling_edge_1 = Edge(dangling_edge_name_1, node1, edge.axis1)
    dangling_edge_2 = Edge(dangling_edge_name_2, node2, edge.axis2)
    node1.add_edge(dangling_edge_1, edge.axis1, True)
    node2.add_edge(dangling_edge_2, edge.axis2, True)
    self.edge_order.remove(edge)
    return [dangling_edge_1, dangling_edge_2]

  def _remove_trace_edge(self, edge: Edge, new_node: Node) -> None:
    """Collapse a trace edge.

    Collapses a trace edge and updates the network.
    Args:
      edge: The edge to contract.
      new_node: The new node created after contraction.

    Returns:
      node: The node that had the contracted edge.
    Raises:
      ValueError: If edge is not a trace edge.
    """
    if edge.is_dangling():
      raise ValueError("Attempted to remove dangling edge '{}'.".format(edge))
    if edge.node1 is not edge.node2:
      raise ValueError("Edge '{}' is not a trace edge.".format(edge))
    axes = sorted([edge.axis1, edge.axis2])
    node_edges = edge.node1.edges[:]
    node_edges.pop(axes[0])
    node_edges.pop(axes[1] - 1)
    seen_edges = set()
    for tmp_edge in node_edges:
      if tmp_edge in seen_edges:
        continue
      else:
        seen_edges.add(tmp_edge)
      if tmp_edge.node1 is edge.node1:
        to_reduce = 0
        to_reduce += 1 if tmp_edge.axis1 > axes[0] else 0
        to_reduce += 1 if tmp_edge.axis1 > axes[1] else 0
        tmp_edge.axis1 -= to_reduce
        tmp_edge.node1 = new_node
      if tmp_edge.node2 is edge.node1:
        to_reduce = 0
        to_reduce += 1 if tmp_edge.axis2 > axes[0] else 0
        to_reduce += 1 if tmp_edge.axis2 > axes[1] else 0
        tmp_edge.axis2 -= to_reduce
        tmp_edge.node2 = new_node
    # Update edges for the new node.
    for i, e in enumerate(node_edges):
      new_node.add_edge(e, i)
    self.nodes_set.remove(edge.node1)

  def _remove_edge(self, edge: Edge, new_node: Node) -> None:
    """Collapse an edge in the network.

    Collapses an edge and updates the rest of the network.

    Args:
      edge: The edge to contract.
      new_node: The new node that represents the contraction of the two old
        nodes.

    Raises:
      Value Error: If edge isn't in the network.
    """
    # Assert that the edge isn't a dangling edge.
    if edge.is_dangling():
      raise ValueError("Attempted to remove dangling edge '{}'.".format(edge))
    if edge.node1 is edge.node2:
      self._remove_trace_edge(edge, new_node)
    # Collapse the nodes into a new node and remove the edge.
    node1 = edge.node1
    node2 = edge.node2
    node1_edges = edge.node1.edges[:]
    node2_edges = edge.node2.edges[:]
    node1_axis = edge.axis1
    node2_axis = edge.axis2
    # Redefine all other edges.
    num_added_front_edges = len(node1_edges) - 1
    for i, tmp_edge in enumerate(node1_edges[:node1_axis]):
      tmp_edge.update_axis(
          old_axis=i, old_node=node1, new_axis=i, new_node=new_node)
    for i, tmp_edge in enumerate(node1_edges[node1_axis + 1:]):
      tmp_edge.update_axis(
          old_axis=i + node1_axis + 1,
          old_node=node1,
          new_axis=i + node1_axis,
          new_node=new_node)
    for i, tmp_edge in enumerate(node2_edges[:node2_axis]):
      tmp_edge.update_axis(
          old_axis=i,
          old_node=node2,
          new_axis=i + num_added_front_edges,
          new_node=new_node)
    for i, tmp_edge in enumerate(node2_edges[node2_axis + 1:]):
      tmp_edge.update_axis(
          old_axis=i + node2_axis + 1,
          old_node=node2,
          new_axis=i + node2_axis + num_added_front_edges,
          new_node=new_node)

    node1_edges.pop(node1_axis)
    node2_edges.pop(node2_axis)
    new_edges = node1_edges + node2_edges
    for i, e in enumerate(new_edges):
      new_node.add_edge(e, i)

    # Remove nodes
    self.nodes_set.remove(node1)
    self.nodes_set.remove(node2)

  def _contract_trace(self, edge: Edge, name: Optional[Text] = None) -> Node:
    """Contract a trace edge connecting in the TensorNetwork.

    Args:
      edge: The edge name or object to contract next.
      name: Name to give to the new node. If None, a name will automatically be
        generated.

    Returns:
      new_node: The new node created after the contraction.
    Raise:
      ValueError: When edge is a dangling edge.
    """
    if edge.is_dangling():
      raise ValueError("Attempted to contract dangling edge '{}'".format(edge))
    elif edge.node1 is not edge.node2:
      raise ValueError("Can not take trace of edge '{}'. This edge connects to "
                       "two different nodes: '{}' and '{}".format(
                           edge, edge.node1, edge.node2))
    axes = sorted([edge.axis1, edge.axis2])
    dims = len(edge.node1.tensor.shape)
    permutation = sorted(set(range(dims)) - set(axes)) + axes
    new_tensor = tf.linalg.trace(
        tf.transpose(edge.node1.tensor, perm=permutation))
    new_node = self.add_node(new_tensor, name)
    self._remove_trace_edge(edge, new_node)
    return new_node

  def contract(self, edge: Edge, name: Optional[Text] = None) -> Node:
    """Contract an edge connecting two nodes in the TensorNetwork.

    Args:
      edge: The edge contract next.
      name: Name of the new node created.

    Returns:
      new_node: The new node created after the contraction.

    Raises:
      ValueError: When edge is a dangling edge or if it already has been
        contracted.
    """
    if not edge.is_being_used() or edge.node1 not in self.nodes_set:
      raise ValueError("Attempting to contract edge '{}' that is not part of "
                       "the network.".format(edge))
    if edge.is_dangling():
      raise ValueError("Attempting to contract dangling edge")
    if edge.node1 is edge.node2:
      return self._contract_trace(edge, name)
    new_tensor = tensordot2.tensordot(edge.node1.tensor, edge.node2.tensor,
                              [[edge.axis1], [edge.axis2]])
    new_node = self.add_node(new_tensor, name)
    self._remove_edge(edge, new_node)
    return new_node

  def outer_product(self, node1: Node, node2: Node,
                    name: Optional[Text] = None) -> Node:
    """Calcuates an outer product of the two nodes.

    This causes the nodes to combine their edges and axes, so the shapes are
    combined. For example, if `a` had a shape (2, 3) and `b` had a shape
    (4, 5, 6), then the node `net.outer_product(a, b) will have shape
    (2, 3, 4, 5, 6).

    Args:
      node1: The first node. The axes on this node will be on the left side of
        the new node.
      node2: The second node. The axes on this node will be on the right side of
        the new node.
      name: Optional name to give the new node created.

    Returns:
      new_node: A new node. It's shape will be node1.shape + node2.shape
    """
    new_tensor = tensordot2.tensordot(node1.tensor, node2.tensor, 0)
    new_node = self.add_node(new_tensor, name)
    additional_axes = len(node1.tensor.shape)
    for i, edge in enumerate(node1.edges):
      edge.update_axis(i, node1, i, new_node)
    for i, edge in enumerate(node2.edges):
      edge.update_axis(i, node2, i + additional_axes, new_node)
    # Remove the nodes from the set.
    self.nodes_set.remove(node1)
    self.nodes_set.remove(node2)
    for i, edge in enumerate(node1.edges + node2.edges):
      new_node.add_edge(edge, i)
    return new_node

  def get_final_node(self) -> Node:
    """Get the final node of a fully contracted network.

    Note: The network must already be fully contracted.

    Returns:
      final_node: The final node in the network.
    Raises:
      ValueError: If this network has more than 1 remaining node or if any of
        the remaining edges are dangling.
    """
    if len(self.nodes_set) != 1:
      raise ValueError("This tensor network has more than 1 node.")
    node = next(iter(self.nodes_set))
    for e in node.edges:
      if not e.is_dangling():
        raise ValueError("This network is not fully contracted. "
                         "Edge '{}' has not been contracted.".format(e))
    return next(iter(self.nodes_set))

  def get_all_nondangling(self):
    """Return the set of all non-dangling edges."""
    edges = set()
    for node in self.nodes_set:
      edges |= node.get_all_nondangling()
    return edges

  def outer_product_final_nodes(self, edge_order: List[Edge]) -> Node:
    """Get the outer product of the final nodes.

    For example, if after all contractions, there were 3 nodes remaining with
    shapes (2, 3), (4, 5, 6), and (7) respectively, the newly returned node
    will have shape (2, 3, 4, 5, 6, 7).

    Args:
      edge_order: Edge order for the final node.

    Returns:
      final_node: The outer product of the remaining nodes.

    Raises:
      ValueError: If any of the remaining nodes are not fully contracted.
    """
    nodes = list(self.nodes_set)
    for node in nodes:
      if node.has_nondangling_edge():
        raise ValueError("Node '{}' has a non-dangling edge remaining.")
    final_node = nodes[0]
    for node in nodes[1:]:
      final_node = self.outer_product(final_node, node)
    return final_node.reorder_edges(edge_order)

  def check_connected(self) -> None:
    """Check if the network is connected."""
    # Fastest way to get a single item from a set.
    node = next(iter(self.nodes_set))
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
    if self.nodes_set != seen_nodes:
      raise ValueError("Non-connected graph")

  def _flatten_trace_edges(self, edges: List[Edge],
                           new_edge_name: Optional[Text]) -> Edge:
    """Flatten trace edges into single edge.

    Args:
      edges: List of trace edges to flatten
      new_edge_name: Optional name of the new edge created.

    Returns:
      new_edge: The new edge that represents the flattening of the given edges.
    """
    node = edges[0].node1  # We are in the trace case, so this is the only node.
    # Flatten all of the edge's axes into a a single list.
    perm_back = [min(e.axis1, e.axis2) for e in edges]
    perm_back += [max(e.axis1, e.axis2) for e in edges]
    perm_front = set(range(len(node.edges))) - set(perm_back)
    perm_front = sorted(perm_front)
    perm = perm_front + perm_back
    new_dim = tf.reduce_prod([tf.shape(node.tensor)[e.axis1] for e in edges])
    node.reorder_axes(perm)
    unaffected_shape = tf.shape(node.tensor)[:len(perm_front)]
    new_shape = tf.concat([unaffected_shape, [new_dim, new_dim]], axis=-1)
    node.tensor = tf.reshape(node.tensor, new_shape)
    edge1 = Edge("TraceFront", node, len(perm_front))
    edge2 = Edge("TraceBack", node, len(perm_front) + 1)
    node.edges = node.edges[:len(perm_front)] + [edge1, edge2]
    new_edge = self.connect(edge1, edge2, new_edge_name)
    node.axis_names = None
    return new_edge

  def flatten_edges(self,
                    edges: List[Edge],
                    new_edge_name: Optional[Text] = None) -> Edge:
    """Flatten edges into single edge.

    If two nodes have multiple edges connecting them, it may be
    benifitial to flatten these edges into a single edge to avoid having several
    unnecessary trace edges. This can speed up computation time and reduce
    memory cost.

    Warning: This will remove all axes names.

    Args:
      edges: A list of edges to flatten.
      new_edge_name: Optional name to give to the newly created edge.

    Returns:
      new_edge: The new flattened edge.

    Raises:
      ValueError: If edges is an empty list.
      ValueError: If not all of the edges connect to the same node(s).
      ValueError: If one of the nodes connecting to these edges does not have
        edge definitions for all of its axes.
    """
    if not edges:
      raise ValueError("At least 1 edge must be given.")
    if len(edges) == 1:
      return edges[0]  # Don't bother with reshaping.
    # Set equality is transitive (a=b, b=c, therefore a=c) so it is only
    # necessary to compare the first edge against the rest.
    expected_nodes = set(edges[0].get_nodes())
    for edge in edges:
      if expected_nodes != set(edge.get_nodes()):
        raise ValueError(
            "Two edges do not share the same nodes. "
            "'{}'s nodes: '{}', '{}'. '{}'s nodes: '{}', '{}'".format(
                edges[0], edges[0].node1, edges[0].node2, edge, edge.node1,
                edge.node2))
    if len(expected_nodes) == 1:
      return self._flatten_trace_edges(edges, new_edge_name)
    # Flatten standard or dangling edges.
    new_dangling_edges = []
    for node in expected_nodes:
      # Required for dangling case.
      if node is None:
        continue
      perm_back = []
      for edge in edges:
        # There will only be 1 edge since we are in the standard edge case.
        perm_back.append(node.edges.index(edge))
      perm_front = sorted(set(range(len(node.edges))) - set(perm_back))
      node.reorder_axes(perm_front + perm_back)
      old_tensor_shape = tf.shape(node.tensor)
      # Calculate the new axis dimension as a product of the other
      # axes dimensions.
      flattened_axis_dim = tf.reduce_prod(old_tensor_shape[len(perm_front):])
      new_tensor_shape = tf.concat(
          [old_tensor_shape[:len(perm_front)], [flattened_axis_dim]], axis=-1)
      new_tensor = tf.reshape(node.tensor, new_tensor_shape)
      # Modify the node in place. Currently, this is they only method that
      # modifies a node's tensor.
      node.tensor = new_tensor
      # This Edge is required for the connect call later.
      edge = Edge(new_edge_name, node, len(perm_front))
      node.edges = node.edges[:len(perm_front)] + [edge]
      new_dangling_edges.append(edge)
      # TODO: Allow renaming of the new axis.
      node.axis_names = None
    node1, node2 = tuple(expected_nodes)
    # Sets are returned in a random order, so this is how we deal with
    # dangling edges.
    if node1 is None or node2 is None:
      return new_dangling_edges[0]
    return self.connect(new_dangling_edges[0],
                        new_dangling_edges[1], new_edge_name)

  def flatten_edges_between(self, node1: Node, node2: Node) -> Optional[Edge]:
    """Flatten all of the edges between the given two nodes.

    Args:
      node1: The first node.
      node2: The second node.

    Returns:
      new_edge: The flattened edge. If there was only one edge between the two
        nodes, then the original edge is returned. If there where no edges
        between the nodes, a None is returned.
    """
    nodes = {node1, node2}
    shared_edges = set()
    # Assuming the network is well formed, all of the edges shared by
    # these two nodes will be stored in just one of the nodes, so we only
    # have to do this loop once.
    for edge in node1.edges:
      if set(edge.get_nodes()) == nodes:
        shared_edges.add(edge)
    if shared_edges:
      return self.flatten_edges(list(shared_edges))
    else:
      return None

  def flatten_all_edges(self) -> List[Edge]:
    """Flatten all edges in the network.

    Returns:
      flattened_edges: A list of all the flattened edges. If there was only one
      edge between two given nodes, that original edge is included in this list.
    """
    nodes = list(self.nodes_set)
    flattened_edges = []
    for i, node1 in enumerate(nodes):
      # We purposely do [i:] instead of [i + 1:] to allow flattening of trace
      # edges.
      for node2 in nodes[i:]:
        flat_edge = self.flatten_edges_between(node1, node2)
        if flat_edge:
          flattened_edges.append(flat_edge)
    return flattened_edges

  def contract_between(self,
                       node1: Node,
                       node2: Node,
                       name: Optional[Text] = None,
                       allow_outer_product: bool = False) -> Node:
    """Contract all of the edges between the two given nodes.

    Args:
      node1: The first node.
      node2: The second node.
      name: Name to give to the new node created.
      allow_outer_product: Optional boolean. If two nodes do not share any edges
        and `allow_outer_product` is set to `True`, then we return the outer
        product of the two nodes. Else, we raise a `ValueError`.

    Returns:
      new_node: The new node created.

    Raises:
      ValueError: If no edges are found between node1 and node2 and
        `allow_outer_product` is set to `False`.
    """
    flat_edge = self.flatten_edges_between(node1, node2)
    if not flat_edge:
      if allow_outer_product:
        return self.outer_product(node1, node2)
      else:
        raise ValueError("No edges found between nodes '{}' and '{}' "
                         "and allow_outer_product=False.".format(node1, node2))
    return self.contract(flat_edge, name)

  def contract_parallel(self, edge: Edge) -> Node:
    """Contract all edges parallel to this edge.

    This method calls `contract_between` with the nodes connected by the edge.

    Args:
      edge: The edge to contract.
    Returns:
      The new node created after contraction.
    """
    if edge.is_dangling():
      raise ValueError("Attempted to contract dangling edge: '{}'".format(edge))
    return self.contract_between(edge.node1, edge.node2)

  def split_node(self,
                 node: Node,
                 left_edges: List[Edge],
                 right_edges: List[Edge],
                 max_singular_values: Optional[int] = None,
                 max_truncation_err: Optional[float] = None
                ) -> Tuple[Node, Node, tf.Tensor]:
    """Split a Node using Singular Value Decomposition.

    Let M be the matrix created by flattening left_edges and right_edges into
    2 axes. Let U S V* = M be the Singular Value Decomposition of M.
    This will split the network into 2 nodes. The left node's tensor will be
    U * sqrt(S) and the right node's tensor will be sqrt(S) * (V*) where V* is
    the adjoint of V.

    Args:
      node: The node you want to split.
      left_edges: The edges you want connected to the new left node.
      right_edges: The edges you want connected to the new right node.
      max_singular_values: The maximum number of singular values to keep.
      max_truncation_err: The maximum allowed truncation error.

    Returns:
      left_node: A new node created that connects to all of the `left_edges`.
      right_node: A new node created that connects to all of the `right_edges`.
      truncated_singular_values: A vector of the dropped smallest singular
        values.
    """
    node.reorder_edges(left_edges + right_edges)
    u, s, vh, trun_vals = decompositions.svd_decomposition(
        node.tensor, len(left_edges), max_singular_values, max_truncation_err)
    sqrt_s = tf.sqrt(s)
    u_s = u * sqrt_s
    # We have to do this since we are doing element-wise multiplication against
    # the first axis of vh. If we don't, it's possible one of the other axes of
    # vh will be the same size as sqrt_s and would multiply across that axis
    # instead, which is bad.
    sqrt_s_broadcast_shape = tf.concat(
        [tf.shape(sqrt_s), [1] * (len(vh.shape) - 1)], axis=-1)
    vh_s = vh * tf.reshape(sqrt_s, sqrt_s_broadcast_shape)
    left_node = self.add_node(u_s)
    for i, edge in enumerate(left_edges):
      left_node.add_edge(edge, i)
      edge.update_axis(i, node, i, left_node)
    right_node = self.add_node(vh_s)
    for i, edge in enumerate(right_edges):
      # i + 1 to account for the new edge.
      right_node.add_edge(edge, i + 1)
      edge.update_axis(i + len(left_edges), node, i + 1, right_node)
    self.connect(left_node[-1], right_node[0])
    self.nodes_set.remove(node)
    return left_node, right_node, trun_vals

  def split_node_full_svd(self,
                          node: Node,
                          left_edges: List[Edge],
                          right_edges: List[Edge],
                          max_singular_values: Optional[int] = None,
                          max_truncation_err: Optional[float] = None
                         ) -> Tuple[Node, Node, Node, tf.Tensor]:
    """Split a node by doing a full singular value decomposition.

    Let M be the matrix created by flattening left_edges and right_edges into
    2 axes. Let U S V* = M be the Singular Value Decomposition of M.
    The left most node will be U tensor of the SVD, the middle node is
    the diagonal matrix of the singular values, ordered largest to smallest,
    and the right most node will be the V* tensor of the SVD.

    Args:
      node: The node you want to split.
      left_edges: The edges you want connected to the new left node.
      right_edges: The edges you want connected to the new right node.
      max_singular_values: The maximum number of singular values to keep.
      max_truncation_err: The maximum allowed truncation error.

    Returns:
      left_node: The new left node created. Its underlying tensor is the same
        as the U tensor from the SVD.
      singular_values_node: The new node representing the diagonal matrix of
        singular values.
      right_node: The new right node created. Its underlying tensor is the same
        as the V* tensor from the SVD.
      truncated_singular_values: The vector of truncated singular values.
    """
    node.reorder_edges(left_edges + right_edges)
    u, s, vh, trun_vals = decompositions.svd_decomposition(
        node.tensor, len(left_edges), max_singular_values, max_truncation_err)
    left_node = self.add_node(u)
    singular_values_node = self.add_node(tf.linalg.diag(s))
    right_node = self.add_node(vh)
    for i, edge in enumerate(left_edges):
      left_node.add_edge(edge, i)
      edge.update_axis(i, node, i, left_node)
    for i, edge in enumerate(right_edges):
      # i + 1 to account for the new edge.
      right_node.add_edge(edge, i + 1)
      edge.update_axis(i + len(left_edges), node, i + 1, right_node)
    self.connect(left_node[-1], singular_values_node[0])
    self.connect(singular_values_node[1], right_node[0])
    self.nodes_set.remove(node)
    return left_node, singular_values_node, right_node, trun_vals

  def check_correct(self, check_connected: bool = True) -> None:
    """Check that the network is structured correctly.

    Args:
      check_connected: Check if the network is connected.

    Raises:
      ValueError: If the tensor network is not correctly structured.
    """
    for node in self.nodes_set:
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
              "Edge axes: {}, {}. Node axis: {}.".format(
                  edge, node, edge.axis1, edge.axis2, i))
    if check_connected:
      self.check_connected()

  def __contains__(self, item):
    if isinstance(item, Edge):
      edge = item
      try:
        edge.node1
        edge.node2
      # If we raise a value error, that means the nodes have been garbage
      # collected, and thus the edge no longer is in the network.
      except ValueError:
        return False
      else:
        edge_is_in_network = edge.node1 in self.nodes_set
        edge_is_in_network &= edge in edge.node1.edges
        if not edge.is_dangling():
          edge_is_in_network &= edge.node2 in self.nodes_set
          edge_is_in_network &= edge in edge.node2.edges
        return edge_is_in_network
    elif isinstance(item, Node):
      return item in self.nodes_set
    else:
      raise TypeError("Type '{}' was unexpected. "
                      "Only 'None' and 'Edge' types are allowed.".format(
                          type(item)))
