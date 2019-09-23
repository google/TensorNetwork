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
import h5py
# pylint: disable=line-too-long
from typing import Any, Sequence, List, Set, Optional, Union, Text, Tuple, Type, Dict, BinaryIO, Collection
import numpy as np
import weakref
from tensornetwork import config
# pylint:disable=useless-import-alias
import tensornetwork.network_components as network_components
from tensornetwork.backends import backend_factory

Tensor = Any
string_type = h5py.special_dtype(vlen=str)


def get_shared_edges(
    node1: network_components.BaseNode,
    node2: network_components.BaseNode) -> Set[network_components.Edge]:
  """Get all edges shared between two nodes.

  Args:
    node1: The first node.
    node2: The second node.

  Returns:
    A (possibly empty) `set` of `Edge`s shared by the nodes.
  """
  nodes = {node1, node2}
  shared_edges = set()
  # Assuming the network is well formed, all of the edges shared by
  # these two nodes will be stored in just one of the nodes, so we only
  # have to do this loop once.
  for edge in node1.edges:
    if set(edge.get_nodes()) == nodes:
      shared_edges.add(edge)
  return shared_edges


def get_parallel_edges(
    edge: network_components.Edge) -> Set[network_components.Edge]:
  """
  Get all of the edges parallel to the given `edge`.
  Args:
    edge: The given edge.

  Returns:
    A `set` of all of the edges parallel to the given edge 
    (including the given edge).
  """
  return get_shared_edges(edge.node1, edge.node2)


def _flatten_trace_edges(edges: List[network_components.Edge],
                         new_edge_name: Optional[Text] = None
                        ) -> network_components.Edge:
  """Flatten trace edges into single edge.

  Args:
    edges: List of trace edges to flatten
    new_edge_name: Optional name of the new edge created.

  Returns:
    The new edge that represents the flattening of the given edges.
  """
  node = edges[0].node1  # We are in the trace case, so this is the only node.
  backend = node.backend
  # Flatten all of the edge's axes into a a single list.
  perm_back = [min(e.axis1, e.axis2) for e in edges]
  perm_back += [max(e.axis1, e.axis2) for e in edges]
  perm_front = set(range(len(node.edges))) - set(perm_back)
  perm_front = sorted(perm_front)
  perm = perm_front + perm_back
  new_dim = backend.prod([backend.shape(node.tensor)[e.axis1] for e in edges])
  node.reorder_axes(perm)
  unaffected_shape = backend.shape(node.tensor)[:len(perm_front)]
  new_shape = backend.concat([unaffected_shape, [new_dim, new_dim]], axis=-1)
  node.tensor = backend.reshape(node.tensor, new_shape)
  edge1 = network_components.Edge(
      node1=node, axis1=len(perm_front), name="TraceFront")
  edge2 = network_components.Edge(
      node1=node, axis1=len(perm_front) + 1, name="TraceBack")
  node.edges = node.edges[:len(perm_front)] + [edge1, edge2]
  new_edge = connect(edge1, edge2, new_edge_name)
  # pylint: disable=expression-not-assigned
  [edge.disable() for edge in edges]  #disable edges!
  return new_edge


def flatten_edges(edges: List[network_components.Edge],
                  new_edge_name: Optional[Text] = None
                 ) -> network_components.Edge:
  """Flatten edges into single edge.

  If two nodes have multiple edges connecting them, it may be
  beneficial to flatten these edges into a single edge to avoid having several
  unnecessary trace edges. This can speed up computation time and reduce
  memory cost.

  Warning: This will remove all axes names.

  Args:
    edges: A list of edges to flatten.
    backend: A backend object
    new_edge_name: Optional name to give to the newly created edge.

  Returns:
    The new flattened edge.

  Raises:
    ValueError: If edges is an empty list.
    ValueError: If not all of the edges connect to the same node(s).
    ValueError: If one of the nodes connecting to these edges does not have
      edge definitions for all of its axes.
  """
  if not edges:
    raise ValueError("At least 1 edge must be given.")

  backends = [edge.node1.backend for edge in edges] + [
      edge.node2.backend for edge in edges if edge.node2 is not None
  ]

  if not all([b.name == backends[0].name for b in backends]):
    raise ValueError("Not all backends are the same.")
  backend = backends[0]
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
    return _flatten_trace_edges(edges, new_edge_name)  #disables edges
  # Flatten standard or dangling edges.
  new_dangling_edges = []
  for node in expected_nodes:
    # Required for dangling case.
    if node is None:
      continue
    axis_names = node.axis_names
    perm_back = []
    for edge in edges:
      # There will only be 1 edge since we are in the standard edge case.
      perm_back.append(node.edges.index(edge))
    perm_front = sorted(set(range(len(node.edges))) - set(perm_back))
    node.reorder_axes(perm_front + perm_back)
    old_tensor_shape = backend.shape(node.tensor)
    # Calculate the new axis dimension as a product of the other
    # axes dimensions.
    flattened_axis_dim = backend.prod(old_tensor_shape[len(perm_front):])
    new_tensor_shape = backend.concat(
        [old_tensor_shape[:len(perm_front)], [flattened_axis_dim]], axis=-1)
    new_tensor = backend.reshape(node.tensor, new_tensor_shape)
    # Modify the node in place. Currently, this is they only method that
    # modifies a node's tensor.
    node.tensor = new_tensor
    # This Edge is required for the connect call later.
    edge = network_components.Edge(
        node1=node, axis1=len(perm_front), name=new_edge_name)
    # Do not set the signature of 'edge' since it is dangling.
    node.edges = node.edges[:len(perm_front)] + [edge]
    new_dangling_edges.append(edge)
    # TODO: Allow renaming of the new axis.
    if axis_names:
      node.axis_names = [axis_names[n] for n in range(len(node.edges))]
    else:
      node.axis_names = None

  node1, node2 = tuple(expected_nodes)
  # Sets are returned in a random order, so this is how we deal with
  # dangling edges.
  # pylint: disable=expression-not-assigned
  [edge.disable() for edge in edges]  #disable edges!
  if node1 is None or node2 is None:
    return new_dangling_edges[0]

  return connect(new_dangling_edges[0], new_dangling_edges[1], new_edge_name)


def flatten_edges_between(
    node1: network_components.BaseNode,
    node2: network_components.BaseNode,
) -> Optional[network_components.Edge]:
  """Flatten all of the edges between the given two nodes.

  Args:
    node1: The first node.
    node2: The second node.

  Returns:
    The flattened `Edge` object. If there was only one edge between the two
      nodes, then the original edge is returned. If there where no edges
      between the nodes, a None is returned.
  """
  shared_edges = get_shared_edges(node1, node2)
  if shared_edges:
    return flatten_edges(list(shared_edges))
  return None


def flatten_all_edges(nodes: Collection[network_components.BaseNode]
                     ) -> List[network_components.Edge]:
  """Flatten all edges in the network.

  Returns:
    A list of all the flattened edges. If there was only one edge between 
    two given nodes, that original edge is included in this list.
  """
  flattened_edges = []
  for edge in get_all_nondangling(nodes):
    if not edge.is_disabled:
      flat_edge = flatten_edges_between(edge.node1, edge.node2)
      flattened_edges.append(flat_edge)
  return flattened_edges


def _remove_trace_edge(edge: network_components.Edge,
                       new_node: network_components.BaseNode) -> None:
  """Collapse a trace edge. `edge` is disabled before returning.

  Take a trace edge (i.e. with edge.node1 = edge.node2),
  remove it, update the axis numbers of all remaining edges
  and move them to `new_node`.

  Args:
    edge: The edge to contract.
    new_node: The new node created after contraction.

  Returns:
    None

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
  edge.node1.fresh_edges(edge.node1.axis_names)
  edge.disable()  #disabled edge!


def _remove_edges(edges: Set[network_components.Edge],
                  node1: network_components.BaseNode,
                  node2: network_components.BaseNode,
                  new_node: network_components.BaseNode) -> None:
  """

  Takes a set of `edges` shared between `node1` and `node2` to be contracted 
  over, and moves all other uncontracted edges from `node1` and `node2` to 
  `new_node`.
  The nodes that currently share the edges in `edges` must be supplied as
  `node1` and `node2`. The ordering of `node1` and `node2` must match the
  axis ordering of `new_node` (as determined by the contraction procedure).
  `node1` and `node2` get both a fresh set edges.
  `edges` are disabled before returning.
  Args:
    edges: The edges to contract.
    node1: The old node that supplies the first edges of `new_node`.
    node2: The old node that supplies the last edges of `new_node`.
    new_node: The new node that represents the contraction of the two old
      nodes.
  Returns:
    node1, node2L
  Raises:
    Value Error: If edge isn't in the network.
  """
  if node1 is node2:
    raise ValueError(
        "node1 and node2 are the same ('{}' == '{}'), but trace edges cannot "
        "be removed by _remove_edges.".format(node1, node2))

  node1_edges = node1.edges[:]
  node2_edges = node2.edges[:]

  nodes_set = set([node1, node2])
  for edge in edges:
    if edge.is_dangling():
      raise ValueError("Attempted to remove dangling edge '{}'.".format(edge))
    if set([edge.node1, edge.node2]) != nodes_set:
      raise ValueError(
          "Attempted to remove edges belonging to different node pairs: "
          "'{}' != '{}'.".format(nodes_set, set([edge.node1, edge.node2])))

  node1_axis_names = node1.axis_names
  node2_axis_names = node2.axis_names

  remaining_edges = []
  for (i, edge) in enumerate(node1_edges):
    if edge not in edges:  # NOTE: Makes the cost quadratic in # edges
      edge.update_axis(
          old_node=node1,
          old_axis=i,
          new_axis=len(remaining_edges),
          new_node=new_node)
      remaining_edges.append(edge)

  for (i, edge) in enumerate(node2_edges):
    if edge not in edges:
      edge.update_axis(
          old_node=node2,
          old_axis=i,
          new_axis=len(remaining_edges),
          new_node=new_node)
      remaining_edges.append(edge)

  for (i, edge) in enumerate(remaining_edges):
    new_node.add_edge(edge, i)

  node1.fresh_edges(node1_axis_names)
  node2.fresh_edges(node2_axis_names)
  # pylint: disable=expression-not-assigned
  [edge.disable() for edge in edges]  #disabled edges!


def _contract_trace(edge: network_components.Edge,
                    name: Optional[Text] = None) -> network_components.BaseNode:
  """Contract a trace edge.
  `edge` is disabled before returning.
  Args:
    edge: The edge name or object to contract next.
    net: An optional TensorNetwork
    name: Name to give to the new node. If None, a name will automatically be
      generated.

  Returns:
    The new node created after the contraction.

  Raise:
    ValueError: When edge is a dangling edge.
  """
  if edge.is_dangling():
    raise ValueError("Attempted to contract dangling edge '{}'".format(edge))
  if edge.node1 is not edge.node2:
    raise ValueError("Can not take trace of edge '{}'. This edge connects to "
                     "two different nodes: '{}' and '{}".format(
                         edge, edge.node1, edge.node2))
  backend = edge.node1.backend
  axes = sorted([edge.axis1, edge.axis2])
  dims = len(edge.node1.tensor.shape)
  permutation = sorted(set(range(dims)) - set(axes)) + axes
  new_tensor = backend.trace(
      backend.transpose(edge.node1.tensor, perm=permutation))

  new_node = network_components.Node(
      new_tensor, name=name, backend=backend.name)
  _remove_trace_edge(edge, new_node)  #disables edge
  return new_node


def contract(
    edge: network_components.Edge,
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
  """
  Contract an edge connecting two nodes in the TensorNetwork.
  All edges of `node1` and `node2` are passed on to the new node, 
  and `node1` and `node2` get a new set of dangling edges.
  `edge is disabled before returning.
  Args:
    edge: The edge contract next.
    name: Name of the new node created.

  Returns:
    The new node created after the contraction.

  Raises:
    ValueError: When edge is a dangling edge or if it already has been
      contracted.
  """
  for node in [edge.node1, edge.node2]:
    if (node is not None) and (not hasattr(node, 'backend')):
      raise TypeError('Node {} of type {} has no `backend`'.format(
          node, type(node)))

  if edge.node1:
    backend = edge.node1.backend
  else:
    raise ValueError("edge {} has no nodes. "
                     "Cannot perfrom a contraction".format(edge.name))

  if edge.is_dangling():
    raise ValueError("Attempting to contract dangling edge")
  backend = edge.node1.backend
  if edge.node1 is edge.node2:
    return _contract_trace(edge, name)
  new_tensor = backend.tensordot(edge.node1.tensor, edge.node2.tensor,
                                 [[edge.axis1], [edge.axis2]])
  new_node = network_components.Node(
      tensor=new_tensor, name=name, axis_names=axis_names, backend=backend.name)
  # edge.node1 and edge.node2 get new edges in _remove_edges
  _remove_edges(set([edge]), edge.node1, edge.node2, new_node)
  return new_node


def outer_product_final_nodes(
    nodes: Collection[network_components.BaseNode],
    edge_order: List[network_components.Edge]) -> network_components.BaseNode:
  """Get the outer product of `nodes`

  For example, if there are 3 nodes remaining in `nodes` with 
  shapes :math:`(2, 3)`, :math:`(4, 5, 6)`, and :math:`(7)`
  respectively, the newly returned node will have shape 
  :math:`(2, 3, 4, 5, 6, 7)`.

  Args:
    nodes: A collection of nodes.
    edge_order: Edge order for the final node.

  Returns:
    The outer product of the remaining nodes.

  Raises:
    ValueError: If any of the remaining nodes are not fully contracted.
  """
  nodes = list(nodes)
  for node in nodes:
    if node.has_nondangling_edge():
      raise ValueError("Node '{}' has a non-dangling edge remaining.")
  final_node = nodes[0]
  for node in nodes[1:]:
    final_node = outer_product(final_node, node)
  return final_node.reorder_edges(edge_order)


def outer_product(
    node1: network_components.BaseNode,
    node2: network_components.BaseNode,
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
  """Calculates an outer product of the two nodes.

  This causes the nodes to combine their edges and axes, so the shapes are
  combined. For example, if `a` had a shape (2, 3) and `b` had a shape
  (4, 5, 6), then the node `net.outer_product(a, b) will have shape
  (2, 3, 4, 5, 6). All edges of `node1` and `node2` are passed on to
  the new node, and `node1` and `node2` get a new set of dangling edges.

  Args:
    node1: The first node. The axes on this node will be on the left side of
      the new node.
    node2: The second node. The axes on this node will be on the right side of
      the new node.
    name: Optional name to give the new node created.
    axis_names: An optional list of names for the axis of the new node
  Returns:
    A new node. Its shape will be node1.shape + node2.shape
  Raises:
    TypeError: If `node1` and `node2` have wrong types.
    ValueError: If `Node` objects `node1` or `node2` have no 
      `TensorNetwork` object.
  """
  for node in [node1, node2]:
    if not hasattr(node, 'backend'):
      raise TypeError('Node {} of type {} has no `backend`'.format(
          node, type(node)))

  if node1.backend.name != node2.backend.name:
    raise ValueError("node {}  and node {} have different backends. "
                     "Cannot perform outer product".format(node1, node2))

  backend = node1.backend
  new_tensor = backend.outer_product(node1.tensor, node2.tensor)
  node1_axis_names = node1.axis_names
  node2_axis_names = node2.axis_names
  new_node = network_components.Node(
      tensor=new_tensor, name=name, axis_names=axis_names, backend=backend.name)
  additional_axes = len(node1.tensor.shape)

  for i, edge in enumerate(node1.edges):
    edge.update_axis(i, node1, i, new_node)
  for i, edge in enumerate(node2.edges):
    edge.update_axis(i, node2, i + additional_axes, new_node)

  for i, edge in enumerate(node1.edges + node2.edges):
    new_node.add_edge(edge, i, True)

  node1.fresh_edges(node1_axis_names)
  node2.fresh_edges(node2_axis_names)

  return new_node


def contract_copy_node(copy_node: network_components.CopyNode,
                       name: Optional[Text] = None
                      ) -> network_components.BaseNode:
  """Contract all edges incident on given copy node.

  Args:
    copy_node: Copy tensor node to be contracted.
    name: Name of the new node created.

  Returns:
    New node representing contracted tensor.

  Raises:
    ValueError: If copy_node has dangling edge(s).
  """
  new_tensor = copy_node.compute_contracted_tensor()
  new_node = network_components.Node(
      new_tensor, name, backend=copy_node.backend.name)

  partners = copy_node.get_partners()
  new_axis = 0
  for partner in partners:
    for edge in partner.edges:
      if edge.node1 is copy_node or edge.node2 is copy_node:
        continue
      old_axis = edge.axis1 if edge.node1 is partner else edge.axis2
      edge.update_axis(
          old_node=partner,
          old_axis=old_axis,
          new_node=new_node,
          new_axis=new_axis)
      new_node.add_edge(edge, new_axis)
      new_axis += 1
  assert len(new_tensor.shape) == new_axis
  copy_node.fresh_edges(copy_node.axis_names)
  return new_node


def contract_between(
    node1: network_components.BaseNode,
    node2: network_components.BaseNode,
    name: Optional[Text] = None,
    allow_outer_product: bool = False,
    output_edge_order: Optional[Sequence[network_components.Edge]] = None,
    axis_names: Optional[List[Text]] = None,
) -> network_components.BaseNode:
  """Contract all of the edges between the two given nodes.

  Args:
    node1: The first node.
    node2: The second node.
    name: Name to give to the new node created.
    allow_outer_product: Optional boolean. If two nodes do not share any edges
      and `allow_outer_product` is set to `True`, then we return the outer
      product of the two nodes. Else, we raise a `ValueError`.
    output_edge_order: Optional sequence of Edges. When not `None`, must
      contain all edges belonging to, but not shared by `node1` and `node2`.
      The axes of the new node will be permuted (if necessary) to match this
      ordering of Edges.
    axis_names: An optional list of names for the axis of the new node
  Returns:
    The new node created.

  Raises:
    ValueError: If no edges are found between node1 and node2 and
      `allow_outer_product` is set to `False`.
  """
  for node in [node1, node2]:
    if not hasattr(node, 'backend'):
      raise TypeError('Node {} of type {} has no `backend`'.format(
          node, type(node)))

  if node1.backend.name != node2.backend.name:
    raise ValueError("node {}  and node {} have different backends. "
                     "Cannot perform outer product".format(node1, node2))

  backend = node1.backend
  # Trace edges cannot be contracted using tensordot.
  if node1 is node2:
    flat_edge = flatten_edges_between(node1, node2)
    if not flat_edge:
      raise ValueError("No trace edges found on contraction of edges between "
                       "node '{}' and itself.".format(node1))
    return contract(flat_edge, name)

  shared_edges = get_shared_edges(node1, node2)
  if not shared_edges:
    if allow_outer_product:
      return outer_product(node1, node2, name=name, axis_names=axis_names)
    raise ValueError("No edges found between nodes '{}' and '{}' "
                     "and allow_outer_product=False.".format(node1, node2))

  # Collect the axis of each node corresponding to each edge, in order.
  # This specifies the contraction for tensordot.
  # NOTE: The ordering of node references in each contraction edge is ignored.
  axes1 = []
  axes2 = []
  for edge in shared_edges:
    if edge.node1 is node1:
      axes1.append(edge.axis1)
      axes2.append(edge.axis2)
    else:
      axes1.append(edge.axis2)
      axes2.append(edge.axis1)

  if output_edge_order:
    # Determine heuristically if output transposition can be minimized by
    # flipping the arguments to tensordot.
    node1_output_axes = []
    node2_output_axes = []
    for (i, edge) in enumerate(output_edge_order):
      if edge in shared_edges:
        raise ValueError(
            "Edge '{}' in output_edge_order is shared by the nodes to be "
            "contracted: '{}' and '{}'.".format(edge, node1, node2))
      edge_nodes = set(edge.get_nodes())
      if node1 in edge_nodes:
        node1_output_axes.append(i)
      elif node2 in edge_nodes:
        node2_output_axes.append(i)
      else:
        raise ValueError(
            "Edge '{}' in output_edge_order is not connected to node '{}' or "
            "node '{}'".format(edge, node1, node2))
    if np.mean(node1_output_axes) > np.mean(node2_output_axes):
      node1, node2 = node2, node1
      axes1, axes2 = axes2, axes1

  new_tensor = backend.tensordot(node1.tensor, node2.tensor, [axes1, axes2])
  new_node = network_components.Node(
      tensor=new_tensor, name=name, axis_names=axis_names, backend=backend.name)
  # node1 and node2 get new edges in _remove_edges
  _remove_edges(shared_edges, node1, node2, new_node)
  if output_edge_order:
    new_node = new_node.reorder_edges(list(output_edge_order))
  return new_node


def contract_parallel(
    edge: network_components.Edge) -> network_components.BaseNode:
  """Contract all edges parallel to this edge.

    This method calls `contract_between` with the nodes connected by the edge.

    Args:
      edge: The edge to contract.

    Returns:
      The new node created after contraction.
    """
  if edge.is_dangling():
    raise ValueError("Attempted to contract dangling edge: '{}'".format(edge))
  return contract_between(edge.node1, edge.node2)


def connect(edge1: network_components.Edge,
            edge2: network_components.Edge,
            name: Optional[Text] = None) -> network_components.Edge:
  for edge in [edge1, edge2]:
    if not edge.is_dangling():
      raise ValueError("Edge '{}' is not a dangling edge. "
                       "This edge points to nodes: '{}' and '{}'".format(
                           edge, edge.node1, edge.node2))
  if edge1 is edge2:
    raise ValueError("Cannot connect and edge '{}' to itself.".format(edge1))

  if edge1.dimension != edge2.dimension:
    raise ValueError("Cannot connect edges of unequal dimension. "
                     "Dimension of edge '{}': {}, "
                     "Dimension of edge '{}': {}.".format(
                         edge1, edge2.dimension, edge2, edge2.dimension))

  #edge1 and edg2 are always dangling in this case
  node1 = edge1.node1
  node2 = edge2.node1
  axis1_num = node1.get_axis_number(edge1.axis1)
  axis2_num = node2.get_axis_number(edge2.axis1)

  new_edge = network_components.Edge(
      node1=node1, axis1=axis1_num, name=name, node2=node2, axis2=axis2_num)

  node1.add_edge(new_edge, axis1_num, override=True)
  node2.add_edge(new_edge, axis2_num, override=True)
  return new_edge


def disconnect(edge, edge1_name: Optional[Text], edge2_name: Optional[Text]
              ) -> Tuple[network_components.Edge, network_components.Edge]:
  """
  Break an existing non-dangling edge.
  This updates both Edge.node1 and Edge.node2 by removing the 
  connecting edge from `Edge.node1.edges` and `Edge.node2.edges`
  and adding new dangling edges instead
  """
  return edge.disconnect(edge1_name, edge2_name)


def conj(
    node: network_components.BaseNode,
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
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

  return network_components.Node(
      backend.conj(node.tensor),
      name=name,
      axis_names=axis_names,
      backend=backend.name)


def transpose(
    node: network_components.BaseNode,
    permutation: Sequence[Union[Text, int]],
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
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

  new_node = network_components.Node(
      node.tensor,
      name=name,
      axis_names=node.axis_names,
      backend=node.backend.name)
  return new_node.reorder_axes(perm)


def split_node(
    node: network_components.BaseNode,
    left_edges: List[network_components.Edge],
    right_edges: List[network_components.Edge],
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[network_components.BaseNode, network_components.BaseNode, Tensor]:
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
    left_name: The name of the new left node. If `None`, a name will be generated
      automatically.
    right_name: The name of the new right node. If `None`, a name will be generated
      automatically.
    edge_name: The name of the new `Edge` connecting the new left and right node. 
      If `None`, a name will be generated automatically. The new axis will get the
      same name as the edge.

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
  left_node = network_components.Node(
      u_s, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = network_components.Node(
      vh_s, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  node.fresh_edges(node.axis_names)
  return left_node, right_node, trun_vals


def split_node_qr(
    node: network_components.BaseNode,
    left_edges: List[network_components.Edge],
    right_edges: List[network_components.Edge],
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[network_components.BaseNode, network_components.BaseNode]:
  """Split a `Node` using QR decomposition

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`QR = M` be the QR Decomposition of
  :math:`M`. This will split the network into 2 nodes. The left node's
  tensor will be :math:`Q` (an orthonormal matrix) and the right node's tensor will be
  :math:`R` (an upper triangular matrix)

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    left_name: The name of the new left node. If `None`, a name will be generated
      automatically.
    right_name: The name of the new right node. If `None`, a name will be generated
      automatically.
    edge_name: The name of the new `Edge` connecting the new left and right node.
      If `None`, a name will be generated automatically.

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
  left_node = network_components.Node(
      q, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = network_components.Node(
      r, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  return left_node, right_node


def split_node_rq(
    node: network_components.BaseNode,
    left_edges: List[network_components.Edge],
    right_edges: List[network_components.Edge],
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[network_components.BaseNode, network_components.BaseNode]:
  """Split a `Node` using RQ (reversed QR) decomposition

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`QR = M^*` be the QR Decomposition of
  :math:`M^*`. This will split the network into 2 nodes. The left node's
  tensor will be :math:`R^*` (a lower triangular matrix) and the right node's tensor will be
  :math:`Q^*` (an orthonormal matrix)

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    left_name: The name of the new left node. If `None`, a name will be generated
      automatically.
    right_name: The name of the new right node. If `None`, a name will be generated
      automatically.
    edge_name: The name of the new `Edge` connecting the new left and right node.
      If `None`, a name will be generated automatically.

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
  left_node = network_components.Node(
      r, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = network_components.Node(
      q, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  return left_node, right_node


def split_node_full_svd(
    node: network_components.BaseNode,
    left_edges: List[network_components.Edge],
    right_edges: List[network_components.Edge],
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    left_name: Optional[Text] = None,
    middle_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    left_edge_name: Optional[Text] = None,
    right_edge_name: Optional[Text] = None,
) -> Tuple[network_components.BaseNode, network_components
           .BaseNode, network_components.BaseNode, Tensor]:
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
    left_name: The name of the new left node. If None, a name will be generated
      automatically.
    middle_name: The name of the new center node. If None, a name will be generated
      automatically.
    right_name: The name of the new right node. If None, a name will be generated
      automatically.
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
  left_node = network_components.Node(
      u, name=left_name, axis_names=left_axis_names, backend=backend.name)
  singular_values_node = network_components.Node(
      backend.diag(s),
      name=middle_name,
      axis_names=center_axis_names,
      backend=backend.name)

  right_node = network_components.Node(
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


def reachable(nodes: Union[network_components
                           .BaseNode, Collection[network_components.BaseNode]],
              strategy: Optional[Text] = 'recursive'
             ) -> Collection[network_components.BaseNode]:
  """
  Computes all nodes reachable from `node` by connected edges.
  Args:
    node: A `BaseNode`
    strategy: The strategy to be used to find all nodes. Currently 
      if `recursive`, use a recursive approach, if `iterative`, use
      and iterative approach, if 'deque' use an iterative approach 
      using itertools.deque.
  Returns:
    A list of `BaseNode` objects that can be reached from `node`
    via connected edges.
  Raises:
    ValueError: If an unknown value for `strategy` is passed.
  """
  if isinstance(nodes, network_components.BaseNode):
    nodes = [nodes]
  reachable_nodes = {nodes[0]}
  if strategy == 'recursive':
    for node in nodes:
      reachable_nodes |= reachable_recursive(node)
  elif strategy == 'iterative':
    for node in nodes:
      reachable_nodes |= reachable_iterative(node)
  elif strategy == 'deque':
    for node in nodes:
      reachable_nodes |= reachable_deque(node)
  else:
    raise ValueError("Unknown value '{}' for `strategy`.".format(strategy))
  return reachable_nodes


def reachable_deque(
    node: network_components.BaseNode) -> Set[network_components.BaseNode]:
  """
  Computes all nodes reachable from `node` by connected edges. This function uses 
  itertools.deque.
  Args:
    node: A `BaseNode`
  Returns:
    A set of `BaseNode` objects that can be reached from `node`
    via connected edges.

  """
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


def reachable_recursive(
    node: network_components.BaseNode) -> Set[network_components.BaseNode]:
  """
  Computes all nodes reachable from `node` by connected edges. This function uses 
  recursion
  Args:
    node: A `BaseNode`
  Returns:
    A set of `BaseNode` objects that can be reached from `node`
    via connected edges.

  """
  reachable_nodes = {node}

  def _reachable(node):
    for edge in node.edges:
      if edge.is_dangling():
        continue
      next_node = edge.node1 if edge.node1 is not node else edge.node2
      if next_node in reachable_nodes:
        continue
      reachable_nodes.add(next_node)
      _reachable(next_node)

  _reachable(node)

  return reachable_nodes


def reachable_iterative(
    node: network_components.BaseNode) -> Set[network_components.BaseNode]:
  """
  Computes all nodes reachable from `node` by connected edges. This function uses 
  an iterative strategy.
  Args:
    node: A `BaseNode`
  Returns:
    A list of `BaseNode` objects that can be reached from `node`
    via connected edges.
  """
  #TODO: this is recursive; use while loop for better performance
  reachable_nodes = {node}
  depth = 0
  while True:
    old_depth = depth
    for edge in node.edges:

      if edge.is_dangling():
        continue
      next_node = edge.node1 if edge.node1 is not node else edge.node2
      if next_node in reachable_nodes:
        continue
      reachable_nodes.add(next_node)
      depth += 1
      node = next_node
      continue

    depth -= 1 if depth == old_depth else 0
    if depth == 0:
      break
  return reachable_nodes


def check_correct(nodes: Collection[network_components.BaseNode],
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


def check_connected(nodes: Collection[network_components.BaseNode]) -> None:
  """
  Check if all nodes in `nodes` are connected.
  Args:
    nodes: A list of nodes.
  Returns:
    None:
  Raises:
    ValueError: If not all nodes in `nodes` are connected.
  """
  if not set(nodes) <= reachable_deque(list(nodes)[0]):
    raise ValueError("Non-connected graph")


def get_all_nondangling(nodes: Collection[network_components.BaseNode]
                       ) -> Set[network_components.Edge]:
  """Return the set of all non-dangling edges."""
  edges = set()
  for node in nodes:
    edges |= node.get_all_nondangling()
  return edges


def get_all_nodes(edges: Collection[network_components.Edge]
                 ) -> Set[network_components.BaseNode]:
  """Return the set of nodes connected to edges."""
  nodes = set()
  for edge in edges:
    if edge.node1 is not None:
      nodes |= {edge.node1}
    if edge.node2 is not None:
      nodes |= {edge.node2}

  return nodes


def get_all_edges(nodes: Collection[network_components.BaseNode]
                 ) -> Set[network_components.Edge]:
  """Return the set of edges of all nodes."""
  edges = set()
  for node in nodes:
    edges |= set(node.edges)
  return edges


class TensorNetwork:
  """Implementation of a TensorNetwork."""

  def __init__(self,
               backend: Optional[Text] = None,
               dtype: Optional[Type[np.number]] = None) -> None:
    """
    Args:
      backend (str): name of the backend. Currently supported 
                     backends are 'numpy', 'tensorflow', 'pytorch', 'jax', 'shell'
      dtype: dtype of the backend of network. If `None`, no dtype checks 
             are performed. Default value is `None`. For backend 
             initialization functions like `zeros`, `ones`, `randn` a 
             dtype of `None` defaults to float64
    """
    if backend is None:
      backend = config.default_backend
    if dtype is None:
      dtype = config.default_dtype
    #backend.dtype is initialized from config.py (currently `None`)
    #if `backend.dtype = None`, the backend dtype is set at the first
    #call to `add_node(tensor)` to `backend.dtype = tensor.dtype`
    #if `dtype` is is set at initialization, all tensors added to
    #the network have to have the same `dtype` as the backend
    self.backend = backend_factory.get_backend(backend, dtype)
    self.nodes_set = set()
    # These increments are only used for generating names.
    self.node_increment = 0
    self.edge_increment = 0

  def _new_edge_name(self, name: Optional[Text]) -> Text:
    self.edge_increment += 1
    if name is None:
      name = "__Edge_{}".format(self.edge_increment)
    return name

  def _new_node_name(self, name: Optional[Text]) -> Text:
    self.node_increment += 1
    if name is None:
      name = "__Node_{}".format(self.node_increment)
    return name

  @property
  def dtype(self) -> Type[np.number]:
    return self.backend.dtype

  # pylint: disable=redefined-outer-name
  def copy(self, conj: bool = False) -> Tuple["TensorNetwork", dict, dict]:
    """

    Return a copy of the TensorNetwork.
    Args:
      conj: Boolean. Whether to conjugate all of the nodes in the
        `TensorNetwork` (useful for calculating norms and reduced density
        matrices).
    Returns:
      A tuple containing:
        TensorNetwork: A copy of the network.
        node_dict: A dictionary mapping the nodes of the original
                   network to the nodes of the copy.
        edge_dict: A dictionary mapping the edges of the original
                   network to the edges of the copy.
    """
    new_net = TensorNetwork(backend=self.backend.name)
    #TODO: add support for copying CopyTensor
    if conj:
      node_dict = {
          node: new_net.add_node(
              self.backend.conj(node.tensor),
              name=node.name,
              axis_names=node.axis_names) for node in self.nodes_set
      }
    else:
      node_dict = {
          node: new_net.add_node(
              node.tensor, name=node.name, axis_names=node.axis_names)
          for node in self.nodes_set
      }
    edge_dict = {}
    for edge in self.get_all_edges():
      node1 = edge.node1
      axis1 = edge.node1.get_axis_number(edge.axis1)

      if not edge.is_dangling():
        node2 = edge.node2
        axis2 = edge.node2.get_axis_number(edge.axis2)
        new_edge = network_components.Edge(node_dict[node1], axis1, edge.name,
                                           node_dict[node2], axis2)
        new_edge.set_signature(edge.signature)
      else:
        new_edge = network_components.Edge(node_dict[node1], axis1, edge.name)

      node_dict[node1].add_edge(new_edge, axis1)
      if not edge.is_dangling():
        node_dict[node2].add_edge(new_edge, axis2)
      edge_dict[edge] = new_edge
    return new_net, node_dict, edge_dict

  def add_subnetwork(self, subnetwork: "TensorNetwork") -> None:
    """Add a subnetwork to an existing network.

    Args:
      subnetwork: A TensorNetwork object. The nodes and edges of this network
        will be merged into the original network.
    """
    if subnetwork.backend.name != self.backend.name:
      raise ValueError("Incompatible backends found: {}, {}".format(
          self.backend.name, subnetwork.backend.name))
    self.nodes_set |= subnetwork.nodes_set
    for node in subnetwork.nodes_set:
      node.set_signature(node.signature + self.node_increment)
      node.network = self
    # Add increment for namings.
    self.node_increment += subnetwork.node_increment
    self.edge_increment += subnetwork.edge_increment

  # TODO: Add pytypes once we figure out why it crashes.
  @classmethod
  def merge_networks(cls, networks: List["TensorNetwork"]) -> "TensorNetwork":
    """Merge several networks into a single network.

    Args:
      networks: An iterable of TensorNetworks.

    Returns:
      A new network created by the merging of all of the given networks.
    """
    backend_dtypes = {net.backend.dtype for net in networks}
    backend_types = {net.backend.name for net in networks}
    if len(backend_types) != 1:
      raise ValueError("Multiple incompatible backends found: {}".format(
          list(backend_types)))
    #check if either all or all but one network have `dtype == None`
    dtypes = {dtype for dtype in backend_dtypes if dtype is not None}
    if len(dtypes) > 1:
      raise ValueError("backends dtypes {} are not compatible".format(dtypes))
    if len(dtypes) == 1:
      final_dtype = list(dtypes)[0]
    else:
      final_dtype = None
    new_network = cls(backend=networks[0].backend.name, dtype=final_dtype)

    for network in networks:
      new_network.add_subnetwork(network)
    return new_network

  def switch_backend(self,
                     new_backend: Text,
                     dtype: Optional[Type[np.number]] = None) -> None:
    """Change this network's backend.

    This will convert all node's tensors to the new backend's Tensor type.
    Args:
      new_backend (str): The new backend.
      dtype (datatype): The dtype of the backend. If None, a defautl dtype according
                         to config.py will be chosen.
    """
    if self.backend.name != "numpy":
      raise NotImplementedError(
          "Can only switch backends when the current "
          "backend is 'numpy'. Current backend is '{}'".format(
              self.backend.name))
    self.backend = backend_factory.get_backend(new_backend, dtype)
    for node in self.nodes_set:
      node.tensor = self.backend.convert_to_tensor(node.tensor)

  def add_node(
      self,
      value: Union[np.ndarray, Tensor, network_components.BaseNode],
      name: Optional[Text] = None,
      axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
    """Create a new node in the network.

    Args:
      value: Either the concrete tensor or an existing `Node` object that
        has no associated `TensorNetwork`. If a concrete tensor is given,
        a new node will be created.
      name: The name of the new node. If None, a name will be generated
        automatically.
      axis_names: Optional list of strings to name each of the axes.

    Returns:
      The new node object.

    Raises:
      ValueError: If `name` already exists in the network.
    """
    given_axis_name = axis_names is not None
    given_node_name = name is not None
    if axis_names is None:
      axis_names = [self._new_edge_name(None) for _ in range(len(value.shape))]
    name = self._new_node_name(name)
    if isinstance(value, network_components.BaseNode):
      new_node = value
      if new_node.network is not None:
        raise ValueError("Given node is already part of a network.")
      new_node.network = self
      if new_node.axis_names is None or given_axis_name:
        new_node.axis_names = axis_names
      if new_node.name is None or given_node_name:
        new_node.name = name
    else:
      value = self.backend.convert_to_tensor(value)
      if self.backend.dtype is None:
        self.backend.dtype = value.dtype
      new_node = network_components.Node(
          value, name, axis_names, backend=self.backend.name, network=self)
      new_node.network = self  #set network manually
    new_node.set_signature(self.node_increment)
    self.nodes_set.add(new_node)
    return new_node

  def add_copy_node(
      self,
      rank: int,
      dimension: int,
      name: Optional[Text] = None,
      axis_names: Optional[List[Text]] = None,
      dtype: Type[np.number] = None) -> network_components.CopyNode:
    """Create a new copy node in the network.

    Copy node represents the copy tensor, i.e. tensor :math:`C` such that
    :math:`C_{ij...k} = 1` if :math:`i = j = ... = k` and 
    :math:`C_{ij...k}= 0` otherwise.

    Args:
      rank: Number of edges of the copy tensor.
      dimension: Dimension of each edge.
      name: The name of the new node. If None, a name will be generated
        automatically.
      axis_names: Optional list of strings to name each of the axes.

    Returns:
      The new node object.

    Raises:
      ValueError: If `name` already exists in the network.
    """
    name = self._new_node_name(name)
    if axis_names is None:
      axis_names = [self._new_edge_name(None) for _ in range(rank)]
    new_node = network_components.CopyNode(
        rank=rank,
        dimension=dimension,
        name=name,
        axis_names=axis_names,
        network=self,
        backend=self.backend.name,
        dtype=dtype)
    new_node.set_signature(self.node_increment)
    self.nodes_set.add(new_node)
    return new_node

  def connect(self,
              edge1: network_components.Edge,
              edge2: network_components.Edge,
              name: Optional[Text] = None) -> network_components.Edge:
    """Join two dangling edges into a new edge.

    Args:
      edge1: The first dangling edge.
      edge2: The second dangling edge.
      name: Optional name to give the new edge.

    Returns:
      A new edge created by joining the two dangling edges together.

    Raises:
      ValueError: If either edge1 or edge2 is not a dangling edge or if edge1
        and edge2 are the same edge.
    """
    name = self._new_edge_name(name)
    new_edge = connect(edge1, edge2, name)
    new_edge.set_signature(self.edge_increment)
    return new_edge

  def disconnect(self,
                 edge: network_components.Edge,
                 dangling_edge_name_1: Optional[Text] = None,
                 dangling_edge_name_2: Optional[Text] = None
                ) -> Tuple[network_components.Edge, network_components.Edge]:
    """Break a edge into two dangling edges.

    Args:
      edge: An edge to break.
      dangling_edge_name_1: Optional name to give the new dangling edge 1.
      dangling_edge_name_2: Optional name to give the new dangling edge 2.

    Returns:
      A tuple of the two new dangling edges.

    Raises:
     ValueError: If input edge is a dangling one.
    """
    return disconnect(edge, dangling_edge_name_1, dangling_edge_name_2)

  def _remove_trace_edge(self, edge: network_components.Edge,
                         new_node: network_components.BaseNode) -> None:
    """Collapse a trace edge.

    Collapses a trace edge and updates the network.

    Args:
      edge: The edge to contract.
      new_node: The new node created after contraction.

    Returns:
      None

    Raises:
      ValueError: If edge is not a trace edge.
    """

    _remove_trace_edge(edge, new_node)
    node = edge.node1
    self.nodes_set.remove(edge.node1)
    node.disable()

  def _remove_edges(self, edges: Set[network_components.Edge],
                    node1: network_components.BaseNode,
                    node2: network_components.BaseNode,
                    new_node: network_components.BaseNode) -> None:
    """Collapse a list of edges shared by two nodes in the network.

    Collapses the edges and updates the rest of the network.
    The nodes that currently share the edges in `edges` must be supplied as
    `node1` and `node2`. The ordering of `node1` and `node2` must match the
    axis ordering of `new_node` (as determined by the contraction procedure).

    Args:
      edges: The edges to contract.
      node1: The old node that supplies the first edges of `new_node`.
      node2: The old node that supplies the last edges of `new_node`.
      new_node: The new node that represents the contraction of the two old
        nodes.

    Raises:
      Value Error: If edge isn't in the network.
    """
    _remove_edges(edges, node1, node2, new_node)

    if node1 in self.nodes_set:
      self.nodes_set.remove(node1)
    if node2 in self.nodes_set:
      self.nodes_set.remove(node2)
    if not node1.is_disabled:
      node1.disable()
    if not node1.is_disabled:
      node2.disable()

  def _contract_trace(self,
                      edge: network_components.Edge,
                      name: Optional[Text] = None
                     ) -> network_components.BaseNode:
    """Contract a trace edge connecting in the TensorNetwork.

    Args:
      edge: The edge name or object to contract next.
      name: Name to give to the new node. If None, a name will automatically be
        generated.

    Returns:
      The new node created after the contraction.

    Raise:
      ValueError: When edge is a dangling edge.
    """
    new_node = self.add_node(_contract_trace(edge, name))
    node = edge.node1
    self.nodes_set.remove(edge.node1)
    node.disable()
    self.nodes_set.add(new_node)
    return new_node

  def contract(self, edge: network_components.Edge,
               name: Optional[Text] = None) -> network_components.BaseNode:
    """Contract an edge connecting two nodes in the TensorNetwork.

    Args:
      edge: The edge contract next.
      name: Name of the new node created.

    Returns:
      The new node created after the contraction.

    Raises:
      ValueError: When edge is a dangling edge or if it already has been
        contracted.
    """
    new_node = self.add_node(contract(edge, name, axis_names=None))
    # disable nodes
    if edge.node1 is edge.node2:
      node = edge.node1
      self.nodes_set.remove(edge.node1)
      node.disable()
    else:
      node1 = edge.node1
      node2 = edge.node2
      self.nodes_set.remove(edge.node1)
      self.nodes_set.remove(edge.node2)
      node1.disable()
      node2.disable()
    return new_node

  def contract_copy_node(self,
                         copy_node: network_components.CopyNode,
                         name: Optional[Text] = None
                        ) -> network_components.BaseNode:
    """Contract all edges incident on given copy node.

    Args:
      copy_node: Copy tensor node to be contracted.
      name: Name of the new node created.

    Returns:
      New node representing contracted tensor.

    Raises:
      ValueError: If copy_node has dangling edge(s).
    """
    partners = copy_node.get_partners()
    new_node = self.add_node(contract_copy_node(copy_node, name))
    # Remove nodes
    for partner in partners:
      if partner in self.nodes_set:
        self.nodes_set.remove(partner)
    for partner in partners:
      if not partner.is_disabled:
        partner.disable()

    self.nodes_set.remove(copy_node)
    copy_node.disable()
    return new_node

  def outer_product(self,
                    node1: network_components.BaseNode,
                    node2: network_components.BaseNode,
                    name: Optional[Text] = None) -> network_components.BaseNode:
    """Calculates an outer product of the two nodes.

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
      A new node. Its shape will be node1.shape + node2.shape
    """
    new_node = self.add_node(outer_product(node1, node2, name, axis_names=None))
    # Remove the nodes from the set.
    if node1 in self.nodes_set:
      self.nodes_set.remove(node1)
    if node2 in self.nodes_set:
      self.nodes_set.remove(node2)
    if not node1.is_disabled:
      node1.disable()
    if not node2.is_disabled:
      node2.disable()

    return new_node

  def get_final_node(self) -> network_components.BaseNode:
    """Get the final node of a fully contracted network.

    Note: The network must already be fully contracted to a single node.

    Returns:
      The final node in the network.

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
    return get_all_nondangling(self.nodes_set)

  def get_all_edges(self):
    """Return the set of all edges."""
    return get_all_edges(self.nodes_set)

  def outer_product_final_nodes(self, edge_order: List[network_components.Edge]
                               ) -> network_components.BaseNode:
    """Get the outer product of the final nodes.

    For example, if after all contractions, there were 3 nodes remaining with
    shapes :math:`(2, 3)`, :math:`(4, 5, 6)`, and :math:`(7)`
    respectively, the newly returned node will have shape 
    :math:`(2, 3, 4, 5, 6, 7)`.

    Args:
      edge_order: Edge order for the final node.

    Returns:
      The outer product of the remaining nodes.

    Raises:
      ValueError: If any of the remaining nodes are not fully contracted.
    """
    return outer_product_final_nodes(self.nodes_set, edge_order)

  def check_connected(self) -> None:
    """Check if the network is connected."""
    # Fastest way to get a single item from a set.
    check_connected(self.nodes_set)

  def _flatten_trace_edges(
      self, edges: List[network_components.Edge],
      new_edge_name: Optional[Text]) -> network_components.Edge:
    """Flatten trace edges into single edge.

    Args:
      edges: List of trace edges to flatten
      new_edge_name: Optional name of the new edge created.

    Returns:
      The new edge that represents the flattening of the given edges.
    """
    return _flatten_trace_edges(edges, new_edge_name)

  def flatten_edges(
      self,
      edges: List[network_components.Edge],
      new_edge_name: Optional[Text] = None) -> network_components.Edge:
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
      The new flattened edge.

    Raises:
      ValueError: If edges is an empty list.
      ValueError: If not all of the edges connect to the same node(s).
      ValueError: If one of the nodes connecting to these edges does not have
        edge definitions for all of its axes.
    """
    return flatten_edges(edges, new_edge_name)

  def get_shared_edges(
      self, node1: network_components.BaseNode,
      node2: network_components.BaseNode) -> Set[network_components.Edge]:
    """Get all edges shared between two nodes.

    Args:
      node1: The first node.
      node2: The second node.

    Returns:
      A (possibly empty) `set` of `Edge`s shared by the nodes.
    """
    return get_shared_edges(node1, node2)

  def get_parallel_edges(
      self, edge: network_components.Edge) -> Set[network_components.Edge]:
    """Get all of the edges parallel to the given `edge`.

    Args:
      edge: The given edge.

    Returns:
      A `set` of all of the edges parallel to the given edge 
      (including the given edge).
  """
    return self.get_shared_edges(edge.node1, edge.node2)

  def flatten_edges_between(
      self, node1: network_components.BaseNode,
      node2: network_components.BaseNode) -> Optional[network_components.Edge]:
    """Flatten all of the edges between the given two nodes.

    Args:
      node1: The first node.
      node2: The second node.

    Returns:
      The flattened `Edge` object. If there was only one edge between the two
        nodes, then the original edge is returned. If there where no edges
        between the nodes, a None is returned.
    """
    return flatten_edges_between(node1, node2)

  def flatten_all_edges(self) -> List[Optional[network_components.Edge]]:
    """Flatten all edges in the network.

    Returns:
      A list of all the flattened edges. If there was only one edge between 
      two given nodes, that original edge is included in this list.
    """
    flattened_edges = []
    for edge in self.get_all_nondangling():
      if edge in self:
        flat_edge = self.flatten_edges_between(edge.node1, edge.node2)
        flattened_edges.append(flat_edge)
    return flattened_edges

  def contract_between(
      self,
      node1: network_components.BaseNode,
      node2: network_components.BaseNode,
      name: Optional[Text] = None,
      allow_outer_product: bool = False,
      output_edge_order: Optional[Sequence[network_components.Edge]] = None,
  ) -> network_components.BaseNode:
    """Contract all of the edges between the two given nodes.

    Args:
      node1: The first node.
      node2: The second node.
      name: Name to give to the new node created.
      allow_outer_product: Optional boolean. If two nodes do not share any edges
        and `allow_outer_product` is set to `True`, then we return the outer
        product of the two nodes. Else, we raise a `ValueError`.
      output_edge_order: Optional sequence of Edges. When not `None`, must 
        contain all edges belonging to, but not shared by `node1` and `node2`.
        The axes of the new node will be permuted (if necessary) to match this
        ordering of Edges.

    Returns:
      The new node created.

    Raises:
      ValueError: If no edges are found between node1 and node2 and
        `allow_outer_product` is set to `False`.
    """
    new_node = self.add_node(
        contract_between(
            node1,
            node2,
            name,
            allow_outer_product,
            output_edge_order,
            axis_names=None))
    if node1 in self.nodes_set:
      self.nodes_set.remove(node1)
    if node2 in self.nodes_set:
      self.nodes_set.remove(node2)
    if not node1.is_disabled:
      node1.disable()
    if not node2.is_disabled:
      node2.disable()

    return new_node

  def contract_parallel(
      self, edge: network_components.Edge) -> network_components.BaseNode:
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

  def split_node(
      self,
      node: network_components.BaseNode,
      left_edges: List[network_components.Edge],
      right_edges: List[network_components.Edge],
      max_singular_values: Optional[int] = None,
      max_truncation_err: Optional[float] = None,
      left_name: Optional[Text] = None,
      right_name: Optional[Text] = None,
      edge_name: Optional[Text] = None,
  ) -> Tuple[network_components.BaseNode, network_components.BaseNode, Tensor]:
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
      left_name: The name of the new left node. If `None`, a name will be generated
        automatically.
      right_name: The name of the new right node. If `None`, a name will be generated
        automatically.
      edge_name: The name of the new `Edge` connecting the new left and right node. 
        If `None`, a name will be generated automatically.

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
    left, right, trun_vals = split_node(node, left_edges, right_edges,
                                        max_singular_values, max_truncation_err,
                                        left_name, right_name, edge_name)
    left_node = self.add_node(left)
    right_node = self.add_node(right)

    self.nodes_set.remove(node)
    node.disable()
    return left_node, right_node, trun_vals

  def split_node_qr(
      self,
      node: network_components.BaseNode,
      left_edges: List[network_components.Edge],
      right_edges: List[network_components.Edge],
      left_name: Optional[Text] = None,
      right_name: Optional[Text] = None,
      edge_name: Optional[Text] = None,
  ) -> Tuple[network_components.BaseNode, network_components.BaseNode]:
    """Split a `Node` using QR decomposition

    Let M be the matrix created by flattening left_edges and right_edges into
    2 axes. Let :math:`QR = M` be the QR Decomposition of 
    :math:`M`. This will split the network into 2 nodes. The left node's 
    tensor will be :math:`Q` (an orthonormal matrix) and the right node's tensor will be 
    :math:`R` (an upper triangular matrix)

    Args:
      node: The node you want to split.
      left_edges: The edges you want connected to the new left node.
      right_edges: The edges you want connected to the new right node.
      left_name: The name of the new left node. If `None`, a name will be generated
        automatically.
      right_name: The name of the new right node. If `None`, a name will be generated
        automatically.
      edge_name: The name of the new `Edge` connecting the new left and right node. 
        If `None`, a name will be generated automatically.

    Returns:
      A tuple containing:
        left_node: 
          A new node created that connects to all of the `left_edges`.
          Its underlying tensor is :math:`Q`
        right_node: 
          A new node created that connects to all of the `right_edges`.
          Its underlying tensor is :math:`R`
    """
    q, r = split_node_qr(node, left_edges, right_edges, left_name, right_name,
                         edge_name)
    left_node = self.add_node(q)
    right_node = self.add_node(r)

    self.nodes_set.remove(node)
    node.disable()
    return left_node, right_node

  def split_node_rq(
      self,
      node: network_components.BaseNode,
      left_edges: List[network_components.Edge],
      right_edges: List[network_components.Edge],
      left_name: Optional[Text] = None,
      right_name: Optional[Text] = None,
      edge_name: Optional[Text] = None,
  ) -> Tuple[network_components.BaseNode, network_components.BaseNode]:
    """Split a `Node` using RQ (reversed QR) decomposition

    Let M be the matrix created by flattening left_edges and right_edges into
    2 axes. Let :math:`QR = M^*` be the QR Decomposition of 
    :math:`M^*`. This will split the network into 2 nodes. The left node's 
    tensor will be :math:`R^*` (a lower triangular matrix) and the right node's tensor will be 
    :math:`Q^*` (an orthonormal matrix)

    Args:
      node: The node you want to split.
      left_edges: The edges you want connected to the new left node.
      right_edges: The edges you want connected to the new right node.
      left_name: The name of the new left node. If `None`, a name will be generated
        automatically.
      right_name: The name of the new right node. If `None`, a name will be generated
        automatically.
      edge_name: The name of the new `Edge` connecting the new left and right node. 
        If `None`, a name will be generated automatically.

    Returns:
      A tuple containing:
        left_node: 
          A new node created that connects to all of the `left_edges`.
          Its underlying tensor is :math:`Q`
        right_node: 
          A new node created that connects to all of the `right_edges`.
          Its underlying tensor is :math:`R`
    """
    r, q = split_node_rq(node, left_edges, right_edges, left_name, right_name,
                         edge_name)
    left_node = self.add_node(r)
    right_node = self.add_node(q)

    self.nodes_set.remove(node)
    node.disable()
    return left_node, right_node

  def split_node_full_svd(
      self,
      node: network_components.BaseNode,
      left_edges: List[network_components.Edge],
      right_edges: List[network_components.Edge],
      max_singular_values: Optional[int] = None,
      max_truncation_err: Optional[float] = None,
      left_name: Optional[Text] = None,
      middle_name: Optional[Text] = None,
      right_name: Optional[Text] = None,
      left_edge_name: Optional[Text] = None,
      right_edge_name: Optional[Text] = None,
  ) -> Tuple[network_components.BaseNode, network_components
             .BaseNode, network_components.BaseNode, Tensor]:
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
      left_name: The name of the new left node. If None, a name will be generated
        automatically.
      middle_name: The name of the new center node. If None, a name will be generated
        automatically.
      right_name: The name of the new right node. If None, a name will be generated
        automatically.
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
    U, S, V, trun_vals = split_node_full_svd(
        node, left_edges, right_edges, max_singular_values, max_truncation_err,
        left_name, middle_name, right_name, left_edge_name, right_edge_name)
    left_node = self.add_node(U)
    singular_values_node = self.add_node(S)
    right_node = self.add_node(V)

    self.nodes_set.remove(node)
    node.disable()
    return left_node, singular_values_node, right_node, trun_vals

  def remove_node(self, node: network_components.BaseNode
                 ) -> Tuple[Dict[Text, network_components
                                 .Edge], Dict[int, network_components.Edge]]:
    """Remove a node from the network.

    Args:
      node: The node to be removed.

    Returns:
      broken_edges_by_name: A Dictionary mapping `node`'s axis names to
        the newly broken edges.
      broken_edges_by_axis: A Dictionary mapping `node`'s axis numbers
        to the newly broken edges.

    Raises:
      ValueError: If the node isn't in the network.
    """
    if node not in self:
      raise ValueError("Node '{}' is not in the network.".format(node))
    broken_edges_by_name = {}
    broken_edges_by_axis = {}
    print(len(node.axis_names))
    for i, name in enumerate(node.axis_names):
      print(i, name)
      if not node[i].is_dangling() and not node[i].is_trace():
        edge1, edge2 = self.disconnect(node[i])
        new_broken_edge = edge1 if edge1.node1 is not node else edge2
        broken_edges_by_axis[i] = new_broken_edge
        broken_edges_by_name[name] = new_broken_edge

    self.nodes_set.remove(node)
    node.disable()
    return broken_edges_by_name, broken_edges_by_axis

  def check_correct(self, check_connections: bool = True) -> None:
    """Check that the network is structured correctly.

    Args:
      check_connections: Check if the network is connected.

    Raises:
      ValueError: If the tensor network is not correctly structured.
    """
    check_correct(self.nodes_set, check_connections)

  def save(self, path: Union[str, BinaryIO]):
    """Serialize the network to disk in hdf5 format.

    Args:
      path: path to folder where network is saved.
    """
    with h5py.File(path, 'w') as net_file:
      net_file.create_dataset('backend', data=self.backend.name)
      nodes_group = net_file.create_group('nodes')
      edges_group = net_file.create_group('edges')

      for node in self.nodes_set:
        node_group = nodes_group.create_group(node.name)
        node._save_node(node_group)

        for edge in node.edges:
          if edge.node1 == node:
            edge_group = edges_group.create_group(edge.name)
            edge._save_edge(edge_group)

  def __contains__(self, item):
    if isinstance(item, network_components.Edge):
      edge = item
      try:
        # pylint: disable=pointless-statement
        edge.node1
        # pylint: disable=pointless-statement
        edge.node2
      # If we raise a value error, that means the nodes have been garbage
      # collected, and thus the edge no longer is in the network.
      except ValueError:
        return False
      else:
        edge_is_in_network = edge.node1 in self.nodes_set
        try:
          edge_is_in_network &= edge in edge.node1.edges
        #if ValueError is raised, edge.node1 has been disabled
        except ValueError:
          return False
        if not edge.is_dangling():
          edge_is_in_network &= edge.node2 in self.nodes_set
          try:
            edge_is_in_network &= edge in edge.node2.edges
          #if ValueError is raised, edge.node2 has been disabled
          except ValueError:
            return False
        return edge_is_in_network
    elif isinstance(item, network_components.BaseNode):
      return item in self.nodes_set
    else:
      raise TypeError("Type '{}' was unexpected. "
                      "Only 'None' and 'Edge' types are allowed.".format(
                          type(item)))
