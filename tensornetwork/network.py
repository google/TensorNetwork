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
import h5py
# pylint: disable=line-too-long
from typing import Any, Sequence, List, Set, Optional, Union, Text, Tuple, Type, Dict, BinaryIO
import numpy as np
from tensornetwork import config
#pylint: disable=useless-import-alias
import tensornetwork.network_components as network_components
#pylint: disable=useless-import-alias
import tensornetwork.network_operations as network_operations
from tensornetwork.backends import backend_factory

Tensor = Any
string_type = h5py.special_dtype(vlen=str)


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
      if (new_node.network is not None) and (new_node.network is not self):
        raise ValueError("Given node is already part of another network.")
      if new_node.backend.name != self.backend.name:
        raise ValueError(
            "Given node '{}' has Node.backend.name='{}' different from TensorNetwork.backend.name='{}'."
            .format(new_node.name, new_node.backend.name, self.backend.name))

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
    new_edge = network_components.connect(edge1, edge2, name)
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
    return network_components.disconnect(edge, dangling_edge_name_1,
                                         dangling_edge_name_2)

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

    network_components._remove_trace_edge(edge, new_node)
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
    network_components._remove_edges(edges, node1, node2, new_node)

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
    # _contract_trace disables contracted edges; if we have to access them,
    # we need to do it beforehand.
    node = edge.node1
    new_node = self.add_node(network_components._contract_trace(edge, name))
    self.nodes_set.remove(node)
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
    #contract disables contracted edges; if we have to access them, we need
    #to do it beforehand.
    trace_edge = edge.node1 is edge.node2
    node1 = edge.node1
    node2 = edge.node2
    new_node = self.add_node(
        network_components.contract(edge, name, axis_names=None))
    # disable nodes
    if trace_edge:
      self.nodes_set.remove(node1)
      node1.disable()
    else:
      self.nodes_set.remove(node1)
      self.nodes_set.remove(node2)
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
    new_node = self.add_node(
        network_components.contract_copy_node(copy_node, name))
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
    new_node = self.add_node(
        network_components.outer_product(node1, node2, name, axis_names=None))
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
    return network_components.get_all_nondangling(self.nodes_set)

  def get_all_edges(self):
    """Return the set of all edges."""
    return network_operations.get_all_edges(self.nodes_set)

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
    return network_components.outer_product_final_nodes(self.nodes_set,
                                                        edge_order)

  def check_connected(self) -> None:
    """Check if the network is connected."""
    # Fastest way to get a single item from a set.
    network_operations.check_connected(self.nodes_set)

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
    return network_components._flatten_trace_edges(edges, new_edge_name)

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
    return network_components.flatten_edges(edges, new_edge_name)

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
    return network_components.get_shared_edges(node1, node2)

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
    return network_components.flatten_edges_between(node1, node2)

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
        network_components.contract_between(
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
    left, right, trun_vals = network_operations.split_node(
        node, left_edges, right_edges, max_singular_values, max_truncation_err,
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
    q, r = network_operations.split_node_qr(node, left_edges, right_edges,
                                            left_name, right_name, edge_name)
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
    r, q = network_operations.split_node_rq(node, left_edges, right_edges,
                                            left_name, right_name, edge_name)
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
    U, S, V, trun_vals = network_operations.split_node_full_svd(
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
    network_operations.check_correct(self.nodes_set, check_connections)

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
