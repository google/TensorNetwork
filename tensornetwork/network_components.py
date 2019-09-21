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
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Type, Union, \
  overload
import typing
import numpy as np
import weakref
from abc import ABC, abstractmethod
import h5py
import tensornetwork as tn
#pylint: disable=useless-import-alias
import tensornetwork.config as config
from tensornetwork.backends import backend_factory

string_type = h5py.special_dtype(vlen=str)
Tensor = Any
# This is required because of the circular dependancy between
# network_components.py and network.py types.
TensorNetwork = Any


class BaseNode(ABC):
  """Base class for nodes. Should be subclassed.

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

  def __init__(self,
               name: Optional[Text] = None,
               axis_names: Optional[List[Text]] = None,
               network: Optional[TensorNetwork] = None,
               backend: Optional["Backend"] = None,
               shape: Optional[Tuple[int]] = None) -> None:
    """Create a node for the TensorNetwork. Should be subclassed before usage
    and a limited number of abstract methods and properties implemented.

    Args:
      name: Name of the node. Used primarily for debugging.
      axis_names: List of names for each of the tensor's axes.
      network: The TensorNetwork this Node belongs to.
      shape: the shape of the tensor, as tuple of integers.

    Raises:
      ValueError: If there is a repeated name in `axis_names` or if the length
        doesn't match the shape of the tensor.
    """
    self.network = network
    self.is_disabled = False
    self.name = name if name is not None else '__unnamed_node__'
    self.backend = backend
    self._shape = shape
    if axis_names is not None:
      self._edges = [
          Edge(node1=self, axis1=i, name=edge_name)
          for i, edge_name in enumerate(axis_names)
      ]
    elif shape is not None:
      self._edges = [
          Edge(node1=self, axis1=i, name="Dangling_{}".format(i))
          for i, _ in enumerate(shape)
      ]
    else:
      raise ValueError("One of axis_names or shape must be provided.")
    if axis_names is not None:
      self.add_axis_names(axis_names)
    else:
      self._axis_names = None

    self._signature = -1

    super().__init__()

  @property
  def dtype(self):
    if self.backend:
      return self.backend.dtype
    return None

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
    if len(axis_names) != len(self.shape):
      raise ValueError("axis_names is not the same length as the tensor shape."
                       "axis_names length: {}, tensor.shape length: {}".format(
                           len(axis_names), len(self.shape)))
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
    if axis_num < 0 or axis_num >= len(self.shape):
      raise ValueError("Axis must be positive and less than rank of the tensor")
    if not self.edges[axis_num].is_dangling() and not override:
      raise ValueError(
          "Node '{}' already has a non-dangling edge for axis {}".format(
              self, axis))
    self.edges[axis_num] = edge

  @abstractmethod
  def get_tensor(self):
    return

  @abstractmethod
  def set_tensor(self, tensor):
    return

  @property
  @abstractmethod
  def shape(self):
    if self._shape is None:
      raise ValueError('Please ensure this Node has a well-defined shape')
    return self._shape

  @property
  @abstractmethod
  def tensor(self) -> Tensor:
    return

  @tensor.setter
  @abstractmethod
  def tensor(self, tensor: Tensor) -> Tensor:
    return

  def get_rank(self) -> int:
    """Return rank of tensor represented by self."""
    return len(self.shape)

  def reorder_edges(self, edge_order: List["Edge"]) -> "BaseNode":
    """Reorder the edges for this given Node.

    This will reorder the node's edges and transpose the underlying tensor
    accordingly.

    Args:
      edge_order: List of edges. The order in the list determines the new edge
        ordering.

    Returns:
      This node post reordering.

    Raises:
      ValueError: If either the list of edges is not the same as expected or
        if you try to reorder with a trace edge.
      AttributeError: If the Node has no tensor.

    """
    if not hasattr(self, '_tensor'):
      raise AttributeError("Please provide a valid tensor for this Node.")

    extra_edges = set(edge_order).difference(set(self.edges))
    if extra_edges:
      raise ValueError("Given edge order does not match expected edges. "
                       "Additional edges that do not belong to node found: "
                       "{}".format(extra_edges))
    missing_edges = set(self.edges).difference(set(edge_order))
    if missing_edges:
      raise ValueError("Given edge order does not match expected edges. "
                       "Missing edges that belong to node found: "
                       "{}".format(missing_edges))
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
    self.tensor = self.backend.transpose(self.tensor, perm=permutation)
    if self.axis_names is not None:
      # Update axis_names:
      tmp_axis_names = []
      for i in permutation:
        tmp_axis_names.append(self.axis_names[i])
      self.axis_names = tmp_axis_names
    return self

  def reorder_axes(self, perm: List[int]) -> "BaseNode":
    """Reorder axes of the node's tensor.

    This will also update all of the node's edges.

    Args:
      perm: Permutation of the dimensions of the node's tensor.

    Returns:
      This node post reordering.

    Raises:
      AttributeError: If the Node has no tensor.
    """
    if not hasattr(self, '_tensor'):
      raise AttributeError("Please provide a valid tensor for this Node.")

    if set(perm) != set(range(len(self.edges))):
      raise ValueError("A full permutation was not passed. "
                       "Permutation passed: {}".format(perm))
    self.tensor = self.backend.transpose(self.tensor, perm=perm)
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
    if axis_num < 0 or axis_num >= len(self.shape):
      raise ValueError("Axis must be positive and less than rank of the tensor")
    return self.shape[axis_num]

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

  def has_dangling_edge(self):
    for e in self.edges:
      if e.is_dangling():
        return True
    return False

  @overload
  def __getitem__(self, key: slice) -> List["Edge"]:
    pass

  @overload
  def __getitem__(self, key: Union[int, Text]) -> "Edge":
    pass

  def __getitem__(self,
                  key: Union[int, Text, slice]) -> Union["Edge", List["Edge"]]:
    if isinstance(key, slice):
      return self.edges[key]
    return self.get_edge(key)

  def __str__(self) -> Text:
    return self.name

  def __lt__(self, other):
    if not isinstance(other, BaseNode):
      raise ValueError("Object {} is not a Node type.".format(other))
    return id(self) < id(other)

  def __matmul__(self, other: "BaseNode") -> "BaseNode":
    if not hasattr(self, '_tensor'):
      raise AttributeError("Please provide a valid tensor for this Node.")
    if not isinstance(other, BaseNode):
      raise TypeError("Cannot use '@' with type '{}'".format(type(other)))
    if self.is_disabled:
      raise ValueError("Cannot use '@' on disabled node {}.".format(self.name))
    if self.network and other.network:
      return self.network.contract_between(self, other)
    return tn.contract_between(self, other)

  @property
  def edges(self):
    if self.is_disabled:
      raise ValueError('Node {} has been disabled. '
                       'Accessing its edges is no longer possible'.format(
                           self.name))
    return self._edges

  @edges.setter
  def edges(self, edges: List):
    if self.is_disabled:
      raise ValueError('Node {} has been disabled.'
                       'Assigning edges is no longer possible'.format(
                           self.name))
    self._edges = edges

  @property
  def axis_names(self):
    return self._axis_names

  @axis_names.setter
  def axis_names(self, axis_names: List[Text]):
    if len(axis_names) != len(self.shape):
      raise ValueError("Expected {} names, only got {}.".format(
          len(self.shape), len(axis_names)))
    self._axis_names = axis_names

  @property
  def signature(self):
    if self.is_disabled:
      raise ValueError('Node {} has been disabled. '
                       'Accessing its signature is no longer possible'.format(
                           self.name))
    return self._signature

  @signature.setter
  def signature(self, signature: int):
    if self.is_disabled:
      raise ValueError('Node {} has been disabled. '
                       'Assigning a signature is no longer possible'.format(
                           self.name))
    self._signature = signature

  def disable(self):
    if self.is_disabled:
      raise ValueError('Node {} is already disabled'.format(self.name))
    if self.network and self in self.network.nodes_set:
      raise ValueError(
          'Node {} is part of a network. Disabelling not allowed'.format(
              self.name))

    self.is_disabled = True

  @classmethod
  @abstractmethod
  def _load_node(cls, node_data: h5py.Group) -> "BaseNode":
    return

  @classmethod
  def _load_node_data(cls, node_data: h5py.Group) -> Tuple[Any, Any, Any, Any]:
    """Common method to enable adding nodes to a network based on hdf5 data.
       Only a common functionality to load node properties is implemented.

    Args:
      node_data: h5py group that contains the serialized node data

    Returns:
      the node's name, signature, shape, axis_names
    """
    name = node_data['name'][()]
    signature = node_data['signature'][()]
    backend = node_data['backend'][()]
    shape = node_data['shape'][()]
    axis_names = node_data['axis_names'][()]
    return name, signature, shape, axis_names, backend

  @abstractmethod
  def _save_node(self, node_group: h5py.Group):
    """Abstract method to enable saving nodes to hdf5.
       Only serializing common properties is implemented. Should be
       overwritten by subclasses.

    Args:
      node_group: h5py group where data is saved
    """
    node_group.create_dataset('type', data=type(self).__name__)
    node_group.create_dataset('signature', data=self.signature)
    node_group.create_dataset('backend', data=self.backend.name)
    node_group.create_dataset('name', data=self.name)
    node_group.create_dataset('shape', data=self.shape)
    if self.axis_names:
      node_group.create_dataset(
          'axis_names',
          dtype=string_type,
          data=np.array(self.axis_names, dtype=object))
    else:  #couldn't find any documentation on saving None
      node_group.create_dataset('axis_names', dtype='i', data=123456789)

    node_group.create_dataset(
        'edges',
        dtype=string_type,
        data=np.array([edge.name for edge in self.edges], dtype=object))

  def fresh_edges(self, axis_names: Optional[List[Text]] = None):
    if not axis_names:
      axis_names = self.axis_names
    if not axis_names:
      axis_names = [str(i) for i in range(len(self.shape))]
    for i in range(len(self.edges)):
      new_edge = Edge(node1=self, axis1=i, name=axis_names[i])
      self.add_edge(new_edge, i, True)


class Node(BaseNode):
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

  def __init__(self,
               tensor: Tensor,
               name: Optional[Text] = None,
               axis_names: Optional[List[Text]] = None,
               network: Optional[TensorNetwork] = None,
               backend: Optional[Text] = None) -> None:
    """Create a node for the TensorNetwork.

    Args:
      tensor: The concrete tensor that is represented by this node. Can be
        either a numpy array or a tensorflow tensor.
      name: Name of the node. Used primarily for debugging.
      axis_names: List of names for each of the tensor's axes.
      backend: The name of the backend.

    Raises:
      ValueError: If there is a repeated name in `axis_names` or if the length
        doesn't match the shape of the tensor.
    """

    if backend is None:
      backend = config.default_backend
    backend = backend_factory.get_backend(backend, dtype=None)
    self._tensor = backend.convert_to_tensor(tensor)

    super().__init__(
        name=name,
        axis_names=axis_names,
        network=network,
        backend=backend,
        shape=backend.shape_tuple(self._tensor))
    if not self.backend.dtype:
      self.backend.dtype = self._tensor.dtype

  def get_tensor(self):
    return self.tensor

  def set_tensor(self, tensor):
    self.tensor = tensor

  @property
  def shape(self):
    if self.is_disabled:
      raise ValueError('Node {} has been disabled. '
                       'Access its shape via self.tensor'.format(self.name))
    return self.backend.shape_tuple(self._tensor)

  @property
  def tensor(self) -> Tensor:
    return self._tensor

  @tensor.setter
  def tensor(self, tensor: Tensor) -> Tensor:
    self._tensor = tensor

  def _save_node(self, node_group: h5py.Group):
    """Method to save a node to hdf5.

    Args:
      node_group: h5py group where data is saved
    """
    super()._save_node(node_group)
    node_group.create_dataset('tensor', data=self._tensor)

  @classmethod
  def _load_node(cls, net: TensorNetwork, node_data: h5py.Group) -> "BaseNode":
    """Add a node to a network based on hdf5 data.

    Args:
      net: The network the node will be added to
      node_data: h5py group that contains the serialized node data

    Returns:
      The added node.
    """
    name, signature, _, axis_names, _ = cls._load_node_data(node_data)
    tensor = node_data['tensor'][()]
    node = net.add_node(
        value=tensor, name=name, axis_names=[ax for ax in axis_names])
    node.set_signature(signature)
    return node


class CopyNode(BaseNode):

  def __init__(self,
               rank: int,
               dimension: int,
               name: Optional[Text] = None,
               axis_names: Optional[List[Text]] = None,
               network: Optional[TensorNetwork] = None,
               backend: Optional["Backend"] = None,
               dtype: Type[np.number] = None) -> None:

    self.rank = rank
    self.dimension = dimension
    self._tensor = None
    if dtype is None:
      dtype = config.default_dtype
    # backend.dtype is initialized from config.py (currently `None`)
    # if `backend.dtype = None`, the backend dtype is set to the type
    # of `tensor`.
    if backend is None:
      backend = config.default_backend
    backend = backend_factory.get_backend(backend, dtype)
    super().__init__(
        name=name,
        axis_names=axis_names,
        network=network,
        backend=backend,
        shape=(dimension,) * rank)

  def get_tensor(self):
    return self.tensor

  def set_tensor(self, tensor):
    self.tensor = tensor

  @property
  def shape(self):
    return (self.dimension,) * self.rank

  @property
  def tensor(self) -> Tensor:
    if self._tensor is None:
      copy_tensor = self.make_copy_tensor(self.rank, self.dimension, self.dtype)
      print(copy_tensor)
      print(self.backend)
      self._tensor = self.backend.convert_to_tensor(copy_tensor)
    return self._tensor

  @tensor.setter
  def tensor(self, tensor: Tensor) -> Tensor:
    self._tensor = tensor

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

  def _get_partner(self, edge: "Edge") -> Tuple[BaseNode, int]:
    if edge.node1 is self:
      assert edge.axis2 is not None
      return edge.node2, edge.axis2
    assert edge.node2 is self
    return edge.node1, edge.axis1

  def get_partners(self) -> Dict[BaseNode, Set[int]]:
    partners = {}  # type: Dict[BaseNode, Set[int]]
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

  def _make_einsum_input_term(self, node: BaseNode, shared_axes: Set[int],
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

  def _make_einsum_expression(self, partners: Dict[BaseNode, Set[int]]) -> str:
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
    return self.backend.einsum(einsum_expression, *tensors)

  # pylint: disable=W0235
  def _save_node(self, node_group: h5py.Group):
    """Method to save a node to hdf5.

    Args:
      node_group: h5py group where data is saved
    """
    super()._save_node(node_group)

  @classmethod
  def _load_node(cls, net: TensorNetwork, node_data: h5py.Group) -> "BaseNode":
    """Add a node to a network based on hdf5 data.

    Args:
      net: The network the node will be added to
      node_data: h5py group that contains the serialized node data

    Returns:
      The added node.
    """
    name, signature, shape, axis_names, _ = cls._load_node_data(node_data)
    node = net.add_copy_node(
        name=name,
        axis_names=[ax for ax in axis_names],
        rank=len(shape),
        dimension=shape[0])
    node.set_signature(signature)
    return node


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
               node1: BaseNode,
               axis1: int,
               name: Optional[Text] = None,
               node2: Optional[BaseNode] = None,
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
    self.is_disabled = False
    if not name:
      name = '__unnamed_edge__'
    self._name = name
    self.node1 = node1
    self._axis1 = axis1
    self.node2 = node2
    self._axis2 = axis2
    self._is_dangling = node2 is None
    self._signature = -1

  def disable(self):
    return  #disable disable for now
    # self._node1 = None
    # self._node2 = None
    # self.is_disabled = True

  @property
  def name(self):
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, accessing its name is no longer possible')
    return self._name

  @name.setter
  def name(self, name):
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, setting its name is no longer possible')
    self._name = name

  @property
  def axis1(self):
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, accessing axis1 is no longer possible')
    return self._axis1

  @axis1.setter
  def axis1(self, axis1: int) -> None:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, setting node1 is no longer possible')
    self._axis1 = axis1

  @property
  def axis2(self):
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, accessing axis2 is no longer possible')
    return self._axis2

  @axis2.setter
  def axis2(self, axis2: int) -> None:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, setting node1 is no longer possible')
    self._axis2 = axis2

  @property
  def signature(self):
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, accessing signature is no longer possible')
    return self._signature

  @signature.setter
  def signature(self, signature: int) -> None:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, setting node1 is no longer possible')
    self._signature = signature

  def set_signature(self, signature: int) -> None:
    if self.is_dangling():
      raise ValueError(
          "Do not set a signature for dangling edge '{}'.".format(self))
    self.signature = signature

  def get_nodes(self) -> List[Optional[BaseNode]]:
    """Get the nodes of the edge."""
    return [self.node1, self.node2]

  def update_axis(self, old_axis: int, old_node: BaseNode, new_axis: int,
                  new_node: BaseNode) -> None:
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
  def node1(self) -> BaseNode:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, accessing node1 is no longer possible')
    if self._node1() is None:
      raise ValueError("node1 for edge '{}' no longer exists.".format(self))
    return self._node1()

  @property
  def node2(self) -> Optional[BaseNode]:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, accessing node2 is no longer possible')
    if self._is_dangling:
      return None
    if self._node2() is None:
      raise ValueError("node2 for edge '{}' no longer exists.".format(self))
    return self._node2()

  @node1.setter
  def node1(self, node: BaseNode) -> None:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, setting node1 is no longer possible')
    # pylint: disable=attribute-defined-outside-init
    self._node1 = weakref.ref(node)

  @node2.setter
  def node2(self, node: Optional[BaseNode]) -> None:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, setting node2 is no longer possible')
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

  def _save_edge(self, edge_group: h5py.Group):
    """Method to save an edge to hdf5.

    Args:
      edge_group: h5py group where data is saved
    """
    edge_group.create_dataset('node1', data=self.node1.name)
    edge_group.create_dataset('axis1', data=self.axis1)
    if self.node2 is not None:
      edge_group.create_dataset('node2', data=self.node2.name)
      edge_group.create_dataset('axis2', data=self.axis2)
    edge_group.create_dataset('signature', data=self.signature)
    edge_group.create_dataset('name', data=self.name)

  # def save(self, edge_group: h5py.Group, reachable: Set):
  #   """Method to save an edge to hdf5.

  #   Args:
  #     edge_group: h5py group where data is saved
  #   """
  #   edge_group.create_dataset('axis1', data=self.axis1)
  #   edge_group.create_dataset('signature', data=self.signature)
  #   edge_group.create_dataset('name', data=self.name)
  #   if self.node2 is not None:
  #     edge_group.create_dataset('axis2', data=self.axis2)
  #     node_group = edge_group.create_group('node2')
  #     self.node2.save(node_group)

  @classmethod
  def _load_edge(cls, edge_data: h5py.Group, nodes_dict: Dict[Text, BaseNode]):
    """Add an edge to a network based on hdf5 data.

    Args:
      edge_data: h5py group that contains the serialized edge data
      nodes: dictionary of node's name, node of all the nodes in the network

    Returns:
      The added edge.
    """
    node1 = nodes_dict[edge_data["node1"][()]]
    axis1 = int(edge_data["axis1"][()])
    if "node2" in list(edge_data.keys()):
      node2 = nodes_dict[edge_data["node2"][()]]
      axis2 = int(edge_data["axis2"][()])
    else:
      node2 = None
      axis2 = None
    signature = edge_data["signature"][()]
    name = edge_data["name"][()]
    edge = cls(node1=node1, axis1=axis1, node2=node2, axis2=axis2, name=name)
    node1.add_edge(edge, axis1)
    if node2 is not None:
      node2.add_edge(edge, axis2)
    if not edge.is_dangling():
      edge.set_signature(signature)
    return edge

  def __xor__(self, other: "Edge") -> "Edge":
    return tn.connect(self, other, self.name)

  def __lt__(self, other):
    if not isinstance(other, Edge):
      raise TypeError("Cannot compare 'Edge' with type {}".format(type(Edge)))
    return self.signature < other.signature

  def __str__(self) -> Optional[Text]:
    if self.name:
      return self.name
    return '__unnamed_edge__'

  def disconnect(
      self,
      edge1_name: Optional[Text] = None,
      edge2_name: Optional[Text] = None) -> Tuple[BaseNode, BaseNode]:
    """
    Break an existing non-dangling edge.
    This updates both Edge.node1 and Edge.node2 by removing the 
    connecting edge from `Edge.node1.edges` and `Edge.node2.edges`
    and adding new dangling edges instead
    Args:
      edge1_name: A name for the new dangling edge at `self.node1`
      edge2_name: A name for the new dangling edge at `self.node2`
    Returns:
      (new_edge1, new_edge2): The new `Edge` objects of 
        `self.node1` and `self.node2`
    """
    if self.is_dangling():
      raise ValueError("Cannot break dangling edge {}.".format(self))
    if not edge1_name:
      edge1_name = '__unnamed_edge__'
    if not edge2_name:
      edge2_name = '__unnamed_edge__'

    node1 = self.node1
    node2 = self.node2

    new_edge1 = Edge(node1=node1, axis1=self.axis1, name=edge1_name)
    new_edge2 = Edge(node1=node2, axis1=self.axis2, name=edge2_name)
    node1.add_edge(new_edge1, self.axis1, override=True)
    node2.add_edge(new_edge2, self.axis2, override=True)
    return new_edge1, new_edge2

  def __or__(self, other: "Edge") -> "Edge":
    """
    Break apart two edges if they are connected
    """
    if self is not other:
      raise ValueError('Cannot break two unconnected edges')
    return self.disconnect('__Edge__', '__Edge__')
