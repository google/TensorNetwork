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
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Type, Union, \
  overload, Sequence, Iterable
import numpy as np
from abc import ABC
from abc import abstractmethod
import h5py

#pylint: disable=useless-import-alias
import tensornetwork.config as config
from tensornetwork import ops
from tensornetwork.backends import backend_factory
from tensornetwork.backends.base_backend import BaseBackend

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
               backend: Optional[BaseBackend] = None,
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
    if network and backend and (network.backend.name != backend.name):
      raise ValueError(
          'network.backend.name={} is different from backend={}'.format(
              network.backend.name, backend.name))
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
      self._axis_names = [str(i) for i in range(len(shape))]

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

  def get_all_dangling(self):
    """Return the set of dangling edges connected to this node."""
    return {edge for edge in self.edges if edge.is_dangling()}

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
    return contract_between(self, other)

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
  def _load_node(cls, net: TensorNetwork, node_data: h5py.Group) -> "BaseNode":
    """Add a node to a network based on hdf5 data.

    Args:
      net: The network the node will be added to. If not `None` the loaded
        node will be added to `net`.
      node_data: h5py group that contains the serialized node data

    Returns:
      The loaded node.
    """
    return

  @classmethod
  def _load_node_data(cls,
                      node_data: h5py.Group) -> Tuple[Any, Any, Any, Any, Any]:
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
    if network:  #if a network is passed, use its backend
      if backend and (network.backend.name != backend):
        raise ValueError(
            'network.backend.name={} is different from backend={}'.format(
                network.backend.name, backend))
      backend_obj = network.backend
    else:
      if not backend:
        backend = config.default_backend
      #use dtype=None here; the backend dtype will be deduced from the
      #tensor dtype.
      backend_obj = backend_factory.get_backend(backend, dtype=None)
    self._tensor = backend_obj.convert_to_tensor(tensor)
    super().__init__(
        name=name,
        axis_names=axis_names,
        network=network,
        backend=backend_obj,
        shape=backend_obj.shape_tuple(self._tensor))
    if self.backend and not self.backend.dtype:
      self.backend.dtype = self._tensor.dtype

    collection = ops.get_current_collection()
    if collection is not None:
      collection.add(self)

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
      net: The network the node will be added to. If not `None` the loaded
        node will be added to `net`.
      node_data: h5py group that contains the serialized node data

    Returns:
      The loaded node.
    """
    name, signature, _, axis_names, backend = cls._load_node_data(node_data)
    tensor = node_data['tensor'][()]
    # pylint: disable=unnecessary-comprehension
    node = Node(
        tensor,
        name=name,
        axis_names=[ax for ax in axis_names],
        backend=backend)

    if net:
      node = net.add_node(node)
    node.set_signature(signature)
    return node


class CopyNode(BaseNode):

  def __init__(self,
               rank: int,
               dimension: int,
               name: Optional[Text] = None,
               axis_names: Optional[List[Text]] = None,
               network: Optional[TensorNetwork] = None,
               backend: Optional[Text] = None,
               dtype: Type[np.number] = np.float64) -> None:
    """
    Initialize a CopyNode:
    Args:
      rank: The rank of the tensor.
      dimension: The dimension of each leg.
      name: A name for the node.
      axis_names:  axis_names for the node.
      network: An optional network for the node.
      backend: An optional backend for the node. If `None`, a default
        backend is used
      dtype: The dtype used to initialize a numpy-copy node.
        Note that this dtype has to be a numpy dtype, and it has to be 
        compatible with the dtype of the backend, e.g. for a tensorflow
        backend with a tf.Dtype=tf.floa32, `dtype` has to be `np.float32`.
    """
    if network:  #if a network is passed, use its backend
      if backend and (network.backend.name != backend):
        raise ValueError(
            'network.backend.name={} is different from backend={}'.format(
                network.backend.name, backend))
      backend_obj = network.backend
    else:
      if not backend:
        backend = config.default_backend
      #use dtype=None here; the backend dtype will be deduced from the
      #tensor dtype.
      backend_obj = backend_factory.get_backend(backend, dtype=None)

    self.rank = rank
    self.dimension = dimension
    self._tensor = None
    self.copy_node_dtype = dtype

    super().__init__(
        name=name,
        axis_names=axis_names,
        network=network,
        backend=backend_obj,
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
      copy_tensor = self.make_copy_tensor(self.rank, self.dimension,
                                          self.copy_node_dtype)
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
    node_group.create_dataset(
        name='copy_node_dtype', data=np.dtype(self.copy_node_dtype).name)

  @classmethod
  def _load_node(cls, net: TensorNetwork, node_data: h5py.Group) -> "CopyNode":
    """Add a node to a network based on hdf5 data.

    Args:
      net: The network the node will be added to. If not `None` the loaded
        node will be added to `net`.
      node_data: h5py group that contains the serialized node data

    Returns:
      The loaded node.
    """
    name, signature, shape, axis_names, backend = cls._load_node_data(node_data)
    copy_node_dtype = np.dtype(node_data['copy_node_dtype'][()])
    # pylint: disable=unnecessary-comprehension
    node = CopyNode(
        rank=len(shape),
        dimension=shape[0],
        name=name,
        axis_names=[ax for ax in axis_names],
        backend=backend,
        dtype=copy_node_dtype)

    if net:
      node = net.add_node(node)

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

  # contraction methods now explicitly disable Edges by setting
  # node1, node2 to None. This makes use of weakref for node1 and node2
  # properties redundant:
  # previously, storage of contracted edges in TensorNetwork caused
  # node1 and node2 refs of those edges to be prevented from garbage
  # collection. Once we set them to None explicitly, they will be garbage
  # collected once their refcount goes to zero.
  def disable(self):
    # pylint: disable=attribute-defined-outside-init
    self._node1 = None
    # pylint: disable=attribute-defined-outside-init
    self._node2 = None
    self.is_disabled = True

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
    if self._node1 is None:
      raise ValueError("node1 for edge '{}' no longer exists.".format(self))
    return self._node1

  @property
  def node2(self) -> Optional[BaseNode]:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, accessing node2 is no longer possible')
    if self._is_dangling:
      return None
    if self._node2 is None:
      raise ValueError("node2 for edge '{}' no longer exists.".format(self))
    return self._node2

  @node1.setter
  def node1(self, node: BaseNode) -> None:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, setting node1 is no longer possible')
    # pylint: disable=attribute-defined-outside-init
    self._node1 = node

  @node2.setter
  def node2(self, node: Optional[BaseNode]) -> None:
    if self.is_disabled:
      raise ValueError(
          'Edge has been disabled, setting node2 is no longer possible')
    # pylint: disable=attribute-defined-outside-init
    self._node2 = node
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
    return connect(self, other, self.name)

  def __lt__(self, other):
    if not isinstance(other, Edge):
      raise TypeError("Cannot compare 'Edge' with type {}".format(type(Edge)))
    return self.signature < other.signature

  def __str__(self) -> Optional[Text]:
    if self.name:
      return self.name
    return '__unnamed_edge__'

  def disconnect(self,
                 edge1_name: Optional[Text] = None,
                 edge2_name: Optional[Text] = None) -> Tuple["Edge", "Edge"]:
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
      edge1_name = '__disconnected_edge1_of_{}__'.format(self.name)
    if not edge2_name:
      edge2_name = '__disconnected_edge2_of_{}__'.format(self.name)

    node1 = self.node1
    node2 = self.node2

    new_edge1 = Edge(node1=node1, axis1=self.axis1, name=edge1_name)
    new_edge2 = Edge(node1=node2, axis1=self.axis2, name=edge2_name)
    node1.add_edge(new_edge1, self.axis1, override=True)
    node2.add_edge(new_edge2, self.axis2, override=True)
    return new_edge1, new_edge2

  def __or__(self, other: "Edge") -> Tuple["Edge", "Edge"]:
    """
    Break apart two edges if they are connected
    """
    if self is not other:
      raise ValueError('Cannot break two unconnected edges')
    return self.disconnect()


def get_shared_edges(node1: BaseNode, node2: BaseNode) -> Set[Edge]:
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


def get_parallel_edges(edge: Edge) -> Set[Edge]:
  """
  Get all of the edges parallel to the given `edge`.
  Args:
    edge: The given edge.

  Returns:
    A `set` of all of the edges parallel to the given edge 
    (including the given edge).
  """
  return get_shared_edges(edge.node1, edge.node2)


def get_all_nondangling(nodes: Iterable[BaseNode]) -> Set[Edge]:
  """Return the set of all non-dangling edges."""
  edges = set()
  for node in nodes:
    edges |= node.get_all_nondangling()
  return edges


def get_all_dangling(nodes: Iterable[BaseNode]) -> Set[Edge]:
  """Return the set of all dangling edges."""
  edges = set()
  for node in nodes:
    edges |= node.get_all_dangling()
  return edges


def _flatten_trace_edges(edges: List[Edge],
                         new_edge_name: Optional[Text] = None) -> Edge:
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
  edge1 = Edge(node1=node, axis1=len(perm_front), name="TraceFront")
  edge2 = Edge(node1=node, axis1=len(perm_front) + 1, name="TraceBack")
  node.edges = node.edges[:len(perm_front)] + [edge1, edge2]
  new_edge = connect(edge1, edge2, new_edge_name)
  # pylint: disable=expression-not-assigned
  [edge.disable() for edge in edges]  #disable edges!
  return new_edge


def flatten_edges(edges: List[Edge],
                  new_edge_name: Optional[Text] = None) -> Edge:
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
    edge = Edge(node1=node, axis1=len(perm_front), name=new_edge_name)
    # Do not set the signature of 'edge' since it is dangling.
    node.edges = node.edges[:len(perm_front)] + [edge]
    new_dangling_edges.append(edge)
    # TODO: Allow renaming of the new axis.
    if axis_names:
      node.axis_names = [axis_names[n] for n in range(len(node.edges))]
    else:
      node.axis_names = [str(n) for n in range(len(node.edges))]

  node1, node2 = tuple(expected_nodes)
  # Sets are returned in a random order, so this is how we deal with
  # dangling edges.
  # pylint: disable=expression-not-assigned
  [edge.disable() for edge in edges]  #disable edges!
  if node1 is None or node2 is None:
    return new_dangling_edges[0]

  return connect(new_dangling_edges[0], new_dangling_edges[1], new_edge_name)


def flatten_edges_between(
    node1: BaseNode,
    node2: BaseNode,
) -> Optional[Edge]:
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


def flatten_all_edges(nodes: Iterable[BaseNode]) -> List[Edge]:
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


def _remove_trace_edge(edge: Edge, new_node: BaseNode) -> None:
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


def _remove_edges(edges: Set[Edge], node1: BaseNode, node2: BaseNode,
                  new_node: BaseNode) -> None:
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


def _contract_trace(edge: Edge, name: Optional[Text] = None) -> BaseNode:
  """Contract a trace edge.
  `edge` is disabled before returning.
  Args:
    edge: The edge name or object to contract next.
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
  name = name if name else edge.node1.name
  new_node = Node(new_tensor, name=name, backend=backend.name)
  _remove_trace_edge(edge, new_node)  #disables edge
  return new_node


def contract(edge: Edge,
             name: Optional[Text] = None,
             axis_names: Optional[List[Text]] = None) -> BaseNode:
  """
  Contract an edge connecting two nodes.
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
  if edge.is_dangling():
    raise ValueError("Attempting to contract dangling edge")

  for node in [edge.node1, edge.node2]:
    if (node is not None) and (not hasattr(node, 'backend')):
      raise TypeError('Node {} of type {} has no `backend`'.format(
          node, type(node)))

  if edge.node1.backend.name != edge.node2.backend.name:
    raise ValueError("edge.node1 {} and edge.node2 {} have different backends "
                     "{} and {}".format(edge.node1.name, edge.node2.name,
                                        edge.node1.backend.name,
                                        edge.node2.backend.name))

  if edge.node1:
    backend = edge.node1.backend
  else:
    raise ValueError("edge {} has no nodes. "
                     "Cannot perfrom a contraction".format(edge.name))

  backend = edge.node1.backend
  if edge.node1 is edge.node2:
    return _contract_trace(edge, name)
  new_tensor = backend.tensordot(edge.node1.tensor, edge.node2.tensor,
                                 [[edge.axis1], [edge.axis2]])
  new_node = Node(
      tensor=new_tensor, name=name, axis_names=axis_names, backend=backend.name)
  # edge.node1 and edge.node2 get new edges in _remove_edges
  _remove_edges(set([edge]), edge.node1, edge.node2, new_node)
  return new_node


def contract_copy_node(copy_node: CopyNode,
                       name: Optional[Text] = None) -> BaseNode:
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
  new_node = Node(new_tensor, name, backend=copy_node.backend.name)

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


def contract_parallel(edge: Edge) -> BaseNode:
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


def connect(edge1: Edge, edge2: Edge, name: Optional[Text] = None) -> Edge:
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

  new_edge = Edge(
      node1=node1, axis1=axis1_num, name=name, node2=node2, axis2=axis2_num)

  node1.add_edge(new_edge, axis1_num, override=True)
  node2.add_edge(new_edge, axis2_num, override=True)
  return new_edge


def disconnect(edge,
               edge1_name: Optional[Text] = None,
               edge2_name: Optional[Text] = None) -> Tuple[Edge, Edge]:
  """
  Break an existing non-dangling edge.
  This updates both Edge.node1 and Edge.node2 by removing the 
  connecting edge from `Edge.node1.edges` and `Edge.node2.edges`
  and adding new dangling edges instead
  """
  return edge.disconnect(edge1_name, edge2_name)


def contract_between(
    node1: BaseNode,
    node2: BaseNode,
    name: Optional[Text] = None,
    allow_outer_product: bool = False,
    output_edge_order: Optional[Sequence[Edge]] = None,
    axis_names: Optional[List[Text]] = None,
) -> BaseNode:
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
    raise ValueError("node {} and node {} have different backends "
                     "{} and {}.".format(node1.name, node2.name,
                                         node1.backend.name,
                                         node2.backend.name))

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
  new_node = Node(
      tensor=new_tensor, name=name, axis_names=axis_names, backend=backend.name)
  # node1 and node2 get new edges in _remove_edges
  _remove_edges(shared_edges, node1, node2, new_node)
  if output_edge_order:
    new_node = new_node.reorder_edges(list(output_edge_order))
  return new_node


def outer_product_final_nodes(nodes: Iterable[BaseNode],
                              edge_order: List[Edge]) -> BaseNode:
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


def outer_product(node1: BaseNode,
                  node2: BaseNode,
                  name: Optional[Text] = None,
                  axis_names: Optional[List[Text]] = None) -> BaseNode:
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
  new_node = Node(
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


class NodeCollection:

  def __init__(self, container):
    if not isinstance(container, (list, set)):
      raise ValueError("Item passed to NodeCollection must be list or set")
    self._container = container

  def add(self, node):
    if isinstance(self._container, set):
      self._container.add(node)
    else:
      self._container.append(node)

  def __enter__(self):
    ops._default_collection_stack.stack.append(self)

  def __exit__(self, exc_type, exc_val, exc_tb):
    ops._default_collection_stack.stack.pop()