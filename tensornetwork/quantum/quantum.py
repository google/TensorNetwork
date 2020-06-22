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
"""Abstractions for quantum vectors and operators.

Quantum mechanics involves a lot of linear algebra on vector spaces that
often have a preferred tensor-product factorization. Tensor networks are
a natural way to represent vectors and operators (matrices) involving
these spaces. Hence we provide some simple abstractions to ease linear
algebra operations in which the vectors and operators are represented by
tensor networks.
"""
from typing import Any, Union, Callable, Optional, Sequence, Collection, Text
from typing import Tuple, Set, List, Type
import numpy as np
from tensornetwork.network_components import AbstractNode, Node, Edge, connect
from tensornetwork.network_components import CopyNode
from tensornetwork.network_operations import get_all_nodes, copy, reachable
from tensornetwork.network_operations import get_subgraph_dangling, remove_node
from tensornetwork.contractors import greedy
Tensor = Any


def quantum_constructor(
    out_edges: Sequence[Edge],
    in_edges: Sequence[Edge],
    ref_nodes: Optional[Collection[AbstractNode]] = None,
    ignore_edges: Optional[Collection[Edge]] = None) -> "QuOperator":
  """Constructs an appropriately specialized QuOperator.

  If there are no edges, creates a QuScalar. If the are only output (input)
  edges, creates a QuVector (QuAdjointVector). Otherwise creates a
  QuOperator.

  Args:
    out_edges: output edges.
    in_edges: in edges.
    ref_nodes: reference nodes for the tensor network (needed if there is a
      scalar component).
    ignore_edges: edges to ignore when checking the dimensionality of the
      tensor network.
  Returns:
    The object.
  """
  if len(out_edges) == 0 and len(in_edges) == 0:
    return QuScalar(ref_nodes, ignore_edges)
  if len(out_edges) == 0:
    return QuAdjointVector(in_edges, ref_nodes, ignore_edges)
  if len(in_edges) == 0:
    return QuVector(out_edges, ref_nodes, ignore_edges)
  return QuOperator(out_edges, in_edges, ref_nodes, ignore_edges)


def identity(space: Sequence[int],
             backend: Optional[Text] = None,
             dtype: Type[np.number] = np.float64) -> "QuOperator":
  """Construct a `QuOperator` representing the identity on a given space.

  Internally, this is done by constructing `CopyNode`s for each edge, with
  dimension according to `space`.

  Args:
    space: A sequence of integers for the dimensions of the tensor product
      factors of the space (the edges in the tensor network).
    backend: Optionally specify the backend to use for computations.
    dtype: The data type (for conversion to dense).
  Returns:
    The desired identity operator.
  """
  nodes = [CopyNode(2, d, backend=backend, dtype=dtype) for d in space]
  out_edges = [n[0] for n in nodes]
  in_edges = [n[1] for n in nodes]
  return quantum_constructor(out_edges, in_edges)


def check_spaces(edges_1: Sequence[Edge], edges_2: Sequence[Edge]) -> None:
  """Check the vector spaces represented by two lists of edges are compatible.

  The number of edges must be the same and the dimensions of each pair of edges
  must match. Otherwise, an exception is raised.

  Args:
    edges_1: List of edges representing a many-body Hilbert space.
    edges_2: List of edges representing a many-body Hilbert space.
  """
  if len(edges_1) != len(edges_2):
    raise ValueError("Hilbert-space mismatch: Cannot connect {} subsystems "
                     "with {} subsystems.".format(len(edges_1), len(edges_2)))

  for (i, (e1, e2)) in enumerate(zip(edges_1, edges_2)):
    if e1.dimension != e2.dimension:
      raise ValueError("Hilbert-space mismatch on subsystems {}: Input "
                       "dimension {} != output dimension {}.".format(
                           i, e1.dimension, e2.dimension))


def eliminate_identities(nodes: Collection[AbstractNode]) -> Tuple[dict, dict]:
  """Eliminates any connected CopyNodes that are identity matrices.

  This will modify the network represented by `nodes`.
  Only identities that are connected to other nodes are eliminated.

  Args:
    nodes: Collection of nodes to search.
  Returns:
    nodes_dict: Dictionary mapping remaining Nodes to any replacements.
    dangling_edges_dict: Dictionary specifying all dangling-edge replacements.
  """
  nodes_dict = {}
  dangling_edges_dict = {}
  for n in nodes:
    if isinstance(
        n, CopyNode) and n.get_rank() == 2 and not (n[0].is_dangling() and
                                                    n[1].is_dangling()):
      old_edges = [n[0], n[1]]
      _, new_edges = remove_node(n)
      if 0 in new_edges and 1 in new_edges:
        e = connect(new_edges[0], new_edges[1])
      elif 0 in new_edges:  # 1 was dangling
        dangling_edges_dict[old_edges[1]] = new_edges[0]
      elif 1 in new_edges:  # 0 was dangling
        dangling_edges_dict[old_edges[0]] = new_edges[1]
      else:
        # Trace of identity, so replace with a scalar node!
        d = n.get_dimension(0)
        # NOTE: Assume CopyNodes have numpy dtypes.
        nodes_dict[n] = Node(np.array(d, dtype=n.dtype), backend=n.backend)
    else:
      for e in n.get_all_dangling():
        dangling_edges_dict[e] = e
      nodes_dict[n] = n

  return nodes_dict, dangling_edges_dict


class QuOperator():
  """Represents a linear operator via a tensor network.

  To interpret a tensor network as a linear operator, some of the dangling
  edges must be designated as `out_edges` (output edges) and the rest as
  `in_edges` (input edges).

  Considered as a matrix, the `out_edges` represent the row index and the
  `in_edges` represent the column index.

  The (right) action of the operator on another then consists of connecting
  the `in_edges` of the first operator to the `out_edges` of the second.

  Can be used to do simple linear algebra with tensor networks.
  """
  __array_priority__ = 100.0  # for correct __rmul__ with scalar ndarrays

  def __init__(self,
               out_edges: Sequence[Edge],
               in_edges: Sequence[Edge],
               ref_nodes: Optional[Collection[AbstractNode]] = None,
               ignore_edges: Optional[Collection[Edge]] = None) -> None:
    """Creates a new `QuOperator` from a tensor network.

    This encapsulates an existing tensor network, interpreting it as a linear
    operator.

    The network is checked for consistency: All dangling edges must either be
    in `out_edges`, `in_edges`, or `ignore_edges`.

    Args:
      out_edges: The edges of the network to be used as the output edges.
      in_edges: The edges of the network to be used as the input edges.
      ref_nodes: Nodes used to refer to parts of the tensor network that are
        not connected to any input or output edges (for example: a scalar
        factor).
      ignore_edges: Optional collection of dangling edges to ignore when
        performing consistency checks.
    """
    # TODO: Decide whether the user must also supply all nodes involved.
    #       This would enable extra error checking and is probably clearer
    #       than `ref_nodes`.
    if len(in_edges) == 0 and len(out_edges) == 0 and not ref_nodes:
      raise ValueError("At least one reference node is required to specify a "
                       "scalar. None provided!")
    self.out_edges = list(out_edges)
    self.in_edges = list(in_edges)
    self.ignore_edges = set(ignore_edges) if ignore_edges else set()
    self.ref_nodes = set(ref_nodes) if ref_nodes else set()
    self.check_network()

  @classmethod
  def from_tensor(cls,
                  tensor: Tensor,
                  out_axes: Sequence[int],
                  in_axes: Sequence[int],
                  backend: Optional[Text] = None) -> "QuOperator":
    """Construct a `QuOperator` directly from a single tensor.

    This first wraps the tensor in a `Node`, then constructs the `QuOperator`
    from that `Node`.

    Args:
      tensor: The tensor.
      out_axes: The axis indices of `tensor` to use as `out_edges`.
      in_axes: The axis indices of `tensor` to use as `in_edges`.
      backend: Optionally specify the backend to use for computations.
    Returns:
      The new operator.
    """
    n = Node(tensor, backend=backend)
    out_edges = [n[i] for i in out_axes]
    in_edges = [n[i] for i in in_axes]
    return cls(out_edges, in_edges, set([n]))

  @property
  def nodes(self) -> Set[AbstractNode]:
    """All tensor-network nodes involved in the operator."""
    return reachable(
        get_all_nodes(self.out_edges + self.in_edges) | self.ref_nodes)

  @property
  def in_space(self) -> List[int]:
    return [e.dimension for e in self.in_edges]

  @property
  def out_space(self) -> List[int]:
    return [e.dimension for e in self.out_edges]

  def is_scalar(self) -> bool:
    return len(self.out_edges) == 0 and len(self.in_edges) == 0

  def is_vector(self) -> bool:
    return len(self.out_edges) > 0 and len(self.in_edges) == 0

  def is_adjoint_vector(self) -> bool:
    return len(self.out_edges) == 0 and len(self.in_edges) > 0

  def check_network(self) -> None:
    """Check that the network has the expected dimensionality.

    This checks that all input and output edges are dangling and that
    there are no other dangling edges (except any specified in
    `ignore_edges`). If not, an exception is raised.
    """
    for (i, e) in enumerate(self.out_edges):
      if not e.is_dangling():
        raise ValueError("Output edge {} is not dangling!".format(i))
    for (i, e) in enumerate(self.in_edges):
      if not e.is_dangling():
        raise ValueError("Input edge {} is not dangling!".format(i))
    for e in self.ignore_edges:
      if not e.is_dangling():
        raise ValueError("ignore_edges contains non-dangling edge: {}".format(
            str(e)))

    known_edges = set(self.in_edges) | set(self.out_edges) | self.ignore_edges
    all_dangling_edges = get_subgraph_dangling(self.nodes)
    if known_edges != all_dangling_edges:
      raise ValueError("The network includes unexpected dangling edges (that "
                       "are not members of ignore_edges).")

  def adjoint(self) -> "QuOperator":
    """The adjoint of the operator.

    This creates a new `QuOperator` with complex-conjugate copies of all
    tensors in the network and with the input and output edges switched.
    """
    nodes_dict, edge_dict = copy(self.nodes, True)
    out_edges = [edge_dict[e] for e in self.in_edges]
    in_edges = [edge_dict[e] for e in self.out_edges]
    ref_nodes = [nodes_dict[n] for n in self.ref_nodes]
    ignore_edges = [edge_dict[e] for e in self.ignore_edges]
    return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

  def trace(self) -> "QuOperator":
    """The trace of the operator."""
    return self.partial_trace(range(len(self.in_edges)))

  def norm(self) -> "QuOperator":
    """The norm of the operator.

    This is the 2-norm (also known as the Frobenius or Hilbert-Schmidt
    norm).
    """
    return (self.adjoint() @ self).trace()

  def partial_trace(self,
                    subsystems_to_trace_out: Collection[int]) -> "QuOperator":
    """The partial trace of the operator.

    Subsystems to trace out are supplied as indices, so that dangling edges
    are connected to eachother as:
      `out_edges[i] ^ in_edges[i] for i in subsystems_to_trace_out`

    This does not modify the original network. The original ordering of the
    remaining subsystems is maintained.

    Args:
      subsystems_to_trace_out: Indices of subsystems to trace out.
    Returns:
      A new QuOperator or QuScalar representing the result.
    """
    out_edges_trace = [self.out_edges[i] for i in subsystems_to_trace_out]
    in_edges_trace = [self.in_edges[i] for i in subsystems_to_trace_out]

    check_spaces(in_edges_trace, out_edges_trace)

    nodes_dict, edge_dict = copy(self.nodes, False)
    for (e1, e2) in zip(out_edges_trace, in_edges_trace):
      edge_dict[e1] = edge_dict[e1] ^ edge_dict[e2]

    # get leftover edges in the original order
    out_edges_trace = set(out_edges_trace)
    in_edges_trace = set(in_edges_trace)
    out_edges = [
        edge_dict[e] for e in self.out_edges if e not in out_edges_trace
    ]
    in_edges = [edge_dict[e] for e in self.in_edges if e not in in_edges_trace]
    ref_nodes = [n for _, n in nodes_dict.items()]
    ignore_edges = [edge_dict[e] for e in self.ignore_edges]

    return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

  def __matmul__(self, other: "QuOperator") -> "QuOperator":
    """The action of this operator on another.

    Given `QuOperator`s `A` and `B`, produces a new `QuOperator` for `A @ B`,
    where `A @ B` means: "the action of A, as a linear operator, on B".

    Under the hood, this produces copies of the tensor networks defining `A`
    and `B` and then connects the copies by hooking up the `in_edges` of
    `A.copy()` to the `out_edges` of `B.copy()`.
    """
    check_spaces(self.in_edges, other.out_edges)

    # Copy all nodes involved in the two operators.
    # We must do this separately for self and other, in case self and other
    # are defined via the same network components (e.g. if self === other).
    nodes_dict1, edges_dict1 = copy(self.nodes, False)
    nodes_dict2, edges_dict2 = copy(other.nodes, False)

    # connect edges to create network for the result
    for (e1, e2) in zip(self.in_edges, other.out_edges):
      _ = edges_dict1[e1] ^ edges_dict2[e2]

    in_edges = [edges_dict2[e] for e in other.in_edges]
    out_edges = [edges_dict1[e] for e in self.out_edges]
    ref_nodes = ([n for _, n in nodes_dict1.items()] +
                 [n for _, n in nodes_dict2.items()])
    ignore_edges = ([edges_dict1[e] for e in self.ignore_edges] +
                    [edges_dict2[e] for e in other.ignore_edges])

    return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

  def __mul__(self, other: Union["QuOperator", AbstractNode,
                                 Tensor]) -> "QuOperator":
    """Scalar multiplication of operators.

    Given two operators `A` and `B`, one of the which is a scalar (it has no
    input or output edges), `A * B` produces a new operator representing the
    scalar multiplication of `A` and `B`.

    For convenience, one of `A` or `B` may be a number or scalar-valued tensor
    or `Node` (it will automatically be wrapped in a `QuScalar`).

    Note: This is a special case of `tensor_product()`.
    """
    if not isinstance(other, QuOperator):
      if isinstance(other, AbstractNode):
        node = other
      else:
        node = Node(other, backend=self.nodes.pop().backend)
      if node.shape:
        raise ValueError("Cannot perform elementwise multiplication by a "
                         "non-scalar tensor.")
      other = QuScalar([node])

    if self.is_scalar() or other.is_scalar():
      return self.tensor_product(other)

    raise ValueError("Elementwise multiplication is only supported if at "
                     "least one of the arguments is a scalar.")

  def __rmul__(
      self, other: Union["QuOperator", AbstractNode, Tensor]) -> "QuOperator":
    """Scalar multiplication of operators.

    See `.__mul__()`.
    """
    return self.__mul__(other)

  def tensor_product(self, other: "QuOperator") -> "QuOperator":
    """Tensor product with another operator.

    Given two operators `A` and `B`, produces a new operator `AB` representing
    `A` ⊗ `B`. The `out_edges` (`in_edges`) of `AB` is simply the
    concatenation of the `out_edges` (`in_edges`) of `A.copy()` with that of
    `B.copy()`:

    `new_out_edges = [*out_edges_A_copy, *out_edges_B_copy]`
    `new_in_edges = [*in_edges_A_copy, *in_edges_B_copy]`

    Args:
      other: The other operator (`B`).
    Returns:
      The result (`AB`).
    """
    nodes_dict1, edges_dict1 = copy(self.nodes, False)
    nodes_dict2, edges_dict2 = copy(other.nodes, False)

    in_edges = ([edges_dict1[e] for e in self.in_edges] +
                [edges_dict2[e] for e in other.in_edges])
    out_edges = ([edges_dict1[e] for e in self.out_edges] +
                 [edges_dict2[e] for e in other.out_edges])
    ref_nodes = ([n for _, n in nodes_dict1.items()] +
                 [n for _, n in nodes_dict2.items()])
    ignore_edges = ([edges_dict1[e] for e in self.ignore_edges] +
                    [edges_dict2[e] for e in other.ignore_edges])

    return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

  def contract(
      self,
      contractor: Callable = greedy,
      final_edge_order: Optional[Sequence[Edge]] = None) -> "QuOperator":
    """Contract the tensor network in place.

    This modifies the tensor network representation of the operator (or vector,
    or scalar), reducing it to a single tensor, without changing the value.

    Args:
      contractor: A function that performs the contraction. Defaults to
        `greedy`, which uses the greedy algorithm from `opt_einsum` to
        determine a contraction order.
      final_edge_order: Manually specify the axis ordering of the final tensor.
    Returns:
      The present object.
    """
    nodes_dict, dangling_edges_dict = eliminate_identities(self.nodes)
    self.in_edges = [dangling_edges_dict[e] for e in self.in_edges]
    self.out_edges = [dangling_edges_dict[e] for e in self.out_edges]
    self.ignore_edges = set(dangling_edges_dict[e] for e in self.ignore_edges)
    self.ref_nodes = set(
        nodes_dict[n] for n in self.ref_nodes if n in nodes_dict)
    self.check_network()

    if final_edge_order:
      final_edge_order = [dangling_edges_dict[e] for e in final_edge_order]
      self.ref_nodes = set(
          [contractor(self.nodes, output_edge_order=final_edge_order)])
    else:
      self.ref_nodes = set([contractor(self.nodes, ignore_edge_order=True)])
    return self

  def eval(self,
           contractor: Callable = greedy,
           final_edge_order: Optional[Sequence[Edge]] = None) -> Tensor:
    """Contracts the tensor network in place and returns the final tensor.

    Note that this modifies the tensor network representing the operator.

    The default ordering for the axes of the final tensor is:
      `*out_edges, *in_edges`.

    If there are any "ignored" edges, their axes come first:
      `*ignored_edges, *out_edges, *in_edges`.

    Args:
      contractor: A function that performs the contraction. Defaults to
        `greedy`, which uses the greedy algorithm from `opt_einsum` to
        determine a contraction order.
      final_edge_order: Manually specify the axis ordering of the final tensor.
        The default ordering is determined by `out_edges` and `in_edges` (see
        above).
    Returns:
      The final tensor representing the operator.
    """
    if not final_edge_order:
      final_edge_order = (
          list(self.ignore_edges) + self.out_edges + self.in_edges)
    self.contract(contractor, final_edge_order)
    nodes = self.nodes
    if len(nodes) != 1:
      raise ValueError("Node count '{}' > 1 after contraction!".format(
          len(nodes)))
    return list(nodes)[0].tensor


class QuVector(QuOperator):
  """Represents a (column) vector via a tensor network."""

  def __init__(self,
               subsystem_edges: Sequence[Edge],
               ref_nodes: Optional[Collection[AbstractNode]] = None,
               ignore_edges: Optional[Collection[Edge]] = None) -> None:
    """Constructs a new `QuVector` from a tensor network.

    This encapsulates an existing tensor network, interpreting it as a (column)
    vector.

    Args:
      subsystem_edges: The edges of the network to be used as the output edges.
      ref_nodes: Nodes used to refer to parts of the tensor network that are
        not connected to any input or output edges (for example: a scalar
        factor).
      ignore_edges: Optional collection of edges to ignore when performing
        consistency checks.
    """
    super().__init__(subsystem_edges, [], ref_nodes, ignore_edges)

  @classmethod
  def from_tensor(cls,
                  tensor: Tensor,
                  subsystem_axes: Optional[Sequence[int]] = None,
                  backend: Optional[Text] = None) -> "QuVector":
    """Construct a `QuVector` directly from a single tensor.

    This first wraps the tensor in a `Node`, then constructs the `QuVector`
    from that `Node`.

    Args:
      tensor: The tensor.
      subsystem_axes: Sequence of integer indices specifying the order in which
        to interpret the axes as subsystems (output edges). If not specified,
        the axes are taken in ascending order.
      backend: Optionally specify the backend to use for computations.
    Returns:
      The new operator.
    """
    n = Node(tensor, backend=backend)
    if subsystem_axes is not None:
      subsystem_edges = [n[i] for i in subsystem_axes]
    else:
      subsystem_edges = n.get_all_edges()
    return cls(subsystem_edges)

  @property
  def subsystem_edges(self) -> List[Edge]:
    return self.out_edges

  @property
  def space(self) -> List[int]:
    return self.out_space

  def projector(self) -> "QuOperator":
    return self @ self.adjoint()

  def reduced_density(self,
                      subsystems_to_trace_out: Collection[int]) -> "QuOperator":
    rho = self.projector()
    return rho.partial_trace(subsystems_to_trace_out)


class QuAdjointVector(QuOperator):
  """Represents an adjoint (row) vector via a tensor network."""

  def __init__(self,
               subsystem_edges: Sequence[Edge],
               ref_nodes: Optional[Collection[AbstractNode]] = None,
               ignore_edges: Optional[Collection[Edge]] = None) -> None:
    """Constructs a new `QuAdjointVector` from a tensor network.

    This encapsulates an existing tensor network, interpreting it as an adjoint
    vector (row vector).

    Args:
      subsystem_edges: The edges of the network to be used as the input edges.
      ref_nodes: Nodes used to refer to parts of the tensor network that are
        not connected to any input or output edges (for example: a scalar
        factor).
      ignore_edges: Optional collection of edges to ignore when performing
        consistency checks.
    """
    super().__init__([], subsystem_edges, ref_nodes, ignore_edges)

  @classmethod
  def from_tensor(cls,
                  tensor: Tensor,
                  subsystem_axes: Optional[Sequence[int]] = None,
                  backend: Optional[Text] = None) -> "QuAdjointVector":
    """Construct a `QuAdjointVector` directly from a single tensor.

    This first wraps the tensor in a `Node`, then constructs the
    `QuAdjointVector` from that `Node`.

    Args:
      tensor: The tensor.
      subsystem_axes: Sequence of integer indices specifying the order in which
        to interpret the axes as subsystems (input edges). If not specified,
        the axes are taken in ascending order.
      backend: Optionally specify the backend to use for computations.
    Returns:
      The new operator.
    """
    n = Node(tensor, backend=backend)
    if subsystem_axes is not None:
      subsystem_edges = [n[i] for i in subsystem_axes]
    else:
      subsystem_edges = n.get_all_edges()
    return cls(subsystem_edges)

  @property
  def subsystem_edges(self) -> List[Edge]:
    return self.in_edges

  @property
  def space(self) -> List[int]:
    return self.in_space

  def projector(self) -> "QuOperator":
    return self.adjoint() @ self

  def reduced_density(self,
                      subsystems_to_trace_out: Collection[int]) -> "QuOperator":
    rho = self.projector()
    return rho.partial_trace(subsystems_to_trace_out)


class QuScalar(QuOperator):
  """Represents a scalar via a tensor network."""

  def __init__(self,
               ref_nodes: Collection[AbstractNode],
               ignore_edges: Optional[Collection[Edge]] = None) -> None:
    """Constructs a new `QuScalar` from a tensor network.

    This encapsulates an existing tensor network, interpreting it as a scalar.

    Args:
      ref_nodes: Nodes used to refer to the tensor network (need not be
        exhaustive - one node from each disconnected subnetwork is sufficient).
        ignore_edges: Optional collection of edges to ignore when performing
        consistency checks.
    """
    super().__init__([], [], ref_nodes, ignore_edges)

  @classmethod
  def from_tensor(cls,
                  tensor: Tensor,
                  backend: Optional[Text] = None) -> "QuScalar":
    """Construct a `QuScalar` directly from a single tensor.

    This first wraps the tensor in a `Node`, then constructs the
    `QuScalar` from that `Node`.

    Args:
      tensor: The tensor.
      backend: Optionally specify the backend to use for computations.
    Returns:
      The new operator.
    """
    n = Node(tensor, backend=backend)
    return cls(set([n]))
