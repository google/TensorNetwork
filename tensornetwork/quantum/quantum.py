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

Quantum mechanics involves a lot of linear algebra on vector spaces that often
have a preferred tensor-product factorization. Tensor networks are a natural
way to represent vectors and operators (matrices) involving these spaces. Hence
we provide some simple abstractions to ease linear algebra operations in which
the vectors and operators are represented by tensor networks.
"""
from typing import Any, Union, Callable, Optional, Sequence, Collection, Text
from tensornetwork.network_components import Node, Edge, connect, CopyNode
from tensornetwork.network_operations import get_all_nodes, copy, reachable
from tensornetwork.network_operations import get_subgraph_dangling
from tensornetwork.contractors import greedy
Tensor = Any


def quantum_constructor(out_edges: Sequence[Edge], in_edges: Sequence[Edge],
                        ref_nodes: Optional[Collection[Node]] = None,
                        ignore_edges: Optional[Collection[Edge]] = None):
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


def identity(shape: Sequence[int]):
  nodes = [CopyNode(2, d) for d in shape]
  out_edges = [n[0] for n in nodes]
  in_edges = [n[1] for n in nodes]
  return quantum_constructor(out_edges, in_edges)


def check_spaces(edges_1: Sequence[Edge], edges_2: Sequence[Edge]):
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


class QuOperator():
  """Represents an operator via a tensor network.

  Can be used to do simple linear algebra with tensor networks.
  """
  def __init__(self, out_edges: Sequence[Edge], in_edges: Sequence[Edge],
               ref_nodes: Optional[Collection[Node]] = None,
               ignore_edges: Optional[Collection[Edge]] = None):
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
  def from_tensor(cls, tensor: Tensor, out_axes: Sequence[int],
                  in_axes: Sequence[int], backend: Optional[Text] = None):
    n = Node(tensor, backend=backend)
    out_edges = [n[i] for i in out_axes]
    in_edges = [n[i] for i in in_axes]
    return cls(out_edges, in_edges, set([n]))

  @property
  def nodes(self):
    """All tensor-network nodes involved in the operator.
    """
    return reachable(
        get_all_nodes(self.out_edges + self.in_edges) | self.ref_nodes)

  @property
  def shape_in(self):
    return [e.dimension for e in self.in_edges]

  @property
  def shape_out(self):
    return [e.dimension for e in self.out_edges]

  def is_scalar(self):
    return len(self.out_edges) == 0 and len(self.in_edges) == 0

  def is_vector(self):
    return len(self.out_edges) > 0 and len(self.in_edges) == 0

  def is_adjoint_vector(self):
    return len(self.out_edges) == 0 and len(self.in_edges) > 0

  def check_network(self):
    """Check that the network has the expected dimensionality.

    This checks that all input and output edges are dangling and that there
    are no other dangling edges (except any specified in `ignore_edges`).
    If not, an exception is raised.
    """
    for (i, e) in enumerate(self.out_edges):
      if not e.is_dangling():
        raise ValueError("Output edge {} is not dangling!".format(i))
    for (i, e) in enumerate(self.in_edges):
      if not e.is_dangling():
        raise ValueError("Input edge {} is not dangling!".format(i))

    known_edges = set(self.in_edges) | set(self.out_edges) | self.ignore_edges
    all_dangling_edges = get_subgraph_dangling(self.nodes)
    if known_edges != all_dangling_edges:
      raise ValueError("The network includes unexpected dangling edges (that "
                       "are not members of ignore_edges).")

  def adjoint(self):
    """The adjoint of the operator.

    This creates a new `QuOperator` with complex-conjugate copies of all
    tensors in the network and with the input and output edges switched.
    """
    nodes_dict, edge_dict = copy(self.nodes, True)
    out_edges = [edge_dict[e] for e in self.in_edges]
    in_edges = [edge_dict[e] for e in self.out_edges]
    ref_nodes = [nodes_dict[n] for n in self.ref_nodes]
    ignore_edges = [edge_dict[e] for e in self.ignore_edges]
    return quantum_constructor(
        out_edges, in_edges, ref_nodes, ignore_edges)

  def trace(self):
    """The trace of the operator.
    """
    return self.partial_trace(range(len(self.in_edges)))

  def norm(self):
    """The norm of the operator.
    This is the 2-norm (also known as the Frobenius or Hilbert-Schmidt norm).
    """
    return (self.adjoint() @ self).trace()

  def partial_trace(self, subsystems_to_trace_out: Collection[int]):
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
    check_spaces(self.in_edges, self.out_edges)

    out_edges_trace = [self.out_edges[i] for i in subsystems_to_trace_out]
    in_edges_trace = [self.in_edges[i] for i in subsystems_to_trace_out]

    nodes_dict, edge_dict = copy(self.nodes, False)
    for (e1, e2) in zip(out_edges_trace, in_edges_trace):
      edge_dict[e1] = edge_dict[e1] ^ edge_dict[e2]

    # get leftover edges in the original order
    out_edges_trace = set(out_edges_trace)
    in_edges_trace = set(in_edges_trace)
    out_edges = [edge_dict[e] for e in self.out_edges
                 if e not in out_edges_trace]
    in_edges = [edge_dict[e] for e in self.in_edges
                if e not in in_edges_trace]
    ref_nodes = [n for _, n in nodes_dict.items()]
    ignore_edges = [edge_dict[e] for e in self.ignore_edges]

    return quantum_constructor(out_edges, in_edges, ref_nodes, ignore_edges)

  def __matmul__(self, other: "QuOperator"):
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

  def __mul__(self, other: Union["QuOperator", Tensor]):
    if not isinstance(other, QuOperator):
      node = Node(other, backend=self.nodes.pop().backend)
      if node.shape:
        raise ValueError("Cannot perform elementwise multiplication by a "
                         "non-scalar tensor.")
      other = QuScalar([node])

    if self.is_scalar() or other.is_scalar():
      return self.tensor_product(other)

    raise ValueError("Elementwise multiplication is only supported if at "
                     "least one of the arguments is a scalar.")

  def __rmul__(self, other: Union["QuOperator", Tensor]):
    return self.__mul__(other)

  def tensor_product(self, other: "QuOperator"):
    """Tensor product with another operator.
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

  def contract(self, contractor: Callable = greedy,
               final_edge_order: Optional[Sequence[Edge]] = None):
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
    if final_edge_order:
      self.ref_nodes = set(
          [contractor(self.nodes, output_edge_order=final_edge_order)])
    else:
      self.ref_nodes = set([contractor(self.nodes, ignore_edge_order=True)])
    return self

  def eval(self, contractor: Callable = greedy,
           final_edge_order: Optional[Sequence[Edge]] = None):
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
      final_edge_order = (list(self.ignore_edges) + self.out_edges +
                          self.in_edges)
    self.contract(contractor, final_edge_order)
    nodes = self.nodes
    if len(nodes) != 1:
      raise ValueError("Node count '{}' > 1 after contraction!".format(
          len(nodes)))
    return list(nodes)[0].tensor


class QuVector(QuOperator):
  def __init__(self, subsystem_edges: Sequence[Edge],
               ref_nodes: Optional[Collection[Node]] = None,
               ignore_edges: Optional[Collection[Edge]] = None):
    super().__init__(subsystem_edges, [], ref_nodes, ignore_edges)

  @classmethod
  def from_tensor(cls, tensor: Tensor, backend: Optional[Text] = None):
    n = Node(tensor, backend=backend)
    return cls(n.get_all_edges())

  @property
  def subsystem_edges(self):
    return self.out_edges

  def projector(self):
    return self @ self.adjoint()

  def reduced_density(self, subsystems_to_trace_out: Collection[int]):
    rho = self.projector()
    return rho.partial_trace(subsystems_to_trace_out)


class QuAdjointVector(QuOperator):
  def __init__(self, subsystem_edges: Sequence[Edge],
               ref_nodes: Optional[Collection[Node]] = None,
               ignore_edges: Optional[Collection[Edge]] = None):
    super().__init__([], subsystem_edges, ref_nodes, ignore_edges)

  @classmethod
  def from_tensor(cls, tensor: Tensor, backend: Optional[Text] = None):
    n = Node(tensor, backend=backend)
    return cls(n.get_all_edges())

  @property
  def subsystem_edges(self):
    return self.in_edges

  def projector(self):
    return self.adjoint() @ self

  def reduced_density(self, subsystems_to_trace_out: Collection[int]):
    rho = self.projector()
    return rho.partial_trace(subsystems_to_trace_out)


class QuScalar(QuOperator):
  def __init__(self, ref_nodes: Collection[Node],
               ignore_edges: Optional[Collection[Edge]] = None):
    super().__init__([], [], ref_nodes, ignore_edges)

  @classmethod
  def from_tensor(cls, tensor: Tensor, backend: Optional[Text] = None):
    n = Node(tensor, backend=backend)
    return cls(set([n]))
