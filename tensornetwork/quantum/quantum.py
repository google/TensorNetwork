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

from tensornetwork.network_components import Edge, connect
from tensornetwork.network_operations import reachable, get_all_nodes, copy
from tensornetwork.network_operations import get_subgraph_dangling


def quantum_constructor(out_edges, in_edges, ref_nodes=None):
  """Constructs an appropriately specialized QuOperator or QuScalar.

  If there are no edges, creates a QuScalar. If the are only output (input)
  edges, construct a QuVector (QuAdjointVector). Otherwise construct a
  QuOperator.

  Args:
    out_edges: output edges.
    in_edges: in edges.
    ref_nodes: reference nodes for the tensor network (in case scalar).
  Returns:
    The object.
  """
  if len(out_edges) == 0 and len(in_edges) == 0:
    return QuScalar(ref_nodes)
  elif len(out_edges) == 0:
    return QuAdjointVector(in_edges)
  elif len(in_edges) == 0:
    return QuVector(out_edges)
  return QuOperator(out_edges, in_edges)


def check_spaces(edges_1, edges_2):
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


class QuScalar():
  def __init__(self, ref_nodes):
    if not ref_nodes:
      raise ValueError("At least one reference node is required for a "
                       "QuScalar. None provided!")
    self.ref_nodes = set(ref_nodes)
    self.check_scalar()

  @property
  def nodes(self):
    """All tensor-network nodes involved in the scalar.
    """
    return reachable(self.ref_nodes)

  def check_scalar(self):
    """Check that the defining network is scalar valued.
    """
    dangling_edges = get_subgraph_dangling(self.nodes)
    return len(dangling_edges) == 0


class QuOperator():
  """Represents an operator via a tensor network.

  Can be used to do simple linear algebra with tensor networks.
  """
  def __init__(self, out_edges, in_edges):
    # TODO: Decide whether the user must also supply all nodes involved.
    #       More flexible if not (e.g. a QuOperator can represent a vector
    #       of operators if there is an extra dangling Edge), better error
    #       checking if so.
    self.out_edges = list(out_edges)
    self.in_edges = list(in_edges)

  @property
  def nodes(self):
    """All tensor-network nodes involved in the operator.
    """
    return reachable(get_all_nodes(self.out_edges + self.in_edges))

  @property
  def shape_in(self):
    return [e.dimension for e in self.in_edges]

  @property
  def shape_out(self):
    return [e.dimension for e in self.out_edges]

  def adjoint(self):
    """The adjoint of the operator.

    This creates a new `QuOperator` with complex-conjugate copies of all
    tensors in the network and with the input and output edges switched.
    """
    _, edge_dict = copy(self.nodes, True)
    out_edges = [edge_dict[e] for e in self.in_edges]
    in_edges = [edge_dict[e] for e in self.out_edges]
    return quantum_constructor(out_edges, in_edges)

  def trace(self):
    """The trace of the operator.
    """
    return self.partial_trace(range(len(self.in_edges)))

  def norm(self):
    """The norm of the operator.
    This is the 2-norm (Frobenius norm), or Hilbert-Schmidt norm.
    """
    return (self.adjoint() @ self).trace()

  def partial_trace(self, subsystems_to_trace_out):
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

    return quantum_constructor(out_edges, in_edges, ref_nodes)

  def __matmul__(self, other):
    check_spaces(self.in_edges, other.out_edges)

    # Copy all nodes involved in the two operators.
    # We must do this separately for self and other, in case self and other
    # were define via the same network components (e.g. if self === other).
    nodes_dict1, edge_dict1 = copy(self.nodes, False)
    nodes_dict2, edge_dict2 = copy(other.nodes, False)

    # connect edges to create network for the result
    for (e1, e2) in zip(self.in_edges, other.out_edges):
      _ = edge_dict1[e1] ^ edge_dict2[e2]

    in_edges = [edge_dict2[e] for e in other.in_edges]
    out_edges = [edge_dict1[e] for e in self.out_edges]
    ref_nodes = [n for _, n in nodes_dict1.items()]
    ref_nodes += [n for _, n in nodes_dict2.items()]

    return quantum_constructor(out_edges, in_edges, ref_nodes)

  def tensor(self, other):
    """Tensor product with another operator.
    """
    _, edges_dict1 = copy(self.nodes, False)
    _, edges_dict2 = copy(other.nodes, False)

    in_edges = ([edges_dict1[e] for e in self.in_edges] +
                [edges_dict2[e] for e in other.in_edges])
    out_edges = ([edges_dict1[e] for e in self.out_edges] +
                 [edges_dict2[e] for e in other.out_edges])

    return quantum_constructor(out_edges, in_edges)


class QuIdentity(QuOperator):
  def __init__(self, num_subsystems):
    edges = [Edge(None, None) for _ in range(num_subsystems)]
    super().__init__(edges, edges)


class QuVector(QuOperator):
  def __init__(self, subsystem_edges):
    super().__init__(subsystem_edges, [])

  @property
  def subsystem_edges(self):
    return self.out_edges

  def projector(self):
    return self @ self.adjoint()

  def reduced_density(self, subsystems_to_trace_out):
    rho = self.projector()
    return rho.partial_trace(subsystems_to_trace_out)


class QuAdjointVector(QuOperator):
  def __init__(self, subsystem_edges):
    super().__init__([], subsystem_edges)

  @property
  def subsystem_edges(self):
    return self.in_edges