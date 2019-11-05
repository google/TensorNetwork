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
"""Abstractions for quantum states and operators."""

from tensornetwork.network_components import connect
from tensornetwork.network_operations import reachable, get_all_nodes, copy, get_subgraph_dangling


def QuantumConstructor(out_edges, in_edges, ref_nodes=None):
  if len(out_edges) == 0 and len(in_edges) == 0:
    return QuantumScalar(ref_nodes)
  elif len(out_edges) == 0:
    return QuantumBra(in_edges)
  elif len(in_edges) == 0:
    return QuantumKet(out_edges)
  return QuantumOperator(out_edges, in_edges)


def check_hilberts(edges_1, edges_2):
  if len(edges_1) != len(edges_2):
    raise ValueError("Hilbert-space mismatch: Cannot connect {} subsystems "
      "with {} subsystems.".format(len(edges_1), len(edges_2)))

  for (i, (e1, e2)) in enumerate(zip(edges_1, edges_2)):
    if e1.dimension != e2.dimension:
      raise ValueError("Hilbert-space mismatch on subsystems {}: Input "
        "dimension {} != output dimension {}.".format(
          i, e1.dimension, e2.dimension))


class QuantumScalar():
  def __init__(self, ref_nodes):
    self.ref_nodes = ref_nodes
    self.check_scalar()

  @property
  def nodes(self):
    """All tensor-network nodes involved in the scalar.
    """
    return reachable(self.ref_nodes)

  def check_scalar(self):
    """Check that the defining network is scalar valued.
    """
    dangling_edges = get_subgraph_dangling(self.nodes())
    return len(dangling_edges) == 0


class QuantumOperator():
  """Represents an operator via a tensor network.

  Can be used to do simple linear algebra with tensor networks.
  """
  def __init__(self, out_edges, in_edges):
    self.out_edges = list(out_edges)
    self.in_edges = list(in_edges)

  @classmethod
  def from_local(cls, operator, out_indices, in_indices, system_edges):
    """Construct a global operator from a local one.
    """
    # FIXME: Would be natural to do this with nodeless Edges, but an Edge
    #        is currently assumed to have at least one Node attached.
    raise NotImplementedError()

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

    This creates a new `QuantumOperator` with complex-conjugate copies of all
    tensors in the network and with the input and output edges switched.
    """
    _, edge_dict = copy(self.nodes, True)
    out_edges = [edge_dict[e] for e in self.in_edges]
    in_edges = [edge_dict[e] for e in self.out_edges]
    return QuantumConstructor(out_edges, in_edges)

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

    Subsystems to trace out are supplied as indicies, so that dangling edges
    are connected to eachother as:
      `out_edges[i] ^ in_edges[i] for i in subsystems_to_trace_out`

    This does not modify the original network. The original ordering of the
    remaining subsystems is maintained.

    Args:
      subsystems_to_trace_out: Indices of subsystems to trace out.
    Returns:
      A new QuantumOperator or QuantumScalar representing the result.
    """
    check_hilberts(self.in_edges, self.out_edges)

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

    return QuantumConstructor(out_edges, in_edges, ref_nodes)

  def __matmul__(self, other):
    check_hilberts(self.in_edges, other.out_edges)

    # copy all nodes involved in the two operators
    net = reachable(get_all_nodes(
      self.out_edges + self.in_edges + other.out_edges + other.in_edges))

    _, edge_dict = copy(net, False)

    # connect edges to create network for the result
    for (e1, e2) in zip(self.in_edges, other.out_edges):
      edge_dict[e1] = edge_dict[e1] ^ edge_dict[e2]

    in_edges = [edge_dict[e] for e in other.in_edges]
    out_edges = [edge_dict[e] for e in self.out_edges]

    return QuantumConstructor(out_edges, in_edges)


class QuantumKet(QuantumOperator):
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


class QuantumBra(QuantumOperator):
  def __init__(self, subsystem_edges):
    super().__init__([], subsystem_edges)

  @property
  def subsystem_edges(self):
    return self.in_edges