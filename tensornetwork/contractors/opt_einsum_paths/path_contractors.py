# pylint: disable=cyclic-import
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
"""Contractors based on `opt_einsum`'s path algorithms."""

import functools
import opt_einsum
import tensornetwork as tn
from tensornetwork.network import TensorNetwork
from tensornetwork.network_components import Edge, BaseNode
from tensornetwork.contractors.opt_einsum_paths import utils
from typing import Any, Optional, Sequence, Iterable, Union


def _base_nodes(nodes: Iterable[BaseNode],
                algorithm: utils.Algorithm,
                output_edge_order: Optional[Sequence[Edge]] = None) -> BaseNode:
  """Base method for all `opt_einsum` contractors.

  Args:
    nodes: A collection of connected nodes.
    algorithm: `opt_einsum` contraction method to use.
    output_edge_order: An optional list of edges. Edges of the
      final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.

  Returns:
    Final node after full contraction.
  """
  tn.check_connected(nodes)
  nodes_set = {node for node in nodes}
  edges = tn.get_all_nondangling(nodes_set)
  #output edge order has to be determinded before any contraction
  #(edges are refreshed after contractions)
  if output_edge_order is None:
    output_edge_order = list(
        (tn.get_all_edges(nodes) - tn.get_all_nondangling(nodes)))

  if set(output_edge_order) != (
      tn.get_all_edges(nodes) - tn.get_all_nondangling(nodes)):
    raise ValueError("output edges are not all dangling.")

  for edge in edges:
    if not edge.is_disabled:  #if its disabled we already contracted it
      if edge.is_trace():
        nodes_set.remove(edge.node1)
        nodes_set.add(tn.contract_parallel(edge))

  if not tn.get_all_nondangling(nodes_set):
    # There's nothing to contract.
    return list(nodes_set)[0]

  # Then apply `opt_einsum`'s algorithm
  path, nodes = utils.get_path(nodes_set, algorithm)
  for a, b in path:
    new_node = nodes[a] @ nodes[b]
    nodes.append(new_node)
    nodes = utils.multi_remove(nodes, [a, b])

  # if the final node has more than one edge,
  # output_edge_order has to be specified
  final_node = nodes[0]  # nodes were connected, we checked this
  if (len(final_node.edges) > 1) and (output_edge_order is None):
    raise ValueError("if the final node has more than one dangling edge"
                     " `output_edge_order` has to be provided")

  final_node.reorder_edges(output_edge_order)
  return final_node


def _base_network(
    net: TensorNetwork,
    algorithm: utils.Algorithm,
    output_edge_order: Optional[Sequence[Edge]] = None) -> TensorNetwork:
  """Base method for all `opt_einsum` contractors.

  Args:
    net: a TensorNetwork object. Should be connected.
    algorithm: `opt_einsum` contraction method to use.
    output_edge_order: An optional list of edges. Edges of the
      final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.

  Returns:
    The network after full contraction.
  """
  net.check_connected()
  # First contract all trace edges
  edges = net.get_all_nondangling()
  for edge in edges:
    if edge in net and edge.is_trace():
      net.contract_parallel(edge)
  if not net.get_all_nondangling():
    # There's nothing to contract.
    return net

  # Then apply `opt_einsum`'s algorithm
  path, nodes = utils.get_path(net, algorithm)
  for a, b in path:
    new_node = nodes[a] @ nodes[b]
    nodes.append(new_node)
    nodes = utils.multi_remove(nodes, [a, b])

  # if the final node has more than one edge,
  # output_edge_order has to be specified
  final_node = net.get_final_node()
  if (len(final_node.edges) <= 1) and (output_edge_order is None):
    output_edge_order = list((net.get_all_edges() - net.get_all_nondangling()))
  elif (len(final_node.edges) > 1) and (output_edge_order is None):
    raise ValueError("if the final node has more than one dangling edge"
                     " `output_edge_order` has to be provided")

  if set(output_edge_order) != (
      net.get_all_edges() - net.get_all_nondangling()):
    raise ValueError("output edges are not all dangling.")

  final_node.reorder_edges(output_edge_order)
  return net


def base(nodes: Union[TensorNetwork, Iterable[BaseNode]],
         algorithm: utils.Algorithm,
         output_edge_order: Optional[Sequence[Edge]] = None
        ) -> Union[BaseNode, TensorNetwork]:

  if isinstance(nodes, TensorNetwork):
    return _base_network(nodes, algorithm, output_edge_order)

  return _base_nodes(nodes, algorithm, output_edge_order)


def optimal(
    nodes: Union[TensorNetwork, Iterable[BaseNode]],
    output_edge_order: Sequence[Edge] = None,
    memory_limit: Optional[int] = None) -> Union[BaseNode, TensorNetwork]:
  """Optimal contraction order via `opt_einsum`.

  This method will find the truly optimal contraction order via
  `opt_einsum`'s depth first search algorithm. Since this search is
  exhaustive, if your network is large (n>10), then the search may
nn  take longer than just contracting in a suboptimal way.

  nArgs:
    net: a TensorNetwork object.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    The network after full contraction.
  """
  alg = functools.partial(opt_einsum.paths.optimal, memory_limit=memory_limit)
  return base(nodes, alg, output_edge_order)


def branch(nodes: Union[TensorNetwork, Iterable[BaseNode]],
           output_edge_order: Sequence[Edge] = None,
           memory_limit: Optional[int] = None,
           nbranch: Optional[int] = None) -> Union[BaseNode, TensorNetwork]:
  """Branch contraction path via `opt_einsum`.

  This method uses the DFS approach of `optimal` while sorting potential
  contractions based on a heuristic cost, in order to reduce time spent
  in exploring paths which are unlikely to be optimal.
  For more details:
    https://optimized-einsum.readthedocs.io/en/latest/branching_path.html

  Args:
    net: a TensorNetwork object.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
       are reordered into `output_edge_order`;
       if final node has more than one edge,
       `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.
    nbranch: Number of best contractions to explore.
      If None it explores all inner products starting with those that
      have the best cost heuristic.

  Returns:
    The network after full contraction.
  """
  alg = functools.partial(
      opt_einsum.paths.branch, memory_limit=memory_limit, nbranch=nbranch)
  return base(nodes, alg, output_edge_order)


def greedy(
    nodes: Union[TensorNetwork, Iterable[BaseNode]],
    output_edge_order: Sequence[Edge] = None,
    memory_limit: Optional[int] = None) -> Union[BaseNode, TensorNetwork]:
  """Greedy contraction path via `opt_einsum`.

  This provides a more efficient strategy than `optimal` for finding
  contraction paths in large networks. First contracts pairs of tensors
  by finding the pair with the lowest cost at each step. Then it performs
  the outer products.
  For more details:
    https://optimized-einsum.readthedocs.io/en/latest/greedy_path.html

  Args:
    net: a TensorNetwork object.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    The network after full contraction.
  """
  alg = functools.partial(opt_einsum.paths.greedy, memory_limit=memory_limit)
  return base(nodes, alg, output_edge_order)


def _auto_nodes(nodes: Iterable[BaseNode],
                output_edge_order: Sequence[Edge] = None,
                memory_limit: Optional[int] = None) -> BaseNode:
  """Chooses one of the above algorithms according to network size.

  Default behavior is based on `opt_einsum`'s `auto` contractor.

  Args:
    nodes: A collection of connected nodes.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    Final node after full contraction.
  """
  n = len(nodes)
  if n <= 0:
    raise ValueError("Cannot contract empty tensor network.")
  if n == 1:
    edges = tn.get_all_nondangling(nodes)

    if output_edge_order is None:
      output_edge_order = list(
          (tn.get_all_edges(nodes) - tn.get_all_nondangling(nodes)))

    final_node = tn.contract_parallel(edges.pop())
    final_node.reorder_edges(output_edge_order)
    return final_node
  if n < 5:
    return optimal(nodes, output_edge_order, memory_limit)
  if n < 7:
    return branch(nodes, output_edge_order, memory_limit)
  if n < 9:
    return branch(nodes, output_edge_order, memory_limit, nbranch=2)
  if n < 15:
    return branch(nodes, output_edge_order, nbranch=1)
  return greedy(nodes, output_edge_order, memory_limit)


def _auto_network(net: TensorNetwork,
                  output_edge_order: Sequence[Edge] = None,
                  memory_limit: Optional[int] = None) -> TensorNetwork:
  """Chooses one of the above algorithms according to network size.

  Default behavior is based on `opt_einsum`'s `auto` contractor.

  Args:
    net: a TensorNetwork object.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    The network after full contraction.
  """
  n = len(net.nodes_set)
  if n <= 0:
    raise ValueError("Cannot contract empty tensor network.")
  if n == 1:
    edges = net.get_all_nondangling()
    net.contract_parallel(edges.pop())
    final_node = net.get_final_node()
    if (len(final_node.edges) <= 1) and (output_edge_order is None):
      output_edge_order = list(
          (net.get_all_edges() - net.get_all_nondangling()))
    elif (len(final_node.edges) > 1) and (output_edge_order is None):
      raise ValueError("if the final node has more than one dangling edge"
                       ", `output_edge_order` has to be provided")

    final_node.reorder_edges(output_edge_order)
    return net
  if n < 5:
    return optimal(net, output_edge_order, memory_limit)
  if n < 7:
    return branch(net, output_edge_order, memory_limit)
  if n < 9:
    return branch(net, output_edge_order, memory_limit, nbranch=2)
  if n < 15:
    return branch(net, output_edge_order, nbranch=1)
  return greedy(net, output_edge_order, memory_limit)


def auto(nodes: Union[TensorNetwork, Iterable[BaseNode]],
         output_edge_order: Sequence[Edge] = None,
         memory_limit: Optional[int] = None) -> Union[BaseNode, TensorNetwork]:

  if isinstance(nodes, TensorNetwork):
    return _auto_network(nodes, output_edge_order, memory_limit)

  return _auto_nodes(nodes, output_edge_order, memory_limit)


def custom(
    nodes: Union[TensorNetwork, Iterable[BaseNode]],
    optimizer: Any,
    output_edge_order: Sequence[Edge] = None,
    memory_limit: Optional[int] = None) -> Union[BaseNode, TensorNetwork]:
  """Uses a custom path optimizer created by the user to calculate paths.

  The custom path optimizer should inherit `opt_einsum`'s `PathOptimizer`.
  For more details:
    https://optimized-einsum.readthedocs.io/en/latest/custom_paths.html

  Args:
    net: a TensorNetwork object.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      output_edge_order` must be provided.
    optimizer: A custom `opt_einsum.PathOptimizer` object.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    The network after full contraction.
  """
  alg = functools.partial(optimizer, memory_limit=memory_limit)
  return base(nodes, alg, output_edge_order)
