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
from typing import Any, Callable, Dict, Optional, List, Set, Sequence
from tensornetwork import network
from tensornetwork import network_components
from tensornetwork.contractors.opt_einsum_paths import utils


def base(net: network.TensorNetwork,
         algorithm: Callable[[List[Set[int]], Set[int], Dict[int, int]], List],
         output_edge_order: Optional[Sequence[network_components.Edge]] = None
        ) -> network.TensorNetwork:
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
  nodes = sorted(net.nodes_set)
  input_sets = utils.get_input_sets(net)
  output_set = utils.get_output_set(net)
  size_dict = utils.get_size_dict(net)
  path = algorithm(input_sets, output_set, size_dict)
  for a, b in path:
    new_node = nodes[a] @ nodes[b]
    nodes.append(new_node)
    nodes = utils.multi_remove(nodes, [a, b])

  # if the final node has more than one edge,
  # output_edge_order has to be specified
  final_node = net.get_final_node()
  if (len(final_node.edges) <= 1) and (output_edge_order is None):
    output_edge_order = list((net.get_all_edges() -
                              net.get_all_nondangling()))
  elif (len(final_node.edges) > 1) and (output_edge_order is None):
    raise ValueError("if the final node has more than one dangling edge"
                     " `output_edge_order` has to be provided")

  if set(output_edge_order) != (
      net.get_all_edges() - net.get_all_nondangling()):
    raise ValueError("output edges are not all dangling.")

  final_node.reorder_edges(output_edge_order)
  return net


def optimal(net: network.TensorNetwork,
            output_edge_order: Sequence[network_components.Edge] = None,
            memory_limit: Optional[int] = None) -> network.TensorNetwork:
  """Optimal contraction order via `opt_einsum`.

  This method will find the truly optimal contraction order via
  `opt_einsum`'s depth first search algorithm. Since this search is
  exhaustive, if your network is large (n>10), then the search may
  take longer than just contracting in a suboptimal way.

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
  alg = functools.partial(opt_einsum.paths.optimal, memory_limit=memory_limit)
  return base(net, alg, output_edge_order)


def branch(net: network.TensorNetwork,
           output_edge_order: Sequence[network_components.Edge] = None,
           memory_limit: Optional[int] = None,
           nbranch: Optional[int] = None) -> network.TensorNetwork:
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
  alg = functools.partial(opt_einsum.paths.branch,
                          memory_limit=memory_limit, nbranch=nbranch)
  return base(net, alg, output_edge_order)


def greedy(net: network.TensorNetwork,
           output_edge_order: Sequence[network_components.Edge] = None,
           memory_limit: Optional[int] = None) -> network.TensorNetwork:
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
  return base(net, alg, output_edge_order)


def auto(net: network.TensorNetwork,
         output_edge_order: Sequence[network_components.Edge] = None,
         memory_limit: Optional[int] = None) -> network.TensorNetwork:
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
    final_node = list(net.nodes_set)
    if (len(final_node[0].edges) <= 1) and (output_edge_order is None):
      output_edge_order = list((net.get_all_edges() -
                                net.get_all_nondangling()))
    elif (len(final_node[0].edges) > 1) and (output_edge_order is None):
      raise ValueError("if the final node has more than one dangling edge"
                       ", `output_edge_order` has to be provided")

    net.nodes_set = set([list(net.nodes_set)[0].reorder_edges(
      output_edge_order)])
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

def custom(net: network.TensorNetwork,
           optimizer: Any,
           output_edge_order: Sequence[network_components.Edge] = None,
           memory_limit: Optional[int] = None) -> network.TensorNetwork:
  """
  Uses a custom path optimizer created by the user to calculate paths.

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
  return base(net, alg, output_edge_order)
