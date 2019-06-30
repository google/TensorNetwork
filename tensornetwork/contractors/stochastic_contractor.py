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
"""Stochastic Network Contraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from typing import Tuple, Set, Optional, Dict
from tensornetwork import network
from tensornetwork import network_components


def find_parallel(edge: network_components.Edge
                 ) -> Tuple[Set[network_components.Edge], int]:
  """Finds all edges shared between the nodes connected with the given edge.

  Args:
    edge: A non-dangling edge between two different nodes.

  Returns:
    parallel_edges: Edges that are parallel to the given edge.
    parallel_dim: Product of sizes of all parallel edges.
  """
  if edge.is_dangling():
    raise ValueError(
        "Cannot find parallel edges for dangling edge {}".format(edge))
  nodes = {edge.node1, edge.node2}
  parallel_dim = 1
  parallel_edges = set()
  for e in edge.node1.edges:
    if set(e.get_nodes()) == nodes:
      parallel_edges.add(e)
      edge_size = list(e.node1.get_tensor().shape)[e.axis1]
      if edge_size is not None:
        parallel_dim *= edge_size
  return parallel_edges, parallel_dim


def contract_trace_edges(
    net: network.TensorNetwork, none_value: int = 1
) -> Tuple[network.TensorNetwork, Dict[network_components.Node, int],
           Dict[network_components.Node, int]]:
  """Contracts trace edges and calculate tensor sizes for every node.

  Tensor size is defined as the product of sizes of each of edges (axes).

  Args:
    net: TensorNetwork to contract all the trace edges of.
    none_value: The value that None dimensions contribute to the tensor size.
      Unit (default) means that None dimensions are neglected.

  Returns:
    A tuple containing:
      net: 
        Given TensorNetwork with all its trace edges contracted.
      node_sizes: 
        Map from nodes in the network to their total size.
      node_sizes_none: 
        Map from nodes that have at least one None dimension to
        their size.
  """
  # Keep node sizes in memory for cost calculation
  node_sizes, node_sizes_none = dict(), dict()
  initial_node_set = set(net.nodes_set)
  for node in initial_node_set:
    trace_edges, flag_none, total_dim = set(), False, 1
    new_node = node
    for edge, dim in zip(node.edges, list(node.get_tensor().shape)):
      if edge.node1 is edge.node2:
        if edge not in trace_edges:
          # Contract trace edge
          new_node = net.contract(edge)
          trace_edges.add(edge)
      else:
        if dim is None:
          total_dim *= none_value
          flag_none = True
        else:
          total_dim *= dim
      if flag_none:
        node_sizes_none[new_node] = total_dim
      else:
        node_sizes[new_node] = total_dim
  return net, node_sizes, node_sizes_none


def stochastic(net: network.TensorNetwork,
               max_rejections: int,
               threshold: Optional[int] = None,
               none_value: int = 1) -> network.TensorNetwork:
  """Contracts a connected network by stochastically picking edges.

  Algorithm 2 in page 7 of https://doi.org/10.1371/journal.pone.0208510.
  Cost calculation is slightly modified here:
  If A and B are the tensors that share the given `edge`, cost is defined as:
  `cost = dims(A @ B) - max(dims(A), dims(B))`, where
  `@` denotes contraction of all shared edges via `contract_between` and
  `dims(X)` is the total dimension of tensor X (product of sizes of all axes).

  Args:
    net: Connected TensorNetwork to contract fully.
    max_rejections: Maximum number of rejections before you increase threshold.
    threshold: Initial value for the threshold.
    none_value: The value of None dimensions in the cost calculation.

  Returns:
    TensorNetwork with a single node after fully contracting.
  """
  net, node_sizes, node_sizes_none = contract_trace_edges(net, none_value)
  if threshold is None:
    # Set threshold as the maximum tensor size in the network
    # ignoring nodes with None sizes.
    threshold = max(node_sizes.values())
  node_sizes.update(node_sizes_none)

  rejections = 0
  nondangling_edges = net.get_all_nondangling()
  while nondangling_edges:
    edge = random.choice(tuple(nondangling_edges))
    shared_edges, shared_dim = find_parallel(edge)
    new_dim = ((node_sizes[edge.node1] // shared_dim) *
               (node_sizes[edge.node2] // shared_dim))
    cost = new_dim - max(node_sizes[edge.node1], node_sizes[edge.node2])
    if cost <= threshold:
      node_sizes.pop(edge.node1)
      node_sizes.pop(edge.node2)
      node_sizes[net.contract_parallel(edge)] = new_dim
      nondangling_edges -= shared_edges
      rejections = 0
    else:
      rejections += 1
      if rejections > max_rejections:
        threshold *= 2
        rejections = 0
  return net
