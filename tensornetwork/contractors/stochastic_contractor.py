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
import numpy as np
from tensornetwork import tensornetwork
from typing import Tuple, Set


def edge_cost(edge: tensornetwork.Edge) -> Tuple[int, Set]:
  """Calculates cost of contracting an edge.

  If A and B are the tensors that share the given `edge`, cost is defined as:
  cost = dims(A * B) - max(dims(A), dims(B)), where
  * denotes contraction of all shared edges (`contract_between`) and
  dims(X) is the total dimension of tensor X (product of sizes of all axes).

  Args:
    edge: Edge to calculate the cost.

  Returns:
    cost: Cost of the given edge.
    shared_edges: Edges that are shared between the nodes connected with
      the given edge.
  """
  # TODO: Verify whether this loss makes sense.
  node1, node2 = edge.node1, edge.node2
  # Calculate dimension of all shared edges
  nodes = {node1, node2}
  shared_dimension = 1
  shared_edges = set()
  for edge in node1.edges:
    if set(edge.get_nodes()) == nodes:
      shared_edges.add(edge)
      edge_size = edge.node1.get_tensor().shape.as_list()[edge.axis1]
      if edge_size is not None:
        shared_dimension *= edge_size

  dimension1 = np.prod([x for x in node1.get_tensor().shape.as_list()
                        if x is not None])
  dimension2 = np.prod([x for x in node2.get_tensor().shape.as_list()
                        if x is not None])
  prod_dimension = ((dimension1 // shared_dimension) *
                    (dimension2 // shared_dimension))
  cost = prod_dimension - max(dimension1, dimension2)
  return cost, shared_edges


def stochastic(network: tensornetwork.TensorNetwork,
               max_rejections: int, threshold: int = 1
              ) -> tensornetwork.TensorNetwork:
  """Contracts a connected network by stochastically picking edges.
  Algorithm 2 in page 7 of https://doi.org/10.1371/journal.pone.0208510.
  Cost calculation is slightly modified here.

  Args:
    network: Connected TensorNetwork to contract fully.
    max_rejections: Maximum number of rejections before you increase threshold.
    threshold: Initial value for the threshold.

  Returns:
    network: TensorNetwork with a single node after fully contracting.
  """
  # TODO: Think about a proper threshold value for our loss.
  rejections = 0
  nondangling_edges = network.get_all_nondangling()
  while nondangling_edges:
    edge = random.choice(tuple(nondangling_edges))
    if edge.node1 is edge.node2:
      network.contract(edge)
      nondangling_edges.remove(edge)
      rejections = 0
    else:
      cost, shared_edges = edge_cost(edge)
      if cost <= threshold:
        network.contract_parallel(edge)
        nondangling_edges -= shared_edges
        rejections = 0
      else:
        rejections += 1
        if rejections > max_rejections:
          threshold *= 2
          rejections = 0
  return network
