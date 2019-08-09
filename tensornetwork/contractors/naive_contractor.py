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
"""Naive Network Contraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Sequence, Optional
from tensornetwork import network
from tensornetwork import network_components


def naive(net: network.TensorNetwork,
          output_edge_order: Sequence[Sequence[network_components.Edge]],
          edge_order: Optional[Sequence[network_components.Edge]] = None,
         ) -> network.TensorNetwork:
  """Contract a TensorNetwork in the order the edges were created.
  if `edge_order` is passed, contract the network in the order of `edge_order`.

  This contraction method will usually be very suboptimal unless the edges were
  created in a deliberate way.

  Args:
    net: A TensorNetwork.
    output_edge_order: A list of lists of edges. Edges of the final nodes in `nodes_set` are reordered into `output_edge_order`
    edge_order: An optional list of edges. Must be equal to all non-dangling
      edges in the net.
  Returns:
    The given TensorNetwork with all non-dangling edges contracted.
  Raises:
    ValueError: If the passed `edge_order` list does not contain all of the
      non-dangling edges of the network.
  """
  if edge_order is None:
    edge_order = sorted(net.get_all_nondangling())
    
  if set(edge_order) != net.get_all_nondangling():
    raise ValueError("Set of passed edges does not match expected set."
                     "Given: {}\nExpected: {}".format(
                          edge_order, net.get_all_nondangling()))

  if set([edge for edge_list in output_edge_order for edge in edge_list]) != (net.get_all_edges() - net.get_all_nondangling()):
    raise ValueError("output edges are not all dangling.")
  
  for edge in edge_order:
    if edge in net:
      net.contract_parallel(edge)
      
  if len(net.nodes_set) != len(output_edge_order):
    raise ValueError("number of elements {0} in `output_edge_order` is not matching"
                     " the number of remaining nodes {1} in `nodes_set`".format(len(output_edge_order), len(net.nodes_set)))
  for order in output_edge_order:
    if not any([set(order) == set(node.edges) for node in net.nodes_set]):
      raise ValueError('elements {0} in `output_edge_order` do not match any node-edges in `node_set`'.format([o.name for o in order]))
    
  node_order_pairs = [(node, order) for order in output_edge_order for node in net.nodes_set  if set(node.edges) == set(order)]    
  net.nodes_set = set([node.reorder_edges(order) for node, order in node_order_pairs])
  return net

