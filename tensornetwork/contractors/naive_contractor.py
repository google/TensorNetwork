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
          edge_order: Optional[Sequence[network_components.Edge]] = None
         ) -> network.TensorNetwork:
  """Contract a TensorNetwork in the order the edges were created.

  This contraction method will usually be very suboptimal unless the edges were
  created in a deliberate way.

  Args:
    net: A TensorNetwork.
    edge_order: An optional list of edges. Must be equal to all non-dangling
      edges in the net.
  Returns:
    The given TensorNetwork with all non-dangling edges contracted.
  Raises:
    ValueError: If any of the edges originally created by `connect` have been
      contracted or flattened.
  """
  if edge_order is None:
    edge_order = net.edge_order
  if set(edge_order) != net.get_all_nondangling():
    raise ValueError("Some non-dangling edges that were orginally created by "
                     "`connect` are no longer in the graph. Please do NOT use"
                     " any edge manipulation methods (contract, flatten, "
                     "split_node, etc) before using the naive contractor.\n"
                     "Original edges missing: {}.\n"
                     "New edges found: {}".format(
                         set(edge_order) - net.get_all_nondangling(),
                         net.get_all_nondangling() - set(edge_order)))
  for edge in edge_order:
    if edge in net:
      net.contract_parallel(edge)
  return net
