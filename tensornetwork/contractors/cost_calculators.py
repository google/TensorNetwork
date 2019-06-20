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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from typing import List, Optional, Tuple
from tensornetwork import network
from tensornetwork import network_components


def cost_contract_between(
  node1: network_components.Node, node2: network_components.Node) -> int:
  """Calculate the memory cost of running `contract_between` on given nodes.

  The "memory cost" is the memory requirement to store the resulting
  tensor after calling `contract_between` on the two given nodes.

  Args:
    node1: The first node.
    node2: The second node.

  Return:
    The memory cost of the resulting contraction.
  """

  if node1 is node2:
    raise NotImplementedError("Trace cost calculation is not implemented.")
  if None in node1.shape or None in node2.shape:
    # We have no idea what the real cost will be, so just assume worst case.
    # This only appiles to code in tensorflow graph mode.
    return sys.maxsize
  contracted_dimensions = 1
  shared_edge_found = False
  for edge in node1.edges:
    if set(edge.get_nodes()) == {node1, node2}:
      shared_edge_found = True
      contracted_dimensions *= edge.node1.shape[edge.axis1]
  if not shared_edge_found:
    raise ValueError(
      "No shared edges found between '{}' and '{}'".format(node1, node2))
  # We take the square as we have to discount the contracted axes twice.
  # The resulting cost is the memory required to store the resulting
  # tensor after the `contract_between` call.
  cost = np.prod(node1.shape + node2.shape) // (contracted_dimensions)**2
  return cost

def cost_contract_parallel(edge: network_components.Edge) -> int:
  """Calculate the memory cost of running `contract_parallel on given edge.

  Args:
    edge: The edge

  Returns:
    The memory required to store the resulting tensor.
  """
  return cost_contract_between(edge.node1, edge.node2)