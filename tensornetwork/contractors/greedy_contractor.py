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
"""Greedy Contraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Tuple
from tensornetwork import network
from tensornetwork import network_components
from tensornetwork.contractors import cost_calculators 

cost_contract_parallel = cost_calculators.cost_contract_parallel


def greedy(net: network.TensorNetwork) -> network.TensorNetwork:
  """Contract the lowest cost pair of nodes first.
  
  Args:
    net: The TensorNetwork to contract.

  Returns:
    The contracted TensorNetwork.
  """
  edges = net.get_all_nondangling()
  # First, contract all of the trace edges.
  for edge in edges:
    if edge in net and edge.is_trace():
      net.contract_parallel(edge)
  # Get the edges again.
  edges = net.get_all_nondangling()
  while edges:
    edge = min(edges, key=lambda x: (cost_contract_parallel(x), x))
    net.contract_parallel(edge)
    edges = net.get_all_nondangling()
  return net
