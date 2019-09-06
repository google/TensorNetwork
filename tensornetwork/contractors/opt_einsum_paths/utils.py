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
"""Helper methods for `path_contractors`."""

from tensornetwork import network
from tensornetwork import network_components
from typing import Any, Callable, Dict, List, Set, Tuple, Union

# `opt_einsum` algorithm method typing
Algorithm = Callable[[List[Set[int]], Set[int], Dict[int, int]],
                     List[Tuple[int]]]


def multi_remove(elems: List[Any], indices: List[int]) -> List[Any]:
  """Remove multiple indicies in a list at once."""
  return [i for j, i in enumerate(elems) if j not in indices]


def get_input_sets(net: network.TensorNetwork
                   ) -> Tuple[Union[List[Set[network_components.Edge]],
                                    List[network_components.Node]]]:
  sorted_nodes = sorted(net.nodes_set, key = lambda n: n.signature)
  input_sets = [set(node.edges) for node in sorted_nodes]
  return input_sets, sorted_nodes


def get_output_set(net: network.TensorNetwork) -> Set[network_components.Edge]:
  dangling_edges = net.get_all_edges() - net.get_all_nondangling()
  return set(dangling_edges)


def get_size_dict(net: network.TensorNetwork
                  ) -> Dict[network_components.Edge, int]:
  return {edge: edge.dimension for edge in net.get_all_edges()}


def get_path(net: network.TensorNetwork, algorithm: Algorithm
             ) -> Tuple[Union[List[Tuple[int]], List[network_components.Node]]]:
  """Calculates the contraction paths using `opt_einsum` methods.

  Args:
    net: TensorNetwork object to contract.
    algorithm: `opt_einsum` method to use for calculating the contraction path.

  Returns:
    The optimal contraction path as returned by `opt_einsum`.
    A list of nodes sorted compatibly with their indices in the path.
  """
  input_sets, sorted_nodes = get_input_sets(net)
  output_set = get_output_set(net)
  size_dict = get_size_dict(net)
  return algorithm(input_sets, output_set, size_dict), sorted_nodes