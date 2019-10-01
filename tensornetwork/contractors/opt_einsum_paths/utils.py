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
import tensornetwork as tn
from tensornetwork.network import TensorNetwork
from tensornetwork.network_components import BaseNode, Edge
from typing import Any, Callable, Dict, List, Set, Tuple, Iterable, Union
# `opt_einsum` algorithm method typing
#Algorithm = Callable[[List[Set[Edge]], Set[Edge], Dict[Edge, Any]], List[
#    Tuple[int, int]]]
Algorithm = Callable[[List[Set[int]], Set[int], Dict[int, int]], List[
    Tuple[int, int]]]


def multi_remove(elems: List[Any], indices: List[int]) -> List[Any]:
  """Remove multiple indicies in a list at once."""
  return [i for j, i in enumerate(elems) if j not in indices]


def _get_path_nodes(nodes: Iterable[BaseNode], algorithm: Algorithm
                   ) -> Tuple[List[Tuple[int, int]], List[BaseNode]]:
  """Calculates the contraction paths using `opt_einsum` methods.

  Args:
    nodes: An iterable of nodes.
    algorithm: `opt_einsum` method to use for calculating the contraction path.

  Returns:
    The optimal contraction path as returned by `opt_einsum`.
  """
  sorted_nodes = sorted(nodes, key=lambda n: n.signature)

  input_sets = [set(node.edges) for node in sorted_nodes]
  output_set = tn.get_all_edges(nodes) - tn.get_all_nondangling(nodes)
  size_dict = {edge: edge.dimension for edge in tn.get_all_edges(nodes)}

  return algorithm(input_sets, output_set, size_dict), sorted_nodes


def _get_path_network(net: TensorNetwork, algorithm: Algorithm
                     ) -> Tuple[List[Tuple[int, int]], List[BaseNode]]:
  """Calculates the contraction paths using `opt_einsum` methods.

  Args:
    net: TensorNetwork object to contract.
    algorithm: `opt_einsum` method to use for calculating the contraction path.

  Returns:
    The optimal contraction path as returned by `opt_einsum`.
  """
  sorted_nodes = sorted(net.nodes_set, key=lambda n: n.signature)

  input_sets = [set(node.edges) for node in sorted_nodes]
  output_set = net.get_all_edges() - net.get_all_nondangling()
  print(output_set)
  size_dict = {edge: edge.dimension for edge in net.get_all_edges()}

  return algorithm(input_sets, output_set, size_dict), sorted_nodes


def get_path(
    nodes: Union[TensorNetwork, Iterable[BaseNode]],
    algorithm: Algorithm) -> Tuple[List[Tuple[int, int]], List[BaseNode]]:
  """Calculates the contraction paths using `opt_einsum` methods.
  Args:
    nodes: TensorNetwork object or an iterable of `BaseNode` objects
      to contract.
    algorithm: `opt_einsum` method to use for calculating the contraction path.
  Returns:
    The optimal contraction path as returned by `opt_einsum`.
  """
  if isinstance(nodes, TensorNetwork):
    return _get_path_network(nodes, algorithm)
  return _get_path_nodes(nodes, algorithm)
