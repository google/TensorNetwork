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
# pylint: disable=line-too-long
from tensornetwork.network_operations import get_all_edges, get_subgraph_dangling
from tensornetwork.network_components import AbstractNode, Edge
from typing import (Any, Callable, Dict, List, Set, Tuple, Iterable, Text)
# `opt_einsum` algorithm method typing
Algorithm = Callable[[List[Set[Edge]], Set[Edge], Dict[Edge, Any]],
                     List[Tuple[int, int]]]


def multi_remove(elems: List[Any], indices: List[int]) -> List[Any]:
  """Remove multiple indicies in a list at once."""
  return [i for j, i in enumerate(elems) if j not in indices]


def get_path(
    nodes: Iterable[AbstractNode],
    algorithm: Algorithm) -> Tuple[List[Tuple[int, int]], List[AbstractNode]]:
  """Calculates the contraction paths using `opt_einsum` methods.

  Args:
    nodes: An iterable of nodes.
    algorithm: `opt_einsum` method to use for calculating the contraction path.

  Returns:
    The optimal contraction path as returned by `opt_einsum`.
  """
  nodes = list(nodes)
  input_sets = [set(node.edges) for node in nodes]
  output_set = get_subgraph_dangling(nodes)
  size_dict = {edge: edge.dimension for edge in get_all_edges(nodes)}

  return algorithm(input_sets, output_set, size_dict), nodes
