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
"""NCON interface to TensorNetwork."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from typing import Any, Sequence, List, Optional, Union, Text, Tuple, Dict, Any
from tensornetwork import network
from tensornetwork import network_components

Tensor = Any


def ncon(tensors: Sequence[Tensor],
         network_structure: Sequence[Sequence],
         con_order: Optional[Sequence] = None,
         out_order: Optional[Sequence] = None) -> Tensor:
  r"""Contracts a list of tensors according to a tensor network specification.

    The network is provided as a list of lists, one for each
    tensor, specifying labels for the edges connected to that tensor.

    If a contraction order `con_order` and an output order `out_order`
    are both provided, the edge labels can be anything.
    Otherwise (`con_order == None or out_order == None`), the edge labels 
    must be nonzero integers and edges will be contracted in ascending order.
    Negative integers denote the (dangling) indices of the output tensor,
    which will be in descending order, e.g. [-1,-2,-3,...].

    For example, matrix multiplication:

    ```python
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[1.0, 1.0], [0.0, 1.0]])
    ncon([A,B], [(-1, 1), (1, -2)])
    ```

    Matrix trace:

    ```python
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    ncon([A], [(1, 1)]) # 5.0
    ```

    Note: The reason `0` is not allowed as an edge label without manually
    specifying the contraction order is to maintain compatibility with the
    [original NCON implementation](https://arxiv.org/abs/1402.0939). However,
    the use of `0` in `con_order` to denote outer products is not (currently) 
    supported in this implementation.

    Args:
      tensors: List of `Tensor`s.
      network_structure: List of lists specifying the tensor network
          structure.
      con_order: List of edge labels specifying the contraction order.
      out_order: List of edge labels specifying the output order.

    Returns:
      A `Tensor` resulting from the contraction of the tensor network.
    """
  tn, con_edges, out_edges = ncon_network(
      tensors, network_structure, con_order=con_order, out_order=out_order)

  # Reverse the list so we can pop from the end: O(1).
  con_edges = con_edges[::-1]
  while con_edges:
    nodes_to_contract = con_edges[-1].get_nodes()
    edges_to_contract = tn.get_shared_edges(*nodes_to_contract)

    # Eat up all parallel edges that are adjacent in the ordering.
    adjacent_parallel_edges = set()
    for edge in reversed(con_edges):
      if edge in edges_to_contract:
        adjacent_parallel_edges.add(edge)
      else:
        break
    con_edges = con_edges[:-len(adjacent_parallel_edges)]

    # In an optimal ordering, all edges connecting a given pair of nodes are
    # adjacent in con_order. If this is not the case, warn the user.
    leftovers = edges_to_contract - adjacent_parallel_edges
    if leftovers:
      warnings.warn(
        "Suboptimal ordering detected. Edges {} are not adjacent in the "
        "contraction order to edges {}, connecting nodes {}. Deviating from "
        "the specified ordering!".format(
          list(map(str, leftovers)),
          list(map(str, adjacent_parallel_edges)),
          list(map(str, nodes_to_contract)))
        )
      con_edges = [e for e in con_edges if e not in edges_to_contract]

    if set(nodes_to_contract) == tn.nodes_set:
      # If this already produces the final output, order the edges
      # here to avoid transposes in some cases.
      tn.contract_between(
        *nodes_to_contract,
        name="con({},{})".format(*nodes_to_contract),
        output_edge_order=out_edges)
    else:
      tn.contract_between(
        *nodes_to_contract,
        name="con({},{})".format(*nodes_to_contract))

  # TODO: More efficient ordering of products based on out_edges
  res_node = tn.outer_product_final_nodes(out_edges)

  return res_node.tensor


def ncon_network(
    tensors: Sequence[Tensor],
    network_structure: Sequence[Sequence],
    con_order: Optional[Sequence] = None,
    out_order: Optional[Sequence] = None) -> Tuple[network.TensorNetwork, List[
        network_components.Edge], List[network_components.Edge]]:
  r"""Creates a TensorNetwork from a list of tensors according to `network`.

    The network is provided as a list of lists, one for each
    tensor, specifying labels for the edges connected to that tensor.

    If a contraction order `con_order` and an output order `out_order`
    are both provided, the edge labels can be anything.
    Otherwise (`con_order == None or out_order == None`), the edge labels 
    must be integers and edges will be contracted in ascending order.
    Negative integers denote the (dangling) indices of the output tensor,
    which will be in descending order, e.g. [-1,-2,-3,...].

    This is used internally by `ncon()`.

    Args:
      tensors: List of `Tensor`s.
      network_structure: List of lists specifying the tensor network.
      con_order: List of edge labels specifying the contraction order.
      out_order: List of edge labels specifying the output order.

    Returns:
      net: `TensorNetwork` with the structure given by `network`.
      con_edges: List of internal `Edge` objects in contraction order.
      out_edges: List of dangling `Edge` objects in output order.
  """
  if len(tensors) != len(network_structure):
    raise ValueError('len(tensors) != len(network_structure)')

  tn, edges = _build_network(tensors, network_structure)

  if con_order is None:
    try:
      con_order = sorted((k for k in edges if k >= 0))
      if con_order and con_order[0] == 0:
        raise ValueError("'0' is not a valid edge label when the "
          "contraction order is not specified separately.")
    except TypeError:
      raise ValueError("Non-integer edge label(s): {}".format(
          list(edges.keys())))
  else:
    if len(con_order) != len(set(con_order)):
      raise ValueError("Duplicate labels in con_order: {}".format(con_order))

  if out_order is None:
    try:
      out_order = sorted((k for k in edges if k < 0), reverse=True)
    except TypeError:
      raise ValueError("Non-integer edge label(s): {}".format(
          list(edges.keys())))
  else:
    if len(out_order) != len(set(out_order)):
      raise ValueError("Duplicate labels in out_order: {}".format(out_order))

  try:
    con_edges = [edges[k] for k in con_order]
    out_edges = [edges[k] for k in out_order]
  except KeyError as err:
    raise ValueError("Order contained an unknown edge label: {}".format(
        err.args[0]))

  if len(con_edges) + len(out_edges) != len(edges):
    raise ValueError(
        "Edges {} were not included in the contraction and output "
        "ordering.".format(
            list(set(edges.keys()) - set(con_order) - set(out_order))))

  for e in con_edges:
    if e.is_dangling():
      raise ValueError(
          "Contraction edge {} appears only once in the network.".format(
              str(e)))

  for e in out_edges:
    if not e.is_dangling():
      raise ValueError(
          "Output edge {} appears more than once in the network.".format(
              str(e)))

  return tn, con_edges, out_edges


def _build_network(
    tensors: Sequence[Tensor], network_structure: Sequence[Sequence]
) -> Tuple[network.TensorNetwork, Dict[Any, network_components.Edge]]:
  tn = network.TensorNetwork()
  nodes = []
  edges = {}
  for i, (tensor, edge_lbls) in enumerate(zip(tensors, network_structure)):
    if len(tensor.shape) != len(edge_lbls):
      raise ValueError(
          "Incorrect number of edge labels specified tensor {}".format(i))

    node = tn.add_node(tensor, name="tensor_{}".format(i))
    nodes.append(node)

    for (axis_num, edge_lbl) in enumerate(edge_lbls):
      if edge_lbl not in edges:
        e = node[axis_num]
        e.set_name(str(edge_lbl))
        edges[edge_lbl] = e
      else:
        # This will raise an error if the edges are not dangling.
        e = tn.connect(edges[edge_lbl], node[axis_num], name=str(edge_lbl))
        edges[edge_lbl] = e
  return tn, edges
