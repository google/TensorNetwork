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
"""Modifications in TensorNetwork core to support batch summation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensornetwork


class BatchTensorNetwork(tensornetwork.TensorNetwork):
  """TensorNetwork that supports batch contraction."""

  _VALID_SUBSCRIPTS = list(
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

  def batched_contract_between(self,
                               node1: tensornetwork.Node,
                               node2: tensornetwork.Node,
                               batch_edge1: tensornetwork.Edge,
                               batch_edge2: tensornetwork.Edge
                              ) -> tensornetwork.Node:
    """Contract between that supports one batch edge in each node.
    
    Uses einsum property: "bij,bjk->bik".
    
    Args:
      node1: First node to contract.
      node2: Second node to contract.
      batch_edge1: The edge of node1 that correspond to its batch index.
      batch_edge2: The edge of node2 that correspond to its batch index.
    
    Returns:
      new_node: Result of the contraction. This node has by default batch_edge1
        as its batch edge. Its edges are in the order of the dangling edges of
        node1 followed by the dangling edges of node2.
    """
    if node1 is node2:
      raise ValueError("Cannot perform batched contraction between "
                       "node '{}' and itself.".format(node1))

    shared_edges = self.get_shared_edges(node1, node2)
    if not shared_edges:
      raise ValueError("No edges found between nodes "
                       "'{}' and '{}'".format(node1, node2))

    if batch_edge1 in shared_edges:
      raise ValueError(
          "Batch edge '{}' is shared between the nodes".format(batch_edge1))
    if batch_edge2 in shared_edges:
      raise ValueError(
          "Batch edge '{}' is shared between the nodes".format(batch_edge2))

    n_shared = len(shared_edges)
    shared_subscripts = dict(zip(shared_edges,
                                 self._VALID_SUBSCRIPTS[:n_shared]))

    res_string, string = [], []
    index = n_shared + 1
    for node, batch_edge in zip([node1, node2], [batch_edge1, batch_edge2]):
      string.append([])
      for edge in node.edges:
        if edge in shared_edges:
          string[-1].append(shared_subscripts[edge])
        elif edge is batch_edge:
          string[-1].append(self._VALID_SUBSCRIPTS[n_shared])
          if node is node1:
            res_string.append(self._VALID_SUBSCRIPTS[n_shared])
        else:
          string[-1].append(self._VALID_SUBSCRIPTS[index])
          res_string.append(self._VALID_SUBSCRIPTS[index])
          index += 1

    string1 = "".join(string[0])
    string2 = "".join(string[1])
    res_string = "".join(res_string)
    einsum_string = "".join([string1, ",", string2, "->", res_string])

    new_tensor = self.backend.einsum(einsum_string, node1.tensor, node2.tensor)
    new_node = self.add_node(new_tensor)

    # Modify batch edge 2 to avoid ValueError in remove
    batch_edge2.node2 = node1
    batch_edge2._is_dangling = False
    shared_edges.add(batch_edge2)
    self._remove_edges(shared_edges, node1, node2, new_node)
    return new_node


def pairwise_reduction(net: BatchTensorNetwork,
                       node: tensornetwork.Node,
                       edge: tensornetwork.Edge
                      ) -> tensornetwork.Node:
  """Parallel contraction of matrix chains.
  
  The operation performed by this function is described in Fig. 4 of the paper
  `TensorNetwork for Machine Learning`. It leads to a more efficient
  implementation of the MPS classifier both in terms of predictions and
  automatic gradient calculation. The idea is that the whole MPS side is saved
  in memory as one node that carries an artificial "space" edge. This function
  removes this additional index by performing the pairwise contractions as
  shown in the Figure.
  
  Args:
    net: TensorNetwork that contains the node we want to reduce.
    node: Node to reduce pairwise. The corresponding tensor should have the
      form (..., space edge, ..., a, b) and matrix multiplications will be
      performed over the last two indices using matmul.
    edge: Space edge of the node.
  
  Returns:
    node: Node after the reduction. Has the shape of given node with the `edge`
      removed.
  """
  # NOTE: This method could be included in the BatchedTensorNetwork class
  # however it seems better to be separated because (at least with the current
  # implementation) it performs a very specialized/non-general operation.
  # It also uses tf.matmul which restricts the backend, however this can be
  # easily generalized since all the backends support batched matmul.
  if not edge.is_dangling():
    raise ValueError("Cannot reduce non-dangling edge '{}'".format(edge))
  if edge.node1 is not node:
    raise ValueError(
        "Edge '{}' does not belong to node '{}'".format(edge, node))

  tensor = node.tensor
  size = int(tensor.shape[edge.axis1])

  # Bring reduction edge in first position
  edge_order = list(range(len(list(tensor.shape))))
  edge_order[0] = edge.axis1
  edge_order[edge.axis1] = 0
  tensor = net.backend.transpose(tensor, edge_order)

  # Remove edge to be reduced from node
  node.edges.pop(edge.axis1)
  for e in node.edges[edge.axis1:]:
    if e.node1 is e.node2:
      raise NotImplementedError("Cannot binary reduce node "
                                "'{}' with trace edge '{}'".format(node, e))
    if e.node1 is node:
      e.axis1 -= 1
    else:
      e.axis2 -= 1

  # Idea from this implementation is from jemisjoky/TorchMPS
  while size > 1:
    half_size = size // 2
    nice_size = 2 * half_size
    leftover = tensor[nice_size:]
    tensor = tf.matmul(tensor[0:nice_size:2], tensor[1:nice_size:2])
    tensor = net.backend.concat([tensor, leftover], axis=0)
    size = half_size + int(size % 2 == 1)

  node.tensor = tensor[0]
  return node
