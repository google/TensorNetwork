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
"""Network contractor which exploits copy tensors."""

from typing import Sequence, Iterable
from tensornetwork.network_components import AbstractNode, contract_copy_node
from tensornetwork import network_components


def bucket(
    nodes: Iterable[AbstractNode],
    contraction_order: Sequence[network_components.CopyNode]
) -> Iterable[AbstractNode]:
  """Contract given nodes exploiting copy tensors.

  This is based on the Bucket-Elimination-based algorithm described in
  `arXiv:quant-ph/1712.05384`_, but avoids explicit construction of the 
  graphical model. Instead, it achieves the efficient contraction of sparse 
  tensors by representing them as subnetworks consisting of lower rank tensors
  and copy tensors. This function assumes that sparse tensors have already been
  decomposed this way by the caller.

  This contractor is efficient on networks with many copy tensors. Time and
  memory requirements are highly sensitive to the requested contraction order.

  Note that the returned tensor network may not be fully contracted if the input
  network doesn't have enough copy nodes. In this case, the client should use
  a different contractor to complete the contraction.

  .. _arXiv:quant-ph/1712.05384:
    https://arxiv.org/abs/1712.05384

  Args:
    nodes: A collection of connected nodes.
    contraction_order: Order in which copy tensors are contracted.

  Returns:
    A new iterable of nodes after contracting copy tensors.
  """
  nodes = set(nodes)
  for copy_node in contraction_order:
    partners = copy_node.get_partners()
    new_node = contract_copy_node(copy_node)
    nodes = nodes.difference(list(partners.keys()) + [copy_node])
    nodes.add(new_node)
  return nodes
