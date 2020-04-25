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
"""Tests for tensornetwork.contractors.bucket."""

from typing import Tuple

import numpy as np

from tensornetwork import network_components, CopyNode, Node
from tensornetwork.contractors import bucket_contractor
from tensornetwork.contractors import greedy

bucket = bucket_contractor.bucket


def add_cnot(
    q0: network_components.Edge,
    q1: network_components.Edge,
    backend: str = "numpy"
) -> Tuple[network_components.CopyNode, network_components.Edge,
           network_components.Edge]:
  """Adds the CNOT quantum gate to tensor network.

  CNOT consists of two rank-3 tensors: a COPY tensor on the control qubit and
  a XOR tensor on the target qubit.

  Args:
    q0: Input edge for the control qubit.
    q1: Input edge for the target qubit.
    backend: backend to use

  Returns:
    Tuple with three elements:
    - copy tensor corresponding to the control qubit
    - output edge for the control qubit and
    - output edge for the target qubit.
  """
  control = CopyNode(rank=3, dimension=2, backend=backend)
  xor = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.float64)
  target = Node(xor, backend=backend)
  network_components.connect(q0, control[0])
  network_components.connect(q1, target[0])
  network_components.connect(control[1], target[1])
  return (control, control[2], target[2])


def test_cnot_gate():
  # Prepare input state: |11>
  q0_in = Node(np.array([0, 1], dtype=np.float64))
  q1_in = Node(np.array([0, 1], dtype=np.float64))
  # Prepare output state: |10>
  q0_out = Node(np.array([0, 1], dtype=np.float64))
  q1_out = Node(np.array([1, 0], dtype=np.float64))
  # Build quantum circuit
  copy_node, q0_t1, q1_t1 = add_cnot(q0_in[0], q1_in[0])
  network_components.connect(q0_t1, q0_out[0])
  network_components.connect(q1_t1, q1_out[0])
  # Contract the network, first using Bucket Elimination, then once
  # no more copy tensors are left to exploit, fall back to the naive
  # contractor.
  contraction_order = (copy_node,)
  net = bucket([q0_in, q1_in, q0_out, q1_out, copy_node], contraction_order)
  result = greedy(net)
  # Verify that CNOT has turned |11> into |10>.
  np.testing.assert_allclose(result.get_tensor(), 1.0)


def test_swap_gate():
  # Prepare input state: 0.6|00> + 0.8|10>
  q0_in = Node(np.array([0.6, 0.8], dtype=np.float64), backend="jax")
  q1_in = Node(np.array([1, 0], dtype=np.float64), backend="jax")
  # Prepare output state: 0.6|00> + 0.8|01>
  q0_out = Node(np.array([1, 0], dtype=np.float64), backend="jax")
  q1_out = Node(np.array([0.6, 0.8], dtype=np.float64), backend="jax")
  # Build quantum circuit: three CNOTs implement a SWAP
  copy_node_1, q0_t1, q1_t1 = add_cnot(q0_in[0], q1_in[0], backend="jax")
  copy_node_2, q1_t2, q0_t2 = add_cnot(q1_t1, q0_t1, backend="jax")
  copy_node_3, q0_t3, q1_t3 = add_cnot(q0_t2, q1_t2, backend="jax")
  network_components.connect(q0_t3, q0_out[0])
  network_components.connect(q1_t3, q1_out[0])
  # Contract the network, first Bucket Elimination, then greedy to complete.
  contraction_order = (copy_node_1, copy_node_2, copy_node_3)
  nodes = [q0_in, q0_out, q1_in, q1_out, copy_node_1, copy_node_2, copy_node_3]
  net = bucket(nodes, contraction_order)
  result = greedy(net)
  # Verify that SWAP has turned |10> into |01> and kept |00> unchanged.
  np.testing.assert_allclose(result.get_tensor(), 1.0)
