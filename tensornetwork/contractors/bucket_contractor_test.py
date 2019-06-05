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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np

from tensornetwork import network
from tensornetwork import network_components
from tensornetwork.contractors import bucket_contractor
from tensornetwork.contractors import naive_contractor

bucket = bucket_contractor.bucket
naive = naive_contractor.naive


def add_cnot(net: network.TensorNetwork,
             q0: network_components.Edge,
             q1: network_components.Edge
) -> Tuple[network_components.CopyNode,
           network_components.Edge,
           network_components.Edge]:
  """Adds the CNOT quantum gate to tensor network.

  CNOT consists of two rank-3 tensors: a COPY tensor on the control qubit and
  a XOR tensor on the target qubit.

  Args:
    net: Tensor network to add CNOT to.
    q0: Input edge for the control qubit.
    q1: Input edge for the target qubit.

  Returns:
    Tuple with three elements:
    - copy tensor corresponding to the control qubit
    - output edge for the control qubit and
    - output edge for the target qubit.
  """
  control = net.add_copy_node(rank=3, dimension=2)
  xor = np.array([[[1, 0], [0, 1]],
                  [[0, 1], [1, 0]]], dtype=np.float64)
  target = net.add_node(xor)
  net.connect(q0, control[0])
  net.connect(q1, target[0])
  net.connect(control[1], target[1])
  return control, control[2], target[2]


def test_cnot_gate():
  net = network.TensorNetwork(backend="numpy")
  # Prepare input state: |11>
  q0_in = net.add_node(np.array([0, 1], dtype=np.float64))
  q1_in = net.add_node(np.array([0, 1], dtype=np.float64))
  # Prepare output state: |10>
  q0_out = net.add_node(np.array([0, 1], dtype=np.float64))
  q1_out = net.add_node(np.array([1, 0], dtype=np.float64))
  # Build quantum circuit
  copy_node, q0_t1, q1_t1 = add_cnot(net, q0_in[0], q1_in[0])
  net.connect(q0_t1, q0_out[0])
  net.connect(q1_t1, q1_out[0])
  # Contract the network, first using Bucket Elimination, then once
  # no more copy tensors are left to exploit, fall back to the naive
  # contractor.
  contraction_order = (copy_node,)
  net = bucket(net, contraction_order)
  net = naive(net)
  result = net.get_final_node()
  # Verify that CNOT has turned |11> into |10>.
  np.testing.assert_allclose(result.get_tensor(), 1.0)

def test_swap_gate():
  net = network.TensorNetwork(backend="jax")
  # Prepare input state: 0.6|00> + 0.8|10>
  q0_in = net.add_node(np.array([0.6, 0.8], dtype=np.float64))
  q1_in = net.add_node(np.array([1, 0], dtype=np.float64))
  # Prepare output state: 0.6|00> + 0.8|01>
  q0_out = net.add_node(np.array([1, 0], dtype=np.float64))
  q1_out = net.add_node(np.array([0.6, 0.8], dtype=np.float64))
  # Build quantum circuit: three CNOTs implement a SWAP
  copy_node_1, q0_t1, q1_t1 = add_cnot(net, q0_in[0], q1_in[0])
  copy_node_2, q1_t2, q0_t2 = add_cnot(net, q1_t1, q0_t1)
  copy_node_3, q0_t3, q1_t3 = add_cnot(net, q0_t2, q1_t2)
  net.connect(q0_t3, q0_out[0])
  net.connect(q1_t3, q1_out[0])
  # Contract the network, first Bucket Elimination, then naive to complete.
  contraction_order = (copy_node_1, copy_node_2, copy_node_3)
  net = bucket(net, contraction_order)
  net = naive(net)
  result = net.get_final_node()
  # Verify that SWAP has turned |10> into |01> and kept |00> unchanged.
  np.testing.assert_allclose(result.get_tensor(), 1.0)
