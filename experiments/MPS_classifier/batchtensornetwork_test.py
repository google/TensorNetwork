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

import pytest
import numpy as np
import tensorflow as tf
from experiments.MPS_classifier import batchtensornetwork

tf.enable_v2_behavior()


@pytest.fixture(
  name="backend", params=["numpy", "tensorflow", "jax"])
def backend_fixure(request):
    return request.param

def test_batched_contract_between_with_matrices(backend):
  net = batchtensornetwork.BatchTensorNetwork(backend=backend)
  a = net.add_node(np.ones([10, 2, 2]))
  b = net.add_node(np.ones([10, 2, 2]))
  net.connect(a[2], b[1])
  c = net.batched_contract_between(a, b, a[0], b[0])
  np.testing.assert_allclose(c.tensor, 2 * np.ones([10, 2, 2]))
  
def test_batched_contract_between_multiple_shared_edges(backend):
  net = batchtensornetwork.BatchTensorNetwork(backend=backend)
  a = net.add_node(np.ones([5, 3, 4, 9]))
  b = net.add_node(np.ones([9, 4, 5]))
  net.connect(a[2], b[1])
  net.connect(a[3], b[0])
  c = net.batched_contract_between(a, b, a[0], b[2])
  np.testing.assert_allclose(c.tensor, 36 * np.ones([5, 3]))
  
def test_pairwise_reduction():
  net = batchtensornetwork.BatchTensorNetwork(backend="tensorflow")
  a = net.add_node(np.ones([10, 2, 2]))
  b = batchtensornetwork.pairwise_reduction(net, a, a[0])
  np.testing.assert_allclose(b.tensor, 512 * np.ones([2, 2]))
  
def test_pairwise_reduction_multiple_edges():
  net = batchtensornetwork.BatchTensorNetwork(backend="tensorflow")
  a = net.add_node(np.ones([3, 5, 6, 2, 2]))
  b = batchtensornetwork.pairwise_reduction(net, a, a[1])
  np.testing.assert_allclose(b.tensor, 16 * np.ones([3, 6, 2, 2]))
