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
"""Tests for tensornetwork.contractors.stochastic_contractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensornetwork import network
from tensornetwork.contractors import stochastic_contractor
tf.compat.v1.enable_v2_behavior()


class StochasticTest(tf.test.TestCase):

  def test_find_parallel(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.ones([4, 5, 2, 3]))
    b = net.add_node(np.ones([3, 2, 3, 5]))
    net.connect(a[2], b[1])
    net.connect(a[1], b[3])
    net.connect(a[3], b[0])
    parallel_edges, parallel_dim = stochastic_contractor.find_parallel(a[2])
    self.assertSetEqual(parallel_edges, {a[1], a[2], a[3]})
    self.assertEqual(parallel_dim, 30)

  def test_contract_trace_edges(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.ones([4, 5, 2, 3, 4]))
    b = net.add_node(np.ones([3, 2, 3, 5]))
    net.connect(a[2], b[1])
    net.connect(a[1], b[3])
    net.connect(b[0], b[2])
    net.connect(a[0], a[4])
    e = a[1]
    net, sizes, sizes_none = stochastic_contractor.contract_trace_edges(net)
    self.assertDictEqual(sizes, {e.node1: 30, e.node2: 10})
    self.assertDictEqual(sizes_none, dict())

  def test_contraction_sanity(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.ones([4, 5, 2]))
    b = net.add_node(np.ones([3, 2, 3]))
    net.connect(a[2], b[1])
    net.connect(b[0], b[2])
    net = stochastic_contractor.stochastic(net, 2)
    net.check_correct()
    res = net.get_final_node()
    self.assertAllClose(res.get_tensor(), 6 * np.ones([4, 5]))

  def test_contraction_parallel_edges(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.ones([4, 5, 2]))
    b = net.add_node(np.ones([3, 2, 3, 5]))
    c = net.add_node(np.ones([
        4,
    ]))
    net.connect(a[2], b[1])
    net.connect(b[0], b[2])
    net.connect(a[1], b[3])
    net.connect(a[0], c[0])
    net = stochastic_contractor.stochastic(net, 2)
    net.check_correct()
    res = net.get_final_node()
    self.assertAllClose(res.get_tensor(), 120)

  def test_contraction_disconnected(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.ones([4, 5, 2]))
    b = net.add_node(np.ones([3, 2, 3]))
    edge1 = a[0]
    net.connect(a[2], b[1])
    net.connect(b[0], b[2])
    c = net.add_node(np.ones([3, 4]))
    d = net.add_node(np.ones([4, 3]))
    edge2 = c[0]
    net.connect(c[1], d[0])
    net = stochastic_contractor.stochastic(net, 2)
    net.check_correct(check_connected=False)
    node1, node2 = edge1.node1, edge2.node1
    self.assertAllClose(node1.get_tensor(), 6 * np.ones([4, 5]))
    self.assertAllClose(node2.get_tensor(), 4 * np.ones([3, 3]))


if __name__ == '__main__':
  tf.test.main()
