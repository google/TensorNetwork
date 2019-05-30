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
"""Tests for tensornetwork.contractors.naive."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensornetwork import network
from tensornetwork.contractors import naive_contractor

naive = naive_contractor.naive


class NaiveTest(tf.test.TestCase):

  def test_sanity_check(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.eye(2))
    b = net.add_node(np.eye(2))
    c = net.add_node(np.eye(2))
    net.connect(a[0], b[1])
    net.connect(b[0], c[1])
    net.connect(c[0], a[1])
    result = naive(net).get_final_node()
    self.assertAllClose(result.get_tensor(), 2.0)

  def test_passed_edge_order(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.eye(2))
    b = net.add_node(np.eye(2))
    c = net.add_node(np.eye(2))
    e1 = net.connect(a[0], b[1])
    e2 = net.connect(b[0], c[1])
    e3 = net.connect(c[0], a[1])
    result = naive(net, [e3, e1, e2]).get_final_node()
    self.assertAllClose(result.get_tensor(), 2.0)

  def test_bad_passed_edges(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.eye(2))
    b = net.add_node(np.eye(2))
    c = net.add_node(np.eye(2))
    e1 = net.connect(a[0], b[1])
    e2 = net.connect(b[0], c[1])
    _ = net.connect(c[0], a[1])
    with self.assertRaises(ValueError):
      naive(net, [e1, e2])

  def test_precontracted_network(self):
    net = network.TensorNetwork(backend="tensorflow")
    a = net.add_node(np.eye(2))
    b = net.add_node(np.eye(2))
    c = net.add_node(np.eye(2))
    net.connect(a[0], b[1])
    net.connect(b[0], c[1])
    edge = net.connect(c[0], a[1])
    net.contract(edge)
    with self.assertRaises(ValueError):
      naive(net)


if __name__ == '__main__':
  tf.test.main()
