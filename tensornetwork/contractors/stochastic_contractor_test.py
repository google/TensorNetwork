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
tf.enable_v2_behavior()

from tensornetwork import tensornetwork
from tensornetwork.contractors import stochastic_contractor


class SimpleStochasticTest(tf.test.TestCase):
  # TODO: More contraction cases

  def test_edge_cost(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones([4, 5, 2]))
    b = net.add_node(np.ones([3, 2, 3]))
    e = net.connect(a[2], b[1])
    self.assertEqual(140, stochastic_contractor.edge_cost(e))

  def test_contraction_sanity(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones([4, 5, 2]))
    b = net.add_node(np.ones([3, 2, 3]))
    net.connect(a[2], b[1])
    net.connect(b[0], b[2])
    net = stochastic_contractor.stochastic(net, 2)
    res = net.get_final_node()
    self.assertAllClose(res.get_tensor(), 6 * np.ones([4, 5]))

  def test_contraction_disconnected(self):
    net = tensornetwork.TensorNetwork()
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
    node1, node2 = edge1.node1, edge2.node1
    self.assertAllClose(node1.get_tensor(), 6 * np.ones([4, 5]))
    self.assertAllClose(node2.get_tensor(), 4 * np.ones([3, 3]))

 
if __name__ == '__main__':
  tf.test.main()
