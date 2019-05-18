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
import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork
from examples.sat import sat_tensornetwork


class SATTensorNetworkTest(tf.test.TestCase):

  def test_sanity_check(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
    ])
    count = tensornetwork.contractors.naive(net).get_final_node().get_tensor()
    self.assertEqual(count.numpy(), 7)

  def test_dual_clauses(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
        (1, -2, 3),
    ])
    count = tensornetwork.contractors.naive(net).get_final_node().get_tensor()
    self.assertEqual(count.numpy(), 6)

  def test_dual_clauses(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
        (1, 2, -3),
        (1, -2, 3),
        (1, -2, -3),
        (-1, 2, 3),
        (-1, 2, -3),
        (-1, -2, 3),
        (-1, -2, -3),
    ])
    count = tensornetwork.contractors.naive(net).get_final_node().get_tensor()
    self.assertEqual(count.numpy(), 0)

  def test_four_variables(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
        (1, 2, 4),
    ])
    count = tensornetwork.contractors.naive(net).get_final_node().get_tensor()
    self.assertEqual(count.numpy(), 13)

  def test_four_variables_four_clauses(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
        (1, 2, 4),
        (-3, -4, 2),
        (-1, 3, -2),
    ])
    count = tensornetwork.contractors.naive(net).get_final_node().get_tensor()
    self.assertEqual(count.numpy(), 9)

  def test_single_variable(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 1, 1),
    ])
    count = tensornetwork.contractors.naive(net).get_final_node().get_tensor()
    self.assertEqual(count.numpy(), 1)

  def test_solutions(self):
    net, edge_order = sat_tensornetwork.sat_tn([
        (1, 2, -3),
    ])
    network = tensornetwork.contractors.naive(net)
    solutions_node = network.get_final_node().reorder_edges(edge_order)
    solutions = solutions_node.get_tensor()
    self.assertEqual(solutions.numpy()[0][0][0], 1)
    # Only unaccepted value.
    self.assertEqual(solutions.numpy()[0][0][1], 0)
    self.assertEqual(solutions.numpy()[0][1][0], 1)
    self.assertEqual(solutions.numpy()[0][1][1], 1)
    self.assertEqual(solutions.numpy()[1][0][0], 1)
    self.assertEqual(solutions.numpy()[1][0][1], 1)
    self.assertEqual(solutions.numpy()[1][1][0], 1)
    self.assertEqual(solutions.numpy()[1][1][1], 1)


if __name__ == '__main__':
  tf.test.main()
