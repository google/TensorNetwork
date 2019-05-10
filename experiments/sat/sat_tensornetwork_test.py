"""Tests for google3.googlex.rolando.tensornetwork.experiments.sat_tensornetwork."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.google as tf
tf.enable_v2_behavior()
from experiments.sat import sat_tensornetwork


class SATTensorNetworkTest(tf.test.TestCase):

  def test_sanity_check(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
    ])
    count = sat_tensornetwork.contract_badly(net).get_tensor()
    self.assertEqual(count.numpy(), 7)

  def test_dual_clauses(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
        (1, -2, 3),
    ])
    count = sat_tensornetwork.contract_badly(net).get_tensor()
    self.assertEqual(count.numpy(), 6)

  def test_four_variables(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
        (1, 2, 4),
    ])
    count = sat_tensornetwork.contract_badly(net).get_tensor()
    self.assertEqual(count.numpy(), 13)

  def test_four_variables_four_clauses(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 2, 3),
        (1, 2, 4),
        (-3, -4, 2),
        (-1, 3, -2),
    ])
    count = sat_tensornetwork.contract_badly(net).get_tensor()
    self.assertEqual(count.numpy(), 9)

  def test_single_variable(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 1, 1),
    ])
    count = sat_tensornetwork.contract_badly(net).get_tensor()
    self.assertEqual(count.numpy(), 1)

  def test_unsatisfiable(self):
    net = sat_tensornetwork.sat_count_tn([
        (1, 1, 1),
        (-1, -1, -1)
    ])
    count = sat_tensornetwork.contract_badly(net).get_tensor()
    self.assertEqual(count.numpy(), 0)

  def test_solutions(self):
    net, edge_order = sat_tensornetwork.sat_tn([
        (1, 2, -3),
    ])
    solutions = sat_tensornetwork.contract_badly(net, edge_order).get_tensor()
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

