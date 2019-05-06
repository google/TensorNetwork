"""Tests for google3.googlex.rolando.tensornetwork.decompositions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import tensorflow as tf
from tensornetwork import decompositions


class DecompositionsTest(tf.test.TestCase):

  def test_expected_shapes(self):
    val = tf.zeros((2, 3, 4, 5))
    u, s, vh, _ = decompositions.svd_decomposition(val, 2)
    self.assertEqual(u.shape, (2, 3, 6))
    self.assertEqual(s.shape, (6,))
    self.assertAllClose(s, np.zeros(6))
    self.assertEqual(vh.shape, (6, 4, 5))

  def test_max_singular_values(self):
    random_matrix = np.random.rand(10, 10)
    unitary1, _, unitary2 = np.linalg.svd(random_matrix)
    singular_values = np.array(range(10))
    val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
    u, s, vh, trun = decompositions.svd_decomposition(
        val, 1, max_singular_values=7)
    self.assertEqual(u.shape, (10, 7))
    self.assertEqual(s.shape, (7,))
    self.assertAllClose(s, np.arange(9, 2, -1))
    self.assertEqual(vh.shape, (7, 10))
    self.assertAllClose(trun, np.arange(2, -1, -1))

  def test_max_truncation_error(self):
    random_matrix = np.random.rand(10, 10)
    unitary1, _, unitary2 = np.linalg.svd(random_matrix)
    singular_values = np.array(range(10))
    val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
    u, s, vh, trun = decompositions.svd_decomposition(
        val, 1, max_truncation_error=math.sqrt(5.1))
    self.assertEqual(u.shape, (10, 7))
    self.assertEqual(s.shape, (7,))
    self.assertAllClose(s, np.arange(9, 2, -1))
    self.assertEqual(vh.shape, (7, 10))
    self.assertAllClose(trun, np.arange(2, -1, -1))

if __name__ == '__main__':
  tf.test.main()

