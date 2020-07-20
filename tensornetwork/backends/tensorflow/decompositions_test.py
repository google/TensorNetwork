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

import math
import numpy as np
import tensorflow as tf
from tensornetwork.backends.tensorflow import decompositions


class DecompositionsTest(tf.test.TestCase):

  def test_expected_shapes(self):
    val = tf.zeros((2, 3, 4, 5))
    u, s, vh, _ = decompositions.svd(tf, val, 2)
    self.assertEqual(u.shape, (2, 3, 6))
    self.assertEqual(s.shape, (6,))
    self.assertAllClose(s, np.zeros(6))
    self.assertEqual(vh.shape, (6, 4, 5))

  def test_expected_shapes_qr(self):
    val = tf.zeros((2, 3, 4, 5))
    for non_negative_diagonal in [True, False]:
      q, r = decompositions.qr(tf, val, 2, non_negative_diagonal)
      self.assertEqual(q.shape, (2, 3, 6))
      self.assertEqual(r.shape, (6, 4, 5))

  def test_expected_shapes_rq(self):
    val = tf.zeros((2, 3, 4, 5))
    for non_negative_diagonal in [True, False]:
      r, q = decompositions.rq(tf, val, 2, non_negative_diagonal)
      self.assertEqual(r.shape, (2, 3, 6))
      self.assertEqual(q.shape, (6, 4, 5))

  def test_rq(self):
    random_matrix = np.random.rand(10, 10)
    for non_negative_diagonal in [True, False]:
      r, q = decompositions.rq(tf, random_matrix, 1, non_negative_diagonal)
      self.assertAllClose(tf.tensordot(r, q, ([1], [0])), random_matrix)

  def test_qr(self):
    random_matrix = np.random.rand(10, 10)
    for non_negative_diagonal in [True, False]:
      q, r = decompositions.qr(tf, random_matrix, 1, non_negative_diagonal)
      self.assertAllClose(tf.tensordot(q, r, ([1], [0])), random_matrix)

  def test_rq_defun(self):
    random_matrix = np.random.rand(10, 10)
    for non_negative_diagonal in [True, False]:
      rq = tf.function(decompositions.rq)
      r, q = rq(tf, random_matrix, 1, non_negative_diagonal)
      self.assertAllClose(tf.tensordot(r, q, ([1], [0])), random_matrix)

  def test_qr_defun(self):
    random_matrix = np.random.rand(10, 10)
    for non_negative_diagonal in [True, False]:
      qr = tf.function(decompositions.qr)
      q, r = qr(tf, random_matrix, 1, non_negative_diagonal)
      self.assertAllClose(tf.tensordot(q, r, ([1], [0])), random_matrix)

  def test_max_singular_values(self):
    random_matrix = np.random.rand(10, 10)
    unitary1, _, unitary2 = np.linalg.svd(random_matrix)
    singular_values = np.array(range(10))
    val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
    u, s, vh, trun = decompositions.svd(
        tf, val, 1, max_singular_values=7)
    self.assertEqual(u.shape, (10, 7))
    self.assertEqual(s.shape, (7,))
    self.assertAllClose(s, np.arange(9, 2, -1))
    self.assertEqual(vh.shape, (7, 10))
    self.assertAllClose(trun, np.arange(2, -1, -1))

  def test_max_singular_values_defun(self):
    random_matrix = np.random.rand(10, 10)
    unitary1, _, unitary2 = np.linalg.svd(random_matrix)
    singular_values = np.array(range(10))
    val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
    svd_decomposition = tf.function(decompositions.svd)
    u, s, vh, trun = svd_decomposition(tf, val, 1, max_singular_values=7)
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
    u, s, vh, trun = decompositions.svd(
        tf, val, 1, max_truncation_error=math.sqrt(5.1))
    self.assertEqual(u.shape, (10, 7))
    self.assertEqual(s.shape, (7,))
    self.assertAllClose(s, np.arange(9, 2, -1))
    self.assertEqual(vh.shape, (7, 10))
    self.assertAllClose(trun, np.arange(2, -1, -1))

  def test_max_truncation_error_relative(self):
    absolute = np.diag([2.0, 1.0, 0.2, 0.1])
    relative = np.diag([2.0, 1.0, 0.2, 0.1])
    max_truncation_err = 0.2
    _, _, _, trunc_sv_absolute = decompositions.svd(
        tf,
        absolute,
        1,
        max_truncation_error=max_truncation_err,
        relative=False)
    _, _, _, trunc_sv_relative = decompositions.svd(
        tf, relative, 1, max_truncation_error=max_truncation_err, relative=True)
    np.testing.assert_almost_equal(trunc_sv_absolute, [0.1])
    np.testing.assert_almost_equal(trunc_sv_relative, [0.2, 0.1])


if __name__ == '__main__':
  tf.test.main()
