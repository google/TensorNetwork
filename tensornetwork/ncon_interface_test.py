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
import numpy as np
import tensorflow as tf
tf.enable_v2_behavior()
from tensornetwork import ncon_interface


class NconTest(tf.test.TestCase):

  def test_sanity_check(self):
    result = ncon_interface.ncon([tf.ones(
        (2, 2)), tf.ones((2, 2))], [(-1, 0), (0, -2)])
    self.assertAllClose(result, tf.ones((2, 2)) * 2)

  def test_order_spec(self):
    a = tf.ones((2, 2))

    result = ncon_interface.ncon([a, a], [(-1, 0), (0, -2)], out_order=[-1, -2])
    self.assertAllClose(result, tf.ones((2, 2)) * 2)

    result = ncon_interface.ncon([a, a], [(-1, 0), (0, -2)], con_order=[0])
    self.assertAllClose(result, tf.ones((2, 2)) * 2)

    result = ncon_interface.ncon(
        [a, a], [(-1, 0), (0, -2)], con_order=[0], out_order=[-1, -2])
    self.assertAllClose(result, tf.ones((2, 2)) * 2)

  def test_order_spec_noninteger(self):
    a = tf.ones((2, 2))
    result = ncon_interface.ncon(
        [a, a], [('o1', 'i'), ('i', 'o2')],
        con_order=['i'],
        out_order=['o1', 'o2'])
    self.assertAllClose(result, tf.ones((2, 2)) * 2)

  def test_invalid_network(self):
    a = tf.ones((2, 2))
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a], [(0, 1), (1, 0), (0, 1)])
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a], [(0, 1), (1, 1)])
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a], [(0, 1), (2, 0)])
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a], [(0, 1), (1, 0.1)])
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a], [(0, 1), (1, 't')])
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a], [(0,), (0, 1)])

  def test_invalid_order(self):
    a = tf.ones((2, 2))
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a], [(0, 1), (1, 0)], con_order=[1, 2])
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a], [(0, 1), (1, 0)], out_order=[-1])
    with self.assertRaises(ValueError):
      ncon_interface.ncon(
          [a, a], [('i1', 'i2'), ('i1', 'i2')], con_order=['i1'], out_order=[])
    with self.assertRaises(ValueError):
      ncon_interface.ncon(
          [a, a], [('i1', 'i2'), ('i1', 'i2')],
          con_order=['i1', 'i2'],
          out_order=['i1'])
    with self.assertRaises(ValueError):
      ncon_interface.ncon(
          [a, a], [('i1', 'i2'), ('i1', 'i2')],
          con_order=['i1', 'i1', 'i2'],
          out_order=[])

  def test_out_of_order_contraction(self):
    a = tf.ones((2, 2, 2))
    with self.assertRaises(ValueError):
      ncon_interface.ncon([a, a, a], [(-1, 0, 2), (0, 2, 1), (1, -2, -3)])

  def test_output_order(self):
    a = np.random.randn(2, 2)
    res = ncon_interface.ncon([a], [(-2, -1)])
    self.assertAllClose(res, a.transpose())

  def test_outer_product(self):
    a = np.array([1, 2, 3])
    b = np.array([1, 2])
    res = ncon_interface.ncon([a, b], [(-1,), (-2,)])
    self.assertAllClose(res, np.kron(a, b).reshape((3, 2)))
    res = ncon_interface.ncon([a, a, a, a], [(0,), (0,), (1,), (1,)])
    self.assertEqual(res.numpy(), 196)

  def test_trace(self):
    a = tf.ones((2, 2))
    res = ncon_interface.ncon([a], [(0, 0)])
    self.assertEqual(res.numpy(), 2)

  def test_small_matmul(self):
    a = np.random.randn(2, 2)
    b = np.random.randn(2, 2)
    res = ncon_interface.ncon([a, b], [(0, -1), (0, -2)])
    self.assertAllClose(res, a.transpose() @ b)

  def test_contraction(self):
    a = np.random.randn(2, 2, 2)
    res = ncon_interface.ncon([a, a, a], [(-1, 0, 1), (0, 1, 2), (2, -2, -3)])
    res_np = a.reshape((2, 4)) @ a.reshape((4, 2)) @ a.reshape((2, 4))
    res_np = res_np.reshape((2, 2, 2))
    self.assertAllClose(res, res_np)


if __name__ == '__main__':
  tf.test.main()
