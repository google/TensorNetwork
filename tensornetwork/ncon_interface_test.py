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
tf.compat.v1.enable_v2_behavior()
from tensornetwork import ncon_interface


def test_sanity_check():
  result = ncon_interface.ncon([tf.ones(
      (2, 2)), tf.ones((2, 2))], [(-1, 1), (1, -2)])
  np.testing.assert_allclose(result, tf.ones((2, 2)) * 2)


def test_order_spec():
  a = tf.ones((2, 2))

  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)], out_order=[-1, -2])
  np.testing.assert_allclose(result, tf.ones((2, 2)) * 2)

  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)], con_order=[1])
  np.testing.assert_allclose(result, tf.ones((2, 2)) * 2)

  result = ncon_interface.ncon(
      [a, a], [(-1, 1), (1, -2)], con_order=[1], out_order=[-1, -2])
  np.testing.assert_allclose(result, tf.ones((2, 2)) * 2)


def test_order_spec_noninteger():
  a = tf.ones((2, 2))
  result = ncon_interface.ncon(
      [a, a], [('o1', 'i'), ('i', 'o2')],
      con_order=['i'],
      out_order=['o1', 'o2'])
  np.testing.assert_allclose(result, tf.ones((2, 2)) * 2)


def test_invalid_network():
  a = tf.ones((2, 2))
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1), (1, 2)])
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 2)])
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (3, 1)])
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 0.1)])
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 't')])
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(0, 1), (1, 0)])
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1,), (1, 2)])


def test_invalid_order():
  a = tf.ones((2, 2))
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)], con_order=[2, 3])
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)], out_order=[-1])
  with pytest.raises(ValueError):
    ncon_interface.ncon(
        [a, a], [('i1', 'i2'), ('i1', 'i2')], con_order=['i1'], out_order=[])
  with pytest.raises(ValueError):
    ncon_interface.ncon(
        [a, a], [('i1', 'i2'), ('i1', 'i2')],
        con_order=['i1', 'i2'],
        out_order=['i1'])
  with pytest.raises(ValueError):
    ncon_interface.ncon(
        [a, a], [('i1', 'i2'), ('i1', 'i2')],
        con_order=['i1', 'i1', 'i2'],
        out_order=[])


def test_out_of_order_contraction():
  a = tf.ones((2, 2, 2))
  with pytest.warns(UserWarning, match='Suboptimal ordering'):
    ncon_interface.ncon([a, a, a], [(-1, 1, 3), (1, 3, 2), (2, -2, -3)])


def test_output_order():
  a = np.random.randn(2, 2)
  res = ncon_interface.ncon([a], [(-2, -1)])
  np.testing.assert_allclose(res, a.transpose())


def test_outer_product():
  a = np.array([1, 2, 3])
  b = np.array([1, 2])
  res = ncon_interface.ncon([a, b], [(-1,), (-2,)])
  np.testing.assert_allclose(res, np.kron(a, b).reshape((3, 2)))
  res = ncon_interface.ncon([a, a, a, a], [(1,), (1,), (2,), (2,)])
  assert res.numpy() == 196


def test_trace():
  a = tf.ones((2, 2))
  res = ncon_interface.ncon([a], [(1, 1)])
  assert res.numpy() == 2


def test_small_matmul():
  a = np.random.randn(2, 2)
  b = np.random.randn(2, 2)
  res = ncon_interface.ncon([a, b], [(1, -1), (1, -2)])
  np.testing.assert_allclose(res, a.transpose() @ b)


def test_contraction():
  a = np.random.randn(2, 2, 2)
  res = ncon_interface.ncon([a, a, a], [(-1, 1, 2), (1, 2, 3), (3, -2, -3)])
  res_np = a.reshape((2, 4)) @ a.reshape((4, 2)) @ a.reshape((2, 4))
  res_np = res_np.reshape((2, 2, 2))
  np.testing.assert_allclose(res, res_np)
