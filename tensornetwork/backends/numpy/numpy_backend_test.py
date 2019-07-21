"""Tests for graphmode_tensornetwork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensornetwork
import pytest


def test_tensordot():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(2*np.ones((2, 3, 4)))
  b = net.add_node(np.ones((2, 3, 4)))
  actual = net.backend.tensordot(a.tensor, b.tensor,  ((1, 2), (1, 2)))
  expected = np.array([[24.0, 24.0], [24.0, 24.0]])
  np.testing.assert_allclose(expected, actual)

def test_reshape():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(np.ones((2, 3, 4)))
  actual = net.backend.shape_tuple(net.backend.reshape(a.tensor,
                                                       np.array((6, 4, 1))))
  expected = (6, 4, 1)
  assert actual == expected

def test_transpose():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
  actual = net.backend.transpose(a.tensor, [2, 0, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)

def test_concat():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(2*np.ones((1, 3, 1)))
  b = net.add_node(np.ones((1, 2, 1)))
  expected = net.backend.concat((a.tensor, b.tensor), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)

def test_shape():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(np.ones([2, 3, 4]))
  assert isinstance(net.backend.shape(a.tensor), tuple)
  actual = net.backend.shape(a.tensor)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(np.ones([2, 3, 4]))
  actual = net.backend.shape_tuple(a.tensor)
  expected = (2, 3, 4)
  assert actual == expected

def test_prod():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(2*np.ones([1, 2, 3, 4]))
  actual = np.array(net.backend.prod(a.tensor))
  expected = 2**24
  assert actual == expected

def test_sqrt():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(np.array([4., 9.]))
  actual = net.backend.sqrt(a.tensor)
  expected = np.array([2, 3])
  np.testing.assert_allclose(expected, actual)

def test_diag():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(np.array([[1, 2, 3], [4, 5, 6]]))
  with pytest.raises(TypeError):
    assert net.backend.diag(a.tensor)
  net = tensornetwork.TensorNetwork('numpy')
  b = net.add_node(np.array([1, 2, 3]))
  actual = net.backend.diag(b.tensor)
  expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
  np.testing.assert_allclose(expected, actual)

def test_convert_to_tensor():
  net = tensornetwork.TensorNetwork('numpy')
  array = np.ones((2, 3, 4))
  a = net.add_node(array)
  actual = net.backend.convert_to_tensor(array)
  expected = np.ones((2, 3, 4))
  assert isinstance(actual, type(a.tensor))
  np.testing.assert_allclose(expected, actual)

def test_trace():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(np.array([[1, 2, 3], [4, 5, 6]]))
  actual = net.backend.trace(a.tensor)
  expected = 6
  np.testing.assert_allclose(expected, actual)

def test_outer_product():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(2 * np.ones((2, 1)))
  b = net.add_node(np.ones((1, 2, 2)))
  actual = net.backend.outer_product(a.tensor, b.tensor)
  expected = np.array([[[[[2.0, 2.0], [2.0, 2.0]]]],
                       [[[[2.0, 2.0], [2.0, 2.0]]]]])
  np.testing.assert_allclose(expected, actual)

def test_einsum():
  net = tensornetwork.TensorNetwork('numpy')
  a = net.add_node(2 * np.ones((2, 1)))
  b = net.add_node(np.ones((1, 2, 2)))
  actual = net.backend.einsum('ij,jil->l', a.tensor, b.tensor)
  expected = np.array([4.0, 4.0])
  np.testing.assert_allclose(expected, actual)
