"""Tests for graphmode_tensornetwork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest
from tensornetwork.backends.numpy import numpy_backend


def test_tensordot():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2*np.ones((2, 3, 4)))
  b = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.tensordot(a, b,  ((1, 2), (1, 2)))
  expected = np.array([[24.0, 24.0], [24.0, 24.0]])
  np.testing.assert_allclose(expected, actual)

def test_reshape():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.shape_tuple(backend.reshape(a, np.array((6, 4, 1))))
  assert actual == (6, 4, 1)

def test_transpose():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.array([[[1., 2.], [3., 4.]],
                                          [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a, [2, 0, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)

def test_concat():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2*np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)

def test_shape():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  assert isinstance(backend.shape(a), tuple)
  actual = backend.shape(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)

def test_prod():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2*np.ones([1, 2, 3, 4]))
  actual = np.array(backend.prod(a))
  assert actual == 2**24

def test_sqrt():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.sqrt(a)
  expected = np.array([2, 3])
  np.testing.assert_allclose(expected, actual)

def test_diag():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
  with pytest.raises(TypeError):
    assert backend.diag(a)
  b = backend.convert_to_tensor(np.array([1, 2, 3]))
  actual = backend.diag(b)
  expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
  np.testing.assert_allclose(expected, actual)

def test_convert_to_tensor():
  backend = numpy_backend.NumPyBackend()
  array = np.ones((2, 3, 4))
  actual = backend.convert_to_tensor(array)
  expected = np.ones((2, 3, 4))
  assert isinstance(actual, type(expected))
  np.testing.assert_allclose(expected, actual)

def test_trace():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
  actual = backend.trace(a)
  np.testing.assert_allclose(actual, 6)

def test_outer_product():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.outer_product(a, b)
  expected = np.array([[[[[2.0, 2.0], [2.0, 2.0]]]],
                       [[[[2.0, 2.0], [2.0, 2.0]]]]])
  np.testing.assert_allclose(expected, actual)

def test_einsum():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.einsum('ij,jil->l', a, b)
  expected = np.array([4.0, 4.0])
  np.testing.assert_allclose(expected, actual)

def test_convert_bad_test():
  backend = numpy_backend.NumPyBackend()
  with pytest.raises(ValueError):
    backend.convert_to_tensor(tf.ones((2, 2)))