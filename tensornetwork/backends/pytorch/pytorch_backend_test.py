"""Tests for graphmode_tensornetwork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensornetwork.backends.pytorch import pytorch_backend
import torch


def test_tensordot():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2*np.ones((2, 3, 4)))
  b = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.tensordot(a, b,  ((1, 2), (1, 2)))
  expected = np.array([[24.0, 24.0], [24.0, 24.0]])
  np.testing.assert_allclose(expected, actual)


def test_reshape():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.shape_tuple(backend.reshape(a, (6, 4, 1)))
  assert actual == (6, 4, 1)


def test_transpose():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.array([[[1., 2.], [3., 4.]],
                                          [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a, [2, 0, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)


def test_concat():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2*np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_shape():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  assert isinstance(backend.shape(a), torch.Tensor)
  actual = backend.shape(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)


def test_prod():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2*np.ones([1, 2, 3, 4]))
  actual = np.array(backend.prod(a))
  assert actual == 2**24


def test_sqrt():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.sqrt(a)
  expected = np.array([2, 3])
  np.testing.assert_allclose(expected, actual)


def test_diag():
  backend = pytorch_backend.PyTorchBackend()
  b = backend.convert_to_tensor(np.array([1, 2, 3]))
  actual = backend.diag(b)
  expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
  np.testing.assert_allclose(expected, actual)


def test_convert_to_tensor():
  backend = pytorch_backend.PyTorchBackend()
  array = np.ones((2, 3, 4))
  actual = backend.convert_to_tensor(array)
  expected = torch.ones((2, 3, 4))
  assert isinstance(actual, type(expected))
  np.testing.assert_allclose(expected, actual)


def test_trace():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.array([[[1., 2.], [3., 4.]],
                                          [[5., 6.], [7., 8.]],
                                          [[9., 10.], [11., 12.]]]))
  actual = backend.trace(a)
  expected = np.array([5., 13., 21.])
  np.testing.assert_allclose(expected, actual)
  a = backend.convert_to_tensor(np.array([[1., 2.], [3., 4.]]))
  actual = backend.trace(a)
  np.testing.assert_allclose(actual, 5)


def test_outer_product():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.outer_product(a, b)
  expected = np.array([[[[[2.0, 2.0], [2.0, 2.0]]]],
                       [[[[2.0, 2.0], [2.0, 2.0]]]]])
  np.testing.assert_allclose(expected, actual)
