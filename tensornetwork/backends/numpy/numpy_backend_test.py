"""Tests for graphmode_tensornetwork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest
from tensornetwork.backends.numpy import numpy_backend
import tensornetwork.config as config_file


np_randn_dtypes = [np.float32, np.float16, np.float64]
np_dtypes = np_randn_dtypes + [np.complex64, np.complex128]

def test_tensordot():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 3, 4)))
  b = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.tensordot(a, b, ((1, 2), (1, 2)))
  expected = np.array([[24.0, 24.0], [24.0, 24.0]])
  np.testing.assert_allclose(expected, actual)


def test_reshape():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.shape_tuple(backend.reshape(a, np.array((6, 4, 1))))
  assert actual == (6, 4, 1)


def test_transpose():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(
      np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a, [2, 0, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)


def test_concat():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
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
  a = backend.convert_to_tensor(2 * np.ones([1, 2, 3, 4]))
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
  a = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  with pytest.raises(TypeError):
    assert backend.diag(a)
  b = backend.convert_to_tensor(np.array([1.0, 2, 3]))
  actual = backend.diag(b)
  expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
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
  a = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  actual = backend.trace(a)
  np.testing.assert_allclose(actual, 6)


def test_outer_product():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.outer_product(a, b)
  expected = np.array([[[[[2.0, 2.0], [2.0, 2.0]]]], [[[[2.0, 2.0], [2.0,
                                                                     2.0]]]]])
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


def test_norm():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.ones((2, 2)))
  assert backend.norm(a) == 2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  a = backend.eye(N=4, M=5)
  np.testing.assert_allclose(np.eye(N=4, M=5, dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  a = backend.ones((4, 4))
  np.testing.assert_allclose(np.ones((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  a = backend.zeros((4, 4))
  np.testing.assert_allclose(np.zeros((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  a = backend.randn((4, 4))
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye_dtype(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  dtype_2 = np.float32
  a = backend.eye(N=4, M=4, dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones_dtype(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  dtype_2 = np.float32
  a = backend.ones((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros_dtype(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  dtype_2 = np.float32
  a = backend.zeros((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_dtype(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  dtype_2 = np.float32
  a = backend.randn((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye_dtype_2(dtype):
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  a = backend.eye(N=4, M=4)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones_dtype_2(dtype):
  dtype = np.float32
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  a = backend.ones((4, 4))
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros_dtype_2(dtype):
  dtype = np.float32
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  a = backend.zeros((4, 4))
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_dtype_2(dtype):
  dtype = np.float32
  backend = numpy_backend.NumPyBackend(dtype=dtype)
  a = backend.randn((4, 4))
  assert a.dtype == dtype


def test_conj():
  backend = numpy_backend.NumPyBackend(np.complex128)
  real = np.random.rand(2, 2, 2)
  imag = np.random.rand(2, 2, 2)
  a = backend.convert_to_tensor(real + 1j * imag)
  actual = backend.conj(a)
  expected = real - 1j * imag
  np.testing.assert_allclose(expected, actual)


def test_backend_dtype_exception():
  backend = numpy_backend.NumPyBackend(dtype=np.float32)
  tensor = np.random.rand(2, 2, 2)
  with pytest.raises(TypeError):
    _ = backend.convert_to_tensor(tensor)
