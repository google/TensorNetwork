# pytype: skip-file
"""Tests for graphmode_tensornetwork."""
import tensorflow as tf
import numpy as np
import jax
import pytest
from tensornetwork.backends.jax import jax_backend

np_randn_dtypes = [np.float32, np.float16, np.float64]
np_dtypes = np_randn_dtypes + [np.complex64, np.complex128]


@pytest.mark.parametrize("dtype", np_dtypes)
def test_dtype(dtype):
  backend = jax_backend.JaxBackend(dtype)
  assert backend.dtype == np.dtype(dtype)
  assert isinstance(backend.dtype, np.dtype)

  backend = jax_backend.JaxBackend(np.dtype(dtype))
  assert backend.dtype == np.dtype(dtype)
  assert isinstance(backend.dtype, np.dtype)


def test_tensordot():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 3, 4)))
  b = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.tensordot(a, b, ((1, 2), (1, 2)))
  expected = np.array([[24.0, 24.0], [24.0, 24.0]])
  np.testing.assert_allclose(expected, actual)


def test_reshape():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.shape_tuple(backend.reshape(a, np.array((6, 4, 1))))
  assert actual == (6, 4, 1)


def test_transpose():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(
      np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a, [2, 0, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)


def test_concat():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_shape():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  assert isinstance(backend.shape(a), tuple)
  actual = backend.shape(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)


def test_prod():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones([1, 2, 3, 4]))
  actual = np.array(backend.prod(a))
  assert actual == 2**24


def test_sqrt():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.sqrt(a)
  expected = np.array([2, 3])
  np.testing.assert_allclose(expected, actual)


def test_diag():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  with pytest.raises(TypeError):
    assert backend.diag(a)
  b = backend.convert_to_tensor(np.array([1.0, 2, 3]))
  actual = backend.diag(b)
  expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
  np.testing.assert_allclose(expected, actual)


def test_convert_to_tensor():
  backend = jax_backend.JaxBackend()
  array = np.ones((2, 3, 4))
  actual = backend.convert_to_tensor(array)
  expected = jax.jit(lambda x: x)(array)
  assert isinstance(actual, type(expected))
  np.testing.assert_allclose(expected, actual)


def test_trace():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  actual = backend.trace(a)
  np.testing.assert_allclose(actual, 6)


def test_outer_product():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.outer_product(a, b)
  expected = np.array([[[[[2.0, 2.0], [2.0, 2.0]]]], [[[[2.0, 2.0], [2.0,
                                                                     2.0]]]]])
  np.testing.assert_allclose(expected, actual)


def test_einsum():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.einsum('ij,jil->l', a, b)
  expected = np.array([4.0, 4.0])
  np.testing.assert_allclose(expected, actual)


def test_convert_bad_test():
  backend = jax_backend.JaxBackend()
  with pytest.raises(TypeError):
    backend.convert_to_tensor(tf.ones((2, 2)))


def test_norm():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.ones((2, 2)))
  assert backend.norm(a) == 2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.eye(N=4, M=5)
  np.testing.assert_allclose(np.eye(N=4, M=5, dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.ones((4, 4))
  np.testing.assert_allclose(np.ones((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.zeros((4, 4))
  np.testing.assert_allclose(np.zeros((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.randn((4, 4))
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_randn_non_zero_imag(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.randn((4, 4))
  assert np.linalg.norm(np.imag(a)) != 0.0


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye_dtype(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  dtype_2 = np.float32
  a = backend.eye(N=4, M=4, dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones_dtype(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  dtype_2 = np.float32
  a = backend.ones((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros_dtype(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  dtype_2 = np.float32
  a = backend.zeros((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_dtype(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  dtype_2 = np.float32
  a = backend.randn((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye_dtype_2(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.eye(N=4, M=4)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones_dtype_2(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.ones((4, 4))
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros_dtype_2(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.zeros((4, 4))
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_dtype_2(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.randn((4, 4))
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_seed(dtype):
  backend = jax_backend.JaxBackend(dtype=dtype)
  a = backend.randn((4, 4), seed=10)
  b = backend.randn((4, 4), seed=10)
  np.testing.assert_allclose(a, b)


def test_conj():
  backend = jax_backend.JaxBackend(np.complex128)
  real = np.random.rand(2, 2, 2)
  imag = np.random.rand(2, 2, 2)
  a = backend.convert_to_tensor(real + 1j * imag)
  actual = backend.conj(a)
  expected = real - 1j * imag
  np.testing.assert_allclose(expected, actual)


def test_backend_dtype_exception():
  backend = jax_backend.JaxBackend(dtype=np.float32)
  tensor = np.random.rand(2, 2, 2)
  with pytest.raises(TypeError):
    _ = backend.convert_to_tensor(tensor)

def test_scalar_multiply():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.scalar_multiply(a, 2)
  expected = np.array([8, 18])
  np.testing.assert_allclose(expected, actual)

def test_scalar_divide():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.scalar_divide(a, 2)
  expected = np.array([2, 4.5])
  np.testing.assert_allclose(expected, actual)