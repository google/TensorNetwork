# pytype: skip-file
"""Tests for graphmode_tensornetwork."""
import tensorflow as tf
import numpy as np
import jax
import pytest
from tensornetwork.backends.jax import jax_backend

np_randn_dtypes = [np.float32, np.float16, np.float64]
np_dtypes = np_randn_dtypes + [np.complex64, np.complex128]


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


def test_shape_concat():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.shape_concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_shape_tensor():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  assert isinstance(backend.shape_tensor(a), tuple)
  actual = backend.shape_tensor(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)


def test_shape_prod():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones([1, 2, 3, 4]))
  actual = np.array(backend.shape_prod(a))
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


@pytest.mark.skip(reason="TODO(chaseriley): Add type checking.")
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
  backend = jax_backend.JaxBackend()
  a = backend.eye(N=4, M=5, dtype=dtype)
  np.testing.assert_allclose(np.eye(N=4, M=5, dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.ones((4, 4), dtype=dtype)
  np.testing.assert_allclose(np.ones((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.zeros((4, 4), dtype=dtype)
  np.testing.assert_allclose(np.zeros((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_random_uniform(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.random_uniform((4, 4), dtype=dtype)
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_randn_non_zero_imag(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert np.linalg.norm(np.imag(a)) != 0.0


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_random_uniform_non_zero_imag(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.random_uniform((4, 4), dtype=dtype)
  assert np.linalg.norm(np.imag(a)) != 0.0


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye_dtype(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.eye(N=4, M=4, dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones_dtype(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.ones((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros_dtype(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.zeros((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_dtype(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_random_uniform_dtype(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.random_uniform((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_seed(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.randn((4, 4), seed=10, dtype=dtype)
  b = backend.randn((4, 4), seed=10, dtype=dtype)
  np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_random_uniform_seed(dtype):
  backend = jax_backend.JaxBackend()
  a = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  b = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_random_uniform_boundaries(dtype):
  lb = 1.2
  ub = 4.8
  backend = jax_backend.JaxBackend()
  a = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  b = backend.random_uniform((4, 4), (lb, ub), seed=10, dtype=dtype)
  assert((a >= 0).all() and (a <= 1).all() and
         (b >= lb).all() and (b <= ub).all())


def test_random_uniform_behavior():
  seed = 10
  key = jax.random.PRNGKey(seed)
  backend = jax_backend.JaxBackend()
  a = backend.random_uniform((4, 4), seed=seed)
  b = jax.random.uniform(key, (4, 4))
  np.testing.assert_allclose(a, b)


def test_conj():
  backend = jax_backend.JaxBackend()
  real = np.random.rand(2, 2, 2)
  imag = np.random.rand(2, 2, 2)
  a = backend.convert_to_tensor(real + 1j * imag)
  actual = backend.conj(a)
  expected = real - 1j * imag
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def index_update(dtype):
  backend = jax_backend.JaxBackend()
  tensor = backend.randn((4, 2, 3), dtype=dtype, seed=10)
  out = backend.index_update(tensor, tensor > 0.1, 0.0)
  tensor = np.array(tensor)
  tensor[tensor > 0.1] = 0.0
  np.testing.assert_allclose(tensor, out)


def test_base_backend_eigs_not_implemented():
  backend = jax_backend.JaxBackend()
  tensor = backend.randn((4, 2, 3), dtype=np.float64)
  with pytest.raises(NotImplementedError):
    backend.eigs(tensor)


def test_base_backend_eigsh_lanczos_not_implemented():
  backend = jax_backend.JaxBackend()
  tensor = backend.randn((4, 2, 3), dtype=np.float64)
  with pytest.raises(NotImplementedError):
    backend.eigsh_lanczos(tensor)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_index_update(dtype):
  backend = jax_backend.JaxBackend()
  tensor = backend.randn((4, 2, 3), dtype=dtype, seed=10)
  out = backend.index_update(tensor, tensor > 0.1, 0.0)
  np_tensor = np.array(tensor)
  np_tensor[np_tensor > 0.1] = 0.0
  np.testing.assert_allclose(out, np_tensor)
