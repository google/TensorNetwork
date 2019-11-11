# pytype: skip-file
"""Tests for graphmode_tensornetwork."""
import numpy as np
import tensorflow as tf
import pytest
from tensornetwork.backends.tensorflow import tensorflow_backend

tf_randn_dtypes = [tf.float32, tf.float16, tf.float64]
tf_dtypes = tf_randn_dtypes + [tf.complex128, tf.complex64]


def test_tensordot():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 3, 4)))
  b = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.tensordot(a, b, ((1, 2), (1, 2)))
  expected = np.array([[24.0, 24.0], [24.0, 24.0]])
  np.testing.assert_allclose(expected, actual)


def test_reshape():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.shape_tuple(backend.reshape(a, np.array((6, 4, 1))))
  assert actual == (6, 4, 1)


def test_transpose():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(
      np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a, [2, 0, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)


def test_concat():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_shape():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  assert isinstance(backend.shape(a), type(a))
  actual = backend.shape(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)


def test_prod():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(2 * np.ones([1, 2, 3, 4]))
  actual = np.array(backend.prod(a))
  assert actual == 2**24


def test_sqrt():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.sqrt(a)
  expected = np.array([2, 3])
  np.testing.assert_allclose(expected, actual)


def test_diag():
  backend = tensorflow_backend.TensorFlowBackend()
  b = backend.convert_to_tensor(np.array([1.0, 2.0, 3.0]))
  actual = backend.diag(b)
  expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
  np.testing.assert_allclose(expected, actual)


def test_convert_to_tensor():
  backend = tensorflow_backend.TensorFlowBackend()
  array = np.ones((2, 3, 4))
  actual = backend.convert_to_tensor(array)
  expected = tf.ones((2, 3, 4))
  assert isinstance(actual, type(expected))
  np.testing.assert_allclose(expected, actual)


def test_trace():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  actual = backend.trace(a)
  np.testing.assert_allclose(actual, 6)


def test_outer_product():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.outer_product(a, b)
  expected = np.array([[[[[2.0, 2.0], [2.0, 2.0]]]], [[[[2.0, 2.0], [2.0,
                                                                     2.0]]]]])
  np.testing.assert_allclose(expected, actual)


def test_einsum():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.einsum('ij,jil->l', a, b)
  expected = np.array([4.0, 4.0])
  np.testing.assert_allclose(expected, actual)


def test_norm():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.ones((2, 2)))
  assert backend.norm(a).numpy() == 2


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_eye(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.eye(N=4, M=5, dtype=dtype)
  np.testing.assert_allclose(tf.eye(num_rows=4, num_columns=5, dtype=dtype), a)


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_ones(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.ones((4, 4), dtype=dtype)
  np.testing.assert_allclose(tf.ones((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_zeros(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.zeros((4, 4), dtype=dtype)
  np.testing.assert_allclose(tf.zeros((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", tf_randn_dtypes)
def test_randn(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", [tf.complex64, tf.complex128])
def test_randn_non_zero_imag(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert tf.math.greater(tf.linalg.norm(tf.math.imag(a)), 0.0)


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_eye_dtype(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.eye(N=4, M=4, dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_ones_dtype(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.ones((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_zeros_dtype(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.zeros((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", tf_randn_dtypes)
def test_randn_dtype(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", tf_randn_dtypes)
def test_randn_seed(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.randn((4, 4), seed=10, dtype=dtype)
  b = backend.randn((4, 4), seed=10, dtype=dtype)
  np.testing.assert_allclose(a, b)


def test_conj():
  backend = tensorflow_backend.TensorFlowBackend()
  real = np.random.rand(2, 2, 2)
  imag = np.random.rand(2, 2, 2)
  a = backend.convert_to_tensor(real + 1j * imag)
  actual = backend.conj(a)
  expected = real - 1j * imag
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(np.ones((1, 2, 3)), np.ones((1, 2, 3)), np.ones((1, 2, 3))),
    pytest.param(2. * np.ones(()), np.ones((1, 2, 3)), 2. * np.ones((1, 2, 3))),
])
def test_multiply(a, b, expected):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)

  np.testing.assert_allclose(backend.multiply(tensor1, tensor2), expected)


@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])
def test_eigh(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  H = backend.randn((4, 4), dtype)
  H = H + tf.math.conj(tf.transpose(H))

  eta, U = backend.eigh(H)
  eta_ac, U_ac = tf.linalg.eigh(H)
  np.testing.assert_allclose(eta, eta_ac)
  np.testing.assert_allclose(U, U_ac)


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_index_update(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.randn((4, 2, 3), dtype=dtype, seed=10)
  out = backend.index_update(tensor, tensor > 0.1, 0)
  tensor_np = tensor.numpy()
  tensor_np[tensor_np > 0.1] = 0.0
  np.testing.assert_allclose(out, tensor_np)
