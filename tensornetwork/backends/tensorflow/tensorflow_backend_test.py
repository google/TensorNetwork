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


def test_tensordot_int():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(2 * np.ones((3, 3, 3)))
  b = backend.convert_to_tensor(np.ones((3, 3, 3)))
  actual = backend.tensordot(a, b, 1)
  expected = tf.tensordot(a, b, 1)
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


def test_transpose_noperm():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(
      np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a)  # [2, 1, 0]
  actual = backend.transpose(actual, perm=[0, 2, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)


def test_shape_concat():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.shape_concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_slice():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(
      np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
  actual = backend.slice(a, (1, 1), (2, 2))
  expected = np.array([[5., 6.], [8., 9.]])
  np.testing.assert_allclose(expected, actual)


def test_shape_tensor():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  assert isinstance(backend.shape_tensor(a), type(a))
  actual = backend.shape_tensor(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)


def test_shape_prod():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(2 * np.ones([1, 2, 3, 4]))
  actual = np.array(backend.shape_prod(a))
  assert actual == 2**24


def test_sqrt():
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.sqrt(a)
  expected = np.array([2, 3])
  np.testing.assert_allclose(expected, actual)


def test_convert_to_tensor():
  backend = tensorflow_backend.TensorFlowBackend()
  array = np.ones((2, 3, 4))
  actual = backend.convert_to_tensor(array)
  expected = tf.ones((2, 3, 4))
  assert isinstance(actual, type(expected))
  np.testing.assert_allclose(expected, actual)


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


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_random_uniform(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.random_uniform((4, 4), dtype=dtype, seed=10)
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", [tf.complex64, tf.complex128])
def test_randn_non_zero_imag(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert tf.math.greater(tf.linalg.norm(tf.math.imag(a)), 0.0)


@pytest.mark.parametrize("dtype", [tf.complex64, tf.complex128])
def test_random_uniform_non_zero_imag(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.random_uniform((4, 4), dtype=dtype, seed=10)
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


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_random_uniform_dtype(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.random_uniform((4, 4), dtype=dtype, seed=10)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", tf_randn_dtypes)
def test_randn_seed(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.randn((4, 4), seed=10, dtype=dtype)
  b = backend.randn((4, 4), seed=10, dtype=dtype)
  np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_random_uniform_seed(dtype):
  test = tf.test.TestCase()
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  b = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  test.assertAllCloseAccordingToType(a, b)


@pytest.mark.parametrize("dtype", tf_randn_dtypes)
def test_random_uniform_boundaries(dtype):
  test = tf.test.TestCase()
  lb = 1.2
  ub = 4.8
  backend = tensorflow_backend.TensorFlowBackend()
  a = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  b = backend.random_uniform((4, 4), (lb, ub), seed=10, dtype=dtype)
  test.assertAllInRange(a, 0, 1)
  test.assertAllInRange(b, lb, ub)


def test_conj():
  backend = tensorflow_backend.TensorFlowBackend()
  real = np.random.rand(2, 2, 2)
  imag = np.random.rand(2, 2, 2)
  a = backend.convert_to_tensor(real + 1j * imag)
  actual = backend.conj(a)
  expected = real - 1j * imag
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(1, 1, 2),
    pytest.param(2. * np.ones(()), 1. * np.ones((1, 2, 3)), 3. * np.ones(
        (1, 2, 3))),
])
def test_addition(a, b, expected):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.addition(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(1, 1, 0),
    pytest.param(np.ones((1, 2, 3)), np.ones((1, 2, 3)), np.zeros((1, 2, 3))),
])
def test_subtraction(a, b, expected):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.subtraction(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(1, 1, 1),
    pytest.param(np.ones((1, 2, 3)), np.ones((1, 2, 3)), np.ones((1, 2, 3))),
])
def test_multiply(a, b, expected):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.multiply(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(2., 2., 1.),
    pytest.param(
        np.ones(()), 2. * np.ones((1, 2, 3)), 0.5 * np.ones((1, 2, 3))),
])
def test_divide(a, b, expected):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.divide(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])
def test_eigh(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  H = backend.randn((4, 4), dtype)
  H = H + tf.math.conj(tf.transpose(H))

  eta, U = backend.eigh(H)
  eta_ac, U_ac = tf.linalg.eigh(H)
  np.testing.assert_allclose(eta, eta_ac)
  np.testing.assert_allclose(U, U_ac)


@pytest.mark.parametrize("dtype", tf_randn_dtypes)
def test_index_update(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.randn((4, 2, 3), dtype=dtype, seed=10)
  out = backend.index_update(tensor, tensor > 0.1, 0.0)
  tensor_np = tensor.numpy()
  tensor_np[tensor_np > 0.1] = 0.0
  np.testing.assert_allclose(out, tensor_np)


@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])
def test_matrix_inv(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  matrix = backend.randn((4, 4), dtype=dtype, seed=10)
  inverse = backend.inv(matrix)
  m1 = tf.matmul(matrix, inverse)
  m2 = tf.matmul(inverse, matrix)

  np.testing.assert_almost_equal(m1, np.eye(4))
  np.testing.assert_almost_equal(m2, np.eye(4))


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_matrix_inv_raises(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  matrix = backend.randn((4, 4, 4), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.inv(matrix)


def test_eigs_not_implemented():
  backend = tensorflow_backend.TensorFlowBackend()
  with pytest.raises(NotImplementedError):
    backend.eigs(np.ones((2, 2)))


def test_gmres_not_implemented():
  backend = tensorflow_backend.TensorFlowBackend()
  with pytest.raises(NotImplementedError):
    backend.gmres(lambda x: x, np.ones((2)))


def test_eigsh_lanczos_not_implemented():
  backend = tensorflow_backend.TensorFlowBackend()
  with pytest.raises(NotImplementedError):
    backend.eigsh_lanczos(lambda x: x, [])


@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])
def test_broadcast_right_multiplication(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((3,), dtype=dtype, seed=10)
  out = backend.broadcast_right_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out, tensor1 * tensor2)


def test_broadcast_right_multiplication_raises():
  dtype = tf.float64
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((3, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.broadcast_right_multiplication(tensor1, tensor2)


@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])
def test_broadcast_left_multiplication(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.randn((3,), dtype=dtype, seed=10)
  tensor2 = backend.randn((3, 4, 2), dtype=dtype, seed=10)
  out = backend.broadcast_left_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out, np.reshape(tensor1, (3, 1, 1)) * tensor2)


def test_broadcast_left_multiplication_raises():
  dtype = tf.float64
  backend = tensorflow_backend.TensorFlowBackend()
  tensor1 = backend.randn((3, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.broadcast_left_multiplication(tensor1, tensor2)


def test_sparse_shape():
  dtype = tf.float64
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.randn((2, 3, 4), dtype=dtype, seed=10)
  np.testing.assert_allclose(backend.sparse_shape(tensor), tensor.shape)


@pytest.mark.parametrize("dtype,method", [(tf.float64, "sin"),
                                          (tf.complex128, "sin"),
                                          (tf.float64, "cos"),
                                          (tf.complex128, "cos"),
                                          (tf.float64, "exp"),
                                          (tf.complex128, "exp"),
                                          (tf.float64, "log"),
                                          (tf.complex128, "log")])
def test_elementwise_ops(dtype, method):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.randn((4, 2, 1), dtype=dtype, seed=10)
  if method == "log":
    tensor = tf.math.abs(tensor)
  tensor1 = getattr(backend, method)(tensor)
  tensor2 = getattr(tf.math, method)(tensor)
  print(tensor1, tensor2)
  np.testing.assert_almost_equal(tensor1.numpy(), tensor2.numpy())


@pytest.mark.parametrize("dtype,method", [(tf.float64, "expm"),
                                          (tf.complex128, "expm")])
def test_matrix_ops(dtype, method):
  backend = tensorflow_backend.TensorFlowBackend()
  matrix = backend.randn((4, 4), dtype=dtype, seed=10)
  matrix1 = getattr(backend, method)(matrix)
  matrix2 = getattr(tf.linalg, method)(matrix)
  np.testing.assert_almost_equal(matrix1.numpy(), matrix2.numpy())


@pytest.mark.parametrize("dtype,method", [(tf.float64, "expm"),
                                          (tf.complex128, "expm")])
def test_matrix_ops_raises(dtype, method):
  backend = tensorflow_backend.TensorFlowBackend()
  matrix = backend.randn((4, 4, 4), dtype=dtype, seed=10)
  with pytest.raises(ValueError, match=r".*Only matrices.*"):
    getattr(backend, method)(matrix)
  matrix = backend.randn((4, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError, match=r".*N\*N matrix.*"):
    getattr(backend, method)(matrix)


def test_jit():
  backend = tensorflow_backend.TensorFlowBackend()

  def fun(x, A, y):
    return tf.tensordot(x, tf.tensordot(A, y, ([1], [0])), ([0], [0]))

  fun_jit = backend.jit(fun)
  x = tf.convert_to_tensor(np.random.rand(4))
  y = tf.convert_to_tensor(np.random.rand(4))
  A = tf.convert_to_tensor(np.random.rand(4, 4))

  res1 = fun(x, A, y)
  res2 = fun_jit(x, A, y)
  np.testing.assert_allclose(res1, res2)


def test_jit_args():
  backend = tensorflow_backend.TensorFlowBackend()

  def fun(x, A, y):
    return tf.tensordot(x, tf.tensordot(A, y, ([1], [0])), ([0], [0]))

  fun_jit = backend.jit(fun)
  x = tf.convert_to_tensor(np.random.rand(4))
  y = tf.convert_to_tensor(np.random.rand(4))
  A = tf.convert_to_tensor(np.random.rand(4, 4))

  res1 = fun(x, A, y)
  res2 = fun_jit(x, A, y)
  res3 = fun_jit(x, y=y, A=A)
  np.testing.assert_allclose(res1, res2)
  np.testing.assert_allclose(res1, res3)


def test_sum():
  np.random.seed(10)
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = np.random.rand(2, 3, 4)
  a = backend.convert_to_tensor(tensor)
  actual = backend.sum(a, axis=(1, 2))
  expected = np.sum(tensor, axis=(1, 2))
  np.testing.assert_allclose(expected, actual)

  actual = backend.sum(a, axis=(1, 2), keepdims=True)
  expected = np.sum(a, axis=(1, 2), keepdims=True)
  np.testing.assert_allclose(expected, actual)


def test_matmul():
  np.random.seed(10)
  backend = tensorflow_backend.TensorFlowBackend()
  t1 = np.random.rand(10, 2, 3)
  t2 = np.random.rand(10, 3, 4)
  a = backend.convert_to_tensor(t1)
  b = backend.convert_to_tensor(t2)
  actual = backend.matmul(a, b)
  expected = np.matmul(t1, t2)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", tf_dtypes)
@pytest.mark.parametrize("offset", range(-2, 2))
@pytest.mark.parametrize("axis1", [-2, 0])
@pytest.mark.parametrize("axis2", [-1, 0])
def test_diagonal(dtype, offset, axis1, axis2):
  shape = (5, 5, 5, 5)
  backend = tensorflow_backend.TensorFlowBackend()
  array = backend.randn(shape, dtype=dtype, seed=10)
  if axis1 != -2 or axis2 != -1:
    with pytest.raises(NotImplementedError):
      actual = backend.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)
  else:
    actual = backend.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)
    expected = np.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", tf_dtypes)
@pytest.mark.parametrize("k", range(-2, 2))
def test_diagflat(dtype, k):
  backend = tensorflow_backend.TensorFlowBackend()
  array = backend.randn((16,), dtype=dtype, seed=10)
  actual = backend.diagflat(array, k=k)
  # pylint: disable=unexpected-keyword-arg
  expected = tf.linalg.diag(array, k=k)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_abs(dtype):
  shape = (4, 3, 2)
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  actual = backend.abs(tensor)
  expected = tf.math.abs(tensor)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_sign(dtype):
  shape = (4, 3, 2)
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  actual = backend.sign(tensor)
  expected = tf.math.sign(tensor)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", tf_dtypes)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("axis1", [-2, 0])
@pytest.mark.parametrize("axis2", [-1, 0])
def test_trace(dtype, offset, axis1, axis2):
  shape = (5, 5, 5, 5)
  backend = tensorflow_backend.TensorFlowBackend()
  tf_array = backend.randn(shape, dtype=dtype, seed=10)
  array = tf_array.numpy()
  if offset != 0:
    with pytest.raises(NotImplementedError):
      actual = backend.trace(tf_array, offset=offset, axis1=axis1, axis2=axis2)
  elif axis1 == axis2:
    with pytest.raises(ValueError):
      actual = backend.trace(tf_array, offset=offset, axis1=axis1, axis2=axis2)
  else:
    actual = backend.trace(tf_array, offset=offset, axis1=axis1, axis2=axis2)
    expected = np.trace(array, axis1=axis1, axis2=axis2)
    tol = array.size * np.finfo(array.dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("pivot_axis", [-1, 1, 2])
@pytest.mark.parametrize("dtype", tf_dtypes)
def test_pivot(dtype, pivot_axis):
  shape = (4, 3, 2, 8)
  pivot_shape = (np.prod(shape[:pivot_axis]), np.prod(shape[pivot_axis:]))
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  expected = tf.reshape(tensor, pivot_shape)
  actual = backend.pivot(tensor, pivot_axis=pivot_axis)
  np.testing.assert_allclose(expected, actual)

@pytest.mark.parametrize("dtype", tf_dtypes)
def test_item(dtype):
  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.ones(1, dtype=dtype) * 5.0
  assert backend.item(tensor) == 5.0

  backend = tensorflow_backend.TensorFlowBackend()
  tensor = backend.ones((2, 1), dtype=dtype)
  with pytest.raises(ValueError, match="expected"):
    backend.item(tensor)

@pytest.mark.parametrize("dtype", tf_dtypes)
def test_power(dtype):
  shape = (4, 3, 2)
  backend = tensorflow_backend.TensorFlowBackend()
  base_tensor = backend.randn(shape, dtype=dtype, seed=10)
  power_tensor = backend.randn(shape, dtype=dtype, seed=10)
  actual = backend.power(base_tensor, power_tensor)
  expected = tf.math.pow(base_tensor, power_tensor)
  np.testing.assert_allclose(expected, actual)
  power = np.random.rand(1)[0]
  actual = backend.power(base_tensor, power)
  expected = tf.math.pow(base_tensor, power)
  np.testing.assert_allclose(expected, actual)
