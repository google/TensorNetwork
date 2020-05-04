import tensorflow as tf
import numpy as np
import scipy as sp
import pytest
from tensornetwork.backends.numpy import numpy_backend
from unittest.mock import Mock

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


def test_shape_concat():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.shape_concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_slice():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(
      np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
  actual = backend.slice(a, (1, 1), (2, 2))
  expected = np.array([[5., 6.], [8., 9.]])
  np.testing.assert_allclose(expected, actual)


def test_slice_raises_error():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(
      np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
  with pytest.raises(ValueError):
    backend.slice(a, (1, 1), (2, 2, 2))


def test_shape_tensor():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  assert isinstance(backend.shape_tensor(a), tuple)
  actual = backend.shape_tensor(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)


def test_shape_prod():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(2 * np.ones([1, 2, 3, 4]))
  actual = np.array(backend.shape_prod(a))
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
  with pytest.raises(TypeError):
    backend.convert_to_tensor(tf.ones((2, 2)))


def test_norm():
  backend = numpy_backend.NumPyBackend()
  a = backend.convert_to_tensor(np.ones((2, 2)))
  assert backend.norm(a) == 2


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.eye(N=4, M=5, dtype=dtype)
  np.testing.assert_allclose(np.eye(N=4, M=5, dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.ones((4, 4), dtype=dtype)
  np.testing.assert_allclose(np.ones((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.zeros((4, 4), dtype=dtype)
  np.testing.assert_allclose(np.zeros((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.randn((4, 4), dtype=dtype, seed=10)
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_random_uniform(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.random_uniform((4, 4), dtype=dtype, seed=10)
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_randn_non_zero_imag(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.randn((4, 4), dtype=dtype, seed=10)
  assert np.linalg.norm(np.imag(a)) != 0.0


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_random_uniform_non_zero_imag(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.random_uniform((4, 4), dtype=dtype, seed=10)
  assert np.linalg.norm(np.imag(a)) != 0.0


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye_dtype(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.eye(N=4, M=4, dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones_dtype(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.ones((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros_dtype(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.zeros((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_dtype(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.randn((4, 4), dtype=dtype, seed=10)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
def test_random_uniform_dtype(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.random_uniform((4, 4), dtype=dtype, seed=10)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_seed(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.randn((4, 4), seed=10, dtype=dtype)
  b = backend.randn((4, 4), seed=10, dtype=dtype)
  np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_random_uniform_seed(dtype):
  backend = numpy_backend.NumPyBackend()
  a = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  b = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_random_uniform_boundaries(dtype):
  lb = 1.2
  ub = 4.8
  backend = numpy_backend.NumPyBackend()
  a = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  b = backend.random_uniform((4, 4), (lb, ub), seed=10, dtype=dtype)
  assert ((a >= 0).all() and (a <= 1).all() and (b >= lb).all() and
          (b <= ub).all())


def test_random_uniform_behavior():
  backend = numpy_backend.NumPyBackend()
  a = backend.random_uniform((4, 4), seed=10)
  np.random.seed(10)
  b = np.random.uniform(size=(4, 4))
  np.testing.assert_allclose(a, b)


def test_conj():
  backend = numpy_backend.NumPyBackend()
  real = np.random.rand(2, 2, 2)
  imag = np.random.rand(2, 2, 2)
  a = backend.convert_to_tensor(real + 1j * imag)
  actual = backend.conj(a)
  expected = real - 1j * imag
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_valid_init_operator_with_shape(dtype):
  backend = numpy_backend.NumPyBackend()
  D = 16
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x):
    return np.dot(H, x)

  eta1, U1 = backend.eigsh_lanczos(mv, [], init)
  eta2, U2 = np.linalg.eigh(H)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


def test_eigsh_small_number_krylov_vectors():
  backend = numpy_backend.NumPyBackend()
  init = np.array([1, 1], dtype=np.float64)
  H = np.array([[1, 2], [3, 4]], dtype=np.float64)

  def mv(x):
    return np.dot(H, x)

  eta1, _ = backend.eigsh_lanczos(mv, [], init, num_krylov_vecs=1)
  np.testing.assert_allclose(eta1[0], 5)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_lanczos_1(dtype):
  backend = numpy_backend.NumPyBackend()
  D = 16
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x):
    return np.dot(H, x)

  eta1, U1 = backend.eigsh_lanczos(mv, [], init)
  eta2, U2 = np.linalg.eigh(H)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_lanczos_2(dtype):
  backend = numpy_backend.NumPyBackend()
  D = 16
  np.random.seed(10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x):
    return np.dot(H, x)

  eta1, U1 = backend.eigsh_lanczos(mv, [], shape=(D,), dtype=dtype)
  eta2, U2 = np.linalg.eigh(H)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("numeig", [1, 2, 3, 4])
def test_eigsh_lanczos_reorthogonalize(dtype, numeig):
  backend = numpy_backend.NumPyBackend()
  D = 24
  np.random.seed(10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x):
    return np.dot(H, x)

  eta1, U1 = backend.eigsh_lanczos(
      mv, [],
      shape=(D,),
      dtype=dtype,
      numeig=numeig,
      num_krylov_vecs=D,
      reorthogonalize=True,
      ndiag=1,
      tol=10**(-12),
      delta=10**(-12))
  eta2, U2 = np.linalg.eigh(H)

  np.testing.assert_allclose(eta1[0:numeig], eta2[0:numeig])
  for n in range(numeig):
    v2 = U2[:, n]
    v2 /= np.sum(v2)  #fix phases
    v1 = np.reshape(U1[n], (D))
    v1 /= np.sum(v1)

    np.testing.assert_allclose(v1, v2, rtol=10**(-5), atol=10**(-5))


def test_eigsh_lanczos_raises():
  backend = numpy_backend.NumPyBackend()
  with pytest.raises(
      ValueError, match='`num_krylov_vecs` >= `numeig` required!'):
    backend.eigsh_lanczos(lambda x: x, [], numeig=10, num_krylov_vecs=9)
  with pytest.raises(
      ValueError,
      match="Got numeig = 2 > 1 and `reorthogonalize = False`. "
      "Use `reorthogonalize=True` for `numeig > 1`"):
    backend.eigsh_lanczos(lambda x: x, [], numeig=2, reorthogonalize=False)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x, [], shape=(10,), dtype=None)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x, [], shape=None, dtype=np.float64)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x, [])
  with pytest.raises(
      TypeError, match="Expected a `np.ndarray`. Got <class 'list'>"):
    backend.eigsh_lanczos(lambda x: x, [], initial_state=[1, 2, 3])


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(1, 1, 2),
    pytest.param(1., np.ones((1, 2, 3)), 2 * np.ones((1, 2, 3))),
    pytest.param(2. * np.ones(()), 1., 3. * np.ones((1, 2, 3))),
    pytest.param(2. * np.ones(()), 1. * np.ones((1, 2, 3)), 3. * np.ones(
        (1, 2, 3))),
])
def test_addition(a, b, expected):
  backend = numpy_backend.NumPyBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.addition(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(1, 1, 0),
    pytest.param(2., 1. * np.ones((1, 2, 3)), 1. * np.ones((1, 2, 3))),
    pytest.param(np.ones((1, 2, 3)), 1., np.zeros((1, 2, 3))),
    pytest.param(np.ones((1, 2, 3)), np.ones((1, 2, 3)), np.zeros((1, 2, 3))),
])
def test_subtraction(a, b, expected):
  backend = numpy_backend.NumPyBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.subtraction(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(1, 1, 1),
    pytest.param(2., 1. * np.ones((1, 2, 3)), 2. * np.ones((1, 2, 3))),
    pytest.param(np.ones((1, 2, 3)), 1., np.ones((1, 2, 3))),
    pytest.param(np.ones((1, 2, 3)), np.ones((1, 2, 3)), np.ones((1, 2, 3))),
])
def test_multiply(a, b, expected):
  backend = numpy_backend.NumPyBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.multiply(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(2., 2., 1.),
    pytest.param(2., 0.5 * np.ones((1, 2, 3)), 4. * np.ones((1, 2, 3))),
    pytest.param(np.ones(()), 2., 0.5 * np.ones((1, 2, 3))),
    pytest.param(
        np.ones(()), 2. * np.ones((1, 2, 3)), 0.5 * np.ones((1, 2, 3))),
])
def test_divide(a, b, expected):
  backend = numpy_backend.NumPyBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.divide(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


def find(which, vector):
  if which == 'LM':
    index = np.argmax(np.abs(vector))
    val = np.abs(vector[index])
  if which == 'SM':
    index = np.argmin(np.abs(vector))
    val = np.abs(vector[index])
  if which == 'LR':
    index = np.argmax(np.real(vector))
    val = np.real(vector[index])
  if which == 'SR':
    index = np.argmin(np.real(vector))
    val = np.real(vector[index])
  if which == 'LI':
    index = np.argmax(np.imag(vector))
    val = np.imag(vector[index])
  if which == 'SI':
    index = np.argmin(np.imag(vector))
    val = np.imag(vector[index])
  return val, index


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("which", ['LM', 'LR', 'SM', 'SR'])
def test_eigs(dtype, which):

  backend = numpy_backend.NumPyBackend()
  D = 16
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  M = backend.randn((D, D), dtype=dtype, seed=10)

  def mv(x):
    return np.dot(M, x)

  eta1, U1 = backend.eigs(mv, init, numeig=1, which=which)
  eta2, U2 = np.linalg.eig(M)
  val, index = find(which, eta2)
  v2 = U2[:, index]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(find(which, eta1)[0], val)
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("which", ['SI', 'LI'])
def test_eigs_raises_error_for_unsupported_which(which):
  backend = numpy_backend.NumPyBackend()
  A = backend.randn((4, 4), dtype=np.float64)
  with pytest.raises(ValueError):
    backend.eigs(A=A, which=which)


def test_eigs_raises_error_for_incompatible_shapes():
  backend = numpy_backend.NumPyBackend()
  A = backend.randn((4, 4), dtype=np.float64)
  init = backend.randn((3,), dtype=np.float64)
  with pytest.raises(ValueError):
    backend.eigs(A, initial_state=init)


def test_eigs_raises_error_for_unshaped_A():
  backend = numpy_backend.NumPyBackend()
  A = Mock(spec=[])
  print(hasattr(A, "shape"))
  err_msg = "`A` has no  attribute `shape`. Cannot initialize lanczos. " \
            "Please provide a valid `initial_state`"
  with pytest.raises(AttributeError, match=err_msg):
    backend.eigs(A)


def test_eigs_raises_error_for_untyped_A():
  backend = numpy_backend.NumPyBackend()
  A = Mock(spec=[])
  A.shape = Mock(return_value=(2, 2))
  err_msg = "`A` has no  attribute `dtype`. Cannot initialize lanczos. " \
            "Please provide a valid `initial_state` with a `dtype` attribute"
  with pytest.raises(AttributeError, match=err_msg):
    backend.eigs(A)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("which", ['LM', 'LR', 'SM', 'SR'])
def test_eigs_no_init(dtype, which):
  backend = numpy_backend.NumPyBackend()
  D = 16
  np.random.seed(10)
  H = backend.randn((D, D), dtype=dtype, seed=10)

  class LinearOperator:

    def __init__(self, shape, dtype):
      self.shape = shape
      self.dtype = dtype

    def __call__(self, x):
      return np.dot(H, x)

  mv = LinearOperator(shape=((D,), (D,)), dtype=dtype)
  eta1, U1 = backend.eigs(mv, numeig=1, which=which)
  eta2, U2 = np.linalg.eig(H)
  val, index = find(which, eta2)
  v2 = U2[:, index]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(find(which, eta1)[0], val)
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("which", ['LM', 'LR', 'SM', 'SR'])
def test_eigs_init(dtype, which):
  backend = numpy_backend.NumPyBackend()
  D = 16
  np.random.seed(10)
  H = backend.randn((D, D), dtype=dtype, seed=10)
  init = backend.randn((D,), dtype=dtype)

  class LinearOperator:

    def __init__(self, shape, dtype):
      self.shape = shape
      self.dtype = dtype

    def __call__(self, x):
      return np.dot(H, x)

  mv = LinearOperator(shape=((D,), (D,)), dtype=dtype)
  eta1, U1 = backend.eigs(mv, initial_state=init, numeig=1, which=which)
  eta2, U2 = np.linalg.eig(H)
  val, index = find(which, eta2)
  v2 = U2[:, index]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(find(which, eta1)[0], val)
  np.testing.assert_allclose(v1, v2)


def test_eigs_raises_error_for_bad_initial_state():
  backend = numpy_backend.NumPyBackend()
  D = 16
  init = [1] * D
  M = backend.randn((D, D), dtype=np.float64)

  def mv(x):
    return np.dot(M, x)

  with pytest.raises(TypeError):
    backend.eigs(mv, initial_state=init)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigh(dtype):
  backend = numpy_backend.NumPyBackend()
  np.random.seed(10)
  H = backend.randn((4, 4), dtype=dtype, seed=10)
  H = H + np.conj(np.transpose(H))

  eta, U = backend.eigh(H)
  eta_ac, U_ac = np.linalg.eigh(H)
  np.testing.assert_allclose(eta, eta_ac)
  np.testing.assert_allclose(U, U_ac)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_index_update(dtype):
  backend = numpy_backend.NumPyBackend()
  tensor = backend.randn((4, 2, 3), dtype=dtype, seed=10)
  out = backend.index_update(tensor, tensor > 0.1, 0)
  tensor[tensor > 0.1] = 0.0
  np.testing.assert_allclose(tensor, out)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_matrix_inv(dtype):
  backend = numpy_backend.NumPyBackend()
  matrix = backend.randn((4, 4), dtype=dtype, seed=10)
  inverse = backend.inv(matrix)
  m1 = matrix.dot(inverse)
  m2 = inverse.dot(matrix)

  np.testing.assert_almost_equal(m1, np.eye(4))
  np.testing.assert_almost_equal(m2, np.eye(4))


@pytest.mark.parametrize("dtype", np_dtypes)
def test_matrix_inv_raises(dtype):
  backend = numpy_backend.NumPyBackend()
  matrix = backend.randn((4, 4, 4), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.inv(matrix)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_broadcast_right_multiplication(dtype):
  backend = numpy_backend.NumPyBackend()
  tensor1 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((3,), dtype=dtype, seed=10)
  out = backend.broadcast_right_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out, tensor1 * tensor2)


def test_broadcast_right_multiplication_raises():
  dtype = np.float64
  backend = numpy_backend.NumPyBackend()
  tensor1 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((3, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.broadcast_right_multiplication(tensor1, tensor2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_broadcast_left_multiplication(dtype):
  backend = numpy_backend.NumPyBackend()
  tensor1 = backend.randn((3,), dtype=dtype, seed=10)
  tensor2 = backend.randn((3, 4, 2), dtype=dtype, seed=10)
  out = backend.broadcast_left_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out, np.reshape(tensor1, (3, 1, 1)) * tensor2)


def test_broadcast_left_multiplication_raises():
  dtype = np.float64
  backend = numpy_backend.NumPyBackend()
  tensor1 = backend.randn((3, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.broadcast_left_multiplication(tensor1, tensor2)


def test_sparse_shape():
  dtype = np.float64
  backend = numpy_backend.NumPyBackend()
  tensor = backend.randn((2, 3, 4), dtype=dtype, seed=10)
  np.testing.assert_allclose(backend.sparse_shape(tensor), tensor.shape)


@pytest.mark.parametrize("dtype,method", [(np.float64, "sin"),
                                          (np.complex128, "sin"),
                                          (np.float64, "cos"),
                                          (np.complex128, "cos"),
                                          (np.float64, "exp"),
                                          (np.complex128, "exp"),
                                          (np.float64, "log"),
                                          (np.complex128, "log")])
def test_elementwise_ops(dtype, method):
  backend = numpy_backend.NumPyBackend()
  tensor = backend.randn((4, 3, 2), dtype=dtype, seed=10)
  if method == "log":
    tensor = np.abs(tensor)
  tensor1 = getattr(backend, method)(tensor)
  tensor2 = getattr(np, method)(tensor)
  np.testing.assert_almost_equal(tensor1, tensor2)


@pytest.mark.parametrize("dtype,method", [(np.float64, "expm"),
                                          (np.complex128, "expm")])
def test_matrix_ops(dtype, method):
  backend = numpy_backend.NumPyBackend()
  matrix = backend.randn((4, 4), dtype=dtype, seed=10)
  matrix1 = getattr(backend, method)(matrix)
  matrix2 = getattr(sp.linalg, method)(matrix)
  np.testing.assert_almost_equal(matrix1, matrix2)


@pytest.mark.parametrize("dtype,method", [(np.float64, "expm"),
                                          (np.complex128, "expm")])
def test_matrix_ops_raises(dtype, method):
  backend = numpy_backend.NumPyBackend()
  matrix = backend.randn((4, 4, 4), dtype=dtype, seed=10)
  with pytest.raises(ValueError, match=r".*Only matrices.*"):
    getattr(backend, method)(matrix)
  matrix = backend.randn((4, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError, match=r".*N\*N matrix.*"):
    getattr(backend, method)(matrix)
