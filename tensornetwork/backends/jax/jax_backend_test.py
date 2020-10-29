import tensorflow as tf
import numpy as np
import scipy as sp
import jax
import pytest
from tensornetwork.backends.jax import jax_backend
import jax.config as config
import tensornetwork.backends.jax.jitted_functions as jitted_functions
# pylint: disable=no-member
config.update("jax_enable_x64", True)
np_randn_dtypes = [np.float32, np.float16, np.float64]
np_dtypes = np_randn_dtypes + [np.complex64, np.complex128]
np_not_half = [np.float32, np.float64, np.complex64, np.complex128]


def test_tensordot():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 3, 4)))
  b = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.tensordot(a, b, ((1, 2), (1, 2)))
  expected = np.array([[24.0, 24.0], [24.0, 24.0]])
  np.testing.assert_allclose(expected, actual)


def test_tensordot_int():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones((3, 3, 3)))
  b = backend.convert_to_tensor(np.ones((3, 3, 3)))
  actual = backend.tensordot(a, b, 1)
  expected = jax.numpy.tensordot(a, b, 1)
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


def test_transpose_noperm():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(
      np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a) # [2, 1, 0]
  actual = backend.transpose(actual, perm=[0, 2, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)


def test_shape_concat():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.shape_concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_slice():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(
      np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
  actual = backend.slice(a, (1, 1), (2, 2))
  expected = np.array([[5., 6.], [8., 9.]])
  np.testing.assert_allclose(expected, actual)


def test_slice_raises_error():
  backend = jax_backend.JaxBackend()
  a = backend.convert_to_tensor(
      np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
  with pytest.raises(ValueError):
    backend.slice(a, (1, 1), (2, 2, 2))


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


def test_convert_to_tensor():
  backend = jax_backend.JaxBackend()
  array = np.ones((2, 3, 4))
  actual = backend.convert_to_tensor(array)
  expected = jax.jit(lambda x: x)(array)
  assert isinstance(actual, type(expected))
  np.testing.assert_allclose(expected, actual)

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
  with pytest.raises(TypeError, match="Expected"):
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
  assert ((a >= 0).all() and (a <= 1).all() and (b >= lb).all() and
          (b <= ub).all())


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


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_valid_init_operator_with_shape(dtype):
  backend = jax_backend.JaxBackend()
  D = 16
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta1, U1 = backend.eigsh_lanczos(mv, [H], init)
  eta2, U2 = np.linalg.eigh(H)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


def test_eigsh_small_number_krylov_vectors():
  backend = jax_backend.JaxBackend()
  init = np.array([1, 1], dtype=np.float64)
  H = np.array([[1, 2], [2, 4]], dtype=np.float64)

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta1, _ = backend.eigsh_lanczos(mv, [H], init, numeig=1, num_krylov_vecs=2)
  np.testing.assert_almost_equal(eta1, [0])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_lanczos_1(dtype):
  backend = jax_backend.JaxBackend()
  D = 16
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta1, U1 = backend.eigsh_lanczos(mv, [H], init)
  eta2, U2 = np.linalg.eigh(H)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_lanczos_2(dtype):
  backend = jax_backend.JaxBackend()
  D = 16
  np.random.seed(10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta1, U1 = backend.eigsh_lanczos(mv, [H], shape=(D,), dtype=dtype)
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
  backend = jax_backend.JaxBackend()
  D = 24
  np.random.seed(10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta1, U1 = backend.eigsh_lanczos(
      mv, [H],
      shape=(D,),
      dtype=dtype,
      numeig=numeig,
      num_krylov_vecs=D,
      reorthogonalize=True,
      ndiag=1,
      tol=1E-12,
      delta=1E-12)
  eta2, U2 = np.linalg.eigh(H)

  np.testing.assert_allclose(eta1[0:numeig], eta2[0:numeig])
  for n in range(numeig):
    v2 = U2[:, n]
    v2 /= np.sum(v2)  #fix phases
    v1 = np.reshape(U1[n], (D))
    v1 /= np.sum(v1)

    np.testing.assert_allclose(v1, v2, rtol=1E-5, atol=1E-5)


def test_eigsh_lanczos_raises():
  backend = jax_backend.JaxBackend()
  with pytest.raises(
      ValueError, match='`num_krylov_vecs` >= `numeig` required!'):
    backend.eigsh_lanczos(lambda x: x, numeig=10, num_krylov_vecs=9)
  with pytest.raises(
      ValueError,
      match="Got numeig = 2 > 1 and `reorthogonalize = False`. "
      "Use `reorthogonalize=True` for `numeig > 1`"):
    backend.eigsh_lanczos(lambda x: x, numeig=2, reorthogonalize=False)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x, shape=(10,), dtype=None)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x, shape=None, dtype=np.float64)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x)
  with pytest.raises(
      TypeError, match="Expected a `jax.array`. Got <class 'list'>"):
    backend.eigsh_lanczos(lambda x: x, initial_state=[1, 2, 3])


@pytest.mark.parametrize("dtype", np_dtypes)
def test_index_update(dtype):
  backend = jax_backend.JaxBackend()
  tensor = backend.randn((4, 2, 3), dtype=dtype, seed=10)
  out = backend.index_update(tensor, tensor > 0.1, 0.0)
  np_tensor = np.array(tensor)
  np_tensor[np_tensor > 0.1] = 0.0
  np.testing.assert_allclose(out, np_tensor)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_broadcast_right_multiplication(dtype):
  backend = jax_backend.JaxBackend()
  tensor1 = backend.randn((2, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((3,), dtype=dtype, seed=10)
  out = backend.broadcast_right_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out, np.array(tensor1) * np.array(tensor2))


def test_broadcast_right_multiplication_raises():
  backend = jax_backend.JaxBackend()
  tensor1 = backend.randn((2, 3))
  tensor2 = backend.randn((3, 3))
  with pytest.raises(ValueError):
    backend.broadcast_right_multiplication(tensor1, tensor2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_broadcast_left_multiplication(dtype):
  backend = jax_backend.JaxBackend()
  tensor1 = backend.randn((3,), dtype=dtype, seed=10)
  tensor2 = backend.randn((3, 4, 2), dtype=dtype, seed=10)
  out = backend.broadcast_left_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out, np.reshape(tensor1, (3, 1, 1)) * tensor2)


def test_broadcast_left_multiplication_raises():
  dtype = np.float64
  backend = jax_backend.JaxBackend()
  tensor1 = backend.randn((3, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.broadcast_left_multiplication(tensor1, tensor2)


def test_sparse_shape():
  dtype = np.float64
  backend = jax_backend.JaxBackend()
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
  backend = jax_backend.JaxBackend()
  tensor = backend.randn((4, 3, 2), dtype=dtype, seed=10)
  if method == "log":
    tensor = np.abs(tensor)
  tensor1 = getattr(backend, method)(tensor)
  tensor2 = getattr(np, method)(tensor)
  np.testing.assert_almost_equal(tensor1, tensor2)


@pytest.mark.parametrize("dtype,method", [(np.float64, "expm"),
                                          (np.complex128, "expm")])
def test_matrix_ops(dtype, method):
  backend = jax_backend.JaxBackend()
  matrix = backend.randn((4, 4), dtype=dtype, seed=10)
  matrix1 = getattr(backend, method)(matrix)
  matrix2 = getattr(sp.linalg, method)(matrix)
  np.testing.assert_almost_equal(matrix1, matrix2)


@pytest.mark.parametrize("dtype,method", [(np.float64, "expm"),
                                          (np.complex128, "expm")])
def test_matrix_ops_raises(dtype, method):
  backend = jax_backend.JaxBackend()
  matrix = backend.randn((4, 4, 4), dtype=dtype, seed=10)
  with pytest.raises(ValueError, match=r".*Only matrices.*"):
    getattr(backend, method)(matrix)
  matrix = backend.randn((4, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError, match=r".*N\*N matrix.*"):
    getattr(backend, method)(matrix)


def test_jit():
  backend = jax_backend.JaxBackend()

  def fun(x, A, y):
    return jax.numpy.dot(x, jax.numpy.dot(A, y))

  fun_jit = backend.jit(fun)
  x = jax.numpy.array(np.random.rand(4))
  y = jax.numpy.array(np.random.rand(4))
  A = jax.numpy.array(np.random.rand(4, 4))
  res1 = fun(x, A, y)
  res2 = fun_jit(x, A, y)
  np.testing.assert_allclose(res1, res2)


def test_jit_args():
  backend = jax_backend.JaxBackend()

  def fun(x, A, y):
    return jax.numpy.dot(x, jax.numpy.dot(A, y))

  fun_jit = backend.jit(fun)
  x = jax.numpy.array(np.random.rand(4))
  y = jax.numpy.array(np.random.rand(4))
  A = jax.numpy.array(np.random.rand(4, 4))

  res1 = fun(x, A, y)
  res2 = fun_jit(x, A, y)
  res3 = fun_jit(x, y=y, A=A)
  np.testing.assert_allclose(res1, res2)
  np.testing.assert_allclose(res1, res3)


def compare_eigvals_and_eigvecs(U,
                                eta,
                                U_exact,
                                eta_exact,
                                thresh=1E-8,
                                rtol=1E-7,
                                atol=0):
  _, iy = np.nonzero(np.abs(eta[:, None] - eta_exact[None, :]) < thresh)
  U_exact_perm = U_exact[:, iy]
  U_exact_perm = U_exact_perm / np.expand_dims(np.sum(U_exact_perm, axis=0), 0)
  U = U / np.expand_dims(np.sum(U, axis=0), 0)
  np.testing.assert_allclose(U_exact_perm, U, atol=atol, rtol=rtol)
  np.testing.assert_allclose(eta, eta_exact[iy], atol=atol, rtol=rtol)


##############################################################
#                   eigs and eigsh tests                     #
##############################################################
def generate_hermitian_matrix(be, dtype, D):
  H = be.randn((D, D), dtype=dtype, seed=10)
  H += H.T.conj()
  return H


def generate_matrix(be, dtype, D):
  return be.randn((D, D), dtype=dtype, seed=10)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "solver, matrix_generator, exact_decomp, which",
    [(jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LM"),
     (jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LR"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "SA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LM")])
def test_eigs_eigsh_all_eigvals_with_init(dtype, solver, matrix_generator,
                                          exact_decomp, which):
  backend = jax_backend.JaxBackend()
  D = 16
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  H = matrix_generator(backend, dtype, D)

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta, U = solver(mv, [H], init, numeig=D, num_krylov_vecs=D, which=which)
  eta_exact, U_exact = exact_decomp(H)
  compare_eigvals_and_eigvecs(
      np.stack(U, axis=1), eta, U_exact, eta_exact, thresh=1E-8)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "solver, matrix_generator, exact_decomp, which",
    [(jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LM"),
     (jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LR"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "SA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LM")])
def test_eigs_eigsh_all_eigvals_no_init(dtype, solver, matrix_generator,
                                        exact_decomp, which):
  backend = jax_backend.JaxBackend()
  D = 16
  np.random.seed(10)
  H = matrix_generator(backend, dtype, D)

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta, U = solver(
      mv, [H],
      shape=(D,),
      dtype=dtype,
      numeig=D,
      num_krylov_vecs=D,
      which=which)
  eta_exact, U_exact = exact_decomp(H)
  compare_eigvals_and_eigvecs(
      np.stack(U, axis=1), eta, U_exact, eta_exact, thresh=1E-8)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "solver, matrix_generator, exact_decomp, which",
    [(jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LM"),
     (jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LR"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "SA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LM")])
def test_eigs_eigsh_few_eigvals_with_init(dtype, solver, matrix_generator,
                                          exact_decomp, which):
  backend = jax_backend.JaxBackend()
  D = 16
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  H = matrix_generator(backend, dtype, D)

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta, U = solver(
      mv, [H], init, numeig=4, num_krylov_vecs=16, maxiter=50, which=which)
  eta_exact, U_exact = exact_decomp(H)
  compare_eigvals_and_eigvecs(
      np.stack(U, axis=1), eta, U_exact, eta_exact, thresh=1E-8)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "solver, matrix_generator, exact_decomp, which",
    [(jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LM"),
     (jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LR"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "SA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LM")])
def test_eigs_eigsh_few_eigvals_no_init(dtype, solver, matrix_generator,
                                        exact_decomp, which):
  backend = jax_backend.JaxBackend()
  D = 16
  np.random.seed(10)
  H = matrix_generator(backend, dtype, D)

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta, U = solver(
      mv, [H],
      shape=(D,),
      dtype=dtype,
      numeig=4,
      num_krylov_vecs=16,
      which=which)
  eta_exact, U_exact = exact_decomp(H)
  compare_eigvals_and_eigvecs(
      np.stack(U, axis=1), eta, U_exact, eta_exact, thresh=1E-8)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "solver, matrix_generator, exact_decomp, which",
    [(jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LM"),
     (jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LR"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "SA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LM")])
def test_eigs_eigsh_large_ncv_with_init(dtype, solver, matrix_generator,
                                        exact_decomp, which):
  backend = jax_backend.JaxBackend()
  D = 16
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  H = matrix_generator(backend, dtype, D)

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta, U = solver(
      mv, [H], init, numeig=4, num_krylov_vecs=50, maxiter=50, which=which)
  eta_exact, U_exact = exact_decomp(H)
  compare_eigvals_and_eigvecs(
      np.stack(U, axis=1), eta, U_exact, eta_exact, thresh=1E-8)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "solver, matrix_generator, exact_decomp, which",
    [(jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LM"),
     (jax_backend.JaxBackend().eigs, generate_matrix, np.linalg.eig, "LR"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "SA"),
     (jax_backend.JaxBackend().eigsh, generate_hermitian_matrix, np.linalg.eigh,
      "LM")])
def test_eigs_eigsh_large_matrix_with_init(dtype, solver, matrix_generator,
                                           exact_decomp, which):
  backend = jax_backend.JaxBackend()
  D = 1000
  np.random.seed(10)
  init = backend.randn((D,), dtype=dtype, seed=10)
  H = matrix_generator(backend, dtype, D)

  def mv(x, H):
    return jax.numpy.dot(H, x)

  eta, U = solver(
      mv, [H],
      init,
      numeig=4,
      num_krylov_vecs=40,
      maxiter=500,
      which=which,
      tol=1E-5)
  eta_exact, U_exact = exact_decomp(H)
  compare_eigvals_and_eigvecs(
      np.stack(U, axis=1), eta, U_exact, eta_exact, thresh=1E-4, atol=1E-4)


@pytest.mark.parametrize(
    "solver, whichs",
    [(jax_backend.JaxBackend().eigs, ["SM", "SR", "LI", "SI"]),
     (jax_backend.JaxBackend().eigsh, ["SM", "BE"])])
def test_eigs_eigsh_raises(solver, whichs):
  with pytest.raises(
      ValueError, match='`num_krylov_vecs` >= `numeig` required!'):
    solver(lambda x: x, numeig=10, num_krylov_vecs=9)

  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    solver(lambda x: x, shape=(10,), dtype=None)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    solver(lambda x: x, shape=None, dtype=np.float64)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    solver(lambda x: x)
  with pytest.raises(
      TypeError, match="Expected a `jax.array`. Got <class 'list'>"):
    solver(lambda x: x, initial_state=[1, 2, 3])
  for which in whichs:
    with pytest.raises(
        ValueError, match=f"which = {which}"
        f" is currently not supported."):
      solver(lambda x: x, which=which)


def test_eigs_dtype_raises():
  solver = jax_backend.JaxBackend().eigs
  with pytest.raises(TypeError, match="dtype"):
    solver(lambda x: x, shape=(10,), dtype=np.int32)

##################################################################
#############  This test should just not crash    ################
##################################################################
@pytest.mark.parametrize("dtype",
                         [np.float64, np.complex128, np.float32, np.complex64])
def test_eigs_bugfix(dtype):
  backend = jax_backend.JaxBackend()
  D = 200
  mat = jax.numpy.array(np.random.rand(D, D).astype(dtype))
  x = jax.numpy.array(np.random.rand(D).astype(dtype))

  def matvec_jax(vector, matrix):
    return matrix @ vector

  backend.eigs(
      matvec_jax, [mat],
      numeig=1,
      initial_state=x,
      which='LR',
      maxiter=10,
      num_krylov_vecs=100,
      tol=0.0001)


def test_sum():
  np.random.seed(10)
  backend = jax_backend.JaxBackend()
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
  backend = jax_backend.JaxBackend()
  t1 = np.random.rand(10, 2, 3)
  t2 = np.random.rand(10, 3, 4)
  a = backend.convert_to_tensor(t1)
  b = backend.convert_to_tensor(t2)
  actual = backend.matmul(a, b)
  expected = np.matmul(t1, t2)
  np.testing.assert_allclose(expected, actual)
  t3 = np.random.rand(10)
  t4 = np.random.rand(11)
  c = backend.convert_to_tensor(t3)
  d = backend.convert_to_tensor(t4)
  with pytest.raises(ValueError, match="inputs to"):
    backend.matmul(c, d)

def test_gmres_raises():
  backend = jax_backend.JaxBackend()
  dummy_mv = lambda x: x
  N = 10

  b = jax.numpy.zeros((N,))
  x0 = jax.numpy.zeros((N+1),)
  diff = "If x0 is supplied, its shape"
  with pytest.raises(ValueError, match=diff): # x0, b have different sizes
    backend.gmres(dummy_mv, b, x0=x0)

  x0 = jax.numpy.zeros((N,), dtype=jax.numpy.float32)
  b = jax.numpy.zeros((N,), dtype=jax.numpy.float64)
  diff = (f"If x0 is supplied, its dtype, {x0.dtype}, must match b's"
          f", {b.dtype}.")
  with pytest.raises(TypeError, match=diff): # x0, b have different dtypes
    backend.gmres(dummy_mv, b, x0=x0)

  x0 = jax.numpy.zeros((N,))
  b = jax.numpy.zeros((N,)).reshape(2, N//2)
  diff = "If x0 is supplied, its shape"
  with pytest.raises(ValueError, match=diff): # x0, b have different shapes
    backend.gmres(dummy_mv, b, x0=x0)

  num_krylov_vectors = 0
  diff = (f"num_krylov_vectors must be positive, not"
          f"{num_krylov_vectors}.")
  with pytest.raises(ValueError, match=diff): # num_krylov_vectors <= 0
    backend.gmres(dummy_mv, b, num_krylov_vectors=num_krylov_vectors)

  tol = -1.
  diff = (f"tol = {tol} must be positive.")
  with pytest.raises(ValueError, match=diff): # tol < 0
    backend.gmres(dummy_mv, b, tol=tol)

  atol = -1
  diff = (f"atol = {atol} must be positive.")
  with pytest.raises(ValueError, match=diff): # atol < 0
    backend.gmres(dummy_mv, b, atol=atol)

  M = lambda x: x
  diff = "M is not supported by the Jax backend."
  with pytest.raises(NotImplementedError, match=diff):
    backend.gmres(dummy_mv, b, M=M)

  A_kwargs = {"bee": "honey"}
  diff = "A_kwargs is not supported by the Jax backend."
  with pytest.raises(NotImplementedError, match=diff):
    backend.gmres(dummy_mv, b, A_kwargs=A_kwargs)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_gmres_on_small_known_problem(dtype):
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype

  backend = jax_backend.JaxBackend()
  A = jax.numpy.array(([[1, 1], [3, -4]]), dtype=dtype)
  b = jax.numpy.array([3, 2], dtype=dtype)
  x0 = jax.numpy.ones(2, dtype=dtype)
  n_kry = 2

  def A_mv(x):
    return A @ x
  tol = 100*jax.numpy.finfo(dtype).eps
  x, _ = backend.gmres(A_mv, b, x0=x0, num_krylov_vectors=n_kry, tol=tol)
  solution = jax.numpy.array([2., 1.], dtype=dtype)
  eps = jax.numpy.linalg.norm(jax.numpy.abs(solution) - jax.numpy.abs(x))
  assert eps < tol


@pytest.mark.parametrize("dtype", np_dtypes)
def test_gmres_with_args(dtype):
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype

  backend = jax_backend.JaxBackend()
  A = jax.numpy.zeros((2, 2), dtype=dtype)
  B = jax.numpy.array(([[0, 1], [3, 0]]), dtype=dtype)
  C = jax.numpy.array(([[1, 0], [0, -4]]), dtype=dtype)
  b = jax.numpy.array([3, 2], dtype=dtype)
  x0 = jax.numpy.ones(2, dtype=dtype)
  n_kry = 2

  def A_mv(x, B, C):
    return (A + B + C) @ x
  tol = 100*jax.numpy.finfo(dtype).eps
  x, _ = backend.gmres(A_mv, b, A_args=[B, C], x0=x0, num_krylov_vectors=n_kry,
                       tol=tol)
  solution = jax.numpy.array([2., 1.], dtype=dtype)
  eps = jax.numpy.linalg.norm(jax.numpy.abs(solution) - jax.numpy.abs(x))
  assert eps < tol


@pytest.mark.parametrize("dtype", np_dtypes)
def test_gmres_on_larger_random_problem(dtype):
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype
  backend = jax_backend.JaxBackend()
  matshape = (100, 100)
  vecshape = (100,)
  A = backend.randn(matshape, seed=10, dtype=dtype)
  solution = backend.randn(vecshape, seed=10, dtype=dtype)
  def A_mv(x):
    return A @ x
  b = A_mv(solution)
  tol = b.size * jax.numpy.finfo(dtype).eps
  x, _ = backend.gmres(A_mv, b, tol=tol, num_krylov_vectors=100)
  err = jax.numpy.linalg.norm(jax.numpy.abs(x)-jax.numpy.abs(solution))
  rtol = tol*jax.numpy.linalg.norm(b)
  atol = tol
  assert err < max(rtol, atol)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_gmres_not_matrix(dtype):
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype
  backend = jax_backend.JaxBackend()
  matshape = (100, 100)
  vecshape = (100,)
  A = backend.randn(matshape, dtype=dtype, seed=10)
  A = backend.reshape(A, (2, 50, 2, 50))
  solution = backend.randn(vecshape, dtype=dtype, seed=10)
  solution = backend.reshape(solution, (2, 50))
  def A_mv(x):
    return backend.einsum('ijkl,kl', A, x)
  b = A_mv(solution)
  tol = b.size * np.finfo(dtype).eps
  x, _ = backend.gmres(A_mv, b, tol=tol, num_krylov_vectors=100)
  err = jax.numpy.linalg.norm(jax.numpy.abs(x)-jax.numpy.abs(solution))
  rtol = tol*jax.numpy.linalg.norm(b)
  atol = tol
  assert err < max(rtol, atol)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("offset", range(-2, 2))
@pytest.mark.parametrize("axis1", range(0, 3))
@pytest.mark.parametrize("axis2", range(0, 3))
def test_diagonal(dtype, offset, axis1, axis2):
  shape = (5, 5, 5, 5)
  backend = jax_backend.JaxBackend()
  array = backend.randn(shape, dtype=dtype, seed=10)
  if axis1 == axis2:
    with pytest.raises(ValueError):
      actual = backend.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)
  else:
    actual = backend.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)
    expected = jax.numpy.diagonal(array, offset=offset, axis1=axis1,
                                  axis2=axis2)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("offset", range(-2, 2))
def test_diagflat(dtype, offset):
  shape = (5, 5, 5, 5)
  backend = jax_backend.JaxBackend()
  array = backend.randn(shape, dtype=dtype, seed=10)
  actual = backend.diagflat(array, k=offset)
  expected = jax.numpy.diag(jax.numpy.ravel(array), k=offset)
  np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("offset", range(-2, 2))
@pytest.mark.parametrize("axis1", range(0, 3))
@pytest.mark.parametrize("axis2", range(0, 3))
def test_trace(dtype, offset, axis1, axis2):
  shape = (5, 5, 5, 5)
  backend = jax_backend.JaxBackend()
  array = backend.randn(shape, dtype=dtype, seed=10)
  if axis1 == axis2:
    with pytest.raises(ValueError):
      actual = backend.trace(array, offset=offset, axis1=axis1, axis2=axis2)
  else:
    actual = backend.trace(array, offset=offset, axis1=axis1, axis2=axis2)
    expected = jax.numpy.trace(array, offset=offset, axis1=axis1, axis2=axis2)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_abs(dtype):
  shape = (4, 3, 2)
  backend = jax_backend.JaxBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  actual = backend.abs(tensor)
  expected = jax.numpy.abs(tensor)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_sign(dtype):
  shape = (4, 3, 2)
  backend = jax_backend.JaxBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  actual = backend.sign(tensor)
  expected = jax.numpy.sign(tensor)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("pivot_axis", [-1, 1, 2])
@pytest.mark.parametrize("dtype", np_dtypes)
def test_pivot(dtype, pivot_axis):
  shape = (4, 3, 2, 8)
  pivot_shape = (np.prod(shape[:pivot_axis]), np.prod(shape[pivot_axis:]))
  backend = jax_backend.JaxBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  expected = tensor.reshape(pivot_shape)
  actual = backend.pivot(tensor, pivot_axis=pivot_axis)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype, atol", [(np.float32, 1E-6),
                                         (np.float64, 1E-10),
                                         (np.complex64, 1E-6),
                                         (np.complex128, 1E-10)])
def test_inv(dtype, atol):
  shape = (10, 10)
  backend = jax_backend.JaxBackend()
  matrix = backend.randn(shape, dtype=dtype, seed=10)
  inv = backend.inv(matrix)
  np.testing.assert_allclose(inv @ matrix, np.eye(10), atol=atol)
  np.testing.assert_allclose(matrix @ inv, np.eye(10), atol=atol)
  tensor = backend.randn((10, 10, 10), dtype=dtype, seed=10)
  with pytest.raises(ValueError, match="input to"):
    backend.inv(tensor)
