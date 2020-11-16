import numpy as np
from tensornetwork.backends.pytorch import pytorch_backend
import torch
import pytest
from unittest.mock import Mock

torch_dtypes = [torch.float32, torch.float64, torch.int32]
torch_eye_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
torch_randn_dtypes = [torch.float32, torch.float64]


def test_tensordot():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 3, 4)))
  b = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.tensordot(a, b, ((1, 2), (1, 2)))
  expected = np.array([[24.0, 24.0], [24.0, 24.0]])
  np.testing.assert_allclose(expected, actual)


def test_tensordot_int():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2 * np.ones((3, 3, 3)))
  b = backend.convert_to_tensor(np.ones((3, 3, 3)))
  actual = backend.tensordot(a, b, 1)
  expected = torch.tensordot(a, b, 1)
  np.testing.assert_allclose(expected, actual)


def test_reshape():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.ones((2, 3, 4)))
  actual = backend.shape_tuple(backend.reshape(a, (6, 4, 1)))
  assert actual == (6, 4, 1)


def test_transpose():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(
      np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a, [2, 0, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)


def test_transpose_noperm():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(
      np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
  actual = backend.transpose(a) # [2, 1, 0]
  actual = backend.transpose(actual, perm=[0, 2, 1])
  expected = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
  np.testing.assert_allclose(expected, actual)


def test_shape_concat():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 1)))
  expected = backend.shape_concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_slice():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(
      np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
  actual = backend.slice(a, (1, 1), (2, 2))
  expected = np.array([[5., 6.], [8., 9.]])
  np.testing.assert_allclose(expected, actual)


def test_slice_raises_error():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(
      np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
  with pytest.raises(ValueError):
    backend.slice(a, (1, 1), (2, 2, 2))


def test_shape_tensor():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  assert isinstance(backend.shape_tensor(a), torch.Tensor)
  actual = backend.shape_tensor(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)


def test_shape_prod():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2 * np.ones([1, 2, 3, 4]))
  actual = np.array(backend.shape_prod(a))
  assert actual == 2**24


def test_sqrt():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.array([4.0, 9.0]))
  actual = backend.sqrt(a)
  expected = np.array([2, 3])
  np.testing.assert_allclose(expected, actual)


def test_convert_to_tensor():
  backend = pytorch_backend.PyTorchBackend()
  array = np.ones((2, 3, 4))
  actual = backend.convert_to_tensor(array)
  expected = torch.ones((2, 3, 4))
  assert isinstance(actual, type(expected))
  np.testing.assert_allclose(expected, actual)


def test_outer_product():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2 * np.ones((2, 1)))
  b = backend.convert_to_tensor(np.ones((1, 2, 2)))
  actual = backend.outer_product(a, b)
  expected = np.ones((2, 1, 1, 2, 2)) * 2

  np.testing.assert_allclose(expected, actual)


def test_norm():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.ones((2, 2)))
  assert backend.norm(a) == 2


@pytest.mark.parametrize("dtype", torch_eye_dtypes)
def test_eye(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.eye(N=4, M=5, dtype=dtype)
  np.testing.assert_allclose(torch.eye(n=4, m=5, dtype=dtype), a)


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_ones(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.ones((4, 4), dtype=dtype)
  np.testing.assert_allclose(torch.ones((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_zeros(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.zeros((4, 4), dtype=dtype)
  np.testing.assert_allclose(torch.zeros((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_randn(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_random_uniform(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.random_uniform((4, 4), dtype=dtype)
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", torch_eye_dtypes)
def test_eye_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.eye(N=4, M=4, dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_ones_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.ones((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_zeros_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.zeros((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_randn_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.randn((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_random_uniform_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.random_uniform((4, 4), dtype=dtype)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_randn_seed(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.randn((4, 4), seed=10, dtype=dtype)
  b = backend.randn((4, 4), seed=10, dtype=dtype)
  np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_random_uniform_seed(dtype):
  backend = pytorch_backend.PyTorchBackend()
  a = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  b = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  torch.allclose(a, b)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_random_uniform_boundaries(dtype):
  lb = 1.2
  ub = 4.8
  backend = pytorch_backend.PyTorchBackend()
  a = backend.random_uniform((4, 4), seed=10, dtype=dtype)
  b = backend.random_uniform((4, 4), (lb, ub), seed=10, dtype=dtype)
  assert (torch.ge(a, 0).byte().all() and torch.le(a, 1).byte().all() and
          torch.ge(b, lb).byte().all() and torch.le(b, ub).byte().all())


def test_random_uniform_behavior():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.random_uniform((4, 4), seed=10)
  torch.manual_seed(10)
  b = torch.empty((4, 4), dtype=torch.float64).uniform_()
  torch.allclose(a, b)


def test_conj():
  backend = pytorch_backend.PyTorchBackend()
  real = np.random.rand(2, 2, 2)
  a = backend.convert_to_tensor(real)
  actual = backend.conj(a)
  expected = real
  np.testing.assert_allclose(expected, actual)


def test_eigsh_lanczos_0():
  #this test should just not crash
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  D = 4
  init = backend.randn((2, 2, 2), dtype=dtype)
  tmp = backend.randn((8, 8), dtype=dtype)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))
  H = H.reshape([2, 2, 2, 2, 2, 2])

  def mv(x, mat):
    return torch.tensordot(mat, x, ([0, 3, 5], [2, 0, 1])).permute([2, 0, 1])

  backend.eigsh_lanczos(mv, [H], init, num_krylov_vecs=D)


def test_eigsh_lanczos_1():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  D = 24
  init = backend.randn((D,), dtype=dtype)
  tmp = backend.randn((D, D), dtype=dtype)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x, mat):
    return mat.mv(x)

  eta1, U1 = backend.eigsh_lanczos(mv, [H], init, num_krylov_vecs=D)
  eta2, U2 = H.symeig(eigenvectors=True)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


def test_eigsh_small_number_krylov_vectors():
  backend = pytorch_backend.PyTorchBackend()
  init = backend.convert_to_tensor(np.array([1, 1], dtype=np.float64))
  H = backend.convert_to_tensor(np.array([[1, 2], [3, 4]], dtype=np.float64))

  def mv(x, mat):
    return mat.mv(x)

  eta1, _ = backend.eigsh_lanczos(mv, [H], init, num_krylov_vecs=1)
  np.testing.assert_allclose(eta1[0], 5)


@pytest.mark.parametrize("numeig", [1, 2, 3, 4])
def test_eigsh_lanczos_reorthogonalize(numeig):
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  D = 24
  np.random.seed(10)
  tmp = backend.randn((D, D), dtype=dtype, seed=10)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x, mat):
    return mat.mv(x)

  eta1, U1 = backend.eigsh_lanczos(
      mv, [H],
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
    v1 /= torch.sum(v1)

    np.testing.assert_allclose(v1, v2, rtol=10**(-5), atol=10**(-5))


def test_eigsh_lanczos_2():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  D = 16
  tmp = backend.randn((D, D), dtype=dtype)
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x, mat):
    return mat.mv(x)

  eta1, U1 = backend.eigsh_lanczos(
      mv, [H],
      shape=(D,),
      dtype=dtype,
      reorthogonalize=True,
      ndiag=1,
      tol=10**(-12),
      delta=10**(-12))
  eta2, U2 = H.symeig(eigenvectors=True)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2, rtol=10**(-5), atol=10**(-5))


def test_eigsh_lanczos_raises():
  backend = pytorch_backend.PyTorchBackend()
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
    backend.eigsh_lanczos(lambda x: x, shape=None, dtype=torch.float64)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x)
  with pytest.raises(
      TypeError, match="Expected a `torch.Tensor`. Got <class 'list'>"):
    backend.eigsh_lanczos(lambda x: x, initial_state=[1, 2, 3])


@pytest.mark.parametrize("a, b, expected", [
    pytest.param(1, 1, 2),
    pytest.param(
        np.ones((1, 2, 3)), np.ones((1, 2, 3)), 2. * np.ones((1, 2, 3))),
])
def test_addition(a, b, expected):
  backend = pytorch_backend.PyTorchBackend()
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
  backend = pytorch_backend.PyTorchBackend()
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
  backend = pytorch_backend.PyTorchBackend()
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
  backend = pytorch_backend.PyTorchBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)
  result = backend.divide(tensor1, tensor2)

  np.testing.assert_allclose(result, expected)
  assert tensor1.dtype == tensor2.dtype == result.dtype


def test_eigh():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  H = backend.randn((4, 4), dtype)
  H = H + np.conj(np.transpose(H))

  eta, U = backend.eigh(H)
  eta_ac, _ = np.linalg.eigh(H)
  M = U.transpose(1, 0).mm(H).mm(U)
  np.testing.assert_allclose(eta, eta_ac)
  np.testing.assert_almost_equal(np.diag(eta), M)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_index_update(dtype):
  backend = pytorch_backend.PyTorchBackend()
  tensor = backend.randn((4, 2, 3), dtype=dtype, seed=10)
  out = backend.index_update(tensor, tensor > 0.1, 0.0)
  tensor[tensor > 0.1] = 0.0
  np.testing.assert_allclose(out, tensor)


def test_matrix_inv():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  matrix = backend.randn((4, 4), dtype=dtype, seed=10)
  inverse = backend.inv(matrix)
  m1 = matrix.mm(inverse)
  m2 = inverse.mm(matrix)

  np.testing.assert_almost_equal(m1, np.eye(4))
  np.testing.assert_almost_equal(m2, np.eye(4))


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_matrix_inv_raises(dtype):
  backend = pytorch_backend.PyTorchBackend()
  matrix = backend.randn((4, 4, 4), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.inv(matrix)


def test_eigs_not_implemented():
  backend = pytorch_backend.PyTorchBackend()
  with pytest.raises(NotImplementedError):
    backend.eigs(np.ones((2, 2)))


def test_gmres_not_implemented():
  backend = pytorch_backend.PyTorchBackend()
  dummy = backend.zeros(2)
  with pytest.raises(NotImplementedError):
    backend.gmres(lambda x: x, dummy)


def test_broadcast_right_multiplication():
  backend = pytorch_backend.PyTorchBackend()
  tensor1 = backend.randn((2, 4, 3), dtype=torch.float64, seed=10)
  tensor2 = backend.randn((3,), dtype=torch.float64, seed=10)
  out = backend.broadcast_right_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out, tensor1 * tensor2)


def test_broadcast_right_multiplication_raises():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  tensor1 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((3, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.broadcast_right_multiplication(tensor1, tensor2)


def test_broadcast_left_multiplication():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  tensor1 = backend.randn((3,), dtype=dtype, seed=10)
  tensor2 = backend.randn((3, 4, 2), dtype=dtype, seed=10)
  out = backend.broadcast_left_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out, np.reshape(tensor1, (3, 1, 1)) * tensor2)


def test_broadcast_left_multiplication_raises():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  tensor1 = backend.randn((3, 3), dtype=dtype, seed=10)
  tensor2 = backend.randn((2, 4, 3), dtype=dtype, seed=10)
  with pytest.raises(ValueError):
    backend.broadcast_left_multiplication(tensor1, tensor2)


def test_sparse_shape():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend()
  tensor = backend.randn((2, 3, 4), dtype=dtype, seed=10)
  np.testing.assert_allclose(backend.sparse_shape(tensor), tensor.shape)


def test_sum():
  np.random.seed(10)
  backend = pytorch_backend.PyTorchBackend()
  tensor = np.random.rand(2, 3, 4)
  a = backend.convert_to_tensor(tensor)
  actual = backend.sum(a, axis=(1, 2))
  expected = np.sum(tensor, axis=(1, 2))
  np.testing.assert_allclose(expected, actual)


def test_matmul():
  np.random.seed(10)
  backend = pytorch_backend.PyTorchBackend()
  t1 = np.random.rand(10, 2, 3)
  t2 = np.random.rand(10, 3, 4)
  a = backend.convert_to_tensor(t1)
  b = backend.convert_to_tensor(t2)
  actual = backend.matmul(a, b)
  expected = np.matmul(t1, t2)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
@pytest.mark.parametrize("offset", range(-2, 2))
@pytest.mark.parametrize("axis1", [-2, 0])
@pytest.mark.parametrize("axis2", [-1, 0])
def test_diagonal(dtype, offset, axis1, axis2):
  shape = (5, 5, 5, 5)
  backend = pytorch_backend.PyTorchBackend()
  array = backend.randn(shape, dtype=dtype, seed=10)
  if axis1 == axis2:
    with pytest.raises(ValueError):
      actual = backend.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)
  else:
    actual = backend.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)
    expected = np.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
@pytest.mark.parametrize("k", range(-2, 2))
def test_diagflat(dtype, k):
  backend = pytorch_backend.PyTorchBackend()
  array = backend.randn((16,), dtype=dtype, seed=10)
  actual = backend.diagflat(array, k=k)
  expected = torch.diag_embed(array, offset=k)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_abs(dtype):
  shape = (4, 3, 2)
  backend = pytorch_backend.PyTorchBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  actual = backend.abs(tensor)
  expected = torch.abs(tensor)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_sign(dtype):
  shape = (4, 3, 2)
  backend = pytorch_backend.PyTorchBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  actual = backend.sign(tensor)
  expected = torch.sign(tensor)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("axis1", range(0, 3))
@pytest.mark.parametrize("axis2", range(0, 3))
def test_trace(dtype, offset, axis1, axis2):
  shape = (5, 5, 5, 5)
  backend = pytorch_backend.PyTorchBackend()
  array = backend.randn(shape, dtype=dtype, seed=10)
  if offset != 0:
    with pytest.raises(NotImplementedError):
      actual = backend.trace(array, offset=offset, axis1=axis1, axis2=axis2)

  elif axis1 == axis2:
    with pytest.raises(ValueError):
      actual = backend.trace(array, offset=offset, axis1=axis1, axis2=axis2)
  else:
    actual = backend.trace(array, offset=offset, axis1=axis1, axis2=axis2)
    expected = np.trace(array, axis1=axis1, axis2=axis2)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_trace_raises():
  shape = tuple([1] * 30)
  backend = pytorch_backend.PyTorchBackend()
  array = backend.randn(shape, seed=10)
  with pytest.raises(ValueError):
    _ = backend.trace(array)


@pytest.mark.parametrize("pivot_axis", [-1, 1, 2])
@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_pivot(dtype, pivot_axis):
  shape = (4, 3, 2, 8)
  pivot_shape = (np.prod(shape[:pivot_axis]), np.prod(shape[pivot_axis:]))
  backend = pytorch_backend.PyTorchBackend()
  tensor = backend.randn(shape, dtype=dtype, seed=10)
  expected = torch.reshape(tensor, pivot_shape)
  actual = backend.pivot(tensor, pivot_axis=pivot_axis)
  np.testing.assert_allclose(expected, actual)


def test_matmul_rank2():
  np.random.seed(10)
  backend = pytorch_backend.PyTorchBackend()
  t1 = np.random.rand(10, 4)
  t2 = np.random.rand(4, 10)
  a = backend.convert_to_tensor(t1)
  b = backend.convert_to_tensor(t2)
  actual = backend.matmul(a, b)
  expected = np.matmul(t1, t2)
  np.testing.assert_allclose(expected, actual)

@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_item(dtype):
  backend = pytorch_backend.PyTorchBackend()
  tensor = backend.randn((1,), dtype=dtype, seed=10)
  assert backend.item(tensor) == tensor.item()
