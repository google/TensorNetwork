"""Tests for graphmode_tensornetwork."""
import numpy as np
from tensornetwork.backends.pytorch import pytorch_backend
import torch
import pytest

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


def test_concat():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(2 * np.ones((1, 3, 1)))
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
  a = backend.convert_to_tensor(2 * np.ones([1, 2, 3, 4]))
  actual = np.array(backend.prod(a))
  assert actual == 2**24


def test_sqrt():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.array([4.0, 9.0]))
  actual = backend.sqrt(a)
  expected = np.array([2, 3])
  np.testing.assert_allclose(expected, actual)


def test_diag():
  backend = pytorch_backend.PyTorchBackend()
  b = backend.convert_to_tensor(np.array([1.0, 2.0, 3.0]))
  actual = backend.diag(b)
  expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
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
  a = backend.convert_to_tensor(
      np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
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
  expected = np.ones((2, 1, 1, 2, 2)) * 2

  np.testing.assert_allclose(expected, actual)


def test_norm():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.ones((2, 2)))
  assert backend.norm(a) == 2


@pytest.mark.parametrize("dtype", torch_eye_dtypes)
def test_eye(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.eye(N=4, M=5)
  np.testing.assert_allclose(torch.eye(n=4, m=5, dtype=dtype), a)


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_ones(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.ones((4, 4))
  np.testing.assert_allclose(torch.ones((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_zeros(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.zeros((4, 4))
  np.testing.assert_allclose(torch.zeros((4, 4), dtype=dtype), a)


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_randn(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.randn((4, 4))
  assert a.shape == (4, 4)


@pytest.mark.parametrize("dtype", torch_eye_dtypes)
def test_eye_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  dtype_2 = torch.float32
  a = backend.eye(N=4, M=4, dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", torch_eye_dtypes)
def test_eye_two_args(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  dtype_2 = torch.float32
  _ = backend.eye(N=4, dtype=dtype_2)  # a check


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_ones_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  dtype_2 = torch.float32
  a = backend.ones((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_zeros_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  dtype_2 = torch.float32
  a = backend.zeros((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_randn_dtype(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  dtype_2 = torch.float32
  a = backend.randn((4, 4), dtype=dtype_2)
  assert a.dtype == dtype_2


@pytest.mark.parametrize("dtype", torch_eye_dtypes)
def test_eye_dtype_2(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.eye(N=4, M=4)
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_ones_dtype_2(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.ones((4, 4))
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_zeros_dtype_2(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.zeros((4, 4))
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_randn_dtype_2(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.randn((4, 4))
  assert a.dtype == dtype


@pytest.mark.parametrize("dtype", torch_randn_dtypes)
def test_randn_seed(dtype):
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  a = backend.randn((4, 4), seed=10)
  b = backend.randn((4, 4), seed=10)
  np.testing.assert_allclose(a, b)


def test_conj():
  backend = pytorch_backend.PyTorchBackend()
  real = np.random.rand(2, 2, 2)
  a = backend.convert_to_tensor(real)
  actual = backend.conj(a)
  expected = real
  np.testing.assert_allclose(expected, actual)


def test_backend_dtype_exception():
  backend = pytorch_backend.PyTorchBackend(dtype=torch.float32)
  tensor = np.random.rand(2, 2, 2)
  with pytest.raises(TypeError):
    _ = backend.convert_to_tensor(tensor)


def test_eigsh_lanczos_1():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  D = 16
  init = backend.randn((D,))
  tmp = backend.randn((D, D))
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  def mv(x):
    return H.mv(x)

  eta1, U1 = backend.eigsh_lanczos(mv, init)
  eta2, U2 = H.symeig()
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


def test_eigsh_lanczos_2():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  D = 16
  tmp = backend.randn((D, D))
  H = tmp + backend.transpose(backend.conj(tmp), (1, 0))

  class LinearOperator:

    def __init__(self, shape):
      self.shape = shape

    def __call__(self, x):
      return H.mv(x)

  mv = LinearOperator(((D,), (D,)))
  eta1, U1 = backend.eigsh_lanczos(mv)
  eta2, U2 = H.symeig()
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


def test_eigsh_lanczos_raises():
  dtype = torch.float64
  backend = pytorch_backend.PyTorchBackend(dtype=dtype)
  with pytest.raises(AttributeError):
    backend.eigsh_lanczos(lambda x: x)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x, numeig=10, ncv=9)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x, numeig=2, reorthogonalize=False)


@pytest.mark.parametrize("a, b, expected",
                         [pytest.param(np.ones((1, 2, 3)),
                                       np.ones((1, 2, 3)),
                                       np.ones((1, 2, 3))),
                          pytest.param(2. * np.ones(()),
                                       np.ones((1, 2, 3)),
                                       2. * np.ones((1, 2, 3))),
                          ])
def test_multiply(a, b, expected):
  backend = pytorch_backend.PyTorchBackend()
  tensor1 = backend.convert_to_tensor(a)
  tensor2 = backend.convert_to_tensor(b)

  np.testing.assert_allclose(backend.multiply(tensor1, tensor2), expected)

def test_scalar_multiply():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.scalar_multiply(a, 2)
  expected = np.array([8, 18])
  np.testing.assert_allclose(expected, actual)

def test_scalar_divide():
  backend = pytorch_backend.PyTorchBackend()
  a = backend.convert_to_tensor(np.array([4., 9.]))
  actual = backend.scalar_divide(a, 2)
  expected = np.array([2, 4.5])
  np.testing.assert_allclose(expected, actual)