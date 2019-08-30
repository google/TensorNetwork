"""Tests for graphmode_tensornetwork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensornetwork.backends.pytorch import pytorch_backend
import torch
import pytest
import tensornetwork.config as config_file
torch_dtypes = [
    dtype for dtype in config_file.supported_pytorch_dtypes
    if dtype is not torch.bool
]
torch_eye_dtypes = [
    dtype for dtype in config_file.supported_pytorch_dtypes
    if dtype not in (torch.bool, torch.float16)
]
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
