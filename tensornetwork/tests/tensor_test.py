# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import config
import tensorflow as tf
import torch
import pytest
import tensornetwork as tn
from tensornetwork.backends import base_backend
from tensornetwork import backends, backend_contextmanager

#pylint: disable=no-member
config.update("jax_enable_x64", True)
BaseBackend = base_backend.BaseBackend

np_real = [np.float32, np.float16, np.float64]
np_complex = [np.complex64, np.complex128]
np_int = [np.int8, np.int16, np.int32, np.int64]
np_uint = [np.uint8, np.uint16, np.uint32, np.uint64]
np_all_dtypes = np_real + np_complex + np_int + np_uint + [np.bool, None]

torch_supported_dtypes = np_real + np_int + [np.unit8, np.bool, None]


def np_dtype_to_backend(backend, dtype):
  """
  Converts a given np dtype to the equivalent in the given backend, or raises
  TypeError if no such equivalent exists.
  """
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  if backend_obj.name == "numpy":
    return dtype

  if backend_obj.name == "jax":
    A = jnp.ones([1], dtype=dtype)
  elif backend_obj.name == "tensorflow":
    A = tf.tensor([1], dtype=dtype)
  elif backend_obj.name == "pytorch":
    A = torch.tensor([1], dtype=dtype)
  else:
    raise ValueError("Invalid backend ", backend)
  return A.dtype


def safe_zeros(shape, backend, dtype):
  """
  Creates a Tensor of zeros, catching errors that occur when the dtype is
  not supported by the backend. Returns both the Tensor and the backend array.
  """
  init = np.zeros(shape, dtype=dtype)
  if backend == "pytorch" and dtype not in torch_supported_dtypes:
    with pytest.raises(TypeError):
      A = tn.Tensor(init, backend=backend)
  else:
    A = tn.Tensor(init, backend=backend)
  return (A, init)


def safe_arange(shape, backend, dtype):
  """
  Creates a tensor of successive integers, catching errors that occur when the 
  dtype is
  not supported by the backend. Returns both the Tensor and the backend array.
  """
  size = np.cumprod(shape)[0]
  init = np.arange(size, dtype=dtype)
  if backend == "pytorch" and dtype not in torch_supported_dtypes:
    with pytest.raises(TypeError):
      A = tn.Tensor(init, backend=backend)
  else:
    A = tn.Tensor(init, backend=backend)
  return (A, init)


def safe_randn(shape, backend, dtype):
  """
  Creates a tensor of successive integers, catching errors that occur when the 
  dtype is
  not supported by the backend. Returns both the Tensor and the backend array.
  """
  init = np.random.randn(*shape)
  if dtype == np.bool:
    init = np.round(init)
  init = init.astype(dtype)

  if dtype in np_complex:
    init_i = np.random.randn(*shape)
    init = init + 1.0j * init_i.astype(dtype)

  if backend == "pytorch" and dtype not in torch_supported_dtypes:
    with pytest.raises(TypeError):
      A = tn.Tensor(init, backend=backend)
  else:
    A = tn.Tensor(init, backend=backend)
  return (A, init)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_init_tensor_from_numpy_array(backend, dtype):
  """
  Creates a numpy array, initializes a Tensor from it, and checks that all
  its members have been correctly initialized.
  """
  A, init = safe_zeros((2, 3, 1), backend, dtype)
  assert A.backend.name == backend
  np.testing.assert_allclose(A.array, init)
  assert A.shape == init.shape
  assert A.size == init.size
  assert A.ndim == init.ndim


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_init_tensor_from_backend_array(backend, dtype):
  """
  Creates an instance of the backend's array class, initializes a Tensor from
  it, and checks that all its members have been correctly initialized.
  """
  shape = (2, 3, 1)
  if backend == "numpy":
    dtype = np_dtype_to_backend(backend, dtype)
    init = np.zeros(shape, dtype=dtype)
  elif backend == "jax":
    dtype = np_dtype_to_backend(backend, dtype)
    init = jnp.zeros(shape, dtype=dtype)
  elif backend == "tensorflow":
    dtype = np_dtype_to_backend(backend, dtype)
    init = tf.zeros(shape, dtype=dtype)
  elif backend == "pytorch":
    if dtype not in torch_supported_dtypes:
      with pytest.raises(TypeError):
        dtype = np_dtype_to_backend(backend, dtype)
    dtype = np_dtype_to_backend(backend, dtype)
    init = torch.zeros(shape, dtype=dtype)
  else:
    raise ValueError("Unexpected backend ", backend)
  A = tn.Tensor(init, backend=backend)
  assert A.backend.name == backend
  np.testing.assert_allclose(A.array, init)
  assert A.shape == init.shape
  assert A.size == init.size
  assert A.ndim == init.ndim


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_dtype(backend, dtype):
  """ Checks that Tensor.dtype works.
  """
  shape = (2, 3, 1)
  A, init = safe_zeros(shape, backend, dtype)
  assert A.dtype == init.dtype


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_T(backend, dtype):
  """ Checks that Tensor.T works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.T.array, init.T)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_H(backend, dtype):
  """ Checks that Tensor.H works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.H.array, init.conj().T)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_real(backend, dtype):
  """ Checks that Tensor.real works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.real.array, init.real)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_imag(backend, dtype):
  """ Checks that Tensor.imag works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.imag.array, init.imag)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_conj(backend, dtype):
  """ Checks that Tensor.conj() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.conj().array, init.conj())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_conjugate(backend, dtype):
  """ Checks that Tensor.conjugate() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.conjugate().array, init.conjugate())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_copy(backend, dtype):
  """ Checks that Tensor.copy() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.copy().array, init.copy())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_reshape(backend, dtype):
  """ Checks that Tensor.copy() works.
  """
  shape = (2, 3, 1)
  newshape = (6, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.reshape(newshape).array, init.reshape(newshape))


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_transpose(backend, dtype):
  """ Checks that Tensor.copy() works.
  """
  shape = (2, 3, 1)
  permutation = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.transpose(axes=permutation).array,
                             init.transpose(axes=permutation))


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_trace(backend, dtype):
  """ Checks that Tensor.trace() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.trace(), init.trace())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_squeeze(backend, dtype):
  """ Checks that Tensor.squeeze() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.squeeze(), init.squeeze())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_ravel(backend, dtype):
  """ Checks that Tensor.ravel() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.ravel(), init.ravel())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_flatten(backend, dtype):
  """ Checks that Tensor.flatten() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.flatten(), init.flatten())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_hconj(backend, dtype):
  """ Checks that Tensor.hconj() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  np.testing.assert_allclose(A.hconj(), init.hconj())
