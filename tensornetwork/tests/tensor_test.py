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
from tensornetwork.backends import abstract_backend
from tensornetwork import backends, backend_contextmanager

#pylint: disable=no-member
config.update("jax_enable_x64", True)
BaseBackend = abstract_backend.AbstractBackend

np_real = [np.float32, np.float16, np.float64]
np_complex = [np.complex64, np.complex128]
np_int = [np.int8, np.int16, np.int32, np.int64]
np_uint = [np.uint8, np.uint16, np.uint32, np.uint64]
np_all_dtypes = np_real + np_complex + np_int + np_uint + [np.bool, None]

torch_supported_dtypes = np_real + np_int + [np.uint8, np.bool, None]


def np_dtype_to_backend(backend, dtype):
  """
  Converts a given np dtype to the equivalent in the given backend.
  """
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  if backend_obj.name == "numpy":
    return dtype
  A_np = np.ones([1], dtype=dtype)

  if backend_obj.name == "jax":
    A = jnp.array(A_np)
  elif backend_obj.name == "tensorflow":
    A = tf.convert_to_tensor(A_np, dtype=dtype)
  elif backend_obj.name == "pytorch":
    A = torch.tensor(A_np)
  else:
    raise ValueError("Invalid backend ", backend)
  return A.dtype


def safe_zeros(shape, backend, dtype):
  """
  Creates a tensor of zeros, catching errors that occur when the
  dtype is
  not supported by the backend. Returns both the Tensor and the backend array,
  which are both None if the dtype and backend did not match.
  """
  init = np.zeros(shape, dtype=dtype)
  if backend == "pytorch" and dtype not in torch_supported_dtypes:
    with pytest.raises(TypeError):
      A = tn.Tensor(init, backend=backend)
    A = None
    init = None
  else:
    A = tn.Tensor(init, backend=backend)
  return (A, init)


def safe_randn(shape, backend, dtype):
  """
  Creates a random tensor , catching errors that occur when the
  dtype is not supported by the backend. Returns the Tensor and the backend
  array, which are both None if the dtype and backend did not match.
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
    A = None
    init = None
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
  if A is None:
    return
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
  if backend == "pytorch":
    if dtype not in torch_supported_dtypes:
      with pytest.raises(TypeError):
        dtype = np_dtype_to_backend(backend, dtype)
      return

    dtype = np_dtype_to_backend(backend, dtype)
    init = torch.zeros(shape, dtype=dtype)
  elif backend == "numpy":
    dtype = np_dtype_to_backend(backend, dtype)
    init = np.zeros(shape, dtype=dtype)
  elif backend == "jax":
    dtype = np_dtype_to_backend(backend, dtype)
    init = jnp.zeros(shape, dtype=dtype)
  elif backend == "tensorflow":
    dtype = np_dtype_to_backend(backend, dtype)
    init = tf.zeros(shape, dtype=dtype)
  else:
    raise ValueError("Unexpected backend ", backend)
  A = tn.Tensor(init, backend=backend)
  assert A.backend.name == backend
  np.testing.assert_allclose(A.array, init)
  assert A.shape == init.shape
  assert A.size == np.prod(init.shape)
  assert A.ndim == init.ndim


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_dtype(backend, dtype):
  """ Checks that Tensor.dtype works.
  """
  shape = (2, 3, 1)
  A, init = safe_zeros(shape, backend, dtype)
  if A is None:
    return
  if backend != "pytorch":
    assert A.dtype == init.dtype
  else:
    assert A.dtype == torch.tensor(init).dtype


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_T(backend, dtype):
  """ Checks that Tensor.T works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.T.array, init.T)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_H(backend, dtype):
  """ Checks that Tensor.H works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.H.array, init.conj().T)


#  @pytest.mark.parametrize("dtype", np_all_dtypes)
#  def test_tensor_real(backend, dtype):
#    """ Checks that Tensor.real works.
#    """
#    shape = (2, 3, 1)
#    A, init = safe_randn(shape, backend, dtype)
#    np.testing.assert_allclose(A.real.array, init.real)


#  @pytest.mark.parametrize("dtype", np_all_dtypes)
#  def test_tensor_imag(backend, dtype):
#    """ Checks that Tensor.imag works.
#    """
#    shape = (2, 3, 1)
#    A, init = safe_randn(shape, backend, dtype)
#    np.testing.assert_allclose(A.imag.array, init.imag)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_conj(backend, dtype):
  """ Checks that Tensor.conj() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.conj().array, A.backend.conj(init))


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_conjugate(backend, dtype):
  """ Checks that Tensor.conjugate() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.conjugate().array, A.backend.conj(init))


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_copy(backend, dtype):
  """ Checks that Tensor.copy() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.copy().array, init.copy())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_reshape(backend, dtype):
  """ Checks that Tensor.copy() works.
  """
  shape = (2, 3, 1)
  newshape = (6, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.reshape(newshape).array,
                               init.reshape(newshape))


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_transpose(backend, dtype):
  """ Checks that Tensor.copy() works.
  """
  shape = (2, 3, 1)
  permutation = (1, 2, 0)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    test = A.backend.convert_to_tensor(init)
    test = A.backend.transpose(test, perm=permutation)
    np.testing.assert_allclose(A.transpose(perm=permutation).array, test)


#  @pytest.mark.parametrize("dtype", np_all_dtypes)
#  def test_tensor_trace(backend, dtype):
#    """ Checks that Tensor.trace() works.
#    """
#    shape = (2, 3, 1)
#    A, init = safe_randn(shape, backend, dtype)
#    if A is not None:
#      np.testing.assert_allclose(A.trace().array, init.trace())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_squeeze(backend, dtype):
  """ Checks that Tensor.squeeze() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.squeeze().array, init.squeeze())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_ravel(backend, dtype):
  """ Checks that Tensor.ravel() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.ravel().array, init.ravel())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_flatten(backend, dtype):
  """ Checks that Tensor.flatten() works.
  """
  shape = (2, 3, 1)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.flatten().array, init.flatten())


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_tensor_hconj(backend, dtype):
  """ Checks that Tensor.hconj() works.
  """
  shape = (2, 3, 1)
  permutation = (1, 2, 0)
  A, init = safe_randn(shape, backend, dtype)
  if A is not None:
    test = A.backend.convert_to_tensor(init)
    test = A.backend.transpose(A.backend.conj(test), perm=permutation)
    np.testing.assert_allclose(A.hconj(perm=permutation).array, test)
