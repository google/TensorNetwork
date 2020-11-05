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
import tensornetwork
from tensornetwork.backends import abstract_backend
from tensornetwork import backends, backend_contextmanager
from tensornetwork.tests import testing_utils
from tensornetwork import ncon_interface

#pylint: disable=no-member
config.update("jax_enable_x64", True)
BaseBackend = abstract_backend.AbstractBackend


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_init_tensor_from_numpy_array(backend, dtype):
  """ Creates a numpy array, initializes a Tensor from it, and checks that all
  its members have been correctly initialized.
  """
  A, init = testing_utils.safe_zeros((2, 3, 1), backend, dtype)
  if A is None:
    return
  assert A.backend.name == backend
  np.testing.assert_allclose(A.array, init)
  assert A.shape == init.shape
  assert A.size == init.size
  assert A.ndim == init.ndim


@pytest.mark.parametrize("dtype", testing_utils.torch_supported_dtypes)
def test_init_tensor_default_backend(dtype):
  """ Creates a numpy array, initializes a Tensor from it, and checks that all
  its members have been correctly initialized.
  """
  backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  shape = (3, 5, 2)
  testA = backend_obj.zeros(shape, dtype=dtype)
  init = np.zeros(shape, dtype=dtype)
  A = tensornetwork.Tensor(init)
  assert A.backend.name == backend
  np.testing.assert_allclose(A.array, testA)
  assert A.shape == testA.shape
  assert A.size == testA.size
  assert A.ndim == testA.ndim


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_init_tensor_from_backend_array(backend, dtype):
  """
  Creates an instance of the backend's array class, initializes a Tensor from
  it, and checks that all its members have been correctly initialized.
  """
  shape = (2, 3, 1)
  if backend == "pytorch":
    if dtype not in testing_utils.torch_supported_dtypes:
      with pytest.raises(TypeError):
        dtype = testing_utils.np_dtype_to_backend(backend, dtype)
      return

    dtype = testing_utils.np_dtype_to_backend(backend, dtype)
    init = torch.zeros(shape, dtype=dtype)
  elif backend == "numpy":
    dtype = testing_utils.np_dtype_to_backend(backend, dtype)
    init = np.zeros(shape, dtype=dtype)
  elif backend == "jax":
    dtype = testing_utils.np_dtype_to_backend(backend, dtype)
    init = jnp.zeros(shape, dtype=dtype)
  elif backend == "tensorflow":
    dtype = testing_utils.np_dtype_to_backend(backend, dtype)
    init = tf.zeros(shape, dtype=dtype)
  else:
    raise ValueError("Unexpected backend ", backend)
  A = tensornetwork.Tensor(init, backend=backend)
  assert A.backend.name == backend
  np.testing.assert_allclose(A.array, init)
  assert A.shape == init.shape
  assert A.size == np.prod(init.shape)
  assert A.ndim == init.ndim


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_tensor_dtype(backend, dtype):
  """ Checks that Tensor.dtype works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_zeros(shape, backend, dtype)
  if A is None:
    return
  if backend != "pytorch":
    assert A.dtype == init.dtype
  else:
    assert A.dtype == torch.tensor(init).dtype


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_tensor_T(backend, dtype):
  """ Checks that Tensor.T works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.T.array, init.T)


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_tensor_H(backend, dtype):
  """ Checks that Tensor.H works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.H.array, init.conj().T)


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_tensor_conj(backend, dtype):
  """ Checks that Tensor.conj() works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.conj().array, A.backend.conj(init))


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_tensor_conjugate(backend, dtype):
  """ Checks that Tensor.conjugate() works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.conjugate().array, A.backend.conj(init))


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_tensor_copy(backend, dtype):
  """ Checks that Tensor.copy() works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.copy().array, init.copy())


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_tensor_reshape(backend, dtype):
  """ Checks that Tensor.copy() works.
  """
  shape = (2, 3, 1)
  newshape = (6, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.reshape(newshape).array,
                               init.reshape(newshape))


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_tensor_transpose(backend, dtype):
  """ Checks that Tensor.transpose() works.
  """
  shape = (2, 3, 1)
  permutation = (1, 2, 0)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    test = A.backend.convert_to_tensor(init)
    test = A.backend.transpose(test, perm=permutation)
    np.testing.assert_allclose(A.transpose(perm=permutation).array, test)


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_tensor_squeeze(backend, dtype):
  """ Checks that Tensor.squeeze() works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.squeeze().array, init.squeeze())


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_tensor_ravel(backend, dtype):
  """ Checks that Tensor.ravel() works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.ravel().array, init.ravel())


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_tensor_flatten(backend, dtype):
  """ Checks that Tensor.flatten() works.
  """
  shape = (2, 3, 1)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(A.flatten().array, init.flatten())


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_tensor_hconj(backend, dtype):
  """ Checks that Tensor.hconj() works.
  """
  shape = (2, 3, 1)
  permutation = (1, 2, 0)
  A, init = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    test = A.backend.convert_to_tensor(init)
    test = A.backend.transpose(A.backend.conj(test), perm=permutation)
    np.testing.assert_allclose(A.hconj(perm=permutation).array, test)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_multiply(backend, dtype):
  """ Checks that Tensor*Tensor works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B, initB = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    testB = B.backend.convert_to_tensor(initB)
    result = A * B
    result2 = A.backend.multiply(testA, testB)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_scalar_multiply(backend, dtype):
  """ Checks that Tensor*scalar works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B = 2.
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    result = A * B
    result2 = A.backend.multiply(testA, B)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_scalar_rmultiply(backend, dtype):
  """ Checks that scalar*Tensor works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B = 2.
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    result = B * A
    result2 = A.backend.multiply(B, testA)
    np.testing.assert_allclose(result.array, result2)

@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_divide(backend, dtype):
  """ Checks that Tensor/Tensor works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B, _ = testing_utils.safe_zeros(shape, backend, dtype)
  if A is not None:
    B = B + 1
    testA = A.backend.convert_to_tensor(initA)
    result = A / B
    result2 = A.backend.divide(testA, B.array)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_scalar_divide(backend, dtype):
  """ Checks that Tensor/scalar works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B = 2.
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    result = A / B
    result2 = A.backend.divide(testA, B)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_addition(backend, dtype):
  """ Checks that Tensor+Tensor works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B, initB = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    testB = B.backend.convert_to_tensor(initB)
    result = A + B
    result2 = A.backend.addition(testA, testB)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_scalar_addition(backend, dtype):
  """ Checks that Tensor+scalar works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B = 2.
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    result = A + B
    result2 = A.backend.addition(testA, B)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_scalar_raddition(backend, dtype):
  """ Checks that scalar+Tensor works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B = 2.
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    result = B + A
    result2 = A.backend.addition(B, testA)
    np.testing.assert_allclose(result.array, result2)

@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_subtraction(backend, dtype):
  """ Checks that Tensor-Tensor works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B, initB = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    testB = B.backend.convert_to_tensor(initB)
    result = A - B
    result2 = A.backend.subtraction(testA, testB)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_scalar_subtraction(backend, dtype):
  """ Checks that Tensor-scalar works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B = 2.
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    result = A - B
    result2 = A.backend.subtraction(testA, B)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_scalar_rsubtraction(backend, dtype):
  """ Checks that scalar-Tensor works.
  """
  shape = (2, 3, 1)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B = 2.
  if A is not None:
    testA = A.backend.convert_to_tensor(initA)
    result = B - A
    result2 = A.backend.subtraction(B, testA)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_matmul(backend, dtype):
  """ Checks that Tensor@Tensor works.
  """
  shape = (3, 3)
  A, initA = testing_utils.safe_randn(shape, backend, dtype)
  B, initB = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None and B is not None:
    testA = A.backend.convert_to_tensor(initA)
    testB = B.backend.convert_to_tensor(initB)
    result = A @ B
    result2 = A.backend.matmul(testA, testB)
    np.testing.assert_allclose(result.array, result2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_tensor_ops_raise(dtype):
  """ Checks that tensor operators raise the right error.
  """
  shape = (2, 3, 1)
  A, _ = testing_utils.safe_randn(shape, "numpy", dtype)
  B, _ = testing_utils.safe_randn(shape, "jax", dtype)
  with pytest.raises(ValueError):
    _ = A * B
  with pytest.raises(ValueError):
    _ = A + B
  with pytest.raises(ValueError):
    _ = A - B
  with pytest.raises(ValueError):
    _ = A / B
  with pytest.raises(ValueError):
    _ = A @ B


def test_ncon_builder(backend):
  a, _ = testing_utils.safe_randn((2, 2, 2), backend, np.float32)
  b, _ = testing_utils.safe_randn((2, 2, 2), backend, np.float32)
  c, _ = testing_utils.safe_randn((2, 2, 2), backend, np.float32)
  tmp = a(2, 1, -1)
  assert tmp.tensors[0] is a
  assert tmp.axes[0] == [2, 1, -1]
  builder = a(2, 1, -1) @ b(2, 3, -2) @ c(1, 3, -3)
  assert builder.tensors == [a, b, c]
  assert builder.axes == [[2, 1, -1], [2, 3, -2], [1, 3, -3]]
  np.testing.assert_allclose(
      ncon_interface.ncon(
          [a, b, c], 
          [[2, 1, -1], [2, 3, -2], [1, 3, -3]], 
          backend=backend).array,
      ncon_interface.finalize(builder).array)
