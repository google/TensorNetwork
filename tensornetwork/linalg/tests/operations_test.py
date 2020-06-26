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


import numpy as np
import jax
import jax.numpy as jnp
from jax import config
import tensorflow as tf
import torch
import pytest
import tensornetwork
import tensornetwork.linalg.operations
from tensornetwork import backends, backend_contextmanager

#pylint: disable=no-member
config.update("jax_enable_x64", True)

np_real = [np.float32, np.float16, np.float64]
np_complex = [np.complex64, np.complex128]
np_int = [np.int8, np.int16, np.int32, np.int64]
np_uint = [np.uint8, np.uint16, np.uint32, np.uint64]
np_float_dtypes = np_real + np_complex
np_not_bool = np_float_dtypes + np_int + np_uint + [None,]
np_all_dtypes = np_not_bool + [np.bool,]

torch_supported_dtypes = np_real + np_int + [np.uint8, np.bool, None]


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
      A = tensornetwork.Tensor(init, backend=backend)
    A = None
    init = None
  else:
    A = tensornetwork.Tensor(init, backend=backend)
  return (A, init)


def np_dtype_to_backend(backend, dtype):
  """
  Converts a given np dtype to the equivalent in the given backend. Skips
  the present test if the dtype is not supported in the backend.
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
    if dtype not in torch_supported_dtypes:
      pytest.skip("dtype unsupported by PyTorch")

    A = torch.tensor(A_np)
  else:
    raise ValueError("Invalid backend ", backend)
  return A.dtype


def check_contraction_dtype(backend, dtype):
  """
  Skips the test if the backend cannot perform multiply-add with the given
  dtype.
  """
  skip = False
  #  backend_obj = backends.backend_factory.get_backend(backend)
  #  dtype = backend_obj.zeros((1,), dtype=dtype).dtype # handles the string case

  if backend == "tensorflow":
    if dtype in [np.uint8, tf.uint8, np.uint16, tf.uint16, np.int8, tf.int8,
                 np.int16, tf.int16, np.uint32, tf.uint32, np.uint64,
                 tf.uint64]:
      skip = True

  if backend == "pytorch":
    if dtype in [np.float16, torch.float16]:
      skip = True
  if skip:
    pytest.skip("backend does not support multiply-add with this dtype.")


@pytest.mark.parametrize("dtype", np_not_bool)
def test_tensordot_invalid_backend_raises_value_error(backend, dtype):
  """
  Tests that tensordot raises ValueError when fed Tensors with different
  backends. Other failure modes are tested at the backend level.
  """
  backend_names = set(["jax", "numpy", "tensorflow", "pytorch"])
  this_name = set([backend])
  other_backend_names = list(backend_names - this_name)
  shape = (4, 4, 4)
  dtype1 = np_dtype_to_backend(backend, dtype)
  check_contraction_dtype(backend, dtype1)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype1)
  for other_backend in other_backend_names:
    dtype2 = np_dtype_to_backend(other_backend, dtype)
    check_contraction_dtype(other_backend, dtype2)
    tensor2 = tensornetwork.ones(shape, backend=other_backend, dtype=dtype2)
    with pytest.raises(ValueError):
      _ = tensornetwork.tensordot(tensor1, tensor2, [[2, 0, 1], [1, 2, 0]])


@pytest.mark.parametrize("dtype", np_not_bool)
def test_tensordot_vs_backend(backend, dtype):
  """
  Tests that tensordot yields the same result as the backend equivalent.
  """
  shape = (4, 4, 4)
  dtype = np_dtype_to_backend(backend, dtype)
  check_contraction_dtype(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensors = [tensor1, tensor2]
  dims = [[2, 0, 1], [1, 2, 0]]
  result = tensornetwork.tensordot(*tensors, dims)
  backend_obj = backends.backend_factory.get_backend(backend)
  arrays = [t.array for t in tensors]
  backend_result = backend_obj.tensordot(*arrays, axes=dims)
  np.testing.assert_allclose(backend_result, result.array)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_reshape_vs_backend(backend, dtype):
  """
  Tests that reshape yields the same result as the backend equivalent.
  """
  shape = (3, 2, 4)
  dtype = np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  result = tensornetwork.reshape(tensor, (6, 4))
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.reshape(tensor.array, (6, 4))
  assert result.shape == backend_result.shape


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_transpose_vs_backend(backend, dtype):
  """
  Tests that transpose yields the same result as the backend equivalent.
  """
  shape = (3, 2, 4)
  permutation = (1, 2, 0)
  tensor, array = safe_randn(shape, backend, dtype)

  if tensor is not None:
    backend_obj = backends.backend_factory.get_backend(backend)
    test = backend_obj.convert_to_tensor(array)
    test = backend_obj.transpose(test, perm=permutation)
    tensor_test = tensornetwork.transpose(tensor, perm=permutation)
    np.testing.assert_allclose(test, tensor_test.array)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_hconj_vs_backend(backend, dtype):
  """
  Tests that hconj yields the same result as the equivalent backend sequence.
  """
  shape = (3, 2, 4)
  permutation = (1, 2, 0)
  tensor, array = safe_randn(shape, backend, dtype)

  if tensor is not None:
    backend_obj = backends.backend_factory.get_backend(backend)
    test = backend_obj.convert_to_tensor(array)
    test = backend_obj.transpose(test, perm=permutation)
    test = backend_obj.conj(test)
    tensor_test = tensornetwork.hconj(tensor, perm=permutation)
    np.testing.assert_allclose(test, tensor_test.array)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_take_slice_vs_backend(backend, dtype):
  """
  Tests that take_slice yields the same result as the backend equivalent.
  """
  shape = (5, 6, 7)
  dtype = np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  start_indices = (1, 2, 3)
  slice_sizes = (2, 3, 3)
  result = tensornetwork.take_slice(tensor, start_indices, slice_sizes)
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.slice(tensor.array, start_indices, slice_sizes)
  assert result.shape == backend_result.shape


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_concatenate_vs_backend(backend, dtype):
  """
  Tests that concatenate yields the same result as the backend equivalent.
  """
  shape1 = (4, 3)
  shape2 = (4, 5)
  axis = 1
  dtype = np_dtype_to_backend(backend, dtype)
  tensor1 = tensornetwork.ones(shape1, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape2, backend=backend, dtype=dtype)
  tensors = [tensor1, tensor2]
  result = tensornetwork.concatenate(tensors, axis=axis)
  backend_obj = backends.backend_factory.get_backend(backend)
  arrays = [t.array for t in tensors]
  backend_result = backend_obj.shape_concat(arrays, axis=axis)
  np.testing.assert_allclose(result.array, backend_result)


@pytest.mark.parametrize("dtype", np_not_bool)
def test_concatenate_invalid_backend_raises_value_error(backend, dtype):
  """
  Tests that concatenate raises ValueError when fed Tensors with different
  backends. Other failure modes are tested at the backend level.
  """
  backend_names = set(["jax", "numpy", "tensorflow", "pytorch"])
  this_name = set([backend])
  other_backend_names = list(backend_names - this_name)
  shape1 = (4, 3)
  shape2 = (4, 5)
  axis = 1
  dtype1 = np_dtype_to_backend(backend, dtype)
  tensor1 = tensornetwork.ones(shape1, backend=backend, dtype=dtype1)
  for other_backend in other_backend_names:
    dtype2 = np_dtype_to_backend(other_backend, dtype)
    tensor2 = tensornetwork.ones(shape2, backend=other_backend, dtype=dtype2)
    with pytest.raises(ValueError):
      _ = tensornetwork.concatenate([tensor1, tensor2], axis=axis)


@pytest.mark.parametrize("dtype", np_float_dtypes)
@pytest.mark.parametrize("fname", ["sin", "cos", "exp", "log", "conj"])
def test_unary_ops_vs_backend(backend, dtype, fname):
  shape = (4, 5, 6)
  dtype_b = np_dtype_to_backend(backend, dtype)
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_func = getattr(backend_obj, fname)
  tn_func = getattr(tensornetwork.linalg.operations, fname)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype_b)
  if backend == "pytorch" and fname in ["sin", "log", "exp", "cos"]:
    with pytest.raises(NotImplementedError):
      backend_result = backend_func(tensor.array)
    with pytest.raises(NotImplementedError):
      tn_result = tn_func(tensor).array
  else:
    backend_result = backend_func(tensor.array)
    tn_result = tn_func(tensor).array
    np.testing.assert_allclose(backend_result, tn_result)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_prod_vs_backend(backend, dtype):
  shape = (4, 5, 6)
  dtype_b = np_dtype_to_backend(backend, dtype)
  backend_obj = backends.backend_factory.get_backend(backend)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype_b)
  if ((backend == "pytorch" and dtype == np.float16) or 
      (backend == "tensorflow" and dtype == np.bool)):
    pytest.skip("Prod not supported with this dtype and backend.")
  else:
    backend_result = backend_obj.shape_prod(tensor.array)
    tn_result = tensornetwork.prod(tensor).array
    np.testing.assert_allclose(backend_result, tn_result)


@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_sqrt_vs_backend(backend, dtype):
  shape = (4, 5, 6)
  dtype_b = np_dtype_to_backend(backend, dtype)
  backend_obj = backends.backend_factory.get_backend(backend)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype_b)
  if (backend == "pytorch" and dtype == np.float16):
    pytest.skip("Prod not supported with this dtype and backend.")
  else:
    backend_result = backend_obj.sqrt(tensor.array)
    tn_result = tensornetwork.sqrt(tensor).array
    np.testing.assert_allclose(backend_result, tn_result)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_shape(backend, dtype):
  shape = (4, 5, 6)
  dtype_b = np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype_b)
  tn_result = tensornetwork.shape(tensor)
  assert tensor.shape == tn_result


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_einsum_invalid_backends(dtype, backend):
  backend_names = set(["jax", "numpy", "tensorflow", "pytorch"])
  this_name = set([backend])
  other_backend_names = list(backend_names - this_name)
  shape = (4, 3)
  dtype1 = np_dtype_to_backend(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype1)
  for other_backend in other_backend_names:
    dtype2 = np_dtype_to_backend(other_backend, dtype)
    tensor2 = tensornetwork.ones(shape, backend=other_backend, dtype=dtype2)
    for other_other_backend in backend_names:
      dtype3 = np_dtype_to_backend(other_other_backend, dtype)
      tensor3 = tensornetwork.zeros(shape, backend=other_other_backend,
                                    dtype=dtype3)
      with pytest.raises(ValueError):
        _ = tensornetwork.einsum("ba, bc, dc", tensor1, tensor2, tensor3,
                                 optimize=True)


@pytest.mark.parametrize("dtype", np_not_bool)
def test_einsum_vs_backend(dtype, backend):
  shape = (4, 3)
  dtype = np_dtype_to_backend(backend, dtype)
  check_contraction_dtype(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor3 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  result = tensornetwork.einsum("ba, bc, dc", tensor1, tensor2, tensor3,
                                optimize=True)
  backend_obj = backends.backend_factory.get_backend(backend)
  arrays = [t.array for t in [tensor1, tensor2, tensor3]]
  backend_result = backend_obj.einsum("ba, bc, dc", *arrays, optimize=True)
  np.testing.assert_allclose(backend_result, result.array)


@pytest.mark.parametrize("dtype", np_all_dtypes)
def test_outer_invalid_backends(dtype, backend):
  backend_names = set(["jax", "numpy", "tensorflow", "pytorch"])
  this_name = set([backend])
  other_backend_names = list(backend_names - this_name)
  shape = (4, 3)
  dtype1 = np_dtype_to_backend(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype1)
  for other_backend in other_backend_names:
    dtype2 = np_dtype_to_backend(other_backend, dtype)
    tensor2 = tensornetwork.ones(shape, backend=other_backend, dtype=dtype2)
    with pytest.raises(ValueError):
      _ = tensornetwork.outer(tensor1, tensor2)


@pytest.mark.parametrize("dtype", np_not_bool)
def test_outer_vs_backend(dtype, backend):
  shape = (4, 3)
  dtype = np_dtype_to_backend(backend, dtype)
  check_contraction_dtype(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  result = tensornetwork.outer(tensor1, tensor2)
  backend_obj = backends.backend_factory.get_backend(backend)
  arrays = [t.array for t in [tensor1, tensor2]]
  backend_result = backend_obj.outer_product(*arrays)
  np.testing.assert_allclose(backend_result, result.array)
