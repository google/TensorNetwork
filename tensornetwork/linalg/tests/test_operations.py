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
from jax import config
import pytest
import tensornetwork
import tensornetwork.linalg.operations
from tensornetwork.linalg.operations import kron
from tensornetwork.linalg.initialization import random_uniform
from tensornetwork.tensor import Tensor
from tensornetwork.ncon_interface import ncon
from tensornetwork import backends
from tensornetwork.tests import testing_utils

# pylint: disable=no-member
config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_tensordot_invalid_backend_raises_value_error(backend, dtype):
  """
  Tests that tensordot raises ValueError when fed Tensors with different
  backends. Other failure modes are tested at the backend level.
  """
  backend_names = set(["jax", "numpy", "tensorflow", "pytorch"])
  this_name = set([backend])
  other_backend_names = list(backend_names - this_name)
  shape = (4, 4, 4)
  dtype1 = testing_utils.np_dtype_to_backend(backend, dtype)
  testing_utils.check_contraction_dtype(backend, dtype1)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype1)
  for other_backend in other_backend_names:
    dtype2 = testing_utils.np_dtype_to_backend(other_backend, dtype)
    testing_utils.check_contraction_dtype(other_backend, dtype2)
    tensor2 = tensornetwork.ones(shape, backend=other_backend, dtype=dtype2)
    with pytest.raises(ValueError):
      _ = tensornetwork.tensordot(tensor1, tensor2, [[2, 0, 1], [1, 2, 0]])


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_tensordot_vs_backend(backend, dtype):
  """
  Tests that tensordot yields the same result as the backend equivalent.
  """
  shape = (4, 4, 4)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  testing_utils.check_contraction_dtype(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensors = [tensor1, tensor2]
  dims = [[2, 0, 1], [1, 2, 0]]
  result = tensornetwork.tensordot(*tensors, dims)
  backend_obj = backends.backend_factory.get_backend(backend)
  arrays = [t.array for t in tensors]
  backend_result = backend_obj.tensordot(*arrays, axes=dims)
  np.testing.assert_allclose(backend_result, result.array)


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_tensordot_int_vs_backend(backend, dtype):
  """
  Tests that tensordot yields the same result as the backend equivalent.
  """
  shape = (4, 4, 4)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  testing_utils.check_contraction_dtype(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensors = [tensor1, tensor2]
  dim = 1
  result = tensornetwork.tensordot(*tensors, dim)
  backend_obj = backends.backend_factory.get_backend(backend)
  arrays = [t.array for t in tensors]
  backend_result = backend_obj.tensordot(*arrays, axes=dim)
  np.testing.assert_allclose(backend_result, result.array)


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_reshape_vs_backend(backend, dtype):
  """
  Tests that reshape yields the same result as the backend equivalent.
  """
  shape = (3, 2, 4)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  result = tensornetwork.reshape(tensor, (6, 4))
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.reshape(tensor.array, (6, 4))
  assert result.shape == backend_result.shape


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_transpose_vs_backend(backend, dtype):
  """
  Tests that transpose yields the same result as the backend equivalent.
  """
  shape = (3, 2, 4)
  permutation = (1, 2, 0)
  tensor, array = testing_utils.safe_randn(shape, backend, dtype)

  if tensor is not None:
    backend_obj = backends.backend_factory.get_backend(backend)
    test = backend_obj.convert_to_tensor(array)
    test = backend_obj.transpose(test, perm=permutation)
    tensor_test = tensornetwork.transpose(tensor, perm=permutation)
    np.testing.assert_allclose(test, tensor_test.array)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_hconj_vs_backend(backend, dtype):
  """
  Tests that hconj yields the same result as the equivalent backend sequence.
  """
  shape = (3, 2, 4)
  permutation = (1, 2, 0)
  tensor, array = testing_utils.safe_randn(shape, backend, dtype)

  if tensor is not None:
    backend_obj = backends.backend_factory.get_backend(backend)
    test = backend_obj.convert_to_tensor(array)
    test = backend_obj.transpose(test, perm=permutation)
    test = backend_obj.conj(test)
    tensor_test = tensornetwork.hconj(tensor, perm=permutation)
    np.testing.assert_allclose(test, tensor_test.array)


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_take_slice_vs_backend(backend, dtype):
  """
  Tests that take_slice yields the same result as the backend equivalent.
  """
  shape = (5, 6, 7)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  start_indices = (1, 2, 3)
  slice_sizes = (2, 3, 3)
  result = tensornetwork.take_slice(tensor, start_indices, slice_sizes)
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.slice(tensor.array, start_indices, slice_sizes)
  assert result.shape == backend_result.shape


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("fname", ["sin", "cos", "exp", "log", "conj", "sign"])
def test_unary_ops_vs_backend(backend, dtype, fname):
  shape = (4, 5, 6)
  dtype_b = testing_utils.np_dtype_to_backend(backend, dtype)
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


@pytest.mark.parametrize("dtype", testing_utils.np_not_half)
def test_abs_vs_backend(backend, dtype):
  shape = (4, 5, 6)
  dtype_b = testing_utils.np_dtype_to_backend(backend, dtype)
  backend_obj = backends.backend_factory.get_backend(backend)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype_b)
  if (backend == "pytorch" and dtype == np.float16):
    pytest.skip("Prod not supported with this dtype and backend.")
  else:
    backend_result = backend_obj.sqrt(tensor.array)
    tn_result = tensornetwork.sqrt(tensor).array
    np.testing.assert_allclose(backend_result, tn_result)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_sqrt_vs_backend(backend, dtype):
  shape = (4, 5, 6)
  dtype_b = testing_utils.np_dtype_to_backend(backend, dtype)
  backend_obj = backends.backend_factory.get_backend(backend)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype_b)
  if (backend == "pytorch" and dtype == np.float16):
    pytest.skip("Prod not supported with this dtype and backend.")
  else:
    backend_result = backend_obj.sqrt(tensor.array)
    tn_result = tensornetwork.sqrt(tensor).array
    np.testing.assert_allclose(backend_result, tn_result)


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_shape(backend, dtype):
  shape = (4, 5, 6)
  dtype_b = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype_b)
  tn_result = tensornetwork.shape(tensor)
  assert tensor.shape == tn_result


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_einsum_invalid_backends(dtype, backend):
  backend_names = set(["jax", "numpy", "tensorflow", "pytorch"])
  this_name = set([backend])
  other_backend_names = list(backend_names - this_name)
  shape = (4, 3)
  dtype1 = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype1)
  for other_backend in other_backend_names:
    dtype2 = testing_utils.np_dtype_to_backend(other_backend, dtype)
    tensor2 = tensornetwork.ones(shape, backend=other_backend, dtype=dtype2)
    for other_other_backend in backend_names:
      dtype3 = testing_utils.np_dtype_to_backend(other_other_backend, dtype)
      tensor3 = tensornetwork.zeros(shape, backend=other_other_backend,
                                    dtype=dtype3)
      with pytest.raises(ValueError):
        _ = tensornetwork.einsum("ba, bc, dc", tensor1, tensor2, tensor3,
                                 optimize=True)


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_einsum_vs_backend(dtype, backend):
  shape = (4, 3)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  testing_utils.check_contraction_dtype(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor3 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  result = tensornetwork.einsum("ba, bc, dc", tensor1, tensor2, tensor3,
                                optimize=True)
  backend_obj = backends.backend_factory.get_backend(backend)
  arrays = [t.array for t in [tensor1, tensor2, tensor3]]
  backend_result = backend_obj.einsum("ba, bc, dc", *arrays, optimize=True)
  np.testing.assert_allclose(backend_result, result.array)


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_outer_invalid_backends(dtype, backend):
  backend_names = set(["jax", "numpy", "tensorflow", "pytorch"])
  this_name = set([backend])
  other_backend_names = list(backend_names - this_name)
  shape = (4, 3)
  dtype1 = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype1)
  for other_backend in other_backend_names:
    dtype2 = testing_utils.np_dtype_to_backend(other_backend, dtype)
    tensor2 = tensornetwork.ones(shape, backend=other_backend, dtype=dtype2)
    with pytest.raises(ValueError):
      _ = tensornetwork.outer(tensor1, tensor2)


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_outer_vs_backend(dtype, backend):
  shape = (4, 3)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  testing_utils.check_contraction_dtype(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  result = tensornetwork.outer(tensor1, tensor2)
  backend_obj = backends.backend_factory.get_backend(backend)
  arrays = [t.array for t in [tensor1, tensor2]]
  backend_result = backend_obj.outer_product(*arrays)
  np.testing.assert_allclose(backend_result, result.array)


@pytest.mark.parametrize("dtype", testing_utils.np_all_dtypes)
def test_ncon_invalid_backends(dtype, backend):
  backend_names = set(["jax", "numpy", "tensorflow", "pytorch"])
  this_name = set([backend])
  other_backend_names = list(backend_names - this_name)
  shape = (4, 3)
  dtype1 = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype1)
  for other_backend in other_backend_names:
    dtype2 = testing_utils.np_dtype_to_backend(other_backend, dtype)
    tensor2 = tensornetwork.ones(shape, backend=other_backend, dtype=dtype2)
    for other_other_backend in backend_names:
      dtype3 = testing_utils.np_dtype_to_backend(other_other_backend, dtype)
      tensor3 = tensornetwork.zeros(shape, backend=other_other_backend,
                                    dtype=dtype3)
      tensors = [tensor1, tensor2, tensor3]
      idxs = [[1, -1], [1, 2], [-2, 2]]
      with pytest.raises(ValueError):
        _ = ncon(tensors, idxs)


@pytest.mark.parametrize("dtype", testing_utils.np_not_bool)
def test_ncon_vs_backend(dtype, backend):
  shape = (4, 3)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  testing_utils.check_contraction_dtype(backend, dtype)
  tensor1 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor2 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensor3 = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tensors = [tensor1, tensor2, tensor3]
  arrays = [tensor1.array, tensor2.array, tensor3.array]
  idxs = [[1, -1], [1, 2], [-2, 2]]
  result = ncon(tensors, idxs, backend=backend)
  old_result = tensornetwork.ncon(arrays, idxs, backend=backend)
  np.testing.assert_allclose(old_result, result.array)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_diagonal(backend, dtype):
  """ Checks that Tensor.diagonal() works.
  """
  shape = (2, 3, 3)
  A, _ = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(tensornetwork.diagonal(A).array,
                               A.backend.diagonal(A.array))


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_diagflat(backend, dtype):
  """ Checks that Tensor.diagflat() works.
  """
  shape = (2, 3, 3)
  A, _ = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(tensornetwork.diagflat(A).array,
                               A.backend.diagflat(A.array))


@pytest.mark.parametrize("dtype", testing_utils.np_not_half)
def test_trace(backend, dtype):
  """ Checks that Tensor.trace() works.
  """
  shape = (2, 3, 3)
  A, _ = testing_utils.safe_randn(shape, backend, dtype)
  if A is not None:
    np.testing.assert_allclose(tensornetwork.trace(A).array,
                               A.backend.trace(A.array))


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("pivotA", [None, 1, 2, 0, -1])
def test_pivot(backend, dtype, pivotA):
  """ Checks that Tensor.pivot() works.
  """
  shapeA = (2, 3, 4, 2)
  A, _ = testing_utils.safe_randn(shapeA, backend, dtype)
  if A is not None:
    if pivotA is None:
      matrixA = tensornetwork.pivot(A)
      tA = A.backend.pivot(A.array, pivot_axis=-1)
    else:
      matrixA = tensornetwork.pivot(A, pivot_axis=pivotA)
      tA = A.backend.pivot(A.array, pivot_axis=pivotA)
    np.testing.assert_allclose(matrixA.array, tA)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_kron(backend, dtype):
  """ Checks that Tensor.kron() works.
  """
  if (backend == "pytorch" and dtype in (np.complex64, np.complex128)):
    pytest.skip("pytorch support for complex dtypes is currently poor.")

  np.random.seed(10)
  t1 = Tensor(np.random.rand(2, 2).astype(dtype), backend=backend)
  t2 = Tensor(np.random.rand(3, 3).astype(dtype), backend=backend)

  res_kron = kron(t1, t2)
  res_ncon = ncon([t1.array, t2.array], [[-1, -3], [-2, -4]], backend=backend)
  np.testing.assert_allclose(res_kron.array, res_ncon)
  mat1 = res_kron.reshape((6, 6))
  mat2 = np.kron(t1.array, t2.array)
  np.testing.assert_allclose(mat1.array, mat2)

  t1 = Tensor(np.random.rand(2, 2, 2, 2).astype(dtype), backend=backend)
  t2 = Tensor(np.random.rand(3, 3, 3, 3).astype(dtype), backend=backend)
  res_kron = kron(t1, t2)
  res_ncon = ncon([t1.array, t2.array], [[-1, -2, -5, -6], [-3, -4, -7, -8]],
                  backend=backend)
  np.testing.assert_allclose(res_kron.array, res_ncon)
  mat1 = res_kron.reshape((36, 36))
  mat2 = np.kron(
      np.array(t1.array).reshape(4, 4),
      np.array(t2.array).reshape(9, 9))
  np.testing.assert_allclose(mat1.array, mat2)


def test_kron_raises(backend):
  np.random.seed(10)
  t1 = Tensor(np.random.rand(2, 2, 2), backend=backend)
  t2 = Tensor(np.random.rand(3, 3), backend=backend)
  with pytest.raises(ValueError, match="tensorA.ndim"):
    kron(t1, t2)
  with pytest.raises(ValueError, match="tensorB.ndim"):
    kron(t2, t1)

  t1 = Tensor(np.random.rand(2, 2, 2), backend='numpy')
  t2 = Tensor(np.random.rand(3, 3), backend='tensorflow')
  with pytest.raises(ValueError, match="kron"):
    kron(t1, t2)
