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
from tensornetwork.linalg import linalg
import tensornetwork.linalg.initialization
from tensornetwork import backends, backend_contextmanager
from tensornetwork.tests import testing_utils
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.charge import U1Charge
from tensornetwork.block_sparse.blocksparsetensor import BlockSparseTensor
from tensornetwork.tensor import Tensor
from tensornetwork.backends.backend_factory import get_backend
# pylint: disable=no-member
config.update("jax_enable_x64", True)

def get_shape(backend, shape):
  if backend == 'symmetric':
    return [Index(U1Charge.random(s,-1,1), False) for s in shape]
  return shape


def get_shape_hermitian(backend, shape):
  if backend == 'symmetric':
    flows = [True, False]
    c = U1Charge.random(shape[0], -1, 1)
    return [Index(c, flow) for flow in flows]
  return shape

def initialize_tensor(fname, backend, shape, dtype):
  shape = get_shape(backend, shape)
  be = get_backend(backend)
  func = getattr(be, fname)
  return Tensor(func(shape=shape, dtype=dtype), backend=be)

def initialize_hermitian_matrix(backend, shape, dtype):
  shape = get_shape_hermitian(backend, shape)
  be = get_backend(backend)
  arr = be.randn(shape=shape, dtype=dtype)
  H = arr + be.conj(be.transpose(arr))
  return Tensor(H, backend=be)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("backend",['jax', 'symmetric', 'numpy', 'pytorch', 'tensorflow'])
def test_eigh_vs_backend(backend, dtype):
  np.random.seed(10)
  shape = (4, 4)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = initialize_hermitian_matrix(backend, shape, dtype)
  tn_result = linalg.eigh(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.eigh(tensor.array)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    testing_utils.assert_allclose(tn_arr, backend_arr, backend_obj)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_expm_vs_backend(backend, dtype):
  shape = 6
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.eye(shape, backend=backend, dtype=dtype)
  if backend in ["pytorch"]:
    with pytest.raises(NotImplementedError):
      tn_result = linalg.expm(tensor)
  else:
    tn_result = linalg.expm(tensor)
  backend_obj = backends.backend_factory.get_backend(backend)
  if backend in ["pytorch"]:
    with pytest.raises(NotImplementedError):
      backend_result = backend_obj.expm(tensor.array)
  else:
    backend_result = backend_obj.expm(tensor.array)
    np.testing.assert_allclose(tn_result.array, backend_result)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("backend",
                         ['jax', 'symmetric', 'numpy', 'pytorch', 'tensorflow'])
def test_inv_vs_backend(backend, dtype):
  np.random.seed(10)
  shape = (4, 4)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = initialize_hermitian_matrix(backend, shape, dtype)
  tn_result = linalg.inv(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.inv(tensor.array)
  testing_utils.assert_allclose(tn_result.array, backend_result, backend_obj)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("backend",
                         ['jax', 'symmetric', 'numpy', 'pytorch', 'tensorflow'])
def test_norm_vs_backend(backend, dtype):
  np.random.seed(10)
  shape = (6, 8, 6)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = initialize_tensor('randn', backend, shape, dtype)
  tn_result = linalg.norm(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.norm(tensor.array)
  assert backend_result == tn_result


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("backend",
                         ['jax', 'symmetric', 'numpy', 'pytorch', 'tensorflow'])
def test_svd_vs_backend(backend, dtype):
  np.random.seed(10)
  shape = (3, 6, 4, 6)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = initialize_tensor('randn', backend, shape, dtype)
  split_axis = 1
  max_singular_values = 5
  max_trunc_error = 0.1
  relative = True
  tn_result = linalg.svd(tensor, split_axis,
                         max_singular_values=max_singular_values,
                         max_truncation_error=max_trunc_error,
                         relative=relative)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.svd(tensor.array, split_axis,
                                   max_singular_values=max_singular_values,
                                   max_truncation_error=max_trunc_error,
                                   relative=relative)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    testing_utils.assert_allclose(tn_arr, backend_arr, backend_obj)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("backend",
                         ['jax', 'symmetric', 'numpy', 'pytorch', 'tensorflow'])
def test_qr_vs_backend(backend, dtype):
  np.random.seed(10)
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = initialize_tensor('randn', backend, shape, dtype)
  split_axis = 1
  tn_result = linalg.qr(tensor, split_axis, non_negative_diagonal=False)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.qr(tensor.array, split_axis)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    testing_utils.assert_allclose(tn_arr, backend_arr, backend_obj)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("backend",
                         ['jax', 'symmetric', 'numpy', 'pytorch', 'tensorflow'])
def test_rq_vs_backend(backend, dtype):
  np.random.seed(10)
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = initialize_tensor('randn', backend, shape, dtype)
  split_axis = 1
  tn_result = linalg.rq(tensor, split_axis, non_negative_diagonal=False)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.rq(tensor.array, split_axis)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    testing_utils.assert_allclose(tn_arr, backend_arr, backend_obj)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("backend",
                         ['jax', 'symmetric', 'numpy', 'pytorch', 'tensorflow'])
def test_qr_default(backend, dtype):
  np.random.seed(10)
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = initialize_tensor('randn', backend, shape, dtype)
  split_axis = 1
  tn_result = linalg.qr(tensor, split_axis)
  result2 = linalg.qr(tensor, split_axis, non_negative_diagonal=False)
  tn_arrays = [t.array for t in tn_result]
  arrays2 = [t.array for t in result2]
  backend_obj = backends.backend_factory.get_backend(backend)
  for tn_arr, arr2 in zip(tn_arrays, arrays2):
    testing_utils.assert_allclose(tn_arr, arr2, backend_obj)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
@pytest.mark.parametrize("backend",
                         ['jax', 'symmetric', 'numpy', 'pytorch', 'tensorflow'])
def test_rq_default(backend, dtype):
  np.random.seed(10)
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = initialize_tensor('randn', backend, shape, dtype)
  split_axis = 1
  tn_result = linalg.rq(tensor, split_axis)
  result2 = linalg.rq(tensor, split_axis, non_negative_diagonal=False)
  tn_arrays = [t.array for t in tn_result]
  arrays2 = [t.array for t in result2]
  backend_obj = backends.backend_factory.get_backend(backend)
  for tn_arr, arr2 in zip(tn_arrays, arrays2):
    testing_utils.assert_allclose(tn_arr, arr2, backend_obj)

