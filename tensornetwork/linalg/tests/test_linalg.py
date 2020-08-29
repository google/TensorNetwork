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

# pylint: disable=no-member
config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_eigh_vs_backend(backend, dtype):
  shape = (3, 6, 4, 4)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tn_result = linalg.eigh(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.eigh(tensor.array)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    np.testing.assert_allclose(tn_arr, backend_arr)


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
def test_inv_vs_backend(backend, dtype):
  shape = 6
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.eye(shape, backend=backend, dtype=dtype)
  tn_result = linalg.inv(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.inv(tensor.array)
  np.testing.assert_allclose(tn_result.array, backend_result)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_norm_vs_backend(backend, dtype):
  shape = (6, 2, 6)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tn_result = linalg.norm(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.norm(tensor.array)
  assert backend_result == tn_result


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_svd_vs_backend(backend, dtype):
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
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
    np.testing.assert_allclose(tn_arr, backend_arr)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_qr_vs_backend(backend, dtype):
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  split_axis = 1
  tn_result = linalg.qr(tensor, split_axis, non_negative_diagonal=False)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.qr(tensor.array, split_axis)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    np.testing.assert_allclose(tn_arr, backend_arr)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_rq_vs_backend(backend, dtype):
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  split_axis = 1
  tn_result = linalg.rq(tensor, split_axis, non_negative_diagonal=False)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.rq(tensor.array, split_axis)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    np.testing.assert_allclose(tn_arr, backend_arr)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_qr_default(backend, dtype):
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  split_axis = 1
  tn_result = linalg.qr(tensor, split_axis)
  result2 = linalg.qr(tensor, split_axis, non_negative_diagonal=False)
  tn_arrays = [t.array for t in tn_result]
  arrays2 = [t.array for t in result2]
  for tn_arr, arr2 in zip(tn_arrays, arrays2):
    np.testing.assert_allclose(tn_arr, arr2)


@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_rq_default(backend, dtype):
  shape = (3, 6, 4, 2)
  dtype = testing_utils.np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  split_axis = 1
  tn_result = linalg.rq(tensor, split_axis)
  result2 = linalg.rq(tensor, split_axis, non_negative_diagonal=False)
  tn_arrays = [t.array for t in tn_result]
  arrays2 = [t.array for t in result2]
  for tn_arr, arr2 in zip(tn_arrays, arrays2):
    np.testing.assert_allclose(tn_arr, arr2)
