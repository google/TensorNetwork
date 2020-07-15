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
from tensornetwork.linalg import linalg
import tensornetwork.linalg.krylov
import tensornetwork.linalg.initialization
from tensornetwork import backends, backend_contextmanager

#pylint: disable=no-member
config.update("jax_enable_x64", True)

np_real = [np.float32, np.float64]
np_complex = [np.complex64, np.complex128]
np_float_dtypes = np_real + np_complex

torch_supported_dtypes = [np.float32, np.float64]
sparse_backends = ["numpy", "jax"]


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


@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_svd_vs_backend(backend, dtype):
  shape = (3, 6, 4, 2)
  dtype = np_dtype_to_backend(backend, dtype)
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


@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_qr_vs_backend(backend, dtype):
  shape = (3, 6, 4, 2)
  dtype = np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  split_axis = 1
  tn_result = linalg.qr(tensor, split_axis)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.qr(tensor.array, split_axis)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    np.testing.assert_allclose(tn_arr, backend_arr)


@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_rq_vs_backend(backend, dtype):
  shape = (3, 6, 4, 2)
  dtype = np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  split_axis = 1
  tn_result = linalg.rq(tensor, split_axis)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.rq(tensor.array, split_axis)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    np.testing.assert_allclose(tn_arr, backend_arr)


@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_eigh_vs_backend(backend, dtype):
  shape = (3, 6, 4, 4)
  dtype = np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tn_result = linalg.eigh(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.eigh(tensor.array)
  tn_arrays = [t.array for t in tn_result]
  for tn_arr, backend_arr in zip(tn_arrays, backend_result):
    np.testing.assert_allclose(tn_arr, backend_arr)


@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_expm_vs_backend(backend, dtype):
  shape = 6
  dtype = np_dtype_to_backend(backend, dtype)
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


@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_inv_vs_backend(backend, dtype):
  shape = 6
  dtype = np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.eye(shape, backend=backend, dtype=dtype)
  tn_result = linalg.inv(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.inv(tensor.array)
  np.testing.assert_allclose(tn_result.array, backend_result)


@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_norm_vs_backend(backend, dtype):
  shape = (6, 2, 6)
  dtype = np_dtype_to_backend(backend, dtype)
  tensor = tensornetwork.ones(shape, backend=backend, dtype=dtype)
  tn_result = linalg.norm(tensor)
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  backend_result = backend_obj.norm(tensor.array)
  assert backend_result == tn_result


@pytest.mark.parametrize("sparse_backend", sparse_backends)
@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_eigsh_lanczos(sparse_backend, dtype):
  shape = (4, 4)
  tensor = tensornetwork.linalg.initialization.ones(shape,
                                                    backend=sparse_backend,
                                                    dtype=dtype)
  def matvec(B):
    return tensor.array @ B

  x0 = tensornetwork.linalg.initialization.ones(shape, backend=sparse_backend,
                                                dtype=dtype)

  result = tensornetwork.linalg.krylov.eigsh_lanczos(matvec,
                                                     backend=tensor.backend,
                                                     initial_state=x0)
  rev, reV = result
  test_result = tensor.backend.eigsh_lanczos(matvec, initial_state=x0.array)
  tev, teV = test_result

  np.testing.assert_allclose(np.array(rev), np.array(tev))

  for r, t in zip(reV, teV):
    np.testing.assert_allclose(r.array, t)


@pytest.mark.parametrize("sparse_backend", sparse_backends)
@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_eigs(sparse_backend, dtype):
  shape = (4, 4)
  tensor = tensornetwork.linalg.initialization.ones(shape,
                                                    backend=sparse_backend,
                                                    dtype=dtype)
  x0 = tensornetwork.linalg.initialization.ones(shape, backend=sparse_backend,
                                                dtype=dtype)
  def matvec(B):
    return tensor.array @ B

  result = tensornetwork.linalg.krylov.eigs(matvec, backend=sparse_backend,
                                            initial_state=x0)
  rev, reV = result
  test_result = tensor.backend.eigs(matvec, initial_state=x0.array)
  tev, _ = test_result

  for r, t, R in zip(rev, tev, reV):
    np.testing.assert_allclose(np.array(r), np.array(t))
    testR = matvec(R.array) / r
    np.testing.assert_allclose(testR, R.array, rtol=1E-5)



@pytest.mark.parametrize("sparse_backend", sparse_backends)
@pytest.mark.parametrize("dtype", np_float_dtypes)
def test_gmres(dtype, sparse_backend):
  Adat = np.array(([[1, 1], [3, -4]]), dtype=dtype)
  A = tensornetwork.tensor.Tensor(Adat, backend=sparse_backend)
  bdat = np.array([3, 2], dtype=dtype)
  b = tensornetwork.tensor.Tensor(bdat, backend=sparse_backend)
  x0dat = np.ones(2, dtype=dtype)
  x0 = tensornetwork.tensor.Tensor(x0dat, backend=sparse_backend)
  n_kry = 2
  def A_mv(y):
    return A.array @ y

  x, _ = A.backend.gmres(A_mv, bdat, x0=x0dat, num_krylov_vectors=n_kry)
  xT, _ = tensornetwork.linalg.krylov.gmres(A_mv, b, x0=x0,
                                            num_krylov_vectors=n_kry)
  np.testing.assert_allclose(x, xT.array)
