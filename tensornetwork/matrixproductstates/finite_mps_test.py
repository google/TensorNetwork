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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest
import numpy as np
import tensornetwork as tn
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
import tensorflow as tf

from jax.config import config


@pytest.fixture(
    name="backend_dtype_values",
    params=[('numpy', np.float64), ('numpy', np.complex128),
            ('tensorflow', np.float64), ('tensorflow', np.complex128),
            ('pytorch', np.float64), ('jax', np.float64)])
def backend_dtype(request):
  return request.param


def get_random_np(shape, dtype, seed=0):
  np.random.seed(seed)  #get the same tensors every time you call this function
  if dtype is np.complex64:
    return np.random.randn(*shape).astype(
        np.float32) + 1j * np.random.randn(*shape).astype(np.float32)
  if dtype is np.complex128:
    return np.random.randn(*shape).astype(
        np.float64) + 1j * np.random.randn(*shape).astype(np.float64)
  return np.random.randn(*shape).astype(dtype)


@pytest.mark.parametrize("N, pos", [(10, -1), (10, 10)])
def test_finite_mps_init(backend, N, pos):
  D, d = 10, 2
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  with pytest.raises(ValueError):
    FiniteMPS(tensors, center_position=pos, backend=backend)


def test_canonical_finite_mps(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]
  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = FiniteMPS(
      tensors, center_position=N // 2, backend=backend, canonicalize=False)
  assert mps.check_canonical() > 1E-12
  mps.canonicalize()
  assert mps.check_canonical() < 1E-12


def test_local_measurement_finite_mps(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors_1 = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps_1 = FiniteMPS(tensors_1, center_position=0, backend=backend)

  tensors_2 = [np.zeros((1, d, D), dtype=dtype)] + [
      np.zeros((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.zeros((D, d, 1), dtype=dtype)]
  for t in tensors_2:
    t[0, 0, 0] = 1
  mps_2 = FiniteMPS(tensors_2, center_position=0, backend=backend)

  sz = np.diag([0.5, -0.5]).astype(dtype)
  result_1 = np.array(mps_1.measure_local_operator([sz] * N, range(N)))
  result_2 = np.array(mps_2.measure_local_operator([sz] * N, range(N)))
  np.testing.assert_almost_equal(result_1, np.zeros(N))
  np.testing.assert_allclose(result_2, np.ones(N) * 0.5)


def test_correlation_measurement_finite_mps(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors_1 = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps_1 = FiniteMPS(tensors_1, center_position=0, backend=backend)
  mps_1.position(N - 1)
  mps_1.position(0)
  tensors_2 = [np.zeros((1, d, D), dtype=dtype)] + [
      np.zeros((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.zeros((D, d, 1), dtype=dtype)]
  for t in tensors_2:
    t[0, 0, 0] = 1
  mps_2 = FiniteMPS(tensors_2, center_position=0, backend=backend)
  mps_2.position(N - 1)
  mps_2.position(0)

  sz = np.diag([0.5, -0.5]).astype(dtype)
  result_1 = np.array(
      mps_1.measure_two_body_correlator(sz, sz, N // 2, range(N)))
  result_2 = np.array(
      mps_2.measure_two_body_correlator(sz, sz, N // 2, range(N)))
  actual = np.zeros(N)
  actual[N // 2] = 0.25
  np.testing.assert_almost_equal(result_1, actual)
  np.testing.assert_allclose(result_2, np.ones(N) * 0.25)
