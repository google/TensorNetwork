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
import tensornetwork
import pytest
import numpy as np

import tensorflow as tf

from jax.config import config

config.update("jax_enable_x64", True)
tf.compat.v1.enable_v2_behavior()


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


def test_normalization(backend):
  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = tensornetwork.mps.FiniteMPS(tensors, center_position=0, backend=backend)
  mps.position(len(mps) - 1)
  Z = mps.position(0, normalize=True)
  np.testing.assert_allclose(Z, 1.0)


@pytest.mark.parametrize("N, pos", [(10, -1), (10, 10)])
def test_mps_init(backend, N, pos):
  D, d, N = 10, 2, N
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  with pytest.raises(ValueError):
    tensornetwork.mps.FiniteMPS(tensors, center_position=pos, backend=backend)


def test_left_orthonormalization(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = tensornetwork.mps.FiniteMPS(
      tensors, center_position=N - 1, backend=backend)
  mps.position(0)
  mps.position(len(mps) - 1)
  assert all([
      mps.check_orthonormality('left', site) < 1E-12
      for site in range(len(mps))
  ])


def test_right_orthonormalization(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]
  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = tensornetwork.mps.FiniteMPS(tensors, center_position=0, backend=backend)

  mps.position(len(mps) - 1)
  mps.position(0)
  assert all([
      mps.check_orthonormality('right', site) < 1E-12
      for site in range(len(mps))
  ])


def test_apply_one_site_gate(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = tensornetwork.mps.FiniteMPS(tensors, center_position=0, backend=backend)
  gate = get_random_np((2, 2), dtype)
  mps.apply_one_site_gate(gate, 5)
  actual = np.transpose(np.tensordot(tensors[5], gate, ([1], [1])), (0, 2, 1))
  np.testing.assert_allclose(mps.nodes[5].tensor, actual)


def test_apply_two_site_gate(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = tensornetwork.mps.FiniteMPS(tensors, center_position=0, backend=backend)
  gate = get_random_np((2, 2, 2, 2), dtype)
  mps.apply_two_site_gate(gate, 5, 6)
  tmp = np.tensordot(tensors[5], tensors[6], ([2], [0]))
  actual = np.transpose(np.tensordot(tmp, gate, ([1, 2], [2, 3])), (0, 2, 3, 1))
  res = mps.contract_between(mps.nodes[5], mps.nodes[6])
  np.testing.assert_allclose(res.tensor, actual)


def test_local_measurement(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors_1 = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps_1 = tensornetwork.mps.FiniteMPS(
      tensors_1, center_position=0, backend=backend)

  tensors_2 = [np.zeros((1, d, D), dtype=dtype)] + [
      np.zeros((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.zeros((D, d, 1), dtype=dtype)]
  for t in tensors_2:
    t[0, 0, 0] = 1
  mps_2 = tensornetwork.mps.FiniteMPS(
      tensors_2, center_position=0, backend=backend)

  sz = np.diag([0.5, -0.5]).astype(dtype)
  result_1 = np.array(mps_1.measure_local_operator([sz] * N, range(N)))
  result_2 = np.array(mps_2.measure_local_operator([sz] * N, range(N)))
  np.testing.assert_almost_equal(result_1, np.zeros(N))
  np.testing.assert_allclose(result_2, np.ones(N) * 0.5)


def test_correlation_measurement(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors_1 = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps_1 = tensornetwork.mps.FiniteMPS(
      tensors_1, center_position=0, backend=backend)
  mps_1.position(N - 1)
  mps_1.position(0)
  tensors_2 = [np.zeros((1, d, D), dtype=dtype)] + [
      np.zeros((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.zeros((D, d, 1), dtype=dtype)]
  for t in tensors_2:
    t[0, 0, 0] = 1
  mps_2 = tensornetwork.mps.FiniteMPS(
      tensors_2, center_position=0, backend=backend)
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
