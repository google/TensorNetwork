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
from tensornetwork.backends import backend_factory
from tensornetwork.matrixproductstates.base_mps import BaseMPS
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
  mps = BaseMPS(tensors, center_position=0, backend=backend)
  mps.position(len(mps) - 1)
  Z = mps.position(0, normalize=True)
  np.testing.assert_allclose(Z, 1.0)


def test_backend_initialization(backend):
  be = backend_factory.get_backend(backend)
  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = BaseMPS(tensors, center_position=0, backend=be)
  mps.position(len(mps) - 1)
  Z = mps.position(0, normalize=True)
  np.testing.assert_allclose(Z, 1.0)


def test_left_orthonormalization(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = BaseMPS(tensors, center_position=N - 1, backend=backend)
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
  mps = BaseMPS(tensors, center_position=0, backend=backend)

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
  mps = BaseMPS(tensors, center_position=0, backend=backend)
  tensor = mps.nodes[5].tensor
  gate = get_random_np((2, 2), dtype)
  mps.apply_one_site_gate(gate, 5)
  actual = np.transpose(np.tensordot(tensor, gate, ([1], [1])), (0, 2, 1))
  np.testing.assert_allclose(mps.nodes[5].tensor, actual)


def test_apply_two_site_gate(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = BaseMPS(tensors, center_position=0, backend=backend)
  gate = get_random_np((2, 2, 2, 2), dtype)
  tensor1 = mps.nodes[5].tensor
  tensor2 = mps.nodes[6].tensor

  mps.apply_two_site_gate(gate, 5, 6)
  tmp = np.tensordot(tensor1, tensor2, ([2], [0]))
  actual = np.transpose(np.tensordot(tmp, gate, ([1, 2], [2, 3])), (0, 2, 3, 1))
  mps.nodes[5][2] ^ mps.nodes[6][0]
  order = [mps.nodes[5][0], mps.nodes[5][1], mps.nodes[6][1], mps.nodes[6][2]]
  res = tn.contract_between(mps.nodes[5], mps.nodes[6])
  res.reorder_edges(order)
  np.testing.assert_allclose(res.tensor, actual)


def test_mps_switch_backend(backend):
  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), np.float64)] + [
      get_random_np((D, d, D), np.float64) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), np.float64)]
  mps = BaseMPS(tensors, center_position=0, backend="numpy")
  mps.switch_backend(backend)
  assert mps.backend.name == backend
