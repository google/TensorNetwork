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
from tensornetwork.matrixproductstates.infinite_mps import InfiniteMPS
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
def test_infinite_mps_init(backend, N, pos):
  D, d = 10, 2
  tensors = [np.random.randn(2, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  with pytest.raises(ValueError):
    InfiniteMPS(tensors, center_position=pos, backend=backend)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_TMeigs(dtype):
  D, d, N = 10, 2, 10
  imps = InfiniteMPS.random(
      d=[d] * N, D=[D] * (N + 1), dtype=dtype, backend='numpy')
  eta, l = imps.transfer_matrix_eigs('r')
  l2 = imps.unit_cell_transfer_operator('r', l)
  np.testing.assert_allclose(eta * l, l2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("direction", ['left', 'right'])
def test_unitcell_transfer_operator(dtype, direction):
  D, d, N = 10, 2, 10
  imps = InfiniteMPS.random(
      d=[d] * N, D=[D] * (N + 1), dtype=dtype, backend='numpy')
  m = imps.backend.randn((D, D), dtype=dtype, seed=10)
  res1 = imps.unit_cell_transfer_operator(direction, m)
  sites = range(len(imps))
  if direction == 'right':
    sites = reversed(sites)

  for site in sites:
    m = imps.apply_transfer_operator(site, direction, m)
  np.testing.assert_allclose(m, res1)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_InfiniteMPS_canonicalize(dtype):
  D, d, N = 10, 2, 4
  imps = InfiniteMPS.random(
      d=[d] * N, D=[D] * (N + 1), dtype=dtype, backend='numpy')

  imps.canonicalize()
  assert imps.check_canonical() < 1E-12
