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
from tensornetwork.block_sparse.block_tensor import BlockSparseTensor
import tensornetwork.block_sparse.block_tensor as bt
from tensornetwork.block_sparse.charge import U1Charge
from tensornetwork.block_sparse.index import Index
from tensornetwork.backends.symmetric import decompositions
import tensornetwork.backends.numpy.decompositions as np_decompositions
import pytest
import numpy as np

np_dtypes = [np.float64, np.complex128]


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
def test_svd_decompositions(dtype, R, R1):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)

  u, s, v, _ = decompositions.svd_decomposition(bt, A, R1)
  u_dense, s_dense, v_dense, _ = np_decompositions.svd_decomposition(
      np, A.todense(), R1)
  res1 = bt.tensordot(bt.tensordot(u, bt.diag(s), 1), v, 1)
  res2 = np.tensordot(np.tensordot(u_dense, np.diag(s_dense), 1), v_dense, 1)
  np.testing.assert_almost_equal(res1.todense(), res2)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
def test_singular_values(dtype, R, R1):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  _, s, _, _ = decompositions.svd_decomposition(bt, A, R1)
  _, s_dense, _, _ = np_decompositions.svd_decomposition(np, A.todense(), R1)
  np.testing.assert_almost_equal(
      np.sort(s.todense()), np.sort(s_dense[s_dense > 1E-13]))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
def test_max_singular_values(dtype, R, R1):
  D = 30
  max_singular_values = 12
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  _, s, _, _ = decompositions.svd_decomposition(
      bt, A, R1, max_singular_values=max_singular_values)
  assert len(s) == max_singular_values


@pytest.mark.parametrize("dtype", np_dtypes)
def test_max_truncation_error(dtype):
  R = 2
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  random_matrix = BlockSparseTensor.random(
      [Index(charges[n], flows[n]) for n in range(R)], dtype=dtype)

  U, S, V = bt.svd(random_matrix, full_matrices=False)
  svals = np.array(range(1, len(S.data) + 1)).astype(np.float64)
  S.data = svals[::-1]
  val = U @ bt.diag(S) @ V
  trunc = 30
  mask = np.sqrt(np.cumsum(np.square(svals))) >= trunc
  _, S2, _, _ = decompositions.svd_decomposition(
      bt, val, 1, max_truncation_error=trunc)
  np.testing.assert_allclose(S2.data, svals[mask][::-1])


@pytest.mark.parametrize("dtype", np_dtypes)
def test_max_singular_values_larger_than_bond_dimension(dtype):
  R = 2
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  random_matrix = BlockSparseTensor.random(
      [Index(charges[n], flows[n]) for n in range(R)], dtype=dtype)
  U, S, V = bt.svd(random_matrix, full_matrices=False)
  S.data = np.array(range(len(S.data)))
  val = U @ bt.diag(S) @ V
  _, S2, _, _ = decompositions.svd_decomposition(
      bt, val, 1, max_singular_values=40)
  assert S2.shape == S.shape


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
def test_rq_decomposition(dtype, R, R1):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)

  r, q = decompositions.rq_decomposition(bt, A, R1)
  res = bt.tensordot(r, q, 1)
  r_dense, q_dense = np_decompositions.rq_decomposition(np, A.todense(), R1)
  res2 = np.tensordot(r_dense, q_dense, 1)
  np.testing.assert_almost_equal(res.todense(), res2)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
def test_qr_decomposition(dtype, R, R1):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)

  q, r = decompositions.qr_decomposition(bt, A, R1)
  res = bt.tensordot(q, r, 1)
  q_dense, r_dense = np_decompositions.qr_decomposition(np, A.todense(), R1)
  res2 = np.tensordot(q_dense, r_dense, 1)
  np.testing.assert_almost_equal(res.todense(), res2)
