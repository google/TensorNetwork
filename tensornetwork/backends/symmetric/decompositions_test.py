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
from tensornetwork.block_sparse.blocksparsetensor import BlockSparseTensor
from tensornetwork.block_sparse.charge import U1Charge, BaseCharge
from tensornetwork.block_sparse.index import Index
import tensornetwork.block_sparse as bs
from tensornetwork.backends.symmetric import decompositions
import tensornetwork.backends.numpy.decompositions as np_decompositions
import pytest
import numpy as np

np_dtypes = [np.float64, np.complex128]


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_svds(dtype, R, R1, num_charges):
  np.random.seed(10)
  D = 30
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (D, num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)

  u, s, v, _ = decompositions.svd(bs, A, R1)
  u_dense, s_dense, v_dense, _ = np_decompositions.svd(np, A.todense(), R1)
  res1 = bs.tensordot(bs.tensordot(u, bs.diag(s), 1), v, 1)
  res2 = np.tensordot(np.tensordot(u_dense, np.diag(s_dense), 1), v_dense, 1)
  np.testing.assert_almost_equal(res1.todense(), res2)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_singular_values(dtype, R, R1, num_charges):
  np.random.seed(10)
  D = 30
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (D, num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  _, s, _, _ = decompositions.svd(bs, A, R1)
  _, s_dense, _, _ = np_decompositions.svd(np, A.todense(), R1)
  np.testing.assert_almost_equal(
      np.sort(s.todense()), np.sort(s_dense[s_dense > 1E-13]))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_max_singular_values(dtype, R, R1, num_charges):
  np.random.seed(10)
  D = 30
  max_singular_values = 12
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (D, num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  _, s, _, _ = decompositions.svd(
      bs, A, R1, max_singular_values=max_singular_values)
  assert len(s.data) <= max_singular_values


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_max_truncation_error(dtype, num_charges):
  np.random.seed(10)
  R = 2
  D = 30
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (D, num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]

  flows = [True] * R
  random_matrix = BlockSparseTensor.random(
      [Index(charges[n], flows[n]) for n in range(R)], dtype=dtype)

  U, S, V = bs.svd(random_matrix, full_matrices=False)
  svals = np.array(range(1, len(S.data) + 1)).astype(np.float64)
  S.data = svals[::-1]
  val = U @ bs.diag(S) @ V
  trunc = 8
  mask = np.sqrt(np.cumsum(np.square(svals))) >= trunc
  _, S2, _, _ = decompositions.svd(
      bs, val, 1, max_truncation_error=trunc)
  np.testing.assert_allclose(S2.data, svals[mask][::-1])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_max_singular_values_larger_than_bond_dimension(dtype, num_charges):
  np.random.seed(10)
  R = 2
  D = 30
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (D, num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]

  flows = [True] * R
  random_matrix = BlockSparseTensor.random(
      [Index(charges[n], flows[n]) for n in range(R)], dtype=dtype)
  U, S, V = bs.svd(random_matrix, full_matrices=False)
  S.data = np.array(range(len(S.data)))
  val = U @ bs.diag(S) @ V
  _, S2, _, _ = decompositions.svd(
      bs, val, 1, max_singular_values=40)
  assert S2.shape == S.shape


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_rq(dtype, R, R1, num_charges):
  np.random.seed(10)
  D = 30
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (D, num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]

  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)

  r, q = decompositions.rq(bs, A, R1)
  res = bs.tensordot(r, q, 1)
  r_dense, q_dense = np_decompositions.rq(np, A.todense(), R1, False)
  res2 = np.tensordot(r_dense, q_dense, 1)
  np.testing.assert_almost_equal(res.todense(), res2)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1", [(2, 1), (3, 2), (3, 1)])
def test_qr(dtype, R, R1):
  np.random.seed(10)
  D = 30
  charges = [
      U1Charge.random(dimension=D, minval=-5, maxval=5) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)

  q, r = decompositions.qr(bs, A, R1)
  res = bs.tensordot(q, r, 1)
  q_dense, r_dense = np_decompositions.qr(np, A.todense(), R1, False)
  res2 = np.tensordot(q_dense, r_dense, 1)
  np.testing.assert_almost_equal(res.todense(), res2)
