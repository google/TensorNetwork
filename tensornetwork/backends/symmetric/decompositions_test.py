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
from tensornetwork.block_tensor.block_tensor import BlockSparseTensor
import tensornetwork.block_tensor.block_tensor as bt
from tensornetwork.block_tensor.charge import U1Charge
from tensornetwork.block_tensor.index import Index
from tensornetwork.backends.symmetric import decompositions
import tensornetwork.backends.numpy.decompositions as np_decompositions
import pytest
import numpy as np

np_dtypes = [np.float32, np.float16, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1, R2", [(2, 1, 1), (3, 2, 1), (3, 1, 2)])
def test_svd_decompositions(dtype, R, R1, R2):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)])
  A = A.reshape([D**R1, D**R2])

  u, s, v, _ = decompositions.svd_decomposition(bt, A, 1)
  u_dense, s_dense, v_dense, _ = np_decompositions.svd_decomposition(
      np, A.todense(), 1)
  res1 = bt.tensordot(
      bt.tensordot(u, bt.diag(s), ([len(u.shape) - 1], [0])), v, ([1], [0]))
  res2 = np.tensordot(
      np.tensordot(u_dense, np.diag(s_dense), ([len(u_dense.shape) - 1], [0])),
      v_dense, ([1], [0]))
  np.testing.assert_almost_equal(res1.todense(), res2)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1, R2", [(2, 1, 1), (3, 2, 1), (3, 1, 2)])
def test_singular_values(dtype, R, R1, R2):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)])
  A = A.reshape([D**R1, D**R2])

  u, s, v, _ = decompositions.svd_decomposition(bt, A, 1)
  u_dense, s_dense, v_dense, _ = np_decompositions.svd_decomposition(
      np, A.todense(), 1)
  np.testing.assert_almost_equal(
      np.sort(s.todense()), np.sort(s_dense[s_dense > 1E-13]))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1, R2", [(2, 1, 1), (3, 2, 1), (3, 1, 2)])
def test_max_singular_values(dtype, R, R1, R2):
  D = 30
  max_singular_values = 12
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)])
  A = A.reshape([D**R1, D**R2])

  u, s, v, _ = decompositions.svd_decomposition(
      bt, A, 1, max_singular_values=max_singular_values)
  assert len(s) == max_singular_values


def test_max_truncation_error():
  R = 2
  D = 30
  max_singular_values = 12
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  random_matrix = BlockSparseTensor.random(
      [Index(charges[n], flows[n]) for n in range(R)])

  U, S, V = bt.svd(random_matrix, full_matrices=False)
  S.data = np.array(range(len(S.data)))
  val = U @ bt.diag(S) @ V
  SC = np.cumsum(S.data**2)
  ind = len(SC) // 2
  U2, S2, V2, trun = decompositions.svd_decomposition(
      bt, val, 1, max_truncation_error=math.sqrt(SC[ind] - 0.1))
  np.testing.assert_allclose(S2.data, S.data[ind::])


# def test_expected_shapes_qr(self):
#   val = np.zeros((2, 3, 4, 5))
#   q, r = decompositions.qr_decomposition(np, val, 2)
#   self.assertEqual(q.shape, (2, 3, 6))
#   self.assertEqual(r.shape, (6, 4, 5))

# def test_expected_shapes_rq(self):
#   val = np.zeros((2, 3, 4, 5))
#   r, q = decompositions.rq_decomposition(np, val, 2)
#   self.assertEqual(r.shape, (2, 3, 6))
#   self.assertEqual(q.shape, (6, 4, 5))

# def test_rq_decomposition(self):
#   random_matrix = np.random.rand(10, 10)
#   r, q = decompositions.rq_decomposition(np, random_matrix, 1)
#   self.assertAllClose(r.dot(q), random_matrix)

# def test_qr_decomposition(self):
#   random_matrix = np.random.rand(10, 10)
#   q, r = decompositions.qr_decomposition(np, random_matrix, 1)
#   self.assertAllClose(q.dot(r), random_matrix)

# def test_max_singular_values_larger_than_bond_dimension(self):
#   random_matrix = np.random.rand(10, 6)
#   unitary1, _, unitary2 = np.linalg.svd(random_matrix, full_matrices=False)
#   singular_values = np.array(range(6))
#   val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
#   u, s, vh, _ = decompositions.svd_decomposition(
#       np, val, 1, max_singular_values=30)
#   self.assertEqual(u.shape, (10, 6))
#   self.assertEqual(s.shape, (6,))
#   self.assertEqual(vh.shape, (6, 6))

# def test_max_truncation_error(self):
#   random_matrix = np.random.rand(10, 10)
#   unitary1, _, unitary2 = np.linalg.svd(random_matrix)
#   singular_values = np.array(range(10))
#   val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
#   u, s, vh, trun = decompositions.svd_decomposition(
#       np, val, 1, max_truncation_error=math.sqrt(5.1))
#   self.assertEqual(u.shape, (10, 7))
#   self.assertEqual(s.shape, (7,))
#   self.assertAllClose(s, np.arange(9, 2, -1))
#   self.assertEqual(vh.shape, (7, 10))
#   self.assertAllClose(trun, np.arange(2, -1, -1))

# if __name__ == '__main__':
#   tf.test.main()
