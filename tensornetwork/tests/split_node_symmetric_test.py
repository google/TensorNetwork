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

import tensornetwork as tn
import pytest
import numpy as np
from tensornetwork.block_sparse import BlockSparseTensor, Index
from tensornetwork.block_sparse.charge import charge_equal, BaseCharge, U1Charge
from tensornetwork.block_sparse.blocksparse_utils import _find_diagonal_sparse_blocks #pylint: disable=line-too-long


def get_random(shape, num_charges, dtype=np.float64):
  R = len(shape)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (shape[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  return BlockSparseTensor.randn(indices=indices, dtype=dtype)


def get_zeros(shape, num_charges, dtype=np.float64):
  R = len(shape)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (shape[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  return BlockSparseTensor.zeros(indices=indices, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_split_node(dtype, num_charges):
  np.random.seed(111)
  a = tn.Node(
      get_zeros((5, 7, 4, 5, 6), num_charges, dtype), backend='symmetric')

  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right, _ = tn.split_node(a, left_edges, right_edges)
  tn.check_correct({left, right})
  actual = left @ right
  np.testing.assert_allclose(actual.tensor.shape, (5, 7, 4, 5, 6))
  np.testing.assert_allclose(a.tensor.shape, (5, 7, 4, 5, 6))
  np.testing.assert_allclose(left.tensor.data, 0)
  np.testing.assert_allclose(right.tensor.data, 0)
  assert np.all([
      charge_equal(a.tensor._charges[n], actual.tensor._charges[n])
      for n in range(len(a.tensor._charges))
  ])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_split_node_mixed_order(dtype, num_charges):
  np.random.seed(131)
  a = tn.Node(
      get_zeros((5, 3, 4, 5, 6), num_charges, dtype), backend='symmetric')

  left_edges = []
  for i in [0, 2, 4]:
    left_edges.append(a[i])
  right_edges = []
  for i in [1, 3]:
    right_edges.append(a[i])
  left, right, _ = tn.split_node(a, left_edges, right_edges)

  tn.check_correct({left, right})
  actual = left @ right
  np.testing.assert_allclose(actual.tensor.shape, (5, 4, 6, 3, 5))
  np.testing.assert_allclose(a.tensor.shape, (5, 3, 4, 5, 6))

  np.testing.assert_allclose(left.tensor.data, 0)
  np.testing.assert_allclose(right.tensor.data, 0)
  np.testing.assert_allclose(left.tensor.shape[0:3], (5, 4, 6))
  np.testing.assert_allclose(right.tensor.shape[1:], (3, 5))
  new_order = [0, 2, 4, 1, 3]
  assert np.all([
      charge_equal(a.tensor.charges[new_order[n]][0],
                   actual.tensor.charges[n][0])
      for n in range(len(a.tensor._charges))
  ])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_svd_consistency(dtype, num_charges):
  np.random.seed(111)
  original_tensor = get_random((20, 20), num_charges, dtype)
  node = tn.Node(original_tensor, backend='symmetric')
  u, vh, _ = tn.split_node(node, [node[0]], [node[1]])
  final_node = tn.contract_between(u, vh)
  np.testing.assert_allclose(
      final_node.tensor.data, original_tensor.data, rtol=1e-6)
  assert np.all([
      charge_equal(final_node.tensor._charges[n], original_tensor._charges[n])
      for n in range(len(original_tensor._charges))
  ])
