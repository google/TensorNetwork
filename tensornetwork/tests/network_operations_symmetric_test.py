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
from tensornetwork.block_sparse import (U1Charge, BlockSparseTensor, Index,
                                        BaseCharge)
from tensornetwork.block_sparse.charge import charge_equal
from tensornetwork.block_sparse.blocksparse_utils import _find_diagonal_sparse_blocks #pylint: disable=line-too-long
import tensornetwork.linalg
import tensornetwork.linalg.node_linalg


def get_random(shape, num_charges, dtype=np.float64):
  R = len(shape)
  charges = [
      BaseCharge(
          np.random.randint(-5, 5, (shape[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  return BlockSparseTensor.random(indices=indices, dtype=dtype)


def get_square_matrix(shape, num_charges, dtype=np.float64):
  charge = BaseCharge(
      np.random.randint(-5, 5, (shape, num_charges)),
      charge_types=[U1Charge] * num_charges)
  flows = [True, False]
  indices = [Index(charge, flows[n]) for n in range(2)]
  return BlockSparseTensor.random(indices=indices, dtype=dtype)


@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_split_node_full_svd_names(num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((10, 10), num_charges=num_charges), backend='symmetric')
  e1 = a[0]
  e2 = a[1]
  left, s, right, _, = tn.split_node_full_svd(
      a, [e1], [e2],
      left_name='left',
      middle_name='center',
      right_name='right',
      left_edge_name='left_edge',
      right_edge_name='right_edge')
  assert left.name == 'left'
  assert s.name == 'center'
  assert right.name == 'right'
  assert left.edges[-1].name == 'left_edge'
  assert s[0].name == 'left_edge'
  assert s[1].name == 'right_edge'
  assert right.edges[0].name == 'right_edge'


@pytest.mark.parametrize("num_charges", [1, 2])
def test_split_node_rq_names(num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((5, 5, 5, 5, 5), num_charges=num_charges), backend='symmetric')

  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_rq(
      a,
      left_edges,
      right_edges,
      left_name='left',
      right_name='right',
      edge_name='edge')
  assert left.name == 'left'
  assert right.name == 'right'
  assert left.edges[-1].name == 'edge'
  assert right.edges[0].name == 'edge'


@pytest.mark.parametrize("num_charges", [1, 2])
def test_split_node_qr_names(num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((5, 5, 5, 5, 5), num_charges=num_charges), backend='symmetric')
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_qr(
      a,
      left_edges,
      right_edges,
      left_name='left',
      right_name='right',
      edge_name='edge')
  assert left.name == 'left'
  assert right.name == 'right'
  assert left.edges[-1].name == 'edge'
  assert right.edges[0].name == 'edge'


@pytest.mark.parametrize("num_charges", [1, 2])
def test_split_node_names(num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((5, 5, 5, 5, 5), num_charges=num_charges), backend='symmetric')
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right, _ = tn.split_node(
      a,
      left_edges,
      right_edges,
      left_name='left',
      right_name='right',
      edge_name='edge')
  assert left.name == 'left'
  assert right.name == 'right'
  assert left.edges[-1].name == 'edge'
  assert right.edges[0].name == 'edge'


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_split_node_rq_unitarity(dtype, num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_square_matrix(50, num_charges, dtype=dtype), backend='symmetric')
  r, q = tn.split_node_rq(a, [a[0]], [a[1]])
  r[1] | q[0]
  qbar = tn.linalg.node_linalg.conj(q)
  q[1] ^ qbar[1]
  u1 = q @ qbar
  qbar[0] ^ q[0]
  u2 = qbar @ q
  blocks, _, shapes = _find_diagonal_sparse_blocks(u1.tensor.flat_charges,
                                                   u1.tensor.flat_flows,
                                                   len(u1.tensor._order[0]))
  for n, block in enumerate(blocks):
    np.testing.assert_almost_equal(
        np.reshape(u1.tensor.data[block], shapes[:, n]),
        np.eye(N=shapes[0, n], M=shapes[1, n]))

  blocks, _, shapes = _find_diagonal_sparse_blocks(u2.tensor.flat_charges,
                                                   u2.tensor.flat_flows,
                                                   len(u2.tensor._order[0]))
  for n, block in enumerate(blocks):
    np.testing.assert_almost_equal(
        np.reshape(u2.tensor.data[block], shapes[:, n]),
        np.eye(N=shapes[0, n], M=shapes[1, n]))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_split_node_rq(dtype, num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((6, 7, 8, 9, 10), num_charges, dtype=dtype),
      backend='symmetric')
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_rq(a, left_edges, right_edges)
  tn.check_correct([left, right])
  result = tn.contract(left[3])
  np.testing.assert_allclose(result.tensor.data, a.tensor.data)
  assert np.all([
      charge_equal(result.tensor._charges[n], a.tensor._charges[n])
      for n in range(len(a.tensor._charges))
  ])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_split_node_qr_unitarity(dtype, num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_square_matrix(50, num_charges, dtype=dtype), backend='symmetric')
  q, r = tn.split_node_qr(a, [a[0]], [a[1]])
  r[0] | q[1]
  qbar = tn.linalg.node_linalg.conj(q)
  q[1] ^ qbar[1]
  u1 = q @ qbar
  qbar[0] ^ q[0]
  u2 = qbar @ q
  blocks, _, shapes = _find_diagonal_sparse_blocks(u1.tensor.flat_charges,
                                                   u1.tensor.flat_flows,
                                                   len(u1.tensor._order[0]))
  for n, block in enumerate(blocks):
    np.testing.assert_almost_equal(
        np.reshape(u1.tensor.data[block], shapes[:, n]),
        np.eye(N=shapes[0, n], M=shapes[1, n]))

  blocks, _, shapes = _find_diagonal_sparse_blocks(u2.tensor.flat_charges,
                                                   u2.tensor.flat_flows,
                                                   len(u2.tensor._order[0]))
  for n, block in enumerate(blocks):
    np.testing.assert_almost_equal(
        np.reshape(u2.tensor.data[block], shapes[:, n]),
        np.eye(N=shapes[0, n], M=shapes[1, n]))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_split_node_qr(dtype, num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((6, 7, 8, 9, 10), num_charges=num_charges, dtype=dtype),
      backend='symmetric')
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_qr(a, left_edges, right_edges)
  tn.check_correct([left, right])
  result = tn.contract(left[3])
  np.testing.assert_allclose(result.tensor.data, a.tensor.data)
  assert np.all([
      charge_equal(result.tensor._charges[n], a.tensor._charges[n])
      for n in range(len(a.tensor._charges))
  ])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_conj(dtype, num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((6, 7, 8, 9, 10), num_charges=num_charges, dtype=dtype),
      backend='symmetric')
  abar = tn.linalg.node_linalg.conj(a)
  np.testing.assert_allclose(abar.tensor.data, a.backend.conj(a.tensor.data))
  assert np.all([
      charge_equal(abar.tensor._charges[n], a.tensor._charges[n])
      for n in range(len(a.tensor._charges))
  ])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_transpose(dtype, num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((6, 7, 8, 9, 10), num_charges=num_charges, dtype=dtype),
      backend='symmetric')
  order = [a[n] for n in reversed(range(5))]
  transpa = tn.linalg.node_linalg.transpose(a, [4, 3, 2, 1, 0])
  a.reorder_edges(order)
  np.testing.assert_allclose(a.tensor.data, transpa.tensor.data)


def test_switch_backend():
  np.random.seed(10)
  a = tn.Node(np.random.rand(3, 3, 3), name="A", backend="numpy")
  b = tn.Node(np.random.rand(3, 3, 3), name="B", backend="numpy")
  c = tn.Node(np.random.rand(3, 3, 3), name="C", backend="numpy")
  nodes = [a, b, c]
  with pytest.raises(ValueError):
    tn.switch_backend(nodes, 'symmetric')


@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_switch_backend_raises_error(num_charges):
  np.random.seed(10)
  a = tn.Node(
      get_random((3, 3, 3), num_charges=num_charges, dtype=np.float64),
      backend='symmetric')
  with pytest.raises(NotImplementedError):
    tn.switch_backend({a}, 'numpy')


def test_switch_backend_raises_error_2():
  np.random.seed(10)
  a = tn.Node(np.random.rand(3, 3, 3))
  with pytest.raises(ValueError):
    tn.switch_backend({a}, 'symmetric')
