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
from tensornetwork.backend_contextmanager import _default_backend_stack
import pytest
import numpy as np
import tensorflow as tf
import torch
import jax
from tensornetwork.block_sparse import (U1Charge, BlockSparseTensor, Index,
                                        BaseCharge)
from tensornetwork.block_sparse.blocksparse_utils import _find_diagonal_sparse_blocks#pylint: disable=line-too-long
from tensornetwork.backends.abstract_backend import AbstractBackend

np_dtypes = [np.float32, np.float64, np.complex64, np.complex128, np.int32]
tf_dtypes = [tf.float32, tf.float64, tf.complex64, tf.complex128, tf.int32]
torch_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
jax_dtypes = [
    jax.numpy.float32, jax.numpy.float64, jax.numpy.complex64,
    jax.numpy.complex128, jax.numpy.int32
]

def get_random_symmetric(shape, flows, num_charges, seed=10, dtype=np.float64):
  assert np.all(np.asarray(shape) == shape[0])
  np.random.seed(seed)
  R = len(shape)
  charge = BaseCharge(
      np.random.randint(-5, 5, (shape[0], num_charges)),
      charge_types=[U1Charge] * num_charges)

  indices = [Index(charge, flows[n]) for n in range(R)]
  return BlockSparseTensor.random(indices=indices, dtype=dtype)

def get_square_matrix(shape, dtype=np.float64):
  charge = U1Charge(np.random.randint(-5, 5, shape))
  flows = [True, False]
  indices = [Index(charge, flows[n]) for n in range(2)]
  return BlockSparseTensor.random(indices=indices, dtype=dtype)


def get_zeros(shape, dtype=np.float64):
  R = len(shape)
  charges = [U1Charge(np.random.randint(-5, 5, shape[n])) for n in range(R)]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  return BlockSparseTensor.zeros(indices=indices, dtype=dtype)


def get_ones(shape, dtype=np.float64):
  R = len(shape)
  charges = [U1Charge(np.random.randint(-5, 5, shape[n])) for n in range(R)]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  return BlockSparseTensor.ones(indices=indices, dtype=dtype)

@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_network_copy_reordered(dtype, num_charges):
  a = tn.Node(
      get_random_symmetric((30, 30, 30), [False, False, False],
                           num_charges,
                           dtype=dtype),
      backend='symmetric')
  b = tn.Node(
      get_random_symmetric((30, 30, 30), [False, True, False],
                           num_charges,
                           dtype=dtype),
      backend='symmetric')
  c = tn.Node(
      get_random_symmetric((30, 30, 30), [True, False, True],
                           num_charges,
                           dtype=dtype),
      backend='symmetric')

  a[0] ^ b[1]
  a[1] ^ c[2]
  b[2] ^ c[0]

  edge_order = [a[2], c[1], b[0]]
  node_dict, edge_dict = tn.copy({a, b, c})
  tn.check_correct({a, b, c})

  res = a @ b @ c
  res.reorder_edges(edge_order)
  res_copy = node_dict[a] @ node_dict[b] @ node_dict[c]
  res_copy.reorder_edges([edge_dict[e] for e in edge_order])
  np.testing.assert_allclose(res.tensor.data, res_copy.tensor.data)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_small_matmul(dtype, num_charges):
  a = tn.Node(
      get_random_symmetric((100, 100), [True, True], num_charges, dtype=dtype),
      backend='symmetric')
  b = tn.Node(
      get_random_symmetric((100, 100), [False, True], num_charges, dtype=dtype),
      backend='symmetric')

  edge = tn.connect(a[0], b[0], "edge")
  tn.check_correct({a, b})
  c = tn.contract(edge, name="a * b")
  assert list(c.shape) == [100, 100]
  tn.check_correct({c})


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_double_trace(dtype, num_charges):
  a = tn.Node(
      get_random_symmetric((20, 20, 20, 20), [True, False, True, False],
                           num_charges,
                           dtype=dtype),
      backend='symmetric')
  edge1 = tn.connect(a[0], a[1], "edge1")
  edge2 = tn.connect(a[2], a[3], "edge2")
  tn.check_correct({a})
  val = tn.contract(edge1)
  tn.check_correct({val})
  val = tn.contract(edge2)
  tn.check_correct({val})
  adense = tn.Node(a.tensor.todense(), backend='numpy')
  e1 = adense[0] ^ adense[1]
  e2 = adense[2] ^ adense[3]
  tn.contract(e1)
  expected = tn.contract(e2)
  np.testing.assert_almost_equal(val.tensor.data, expected.tensor)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_real_physics(dtype, num_charges):
  # Calcuate the expected value in numpy
  t1 = get_random_symmetric((20, 20, 20, 20), [False, False, False, False],
                            num_charges,
                            dtype=dtype)
  t2 = get_random_symmetric((20, 20, 20), [True, False, True],
                            num_charges,
                            dtype=dtype)
  t3 = get_random_symmetric((20, 20, 20), [True, True, False],
                            num_charges,
                            dtype=dtype)

  t1_dense = t1.todense()
  t2_dense = t2.todense()
  t3_dense = t3.todense()
  adense = tn.Node(t1_dense, name="T", backend='numpy')
  bdense = tn.Node(t2_dense, name="A", backend='numpy')
  cdense = tn.Node(t3_dense, name="B", backend='numpy')

  e1 = tn.connect(adense[2], bdense[0], "edge")
  e2 = tn.connect(cdense[0], adense[3], "edge2")
  e3 = tn.connect(bdense[1], cdense[1], "edge3")
  node_result = tn.contract(e1)
  node_result = tn.contract(e2)
  final_result = tn.contract(e3)

  # Build the network
  a = tn.Node(t1, name="T", backend='symmetric')
  b = tn.Node(t2, name="A", backend='symmetric')
  c = tn.Node(t3, name="B", backend='symmetric')
  e1 = tn.connect(a[2], b[0], "edge")
  e2 = tn.connect(c[0], a[3], "edge2")
  e3 = tn.connect(b[1], c[1], "edge3")
  tn.check_correct(tn.reachable(a))
  node_result = tn.contract(e1)
  tn.check_correct(tn.reachable(node_result))
  node_result = tn.contract(e2)
  tn.check_correct(tn.reachable(node_result))
  val = tn.contract(e3)
  tn.check_correct(tn.reachable(val))
  np.testing.assert_allclose(val.tensor.todense(), final_result.tensor)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_node2_contract_trace(dtype, num_charges):
  a = tn.Node(
      get_random_symmetric((20, 20, 20), [False, True, False],
                           num_charges,
                           dtype=dtype),
      backend='symmetric')
  b = tn.Node(
      get_random_symmetric((20, 20), [True, False], num_charges, dtype=dtype),
      backend='symmetric')

  tn.connect(b[0], a[2])
  trace_edge = tn.connect(a[0], a[1])
  c = tn.contract(trace_edge)
  tn.check_correct({c})


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_flatten_consistent_result(dtype, num_charges):
  a_val = get_random_symmetric((10, 10, 10, 10), [False] * 4,
                               num_charges,
                               dtype=dtype)

  b_val = get_random_symmetric((10, 10, 10, 10), [True] * 4,
                               num_charges,
                               dtype=dtype)

  # Create non flattened example to compare against.
  a_noflat = tn.Node(a_val, backend='symmetric')
  b_noflat = tn.Node(b_val, backend='symmetric')
  e1 = tn.connect(a_noflat[1], b_noflat[3])
  e2 = tn.connect(a_noflat[3], b_noflat[1])
  e3 = tn.connect(a_noflat[2], b_noflat[0])
  a_dangling_noflat = a_noflat[0]
  b_dangling_noflat = b_noflat[2]
  for edge in [e1, e2, e3]:
    noflat_result_node = tn.contract(edge)
  noflat_result_node.reorder_edges([a_dangling_noflat, b_dangling_noflat])
  noflat_result = noflat_result_node.tensor
  # Create network with flattening
  a_flat = tn.Node(a_val, backend='symmetric')
  b_flat = tn.Node(b_val, backend='symmetric')
  e1 = tn.connect(a_flat[1], b_flat[3])
  e2 = tn.connect(a_flat[3], b_flat[1])
  e3 = tn.connect(a_flat[2], b_flat[0])
  a_dangling_flat = a_flat[0]
  b_dangling_flat = b_flat[2]
  final_edge = tn.flatten_edges([e1, e2, e3])
  flat_result_node = tn.contract(final_edge)
  flat_result_node.reorder_edges([a_dangling_flat, b_dangling_flat])
  flat_result = flat_result_node.tensor
  flat_result = flat_result.contiguous()
  noflat_result = noflat_result.contiguous()
  np.testing.assert_allclose(flat_result.data, noflat_result.data)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_flatten_trace_consistent_result(dtype, num_charges):
  a_val = get_random_symmetric((5, 5, 5, 5, 5, 5),
                               [False, False, True, True, True, False],
                               num_charges,
                               dtype=dtype)
  a_noflat = tn.Node(a_val, backend='symmetric')
  e1 = tn.connect(a_noflat[0], a_noflat[4])
  e2 = tn.connect(a_noflat[1], a_noflat[2])
  e3 = tn.connect(a_noflat[3], a_noflat[5])
  for edge in [e1, e2, e3]:
    noflat_result = tn.contract(edge).tensor
  # Create network with flattening
  a_flat = tn.Node(a_val, backend='symmetric')
  e1 = tn.connect(a_flat[0], a_flat[4])
  e2 = tn.connect(a_flat[1], a_flat[2])
  e3 = tn.connect(a_flat[3], a_flat[5])
  final_edge = tn.flatten_edges([e1, e2, e3])
  flat_result = tn.contract(final_edge).tensor
  flat_result = flat_result.contiguous()
  noflat_result = noflat_result.contiguous()
  np.testing.assert_allclose(flat_result.data, noflat_result.data)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_flatten_trace_consistent_tensor(dtype, num_charges):
  a_val = get_random_symmetric((5, 5, 5, 5, 5),
                               [False, False, True, True, True],
                               num_charges,
                               dtype=dtype)
  a = tn.Node(a_val, backend='symmetric')
  e1 = tn.connect(a[0], a[4])
  e2 = tn.connect(a[3], a[2])
  tn.flatten_edges([e2, e1])
  tn.check_correct({a})
  # Check expected values.
  a_final = np.reshape(
      np.transpose(a_val.todense(), (1, 2, 0, 3, 4)), (5, 25, 25))
  np.testing.assert_allclose(a.tensor.todense(), a_final)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_contract_between_trace_edges(dtype, num_charges):
  a_val = get_random_symmetric((50, 50), [False, True],
                               num_charges,
                               dtype=dtype)
  final_val = np.trace(a_val.todense())
  a = tn.Node(a_val, backend='symmetric')
  tn.connect(a[0], a[1])
  b = tn.contract_between(a, a)
  tn.check_correct({b})
  np.testing.assert_allclose(b.tensor.todense(), final_val)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2, 3])
def test_at_operator(dtype, num_charges):
  a = tn.Node(
      get_random_symmetric((50, 50), [False, True], num_charges, dtype=dtype),
      backend='symmetric')
  b = tn.Node(
      get_random_symmetric((50, 50), [False, True], num_charges, dtype=dtype),
      backend='symmetric')
  tn.connect(a[1], b[0])
  c = a @ b
  assert isinstance(c, tn.Node)
  np.testing.assert_allclose(c.tensor.todense(),
                             a.tensor.todense() @ b.tensor.todense())
