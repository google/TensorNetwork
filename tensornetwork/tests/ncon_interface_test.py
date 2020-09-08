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
import pytest
import numpy as np
from tensornetwork.tensor import Tensor
from tensornetwork import ncon_interface

from tensornetwork.ncon_interface import (_get_cont_out_labels,
                                          _canonicalize_network_structure)
from tensornetwork.backends.backend_factory import get_backend
from tensornetwork.contractors import greedy


@pytest.fixture(
    name="backend",
    params=[
        'numpy',
        get_backend('numpy'), 'jax',
        get_backend('jax'), 'pytorch',
        get_backend('pytorch'), 'tensorflow',
        get_backend('tensorflow')
    ])
def backends(request):
  return request.param


def test_sanity_check(backend):
  np.random.seed(10)
  t1, t2 = np.random.rand(2, 2), np.random.rand(2, 2)
  result = ncon_interface.ncon([t1, t2], [(-1, 1), (1, -2)], backend=backend)
  np.testing.assert_allclose(result, t1 @ t2)


def test_node_sanity_check(backend):
  np.random.seed(10)
  t1, t2 = np.random.rand(2, 2), np.random.rand(2, 2)
  n1, n2 = Tensor(t1, backend=backend), Tensor(t2, backend=backend)
  result = ncon_interface.ncon([n1, n2], [(-1, 1), (1, -2)], backend=backend)
  np.testing.assert_allclose(result.array, t1 @ t2)


def test_return_type(backend):
  t1, t2 = np.ones((2, 2)), np.ones((2, 2))
  n1, n2 = Tensor(t1, backend=backend), Tensor(t2, backend=backend)
  result_1 = ncon_interface.ncon([t1, t2], [(-1, 1), (1, -2)], backend=backend)
  result_2 = ncon_interface.ncon([n1, n2], [(-1, 1), (1, -2)], backend=backend)
  result_3 = ncon_interface.ncon([n1, t2], [(-1, 1), (1, -2)], backend=backend)
  assert isinstance(result_1, type(n1.backend.convert_to_tensor(t1)))
  assert isinstance(result_2, Tensor)
  assert isinstance(result_3, type(n1.backend.convert_to_tensor(t1)))


def test_order_spec(backend):
  np.random.seed(10)
  a = np.random.rand(2, 2)
  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                               out_order=[-1, -2],
                               backend=backend)
  np.testing.assert_allclose(result, a @ a)

  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                               con_order=[1],
                               backend=backend)

  np.testing.assert_allclose(result, a @ a)

  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                               con_order=[1],
                               out_order=[-1, -2],
                               backend=backend)
  np.testing.assert_allclose(result, a @ a)

  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                               con_order=[1],
                               out_order=[-2, -1],
                               backend=backend)
  np.testing.assert_allclose(result, (a @ a).T)


def test_node_order_spec(backend):
  np.random.seed(10)
  a = np.random.rand(2, 2)
  node = Tensor(a, backend=backend)
  result = ncon_interface.ncon([node, node], [(-1, 1), (1, -2)],
                               out_order=[-1, -2],
                               backend=backend)

  np.testing.assert_allclose(result.array, a @ a)
  result = ncon_interface.ncon([node, node], [(-1, 1), (1, -2)],
                               con_order=[1],
                               backend=backend)
  np.testing.assert_allclose(result.array, a @ a)

  result = ncon_interface.ncon([node, node], [(-1, 1), (1, -2)],
                               con_order=[1],
                               out_order=[-1, -2],
                               backend=backend)
  np.testing.assert_allclose(result.array, a @ a)

  result = ncon_interface.ncon([node, node], [(-1, 1), (1, -2)],
                               con_order=[1],
                               out_order=[-2, -1],
                               backend=backend)
  np.testing.assert_allclose(result.array, (a @ a).T)


def test_order_spec_noninteger(backend):
  np.random.seed(10)
  a = np.random.rand(2, 2)
  exp = a @ a
  result = ncon_interface.ncon([a, a], [('-o1', 'i'), ('i', '-o2')],
                               con_order=['i'],
                               out_order=['-o1', '-o2'],
                               backend=backend)

  np.testing.assert_allclose(result, exp)
  result = ncon_interface.ncon([a, a], [('-o1', 'i'), ('i', '-o2')],
                               con_order=['i'],
                               out_order=['-o2', '-o1'],
                               backend=backend)

  np.testing.assert_allclose(result, exp.T)


def test_node_order_spec_noninteger(backend):
  np.random.seed(10)
  a = np.random.rand(2, 2)
  exp = a @ a
  node = Tensor(a, backend=backend)
  result = ncon_interface.ncon([node, node], [('-o1', 'i'), ('i', '-o2')],
                               con_order=['i'],
                               out_order=['-o1', '-o2'],
                               backend=backend)
  np.testing.assert_allclose(result.array, exp)
  result = ncon_interface.ncon([node, node], [('-o1', 'i'), ('i', '-o2')],
                               con_order=['i'],
                               out_order=['-o2', '-o1'],
                               backend=backend)
  np.testing.assert_allclose(result.array, exp.T)


def test_output_order(backend):
  np.random.seed(10)
  a = np.random.randn(2, 2)
  res = ncon_interface.ncon([a], [(-2, -1)], backend=backend)
  np.testing.assert_allclose(res, a.transpose())


def test_node_output_order(backend):
  np.random.seed(10)
  t = np.random.randn(2, 2)
  a = Tensor(t, backend=backend)
  res = ncon_interface.ncon([a], [(-2, -1)], backend=backend)
  np.testing.assert_allclose(res.array, t.transpose())


def test_outer_product_1(backend):
  a = np.array([1, 2, 3])
  b = np.array([1, 2])
  res = ncon_interface.ncon([a, b], [(-1,), (-2,)], backend=backend)
  np.testing.assert_allclose(res, np.kron(a, b).reshape((3, 2)))

  res = ncon_interface.ncon([a, a, a, a], [(1,), (1,), (2,), (2,)],
                            backend=backend)
  np.testing.assert_allclose(res, 196)


def test_outer_product_1_mixed_labels(backend):
  a = np.array([1, 2, 3])
  b = np.array([1, 2])
  res = ncon_interface.ncon([a, b], [('-hi',), ('-ho',)], backend=backend)
  np.testing.assert_allclose(res, np.kron(a, b).reshape((3, 2)))

  res = ncon_interface.ncon([a, a, a, a], [('hi',), ('hi',), ('ho',), ('ho',)],
                            backend=backend)
  np.testing.assert_allclose(res, 196)


def test_outer_product_2(backend):
  np.random.seed(10)
  a = np.random.rand(10, 100)
  b = np.random.rand(8)
  res = ncon_interface.ncon([a, b], [(-1, -2), (-3,)],
                            out_order=[-2, -1, -3],
                            backend=backend)
  exp = np.einsum('ij,k->jik', a, b)
  np.testing.assert_allclose(res, exp)

def test_outer_product_2_mixed_labels(backend):
  np.random.seed(10)
  a = np.random.rand(10, 100)
  b = np.random.rand(8)
  res = ncon_interface.ncon([a, b], [(-1, '-hi'), ('-ho',)],
                            out_order=['-hi', -1, '-ho'],
                            backend=backend)
  exp = np.einsum('ij,k->jik', a, b)
  np.testing.assert_allclose(res, exp)


def test_node_outer_product_1(backend):
  t1 = np.array([1, 2, 3])
  t2 = np.array([1, 2])
  a = Tensor(t1, backend=backend)
  b = Tensor(t2, backend=backend)
  res = ncon_interface.ncon([a, b], [(-1,), (-2,)], backend=backend)
  np.testing.assert_allclose(res.array, np.kron(t1, t2).reshape((3, 2)))

  res = ncon_interface.ncon([a, a, a, a], [(1,), (1,), (2,), (2,)],
                            backend=backend)
  np.testing.assert_allclose(res.array, 196)


def test_node_outer_product_1_mixed_labels(backend):
  t1 = np.array([1, 2, 3])
  t2 = np.array([1, 2])
  a = Tensor(t1, backend=backend)
  b = Tensor(t2, backend=backend)
  res = ncon_interface.ncon([a, b], [('-hi',), ('-ho',)], backend=backend)
  np.testing.assert_allclose(res.array, np.kron(t1, t2).reshape((3, 2)))

  res = ncon_interface.ncon([a, a, a, a], [('hi',), ('hi',), ('ho',), ('ho',)],
                            backend=backend)
  np.testing.assert_allclose(res.array, 196)


def test_node_outer_product_2(backend):
  np.random.seed(10)
  t1 = np.random.rand(10, 100)
  t2 = np.random.rand(8)
  a = Tensor(t1, backend=backend)
  b = Tensor(t2, backend=backend)

  res = ncon_interface.ncon([a, b], [(-1, -2), (-3,)],
                            out_order=[-2, -1, -3],
                            backend=backend)
  exp = np.einsum('ij,k->jik', t1, t2)
  np.testing.assert_allclose(res.array, exp)


def test_node_outer_product_2_mixed_labels(backend):
  np.random.seed(10)
  t1 = np.random.rand(10, 100)
  t2 = np.random.rand(8)
  a = Tensor(t1, backend=backend)
  b = Tensor(t2, backend=backend)

  res = ncon_interface.ncon([a, b], [(-1, '-hi'), ('-ho',)],
                            out_order=['-hi', -1, '-ho'],
                            backend=backend)
  exp = np.einsum('ij,k->jik', t1, t2)
  np.testing.assert_allclose(res.array, exp)


def test_trace(backend):
  a = np.ones((2, 2))
  res = ncon_interface.ncon([a], [(1, 1)], backend=backend)
  np.testing.assert_allclose(res, 2)


def test_trace_str_labels(backend):
  a = np.ones((2, 2))
  res = ncon_interface.ncon([a], [('hi', 'hi')], backend=backend)
  np.testing.assert_allclose(res, 2)


def test_node_trace(backend):
  a = Tensor(np.ones((2, 2)), backend=backend)
  res = ncon_interface.ncon([a], [(1, 1)], backend=backend)
  np.testing.assert_allclose(res.array, 2)


def test_node_trace_str_labels(backend):
  a = Tensor(np.ones((2, 2)), backend=backend)
  res = ncon_interface.ncon([a], [('hi', 'hi')], backend=backend)
  np.testing.assert_allclose(res.array, 2)


def test_small_matmul(backend):
  np.random.seed(10)
  a = np.random.randn(2, 2)
  b = np.random.randn(2, 2)
  res = ncon_interface.ncon([a, b], [(1, -1), (1, -2)], backend=backend)
  np.testing.assert_allclose(res, a.transpose() @ b)


def test_small_matmul_mixed_labels(backend):
  np.random.seed(10)
  a = np.random.randn(2, 2)
  b = np.random.randn(2, 2)
  res = ncon_interface.ncon([a, b], [('hi', -1), ('hi', '-ho')],
                            backend=backend)
  np.testing.assert_allclose(res, a.transpose() @ b)


def test_node_small_matmul(backend):
  np.random.seed(10)
  t1 = np.random.randn(2, 2)
  t2 = np.random.randn(2, 2)

  a = Tensor(t1, backend=backend)
  b = Tensor(t2, backend=backend)
  res = ncon_interface.ncon([a, b], [(1, -1), (1, -2)], backend=backend)
  np.testing.assert_allclose(res.array, t1.transpose() @ t2)


def test_node_small_matmul_mixed_labels(backend):
  np.random.seed(10)
  t1 = np.random.randn(2, 2)
  t2 = np.random.randn(2, 2)

  a = Tensor(t1, backend=backend)
  b = Tensor(t2, backend=backend)

  res = ncon_interface.ncon([a, b], [('hi', -1), ('hi', '-ho')],
                            backend=backend)
  np.testing.assert_allclose(res.array, t1.transpose() @ t2)


def test_contraction(backend):
  np.random.seed(10)
  a = np.random.randn(2, 2, 2)
  res = ncon_interface.ncon([a, a, a], [(-1, 1, 2), (1, 2, 3), (3, -2, -3)],
                            backend=backend)
  res_np = a.reshape((2, 4)) @ a.reshape((4, 2)) @ a.reshape((2, 4))
  res_np = res_np.reshape((2, 2, 2))
  np.testing.assert_allclose(res, res_np)


def test_contraction_mixed_labels(backend):
  np.random.seed(10)
  a = np.random.randn(2, 2, 2)
  res = ncon_interface.ncon([a, a, a], [(-1, 'rick', 2), ('rick', 2, 'morty'),
                                        ('morty', -2, -3)],
                            backend=backend)
  res_np = a.reshape((2, 4)) @ a.reshape((4, 2)) @ a.reshape((2, 4))
  res_np = res_np.reshape((2, 2, 2))
  np.testing.assert_allclose(res, res_np)


def test_node_contraction(backend):
  np.random.seed(10)
  tensor = np.random.randn(2, 2, 2)
  a = Tensor(tensor, backend=backend)
  res = ncon_interface.ncon([a, a, a], [(-1, 1, 2), (1, 2, 3), (3, -2, -3)],
                            backend=backend)
  res_np = tensor.reshape((2, 4)) @ tensor.reshape((4, 2)) @ tensor.reshape(
      (2, 4))
  res_np = res_np.reshape((2, 2, 2))
  np.testing.assert_allclose(res.array, res_np)


def test_node_contraction_mixed_labels(backend):
  np.random.seed(10)
  tensor = np.random.randn(2, 2, 2)
  a = Tensor(tensor, backend=backend)
  res = ncon_interface.ncon([a, a, a], [(-1, 'rick', 2), ('rick', 2, 'morty'),
                                        ('morty', -2, -3)],
                            backend=backend)
  res_np = tensor.reshape((2, 4)) @ tensor.reshape((4, 2)) @ tensor.reshape(
      (2, 4))
  res_np = res_np.reshape((2, 2, 2))
  np.testing.assert_allclose(res.array, res_np)


def check(exp, actual):
  for e, a in zip(exp, actual):
    assert e == a


def test_get_cont_out_labels_1():
  network_structure = [[-1, 2, '3', '33', '4', 3, '-33', '-5'],
                       ['-4', -2, '-3', '3', '33', '-5', '4', 2, 3, 6, 'hello']]
  # pylint: disable=line-too-long
  int_cont_labels, str_cont_labels, int_out_labels, str_out_labels = _get_cont_out_labels(
      network_structure)
  exp_int_cont_labels = [2, 3]
  exp_str_cont_labels = ['3', '33', '4']
  exp_int_out_labels = [-1, -2]
  exp_str_out_labels = ['-3', '-33', '-4']

  check(exp_int_cont_labels, int_cont_labels)
  check(exp_str_cont_labels, str_cont_labels)
  check(exp_int_out_labels, int_out_labels)
  check(exp_str_out_labels, str_out_labels)


def test_get_cont_out_labels_2():
  network_structure = [[
      -1, 2, '3', '33', '4', 3, '-33', '-5', 5, -3, '-6', '5'
  ], ['-4', -2, '-3', '3', '33', '-5', '4', '5', 2, 3, 5, -3, 11],
                       [5, -3, '-6', '5', 'ricksanchez']]
  # pylint: disable=line-too-long
  int_cont_labels, str_cont_labels, int_out_labels, str_out_labels = _get_cont_out_labels(
      network_structure)
  exp_int_cont_labels = [2, 3]
  exp_str_cont_labels = ['3', '33', '4']
  exp_int_out_labels = [-1, -2]
  exp_str_out_labels = ['-3', '-33', '-4']

  check(exp_int_cont_labels, int_cont_labels)
  check(exp_str_cont_labels, str_cont_labels)
  check(exp_int_out_labels, int_out_labels)
  check(exp_str_out_labels, str_out_labels)

def test_canonicalize_network_structure():
  network_structure = [[-3, 10, 15, '-5'], [-5, -23, 8, '66', '60'],
                       [3, 4, 5, '6']]
  unique = [-3, 10, 15, '-5', -5, -23, 8, '66', '60', 3, 4, 5]
  labels = [-1, 5, 6, -4, -2, -3, 4, 9, 8, 1, 2, 3, 7]

  exp = [[-1, 5, 6, -4], [-2, -3, 4, 9, 8], [1, 2, 3, 7]]
  actual, mapping = _canonicalize_network_structure(network_structure)
  for u, l in zip(unique, labels):
    assert mapping[u] == l

  for a, b in zip(actual, exp):
    np.testing.assert_allclose(a, b)


def test_batched_outer_product(backend):
  a = np.random.rand(10, 100)
  b = np.random.rand(8, 100)
  res = ncon_interface.ncon([a, b], [(-1, -3), (-2, -3)], backend=backend)
  exp = np.einsum('ik,jk->ijk', a, b)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b], [(-1, -3), (-2, -3)],
                            out_order=[-2, -1, -3],
                            backend=backend)
  exp = np.einsum('ik,jk->jik', a, b)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b], [(-1, -3), (-2, -3)],
                            out_order=[-2, -3, -1],
                            backend=backend)
  exp = np.einsum('ik,jk->jki', a, b)


def test_partial_traces(backend):
  np.random.seed(10)
  a = np.random.rand(4, 4, 4, 4)
  res = ncon_interface.ncon([a, a], [(-1, 1, 1, 3), (2, -2, 2, 3)],
                            backend=backend)
  t1 = np.trace(a, axis1=1, axis2=2)
  t2 = np.trace(a, axis1=0, axis2=2)
  exp = np.tensordot(t1, t2, ([1], [1]))
  np.testing.assert_allclose(res, exp)


def test_batched_traces(backend):
  np.random.seed(10)
  a = np.random.randn(10, 10, 100)
  res = ncon_interface.ncon([a, a], [(1, 1, -1), (2, 2, -1)], backend=backend)
  exp = np.einsum('iik,jjk->k', a, a)
  np.testing.assert_allclose(res, exp)


def test_batched_matmul_1(backend):
  np.random.seed(10)
  a = np.random.randn(10, 11, 100)
  b = np.random.randn(11, 100, 12)
  res = ncon_interface.ncon([a, b], [(-1, 1, -3), (1, -3, -2)], backend=backend)
  exp = np.einsum('ijk,jkm->imk', a, b)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b], [(-1, 1, -3), (1, -3, -2)],
                            out_order=[-2, -1, -3],
                            backend=backend)
  exp = np.einsum('ijk,jkm->mik', a, b)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b], [(-1, 1, -3), (1, -3, -2)],
                            out_order=[-3, -2, -1],
                            backend=backend)
  exp = np.einsum('ijk,jkm->kmi', a, b)
  np.testing.assert_allclose(res, exp)


def test_batched_matmul_2(backend):
  np.random.seed(10)
  batchsize = 10
  a = np.random.randn(2, 4, 4, batchsize)
  b = np.random.randn(4, 3, batchsize, 5)
  c = np.random.randn(batchsize, 5, 4)
  res = ncon_interface.ncon([a, b, c], [(-1, 1, 2, -2), (1, -3, -2, 3),
                                        (-2, 3, 2)],
                            backend=backend)
  exp = np.einsum('abck,bdke,kec->akd', a, b, c)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b, c], [(-1, 1, 2, -2), (1, -3, -2, 3),
                                        (-2, 3, 2)],
                            out_order=[-3, -1, -2],
                            backend=backend)
  exp = np.einsum('abck,bdke,kec->dak', a, b, c)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b, c], [(-1, 1, 2, -2), (1, -3, -2, 3),
                                        (-2, 3, 2)],
                            out_order=[-2, -1, -3],
                            backend=backend)
  exp = np.einsum('abck,bdke,kec->kad', a, b, c)
  np.testing.assert_allclose(res, exp)


def test_batched_matmul_3(backend):
  np.random.seed(10)
  batchsize = 10
  a = np.random.randn(2, 4, 4, batchsize)
  b = np.random.randn(4, 3, batchsize, 5)
  c = np.random.randn(batchsize, 5, 4)
  res = ncon_interface.ncon([a, b, c], [(-1, 1, 2, 4), (1, -2, 4, 3),
                                        (4, 3, 2)],
                            backend=backend)
  exp = np.einsum('abck,bdke,kec->ad', a, b, c)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b, c], [(-1, 1, 2, 4), (1, -2, 4, 3),
                                        (4, 3, 2)],
                            out_order=[-2, -1],
                            backend=backend)
  exp = np.einsum('abck,bdke,kec->da', a, b, c)
  np.testing.assert_allclose(res, exp)


def test_multiple_batched_matmul_1(backend):
  np.random.seed(10)
  batchsize1 = 10
  batchsize2 = 12
  a = np.random.randn(2, 4, 4, batchsize1)
  b = np.random.randn(4, 3, batchsize1, 5)
  c = np.random.randn(batchsize2, 5, 4)
  e = np.random.randn(batchsize2, 3, 6)

  res = ncon_interface.ncon([a, b, c, e], [(-1, 1, 2, -2), (1, 3, -2, 4),
                                           (-3, 4, 2), (-3, 3, -4)],
                            backend=backend)
  exp = np.einsum('abck,bdke,lec,ldf->aklf', a, b, c, e)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b, c, e], [(-1, 1, 2, -2), (1, 3, -2, 4),
                                           (-3, 4, 2), (-3, 3, -4)],
                            out_order=[-3, -1, -2, -4],
                            backend=backend)
  exp = np.einsum('abck,bdke,lec,ldf->lakf', a, b, c, e)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b, c, e], [(-1, 1, 2, -2), (1, 3, -2, 4),
                                           (-3, 4, 2), (-3, 3, -4)],
                            out_order=[-3, -2, -4, -1],
                            backend=backend)
  exp = np.einsum('abck,bdke,lec,ldf->lkfa', a, b, c, e)
  np.testing.assert_allclose(res, exp)


def test_multiple_batched_matmul_2(backend):
  np.random.seed(10)
  batchsize1 = 10
  batchsize2 = 12
  a = np.random.randn(2, 2, batchsize1, 2)
  b = np.random.randn(2, batchsize1, 2, 2)
  c = np.random.randn(batchsize1, 2, 2, 2)
  e = np.random.randn(2, 2, batchsize2, 2)
  f = np.random.randn(2, batchsize2, 2, 2)
  g = np.random.randn(2, batchsize2, 2, 2)

  res = ncon_interface.ncon([a, b, c, e, f, g], [(1, 2, -5, 3), (-1, -5, 2, 1),
                                                 (-5, -3, 7, 6), (4, 5, -6, 3),
                                                 (-2, -6, 6, -4),
                                                 (7, -6, 5, 4)],
                            backend=backend)
  exp = np.einsum('abtc,etba,tfgh,ijqc,kqhl,gqji->ekfltq', a, b, c, e, f, g)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b, c, e, f, g], [(1, 2, -5, 3), (-1, -5, 2, 1),
                                                 (-5, -3, 7, 6), (4, 5, -6, 3),
                                                 (-2, -6, 6, -4),
                                                 (7, -6, 5, 4)],
                            out_order=[-5, -6, -2, -1, -4, -3],
                            backend=backend)
  exp = np.einsum('abtc,etba,tfgh,ijqc,kqhl,gqji->tqkelf', a, b, c, e, f, g)
  np.testing.assert_allclose(res, exp)


def test_multiple_batched_matmul_3(backend):
  np.random.seed(10)
  batchsize1 = 10
  batchsize2 = 12
  a = np.random.randn(2, 2, batchsize1, 2)
  b = np.random.randn(2, batchsize1, 2, 2)
  c = np.random.randn(batchsize1, 2, 2, 2)
  e = np.random.randn(2, 2, batchsize2, 2)
  f = np.random.randn(2, batchsize2, 2, 2)
  g = np.random.randn(2, batchsize2, 2, 2)

  res = ncon_interface.ncon([a, b, c, e, f, g], [(1, 2, 8, 3), (-1, 8, 2, 1),
                                                 (8, -3, 7, 6), (4, 5, -6, 3),
                                                 (-2, -6, 6, -4),
                                                 (7, -6, 5, 4)],
                            backend=backend)

  exp = np.einsum('abtc,etba,tfgh,ijqc,kqhl,gqji->ekflq', a, b, c, e, f, g)
  np.testing.assert_allclose(res, exp)

  res = ncon_interface.ncon([a, b, c, e, f, g], [(1, 2, 8, 3), (-1, 8, 2, 1),
                                                 (8, -3, 7, 6), (4, 5, -6, 3),
                                                 (-2, -6, 6, -4),
                                                 (7, -6, 5, 4)],
                            out_order=[-6, -2, -1, -4, -3],
                            backend=backend)

  exp = np.einsum('abtc,etba,tfgh,ijqc,kqhl,gqji->qkelf', a, b, c, e, f, g)
  np.testing.assert_allclose(res, exp)


def run_tests(a, b, c, backend):

  with pytest.raises(
      ValueError,
      match=r"only alphanumeric values allowed for string labels, "
      r"found \['henry@', 'megan!'\]"):
    ncon_interface.ncon([a, a], [('megan!', 'henry@'), ('henry@', 'megan!')],
                        backend=backend)

  with pytest.raises(
      ValueError,
      match="number of tensors does not "
      "match the number of network connections."):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1), (1, 2)], backend=backend)
  with pytest.raises(
      ValueError,
      match="number of indices does not match "
      "number of labels on tensor 0."):
    ncon_interface.ncon([a, a], [(1,), (1, 2)], backend=backend)

  with pytest.raises(
      ValueError,
      match="only nonzero values are allowed to "
      "specify network structure."):
    ncon_interface.ncon([a, a], [(0, 1), (1, 0)], backend=backend)

  with pytest.raises(
      ValueError,
      match=r"all number type labels in `con_order` have "
      r"to be positive, found \[-1\]"):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=[-1, 2],
                        backend=backend)
  with pytest.raises(
      ValueError,
      match=r"all string type labels in `con_order` "
      r"must be unhyphenized, found \['-hi'\]"):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=['-hi', 2],
                        backend=backend)

  with pytest.raises(
      ValueError,
      match=r"labels \['hi', 1\] appear more than once in `con_order`."):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=['hi', 'hi', 1, 1],
                        backend=backend)
  with pytest.raises(
      ValueError,
      match=r"`con_order = \[3, 4, 5\] is not a valid "
      r"contraction order for contracted labels \[1, 2\]"):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=[3, 4, 5],
                        backend=backend)

  with pytest.raises(
      ValueError,
      match=r"labels \[3, 4\] in `con_order` "
      r"do not appear as contracted labels in `network_structure`."):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=[3, 4],
                        backend=backend)

  with pytest.raises(
      ValueError,
      match=r"all number type labels in `out_order` have "
      r"to be negative, found \[2\]"):
    ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                        out_order=[-1, 2],
                        backend=backend)
  with pytest.raises(
      ValueError,
      match=r"all string type labels in `out_order` "
      r"have to be hyphenized, found \['hi'\]"):
    ncon_interface.ncon([a, a], [('-hi', 1), (1, -2)],
                        out_order=['hi', -2],
                        backend=backend)

  with pytest.raises(
      ValueError,
      match=r"labels \['-hi', -1\] appear more than once in `out_order`."):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        out_order=['-hi', '-hi', -1, -1],
                        backend=backend)

  with pytest.raises(
      ValueError,
      match=r"`out_order` = \[-1, -2, -3\] is not a valid output"
      r" order for open labels \[-1, -2\]"):
    ncon_interface.ncon([a, a], [(-1, 2), (2, -2)],
                        out_order=[-1, -2, -3],
                        backend=backend)

  with pytest.raises(
      ValueError,
      match=r"labels \[-3, -4\] in `out_order` "
      r"do not appear in `network_structure`."):
    ncon_interface.ncon([a, a], [(-1, 2), (2, -2)],
                        out_order=[-3, -4],
                        backend=backend)

  with pytest.raises(
      ValueError,
      match=r"tensor dimensions for labels \[2, 4\] "
      r"are mismatching"):
    ncon_interface.ncon([b, c], [(1, 2, 3, 4), (1, 2, 3, 4)], backend=backend)


def test_invalid_network(backend):
  a = np.ones((2, 2))
  b = np.ones((2, 3, 4, 2))
  c = np.ones((2, 4, 4, 3))
  run_tests(a, b, c, backend)


def test_node_invalid_network(backend):
  a = np.ones((2, 2))
  b = np.ones((2, 3, 4, 2))
  c = np.ones((2, 4, 4, 3))
  run_tests(
      Tensor(a, backend=backend), Tensor(b, backend=backend),
      Tensor(c, backend=backend), backend)


def test_infinite_loop(backend):
  a = np.ones((2, 2, 2))
  b = np.ones((2, 2))
  with pytest.raises(
      ValueError,
      match=r"ncon seems stuck in an infinite loop. \n"
      r"Please check if `con_order` = \[3\] is a valid "
      r"contraction order for \n"
      r"`network_structure` = \[\[3, 1, 2\], \[3, 1\], \[3, 2\]\]"):
    ncon_interface.ncon([a, b, b], [[3, 1, 2], [3, 1], [3, 2]],
                        con_order=[3],
                        check_network=False,
                        backend=backend)
