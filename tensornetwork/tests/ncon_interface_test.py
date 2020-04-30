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
from tensornetwork import BaseNode, Node
from tensornetwork import ncon_interface
from tensornetwork.contractors import greedy


def test_sanity_check(backend):
  t1, t2 = np.ones((2, 2)), np.ones((2, 2))
  result_1 = ncon_interface.ncon([t1, t2], [(-1, 1), (1, -2)], backend=backend)
  np.testing.assert_allclose(result_1, np.ones((2, 2)) * 2)


def test_node_sanity_check(backend):
  t1, t2 = np.ones((2, 2)), np.ones((2, 2))
  n1, n2 = Node(t1, backend=backend), Node(t2, backend=backend)
  result_2 = ncon_interface.ncon([n1, n2], [(-1, 1), (1, -2)], backend=backend)
  np.testing.assert_allclose(result_2.tensor, np.ones((2, 2)) * 2)


def test_return_type(backend):
  t1, t2 = np.ones((2, 2)), np.ones((2, 2))
  n1, n2 = Node(t1, backend=backend), Node(t2, backend=backend)
  result_1 = ncon_interface.ncon([t1, t2], [(-1, 1), (1, -2)], backend=backend)
  result_2 = ncon_interface.ncon([n1, n2], [(-1, 1), (1, -2)], backend=backend)
  result_3 = ncon_interface.ncon([n1, t2], [(-1, 1), (1, -2)], backend=backend)
  assert isinstance(result_1, type(n1.backend.convert_to_tensor(t1)))
  assert isinstance(result_2, BaseNode)
  assert isinstance(result_3, type(n1.backend.convert_to_tensor(t1)))


def test_order_spec(backend):
  a = np.ones((2, 2))
  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                               out_order=[-1, -2],
                               backend=backend)
  np.testing.assert_allclose(result, np.ones((2, 2)) * 2)

  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                               con_order=[1],
                               backend=backend)

  np.testing.assert_allclose(result, np.ones((2, 2)) * 2)

  result = ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                               con_order=[1],
                               out_order=[-1, -2],
                               backend=backend)

  np.testing.assert_allclose(result, np.ones((2, 2)) * 2)


def test_node_order_spec(backend):
  node = Node(np.ones((2, 2)), backend=backend)
  result = ncon_interface.ncon([node, node], [(-1, 1), (1, -2)],
                               out_order=[-1, -2],
                               backend=backend)

  np.testing.assert_allclose(result.tensor, np.ones((2, 2)) * 2)
  result = ncon_interface.ncon([node, node], [(-1, 1), (1, -2)],
                               con_order=[1],
                               backend=backend)

  np.testing.assert_allclose(result.tensor, np.ones((2, 2)) * 2)

  result = ncon_interface.ncon([node, node], [(-1, 1), (1, -2)],
                               con_order=[1],
                               out_order=[-1, -2],
                               backend=backend)

  np.testing.assert_allclose(result.tensor, np.ones((2, 2)) * 2)


def test_order_spec_noninteger(backend):
  a = np.ones((2, 2))
  result = ncon_interface.ncon([a, a], [('o1', 'i'), ('i', 'o2')],
                               con_order=['i'],
                               out_order=['o1', 'o2'],
                               backend=backend)
  np.testing.assert_allclose(result, np.ones((2, 2)) * 2)


def test_node_order_spec_noninteger(backend):
  node = Node(np.ones((2, 2)), backend=backend)
  result = ncon_interface.ncon([node, node], [('o1', 'i'), ('i', 'o2')],
                               con_order=['i'],
                               out_order=['o1', 'o2'],
                               backend=backend)
  np.testing.assert_allclose(result.tensor, np.ones((2, 2)) * 2)


def test_invalid_network(backend):
  a = np.ones((2, 2))
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1), (1, 2)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 2)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (3, 1)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 0.1)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 't')], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(0, 1), (1, 0)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1,), (1, 2)], backend=backend)


def test_node_invalid_network(backend):
  a = Node(np.ones((2, 2)), backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1), (1, 2)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 2)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (3, 1)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 0.1)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 't')], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(0, 1), (1, 0)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1,), (1, 2)], backend=backend)


def test_invalid_order(backend):
  a = np.ones((2, 2))
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=[2, 3],
                        backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        out_order=[-1],
                        backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [('i1', 'i2'), ('i1', 'i2')],
                        con_order=['i1'],
                        out_order=[],
                        backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [('i1', 'i2'), ('i1', 'i2')],
                        con_order=['i1', 'i2'],
                        out_order=['i1'],
                        backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [('i1', 'i2'), ('i1', 'i2')],
                        con_order=['i1', 'i1', 'i2'],
                        out_order=[],
                        backend=backend)


def test_node_invalid_order(backend):
  a = Node(np.ones((2, 2)), backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=[2, 3],
                        backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        out_order=[-1],
                        backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [('i1', 'i2'), ('i1', 'i2')],
                        con_order=['i1'],
                        out_order=[],
                        backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [('i1', 'i2'), ('i1', 'i2')],
                        con_order=['i1', 'i2'],
                        out_order=['i1'],
                        backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [('i1', 'i2'), ('i1', 'i2')],
                        con_order=['i1', 'i1', 'i2'],
                        out_order=[],
                        backend=backend)


def test_out_of_order_contraction(backend):
  a = np.ones((2, 2, 2))
  with pytest.warns(UserWarning, match='Suboptimal ordering'):
    ncon_interface.ncon([a, a, a], [(-1, 1, 3), (1, 3, 2), (2, -2, -3)],
                        backend=backend)


def test_node_out_of_order_contraction(backend):
  a = Node(np.ones((2, 2, 2)), backend=backend)
  with pytest.warns(UserWarning, match='Suboptimal ordering'):
    ncon_interface.ncon([a, a, a], [(-1, 1, 3), (1, 3, 2), (2, -2, -3)],
                        backend=backend)


def test_output_order(backend):
  a = np.random.randn(2, 2)
  res = ncon_interface.ncon([a], [(-2, -1)], backend=backend)
  np.testing.assert_allclose(res, a.transpose())


def test_node_output_order(backend):
  t = np.random.randn(2, 2)
  a = Node(t, backend=backend)
  res = ncon_interface.ncon([a], [(-2, -1)], backend=backend)
  np.testing.assert_allclose(res.tensor, t.transpose())


def test_outer_product(backend):
  if backend == "jax":
    pytest.skip("Jax outer product support is currently broken.")
  a = np.array([1, 2, 3])
  b = np.array([1, 2])
  res = ncon_interface.ncon([a, b], [(-1,), (-2,)], backend=backend)
  np.testing.assert_allclose(res, np.kron(a, b).reshape((3, 2)))

  res = ncon_interface.ncon([a, a, a, a], [(1,), (1,), (2,), (2,)],
                            backend=backend)
  np.testing.assert_allclose(res, 196)


def test_node_outer_product(backend):
  if backend == "jax":
    pytest.skip("Jax outer product support is currently broken.")

  t1 = np.array([1, 2, 3])
  t2 = np.array([1, 2])
  a = Node(t1, backend=backend)
  b = Node(t2, backend=backend)
  res = ncon_interface.ncon([a, b], [(-1,), (-2,)], backend=backend)
  np.testing.assert_allclose(res.tensor, np.kron(t1, t2).reshape((3, 2)))

  res = ncon_interface.ncon([a, a, a, a], [(1,), (1,), (2,), (2,)],
                            backend=backend)
  np.testing.assert_allclose(res.tensor, 196)


def test_trace(backend):
  a = np.ones((2, 2))
  res = ncon_interface.ncon([a], [(1, 1)], backend=backend)
  np.testing.assert_allclose(res, 2)


def test_node_trace(backend):
  a = Node(np.ones((2, 2)), backend=backend)
  res = ncon_interface.ncon([a], [(1, 1)], backend=backend)
  np.testing.assert_allclose(res.tensor, 2)


def test_small_matmul(backend):
  a = np.random.randn(2, 2)
  b = np.random.randn(2, 2)
  res = ncon_interface.ncon([a, b], [(1, -1), (1, -2)], backend=backend)
  np.testing.assert_allclose(res, a.transpose() @ b)


def test_node_small_matmul(backend):
  t1 = np.random.randn(2, 2)
  t2 = np.random.randn(2, 2)

  a = Node(t1, backend=backend)
  b = Node(t2, backend=backend)
  res = ncon_interface.ncon([a, b], [(1, -1), (1, -2)], backend=backend)
  np.testing.assert_allclose(res.tensor, t1.transpose() @ t2)


def test_contraction(backend):
  a = np.random.randn(2, 2, 2)
  res = ncon_interface.ncon([a, a, a], [(-1, 1, 2), (1, 2, 3), (3, -2, -3)],
                            backend=backend)
  res_np = a.reshape((2, 4)) @ a.reshape((4, 2)) @ a.reshape((2, 4))
  res_np = res_np.reshape((2, 2, 2))
  np.testing.assert_allclose(res, res_np)


def test_node_contraction(backend):
  tensor = np.random.randn(2, 2, 2)
  a = Node(tensor, backend=backend)
  res = ncon_interface.ncon([a, a, a], [(-1, 1, 2), (1, 2, 3), (3, -2, -3)],
                            backend=backend)
  res_np = tensor.reshape((2, 4)) @ tensor.reshape((4, 2)) @ tensor.reshape(
      (2, 4))
  res_np = res_np.reshape((2, 2, 2))
  np.testing.assert_allclose(res.tensor, res_np)


def test_backend_network(backend):
  a = np.random.randn(2, 2, 2)
  nodes, _, out_edges = ncon_interface.ncon_network([a, a, a], [(-1, 1, 2),
                                                                (1, 2, 3),
                                                                (3, -2, -3)],
                                                    backend=backend)
  res = greedy(nodes, out_edges).tensor
  res_np = a.reshape((2, 4)) @ a.reshape((4, 2)) @ a.reshape((2, 4))
  res_np = res_np.reshape((2, 2, 2))
  np.testing.assert_allclose(res, res_np)
