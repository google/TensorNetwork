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
from tensornetwork import AbstractNode, Node
from tensornetwork import ncon_interface
from tensornetwork.ncon_interface import _get_cont_out_labels
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
  assert isinstance(result_2, AbstractNode)
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
      match=r"labels \[3, 4\] in `con_order` "
      r"do not appear as contracted labels in `network_structure`."):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=[3, 4],
                        backend=backend)
  with pytest.raises(
      ValueError, match=r"label 2"
      " appears more than once in `con_order`."):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        con_order=[2, 2],
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
      match=r"`con_order = \[-1, 2\] "
      r"is not a valid contraction order for contracted labels \[2\]"):
    ncon_interface.ncon([a, a], [(-1, 2), (2, -2)],
                        con_order=[-1, 2],
                        backend=backend)
  with pytest.raises(
      ValueError,
      match=r"`out_order` = \[2, 2\] is not a valid output"
      r" order for open labels \[\]"):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1)],
                        out_order=[2, 2],
                        backend=backend)
  with pytest.raises(
      ValueError,
      match=r"labels \[-3\] in `out_order` do not "
      r"appear in `network_structure`."):
    ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                        out_order=[-3, -1],
                        backend=backend)
  with pytest.raises(
      ValueError,
      match=r"labels \[1\] in `out_order` appear more "
      r"than once in `network_structure`."):
    ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                        out_order=[1, -1],
                        backend=backend)
  with pytest.raises(
      ValueError, match=r"label -1 appears more than once in `out_order`."):
    ncon_interface.ncon([a, a], [(-1, 1), (1, -2)],
                        out_order=[-1, -1],
                        backend=backend)
  with pytest.raises(
      ValueError,
      match=r'labels \[2\] appear more than twice in `network_structure`.'):
    ncon_interface.ncon([a, a], [(1, 2), (2, 2)], backend=backend)
  with pytest.raises(
      ValueError,
      match=r"open integer labels have to be negative "
      r"integers, found \[3, 2\]"):
    ncon_interface.ncon([a, a], [(1, 2), (3, 1)], backend=backend)
  with pytest.raises(
      ValueError,
      match="only nonzero values are allowed to "
      "specify network structure."):
    ncon_interface.ncon([a, a], [(0, 1), (1, 0)], backend=backend)
  with pytest.raises(
      ValueError,
      match=r"open string labels have to be prepended with '-'; "
      r"found \['1', '2'\]"):
    ncon_interface.ncon([a, a], [('1', 1), (1, '2')], backend=backend)
  with pytest.raises(
      ValueError,
      match=r"open integer labels have to be negative integers, "
      r"found \[2, 1\]"):
    ncon_interface.ncon([a, a], [(1, 3), (3, 2)], backend=backend)
  with pytest.raises(
      ValueError,
      match=r"contracted labels can only be positive integers or strings"
      r", found \[-5\]."):
    ncon_interface.ncon([a, a], [(-1, -5), (-5, -2)], backend=backend)
  with pytest.raises(
      ValueError,
      match=r"contracted labels must"
      r" not be prepended with '-', found \['-5'\]."):
    ncon_interface.ncon([a, a], [(-1, '-5'), ('-5', -2)], backend=backend)


def test_node_invalid_network(backend):
  a = Node(np.ones((2, 2)), backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 1), (1, 2)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (2, 2)], backend=backend)
  with pytest.raises(ValueError):
    ncon_interface.ncon([a, a], [(1, 2), (3, 1)], backend=backend)
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


def test_get_cont_out_labels():
  network_structure = [[-1, 2, '3', '33', '4', 3, '-33'],
                       ['-4', -2, '-3', '3', '33', '4', 2, 3]]
  # pylint: disable=line-too-long
  int_cont_labels, str_cont_labels, int_out_labels, str_out_labels = _get_cont_out_labels(
      network_structure)
  exp_int_cont_labels = [2, 3]
  exp_str_cont_labels = ['3', '33', '4']
  exp_int_out_labels = [-1, -2]
  exp_str_out_labels = ['-3', '-33', '-4']

  def check(exp, actual):
    for e, a in zip(exp, actual):
      assert e == a

  check(exp_int_cont_labels, int_cont_labels)
  check(exp_str_cont_labels, str_cont_labels)
  check(exp_int_out_labels, int_out_labels)
  check(exp_str_out_labels, str_out_labels)
  
def test_partial_traces(backend):
  np.random.seed(10)
  a = np.random.rand(4, 4, 4, 4)
  res = ncon_interface.ncon([a, a], [(-1, 1, 1, 3), (2, -2, 2, 3)],
                            backend=backend)
  t1 = np.trace(a, axis1=1, axis2=2)
  t2 = np.trace(a, axis1=0, axis2=2)
  exp = np.tensordot(t1, t2, ([1], [1]))
  np.testing.assert_allclose(res, exp)
