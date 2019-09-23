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
import tensornetwork
import pytest
import numpy as np
import tensorflow as tf
import torch
import jax

np_dtypes = [np.float32, np.float64, np.complex64, np.complex128, np.int32]
tf_dtypes = [tf.float32, tf.float64, tf.complex64, tf.complex128, tf.int32]
torch_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
jax_dtypes = [
    jax.numpy.float32, jax.numpy.float64, jax.numpy.complex64,
    jax.numpy.complex128, jax.numpy.int32
]


def test_network_copy_reordered(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(3, 3, 3))
  b = net.add_node(np.random.rand(3, 3, 3))
  c = net.add_node(np.random.rand(3, 3, 3))
  a[0] ^ b[1]
  a[1] ^ c[2]
  b[2] ^ c[0]

  edge_order = [a[2], c[1], b[0]]
  net_copy, node_dict, edge_dict = net.copy()
  net_copy.check_correct()

  res = a @ b @ c
  res.reorder_edges(edge_order)
  res_copy = node_dict[a] @ node_dict[b] @ node_dict[c]
  res_copy.reorder_edges([edge_dict[e] for e in edge_order])
  np.testing.assert_allclose(res.tensor, res_copy.tensor)


def test_add_node_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2), "a", axis_names=["e0", "e1"])
  assert a.name == "a"
  assert a[0].name == "e0"
  assert a[1].name == "e1"


def test_add_copy_node_from_node_object(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(
      tensornetwork.CopyNode(
          3, 3, name="TestName", axis_names=['a', 'b', 'c'], backend=backend))
  assert a in net
  assert a.shape == (3, 3, 3)
  assert isinstance(a, tensornetwork.CopyNode)
  assert a.name == "TestName"
  assert a.axis_names == ['a', 'b', 'c']
  b = net.add_node(np.eye(3))
  e = a[0] ^ b[0]
  c = net.contract(e)
  np.testing.assert_allclose(c.tensor, a.tensor)


def test_default_names_add_node_object(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(tensornetwork.CopyNode(3, 3, backend=backend))
  assert a.name is not None
  assert len(a.axis_names) == 3


def test_set_tensor(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones(2))
  np.testing.assert_allclose(a.tensor, np.ones(2))
  a.tensor = np.zeros(2)
  np.testing.assert_allclose(a.tensor, np.zeros(2))


def test_has_nondangling_edge(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones(2))
  assert not a.has_nondangling_edge()
  b = net.add_node(np.ones((2, 2)))
  net.connect(b[0], b[1])
  assert b.has_nondangling_edge()


def test_large_nodes(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros([5, 6, 7, 8, 9]), "a")
  b = net.add_node(np.zeros([5, 6, 7, 8, 9]), "b")
  for i in range(5):
    net.connect(a[i], b[i])
  net.check_correct()


def test_small_matmul(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros([10, 10]), name="a")
  b = net.add_node(np.zeros([10, 10]), name="b")
  edge = net.connect(a[0], b[0], "edge")
  net.check_correct()
  c = net.contract(edge, name="a * b")
  assert list(c.shape) == [10, 10]
  net.check_correct()


def test_double_trace(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones([10, 10, 10, 10]), name="a")
  edge1 = net.connect(a[0], a[1], "edge1")
  edge2 = net.connect(a[2], a[3], "edge2")
  net.check_correct()
  net._contract_trace(edge1)
  net.check_correct()
  val = net._contract_trace(edge2)
  net.check_correct()
  np.testing.assert_allclose(val.tensor, 100.0)


def test_indirect_trace(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones([10, 10]), name="a")
  edge = net.connect(a[0], a[1], "edge")
  net.check_correct()
  val = net.contract(edge)
  net.check_correct()
  np.testing.assert_allclose(val.tensor, 10.0)


def test_real_physics(backend):
  # Calcuate the expected value in numpy
  a_vals = np.ones([2, 3, 4, 5])
  b_vals = np.ones([4, 6, 7])
  c_vals = np.ones([5, 6, 8])
  contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
  contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
  final_result = np.trace(contract2, axis1=0, axis2=4)
  # Build the network
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(a_vals, name="T")
  b = net.add_node(b_vals, name="A")
  c = net.add_node(c_vals, name="B")
  e1 = net.connect(a[2], b[0], "edge")
  e2 = net.connect(c[0], a[3], "edge2")
  e3 = net.connect(b[1], c[1], "edge3")
  net.check_correct()
  node_result = net.contract(e1)
  np.testing.assert_allclose(node_result.tensor, contract1)
  net.check_correct()
  node_result = net.contract(e2)
  np.testing.assert_allclose(node_result.tensor, contract2)
  net.check_correct()
  val = net.contract(e3)
  net.check_correct()
  np.testing.assert_allclose(val.tensor, final_result)


def test_real_physics_with_tensors(backend):
  # Calcuate the expected value in numpy
  a_vals = np.ones([2, 3, 4, 5])
  b_vals = np.ones([4, 6, 7])
  c_vals = np.ones([5, 6, 8])
  contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
  contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
  final_result = np.trace(contract2, axis1=0, axis2=4)
  # Build the network
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones([2, 3, 4, 5]), name="T")
  b = net.add_node(np.ones([4, 6, 7]), name="A")
  c = net.add_node(np.ones([5, 6, 8]), name="B")
  e1 = net.connect(a[2], b[0], "edge")
  e2 = net.connect(c[0], a[3], "edge2")
  e3 = net.connect(b[1], c[1], "edge3")
  net.check_correct()
  node_result = net.contract(e1)
  np.testing.assert_allclose(node_result.tensor, contract1)
  net.check_correct()
  node_result = net.contract(e2)
  np.testing.assert_allclose(node_result.tensor, contract2)
  net.check_correct()
  val = net.contract(e3)
  net.check_correct()
  np.testing.assert_allclose(val.tensor, final_result)


def test_real_physics_naive_contraction(backend):
  # Calcuate the expected value in numpy
  a_vals = np.ones([2, 3, 4, 5])
  b_vals = np.ones([4, 6, 7])
  c_vals = np.ones([5, 6, 8])
  contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
  contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
  final_result = np.trace(contract2, axis1=0, axis2=4)
  # Build the network
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones([2, 3, 4, 5]), name="T")
  b = net.add_node(np.ones([4, 6, 7]), name="A")
  c = net.add_node(np.ones([5, 6, 8]), name="B")
  e1 = net.connect(a[2], b[0], "edge")
  e2 = net.connect(c[0], a[3], "edge2")
  e3 = net.connect(b[1], c[1], "edge3")
  for edge in [e1, e2, e3]:
    net.contract(edge)
  val = net.get_final_node()
  assert list(val.shape) == [8, 2, 3, 7]
  np.testing.assert_allclose(val.tensor, final_result)


def test_with_tensors(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2) * 2, name="T")
  b = net.add_node(np.eye(2) * 3, name="A")
  e1 = net.connect(a[0], b[0], "edge")
  e2 = net.connect(a[1], b[1], "edge2")
  net.check_correct()
  net.contract(e1)
  net.check_correct()
  val = net.contract(e2)
  net.check_correct()
  np.testing.assert_allclose(val.tensor, 12.0)


def test_node2_contract_trace(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros([3, 3, 1]))
  b = net.add_node(np.zeros([1]))
  net.connect(b[0], a[2])
  trace_edge = net.connect(a[0], a[1])
  net._contract_trace(trace_edge)
  net.check_correct()


def test_node_get_dim_bad_axis(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  node = net.add_node(np.eye(2), name="a", axis_names=["1", "2"])
  with pytest.raises(ValueError):
    node.get_dimension(10)


def test_named_axis(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2), axis_names=["alpha", "beta"])
  e = net.connect(a["alpha"], a["beta"])
  b = net.contract(e)
  np.testing.assert_allclose(b.tensor, 2.0)


def test_mixed_named_axis(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2) * 2.0, axis_names=["alpha", "beta"])
  b = net.add_node(np.eye(2) * 3.0)
  e1 = net.connect(a["alpha"], b[0])
  # Axes should still be indexable by numbers even with naming.
  e2 = net.connect(a[1], b[1])
  net.contract(e1)
  result = net.contract(e2)
  np.testing.assert_allclose(result.tensor, 12.0)


def test_duplicate_name(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  with pytest.raises(ValueError):
    net.add_node(np.eye(2), axis_names=["test", "test"])


def test_bad_axis_name_length(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  with pytest.raises(ValueError):
    # This should have 2 names, not 1.
    net.add_node(np.eye(2), axis_names=["need_2_names"])


def test_bad_axis_name_connect(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2), axis_names=["test", "names"])
  with pytest.raises(ValueError):
    a.get_edge("bad_name")


def test_node_edge_ordering(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4)))
  e2 = a[0]
  e3 = a[1]
  e4 = a[2]
  assert a.shape == (2, 3, 4)
  a.reorder_edges([e4, e2, e3])
  net.check_correct()
  assert a.shape == (4, 2, 3)
  assert e2.axis1 == 1
  assert e3.axis1 == 2
  assert e4.axis1 == 0


def test_trace_edge_ordering(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 2, 3)))
  e2 = net.connect(a[1], a[0])
  e3 = a[2]
  with pytest.raises(ValueError):
    a.reorder_edges([e2, e3])


def test_mismatch_edge_ordering(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3)))
  e2_a = a[0]
  b = net.add_node(np.zeros((2,)))
  e_b = b[0]
  with pytest.raises(ValueError):
    a.reorder_edges([e2_a, e_b])


def test_complicated_edge_reordering(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4)))
  b = net.add_node(np.zeros((2, 5)))
  c = net.add_node(np.zeros((3,)))
  d = net.add_node(np.zeros((4, 5)))
  e_ab = net.connect(a[0], b[0])
  e_bd = net.connect(b[1], d[1])
  e_ac = net.connect(a[1], c[0])
  e_ad = net.connect(a[2], d[0])
  net.contract(e_bd)
  a.reorder_edges([e_ac, e_ab, e_ad])
  net.check_correct()
  assert a.shape == (3, 2, 4)


def test_edge_reorder_axis_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4, 5)), axis_names=["a", "b", "c", "d"])
  edge_a = a["a"]
  edge_b = a["b"]
  edge_c = a["c"]
  edge_d = a["d"]
  a.reorder_edges([edge_c, edge_b, edge_d, edge_a])
  assert a.shape == (4, 3, 5, 2)
  assert a.axis_names == ["c", "b", "d", "a"]


def test_add_axis_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2), name="A", axis_names=["ignore1", "ignore2"])
  a.add_axis_names(["a", "b"])
  assert a.axis_names == ["a", "b"]


def test_reorder_axes(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4)))
  b = net.add_node(np.zeros((3, 4, 5)))
  c = net.add_node(np.zeros((2, 4, 5)))
  net.connect(a[0], c[0])
  net.connect(b[0], a[1])
  net.connect(a[2], c[1])
  net.connect(b[2], c[2])
  a.reorder_axes([2, 0, 1])
  net.check_correct()
  assert a.shape == (4, 2, 3)


def test_flatten_consistent_result(backend):
  net_noflat = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((3, 5, 5, 6))
  b_val = np.ones((5, 6, 4, 5))
  # Create non flattened example to compare against.
  a_noflat = net_noflat.add_node(a_val)
  b_noflat = net_noflat.add_node(b_val)
  e1 = net_noflat.connect(a_noflat[1], b_noflat[3])
  e2 = net_noflat.connect(a_noflat[3], b_noflat[1])
  e3 = net_noflat.connect(a_noflat[2], b_noflat[0])
  a_dangling_noflat = a_noflat[0]
  b_dangling_noflat = b_noflat[2]
  for edge in [e1, e2, e3]:
    net_noflat.contract(edge)
  noflat_result_node = net_noflat.get_final_node()
  noflat_result_node.reorder_edges([a_dangling_noflat, b_dangling_noflat])
  noflat_result = noflat_result_node.tensor
  # Create network with flattening
  net_flat = tensornetwork.TensorNetwork(backend=backend)
  a_flat = net_flat.add_node(a_val)
  b_flat = net_flat.add_node(b_val)
  e1 = net_flat.connect(a_flat[1], b_flat[3])
  e2 = net_flat.connect(a_flat[3], b_flat[1])
  e3 = net_flat.connect(a_flat[2], b_flat[0])
  a_dangling_flat = a_flat[0]
  b_dangling_flat = b_flat[2]
  final_edge = net_flat.flatten_edges([e1, e2, e3])
  flat_result_node = net_flat.contract(final_edge)
  flat_result_node.reorder_edges([a_dangling_flat, b_dangling_flat])
  flat_result = flat_result_node.tensor
  np.testing.assert_allclose(flat_result, noflat_result)


def test_flatten_consistent_tensor(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((2, 3, 4, 5))
  b_val = np.ones((3, 5, 4, 2))
  a = net.add_node(a_val)
  b = net.add_node(b_val)
  e1 = net.connect(a[0], b[3])
  e2 = net.connect(b[1], a[3])
  e3 = net.connect(a[1], b[0])
  net.flatten_edges([e3, e1, e2])
  net.check_correct()

  # Check expected values.
  a_final = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
  b_final = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (4, 30))
  np.testing.assert_allclose(a.tensor, a_final)
  np.testing.assert_allclose(b.tensor, b_final)


def test_flatten_trace_consistent_result(backend):
  net_noflat = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((5, 6, 6, 7, 5, 7))
  a_noflat = net_noflat.add_node(a_val)
  e1 = net_noflat.connect(a_noflat[0], a_noflat[4])
  e2 = net_noflat.connect(a_noflat[1], a_noflat[2])
  e3 = net_noflat.connect(a_noflat[3], a_noflat[5])
  for edge in [e1, e2, e3]:
    net_noflat.contract(edge)
  noflat_result = net_noflat.get_final_node().tensor
  # Create network with flattening
  net_flat = tensornetwork.TensorNetwork(backend=backend)
  a_flat = net_flat.add_node(a_val)
  e1 = net_flat.connect(a_flat[0], a_flat[4])
  e2 = net_flat.connect(a_flat[1], a_flat[2])
  e3 = net_flat.connect(a_flat[3], a_flat[5])
  final_edge = net_flat.flatten_edges([e1, e2, e3])
  flat_result = net_flat.contract(final_edge).tensor
  np.testing.assert_allclose(flat_result, noflat_result)


def test_flatten_trace_consistent_tensor(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((5, 3, 4, 4, 5))
  a = net.add_node(a_val)
  e1 = net.connect(a[0], a[4])
  e2 = net.connect(a[3], a[2])
  net.flatten_edges([e2, e1])
  net.check_correct()
  # Check expected values.
  a_final = np.reshape(np.transpose(a_val, (1, 2, 0, 3, 4)), (3, 20, 20))
  np.testing.assert_allclose(a.tensor, a_final)


def test_contract_between_output_order(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((2, 3, 4, 5))
  b_val = np.ones((3, 5, 4, 2))
  c_val = np.ones((2, 2))
  a = net.add_node(a_val)
  b = net.add_node(b_val)
  c = net.add_node(c_val)
  net.connect(a[0], b[3])
  net.connect(b[1], a[3])
  net.connect(a[1], b[0])
  with pytest.raises(ValueError):
    d = net.contract_between(
        a, b, name="New Node", output_edge_order=[a[2], b[2], a[0]])
  net.check_correct(check_connections=False)
  with pytest.raises(ValueError):
    d = net.contract_between(
        a, b, name="New Node", output_edge_order=[a[2], b[2], c[0]])
  net.check_correct(check_connections=False)
  d = net.contract_between(
      a, b, name="New Node", output_edge_order=[b[2], a[2]])
  net.check_correct(check_connections=False)
  a_flat = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
  b_flat = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (4, 30))
  final_val = np.matmul(b_flat, a_flat.T)
  np.testing.assert_allclose(d.tensor, final_val)
  assert d.name == "New Node"


def test_contract_between_trace_edges(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((3, 3))
  final_val = np.trace(a_val)
  a = net.add_node(a_val)
  net.connect(a[0], a[1])
  b = net.contract_between(a, a)
  net.check_correct()
  np.testing.assert_allclose(b.tensor, final_val)


def test_contract_disable(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(2, 2))
  b = net.add_node(np.random.rand(2, 2))
  e = net.connect(a[0], b[0])
  net.contract(e)
  with pytest.raises(ValueError):
    a.edges[0]
  with pytest.raises(ValueError):
    a.edges
  with pytest.raises(ValueError):
    a.signature
  with pytest.raises(ValueError):
    a.shape
  with pytest.raises(ValueError):
    b.edges[0]
  with pytest.raises(ValueError):
    b.edges
  with pytest.raises(ValueError):
    b.signature
  with pytest.raises(ValueError):
    b.shape


def test_contract_between_disable(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(2, 2))
  b = net.add_node(np.random.rand(2, 2))
  net.connect(a[1], b[0])
  net.contract_between(a, b)
  with pytest.raises(ValueError):
    a.edges[0]
  with pytest.raises(ValueError):
    a.edges
  with pytest.raises(ValueError):
    a.signature
  with pytest.raises(ValueError):
    a.shape
  with pytest.raises(ValueError):
    b.edges[0]
  with pytest.raises(ValueError):
    b.edges
  with pytest.raises(ValueError):
    b.signature
  with pytest.raises(ValueError):
    b.shape


def test_double_trace_disable(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  node = net.add_node(np.random.rand(2, 2, 2, 2), name='node1')

  e1 = net.connect(node[0], node[2], name='e1')
  net.connect(node[1], node[3], name='e2')

  # contract disables contracted edges;
  # if we have to access them, we need
  # to do it beforehand.
  node1 = e1.node1
  node2 = e1.node2

  net.contract(e1, name='b')

  with pytest.raises(ValueError):
    node.edges[0]
  with pytest.raises(ValueError):
    node.edges
  with pytest.raises(ValueError):
    node.signature
  with pytest.raises(ValueError):
    node.shape

  with pytest.raises(ValueError):
    node1.edges[0]
  with pytest.raises(ValueError):
    node1.edges
  with pytest.raises(ValueError):
    node1.signature
  with pytest.raises(ValueError):
    node1.shape

  with pytest.raises(ValueError):
    node2.edges[0]
  with pytest.raises(ValueError):
    node2.edges
  with pytest.raises(ValueError):
    node2.signature
  with pytest.raises(ValueError):
    node2.shape


def test_trace_disable(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  node = net.add_node(np.random.rand(2, 2, 2, 2), name='node1')
  e = net.connect(node[0], node[2], name='e1')
  # _contract_trace disables contracted edges;
  # if we have to access them, we need
  # to do it beforehand.
  node1 = e.node1
  node2 = e.node2

  net._contract_trace(e, name='b')

  with pytest.raises(ValueError):
    node.edges[0]
  with pytest.raises(ValueError):
    node.edges
  with pytest.raises(ValueError):
    node.signature
  with pytest.raises(ValueError):
    node.shape

  with pytest.raises(ValueError):
    node1.edges[0]
  with pytest.raises(ValueError):
    node1.edges
  with pytest.raises(ValueError):
    node1.signature
  with pytest.raises(ValueError):
    node1.shape
  with pytest.raises(ValueError):
    node2.edges[0]
  with pytest.raises(ValueError):
    node2.edges
  with pytest.raises(ValueError):
    node2.signature
  with pytest.raises(ValueError):
    node2.shape


def test_weakref(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  e = net.connect(a[0], b[0])
  del a
  del b
  net.contract(e)
  with pytest.raises(ValueError):
    e.node1
  with pytest.raises(ValueError):
    e.node2


# This test no longer tests weakref behaviour
# since contraction methods now automatically
# disable contracted edges by setting Edge.node1 and
# Edge.node2 to None.
def test_weakref_complex(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  c = net.add_node(np.eye(2))
  e1 = net.connect(a[0], b[0])
  e2 = net.connect(b[1], c[0])
  net.contract(e1)
  net.contract(e2)
  # This now raises an exception because we contract disables contracted edges
  # and raises a ValueError if we try to access the nodes
  with pytest.raises(ValueError):
    e1.node1
  # This raises an exception since the intermediate node created when doing
  # `net.contract(e2)` was garbage collected.
  with pytest.raises(ValueError):
    e2.node1


def test_edge_disable_complex(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  c = net.add_node(np.eye(2))
  e1 = net.connect(a[0], b[0])
  e2 = net.connect(b[1], c[0])
  net.contract(e1)
  net.contract(e2)
  # This now raises an exception because we contract disables contracted edges
  # and raises a ValueError if we try to access the nodes
  with pytest.raises(ValueError):
    e1.node1
  # This raises an exception since the intermediate node created when doing
  # `net.contract(e2)` was garbage collected.
  with pytest.raises(ValueError):
    e2.node1


def test_set_node2(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  e = net.connect(a[0], b[0])
  # You should never do this, but if you do, we should handle
  # it gracefully.
  e.node2 = None
  assert e.is_dangling()


def test_contracted_node_not_in_network(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2)))
  e = net.connect(a[0], a[1])
  net.contract(e)
  assert a not in net


def test_set_default(backend):
  tensornetwork.set_default_backend(backend)
  assert tensornetwork.config.default_backend == backend
  net = tensornetwork.TensorNetwork()
  assert net.backend.name == backend


def test_copy_tensor(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([1, 2, 3], dtype=np.float64))
  b = net.add_node(np.array([10, 20, 30], dtype=np.float64))
  c = net.add_node(np.array([5, 6, 7], dtype=np.float64))
  d = net.add_node(np.array([1, -1, 1], dtype=np.float64))
  cn = net.add_copy_node(rank=4, dimension=3)
  edge1 = net.connect(a[0], cn[0])
  edge2 = net.connect(b[0], cn[1])
  edge3 = net.connect(c[0], cn[2])
  edge4 = net.connect(d[0], cn[3])

  result = cn.compute_contracted_tensor()
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 50 - 240 + 630)

  for edge in [edge1, edge2, edge3, edge4]:
    net.contract(edge)
  val = net.get_final_node()
  result = val.tensor
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 50 - 240 + 630)


# Include 'tensorflow' (by removing the decorator) once #87 is fixed.
@pytest.mark.parametrize('backend', ('numpy', 'jax'))
def test_copy_tensor_parallel_edges(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.diag([1., 2, 3]))
  b = net.add_node(np.array([10, 20, 30], dtype=np.float64))
  cn = net.add_copy_node(rank=3, dimension=3)
  edge1 = net.connect(a[0], cn[0])
  edge2 = net.connect(a[1], cn[1])
  edge3 = net.connect(b[0], cn[2])

  result = cn.compute_contracted_tensor()
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 10 + 40 + 90)

  for edge in [edge1, edge2, edge3]:
    net.contract(edge)
  val = net.get_final_node()
  result = val.tensor
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 10 + 40 + 90)


def test_contract_copy_node_connected_neighbors(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([[1, 2, 3], [10, 20, 30]], dtype=np.float64))
  b = net.add_node(np.array([[2, 1, 1], [2, 2, 2]], dtype=np.float64))
  c = net.add_node(np.array([3, 4, 4], dtype=np.float64))
  cn = net.add_copy_node(rank=3, dimension=3)
  net.connect(a[0], b[0])
  net.connect(a[1], cn[0])
  net.connect(b[1], cn[1])
  net.connect(c[0], cn[2])

  n = net.contract_copy_node(cn)

  with pytest.raises(ValueError):
    val = net.get_final_node()
  assert len(net.nodes_set) == 1
  assert len(n.edges) == 2
  assert n.edges[0] == n.edges[1]

  net.contract_parallel(n.edges[0])
  val = net.get_final_node()
  result = val.tensor
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 26 + 460)


def test_bad_backend():
  with pytest.raises(ValueError):
    tensornetwork.TensorNetwork("NOT_A_BACKEND")


def test_remove_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2)), axis_names=["test", "names", "ignore"])
  b = net.add_node(np.ones((2, 2)))
  c = net.add_node(np.ones((2, 2)))
  net.connect(a["test"], b[0])
  net.connect(a[1], c[0])
  broken_edges_name, broken_edges_axis = net.remove_node(a)
  assert a not in net
  assert "test" in broken_edges_name
  assert broken_edges_name["test"] is b[0]
  assert "names" in broken_edges_name
  assert broken_edges_name["names"] is c[0]
  assert "ignore" not in broken_edges_name
  assert 0 in broken_edges_axis
  assert 1 in broken_edges_axis
  assert 2 not in broken_edges_axis
  assert broken_edges_axis[0] is b[0]
  assert broken_edges_axis[1] is c[0]


def test_remove_node_trace_edge(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2)))
  b = net.add_node(np.ones(2))
  net.connect(a[0], b[0])
  net.connect(a[1], a[2])
  _, broken_edges = net.remove_node(a)
  assert 0 in broken_edges
  assert 1 not in broken_edges
  assert 2 not in broken_edges
  assert broken_edges[0] is b[0]


def test_subnetwork_signatures(backend):
  net1 = tensornetwork.TensorNetwork(backend=backend)
  net2 = tensornetwork.TensorNetwork(backend=backend)
  a = net1.add_node(np.eye(2))
  assert a.signature == 1
  b = net2.add_node(np.eye(2))
  assert b.signature == 1
  net1.add_subnetwork(net2)
  assert b.signature == 2


def test_edges_signatures(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2,) * 5))
  b = net.add_node(np.ones((2,) * 5))
  for i in range(5):
    assert a[i].signature == -1
    assert b[i].signature == -1
  for i, index in enumerate({1, 3, 4}):
    edge = net.connect(a[index], b[index])
    # Add 11 to account for the the original 10
    # edges and the 1 indexing.
    assert edge.signature == i + 11


def test_at_operator(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2,)))
  b = net.add_node(np.ones((2,)))
  net.connect(a[0], b[0])
  c = a @ b
  assert isinstance(c, tensornetwork.Node)
  np.testing.assert_allclose(c.tensor, 2.0)


def test_at_operator_out_of_network(backend):
  net1 = tensornetwork.TensorNetwork(backend=backend)
  net2 = tensornetwork.TensorNetwork(backend=backend)
  a = net1.add_node(np.ones((2,)))
  b = net2.add_node(np.ones((2,)))
  with pytest.raises(ValueError):
    a = a @ b


def test_edge_sorting(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  c = net.add_node(np.eye(2))
  e1 = net.connect(a[0], b[1])
  e2 = net.connect(b[0], c[1])
  e3 = net.connect(c[0], a[1])
  sorted_edges = sorted([e2, e3, e1])
  assert sorted_edges == [e1, e2, e3]


def test_connect_alias(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2)))
  b = net.add_node(np.ones((2, 2)))
  e = a[0] ^ b[0]
  assert set(e.get_nodes()) == {a, b}
  assert e is a[0]
  assert e is b[0]
  assert e in net


def test_remove_after_flatten(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2)))
  b = net.add_node(np.ones((2, 2)))
  net.connect(a[0], b[0])
  net.connect(a[1], b[1])
  net.flatten_all_edges()
  net.remove_node(a)


def test_split_node_full_svd_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(10, 10))
  e1 = a[0]
  e2 = a[1]
  left, s, right, _, = net.split_node_full_svd(
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


def test_split_node_rq_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4, 5, 6)))
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = net.split_node_rq(
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


def test_split_node_qr_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4, 5, 6)))
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = net.split_node_qr(
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


def test_split_node_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4, 5, 6)))
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right, _ = net.split_node(
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


def test_split_node_rq(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(2, 3, 4, 5, 6))
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, _ = net.split_node_rq(a, left_edges, right_edges)
  net.check_correct()
  np.testing.assert_allclose(a.tensor, net.contract(left[3]).tensor)


def test_split_node_qr_unitarity_complex(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")
  if backend == "jax":
    pytest.skip("Complex QR crashes jax")

  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3))
  q, _ = net.split_node_qr(a, [a[0]], [a[1]])
  net_1 = tensornetwork.TensorNetwork(backend=backend)
  n1 = net_1.add_node(q.tensor)
  n2 = net_1.add_node(net_1.backend.conj(q.tensor))
  n1[1] ^ n2[1]
  u1 = net_1.contract_between(n1, n2)

  net_2 = tensornetwork.TensorNetwork(backend=backend)
  n1 = net_2.add_node(q.tensor)
  n2 = net_2.add_node(net_2.backend.conj(q.tensor))
  n2[0] ^ n1[0]
  u2 = net_2.contract_between(n1, n2)

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_qr_unitarity_float(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(3, 3))
  q, _ = net.split_node_qr(a, [a[0]], [a[1]])
  net_1 = tensornetwork.TensorNetwork(backend=backend)
  n1 = net_1.add_node(q.tensor)
  n2 = net_1.add_node(net_1.backend.conj(q.tensor))
  n1[1] ^ n2[1]
  u1 = net_1.contract_between(n1, n2)

  net_2 = tensornetwork.TensorNetwork(backend=backend)
  n1 = net_2.add_node(q.tensor)
  n2 = net_2.add_node(q.tensor)
  n2[0] ^ n1[0]
  u2 = net_2.contract_between(n1, n2)

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_qr(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(2, 3, 4, 5, 6))
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, _ = net.split_node_qr(a, left_edges, right_edges)
  net.check_correct()
  np.testing.assert_allclose(a.tensor, net.contract(left[3]).tensor)


def test_disable_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(2, 3, 4, 5, 6))
  with pytest.raises(ValueError):
    a.disable()


@pytest.mark.parametrize("backend,dtype", [
    *list(zip(['numpy'] * len(np_dtypes), np_dtypes)),
    *list(zip(['tensorflow'] * len(tf_dtypes), tf_dtypes)),
    *list(zip(['pytorch'] * len(torch_dtypes), torch_dtypes)),
    *list(zip(['jax'] * len(jax_dtypes), jax_dtypes)),
])
def test_network_backend_dtype_1(backend, dtype):
  net = tensornetwork.TensorNetwork(backend=backend, dtype=dtype)
  n1 = net.add_node(net.backend.zeros((2, 2)))
  assert n1.tensor.dtype == dtype


@pytest.mark.parametrize("backend,dtype", [('numpy', np.float32),
                                           ('tensorflow', tf.float32),
                                           ('pytorch', torch.float32),
                                           ('jax', np.float32)])
def test_network_backend_dtype_3(backend, dtype):
  net = tensornetwork.TensorNetwork(backend=backend, dtype=dtype)
  with pytest.raises(TypeError):
    net.add_node(np.random.rand(3, 3))
