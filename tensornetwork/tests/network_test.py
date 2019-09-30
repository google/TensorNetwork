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
from jax.config import config

np_dtypes = [np.float32, np.float64, np.complex64, np.complex128, np.int32]
tf_dtypes = [tf.float32, tf.float64, tf.complex64, tf.complex128, tf.int32]
torch_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
jax_dtypes = [
    jax.numpy.float32, jax.numpy.float64, jax.numpy.complex64,
    jax.numpy.complex128, jax.numpy.int32
]


@pytest.mark.parametrize("backend, dtype", [
    *list(zip(['numpy'] * len(np_dtypes), np_dtypes)),
    *list(zip(['tensorflow'] * len(tf_dtypes), tf_dtypes)),
    *list(zip(['pytorch'] * len(torch_dtypes), torch_dtypes)),
    *list(zip(['jax'] * len(jax_dtypes), jax_dtypes)),
])
def test_network_init(backend, dtype):
  net = tensornetwork.TensorNetwork(backend=backend, dtype=dtype)
  assert net.nodes_set == set()
  assert net.dtype == dtype
  assert net.node_increment == 0
  assert net.edge_increment == 0


def test_network_init_no_dtype(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  assert net.dtype is None


def test_new_edge_name(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  net.edge_increment = 7
  assert net._new_edge_name(None) == "__Edge_8"
  assert net._new_edge_name("new_name") == "new_name"
  assert net._new_edge_name(None) == "__Edge_10"


def test_new_node_name(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  net.node_increment = 7
  assert net._new_node_name(None) == "__Node_8"
  assert net._new_node_name("new_name") == "new_name"
  assert net._new_node_name(None) == "__Node_10"


def test_dtype(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  assert net.dtype is None


def test_network_copy_conj(backend):
  if backend == "pytorch":
    pytest.skip("Pytorch does not support complex numbers")
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([1.0 + 2.0j, 2.0 - 1.0j]))
  _, nodes, _ = net.copy(conj=True)
  np.testing.assert_allclose(nodes[a].tensor, np.array([1.0 - 2.0j,
                                                        2.0 + 1.0j]))


def test_network_copy(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(3, 3, 3))
  b = net.add_node(np.random.rand(3, 3, 3))
  c = net.add_node(np.random.rand(3, 3, 3))
  a[0] ^ b[1]
  a[1] ^ c[2]
  b[2] ^ c[0]
  net_copy, node_dict, _ = net.copy()
  net_copy.check_correct()

  res = a @ b @ c
  res_copy = node_dict[a] @ node_dict[b] @ node_dict[c]
  np.testing.assert_allclose(res.tensor, res_copy.tensor)


def test_network_copy_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(3, 3, 3), name='a')
  b = net.add_node(np.random.rand(3, 3, 3), name='b')
  c = net.add_node(np.random.rand(3, 3, 3), name='c')
  a[0] ^ b[1]
  b[2] ^ c[0]
  _, node_dict, edge_dict = net.copy()
  for node in net.nodes_set:
    assert node_dict[node].name == node.name
  for edge in net.get_all_edges():
    assert edge_dict[edge].name == edge.name


def test_network_copy_identities(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.random.rand(3, 3, 3), name='a')
  b = net.add_node(np.random.rand(3, 3, 3), name='b')
  c = net.add_node(np.random.rand(3, 3, 3), name='c')
  a[0] ^ b[1]
  b[2] ^ c[0]
  _, node_dict, edge_dict = net.copy()
  for node in net.nodes_set:
    assert not node_dict[node] is node
  for edge in net.get_all_edges():
    assert not edge_dict[edge] is edge


def test_add_subnetwork(backend):
  net1 = tensornetwork.TensorNetwork(backend=backend)
  net2 = tensornetwork.TensorNetwork(backend=backend)
  a = net1.add_node(np.eye(2) * 2)
  b = net1.add_node(np.eye(2) * 3)
  e1 = net1.connect(a[0], b[0])
  c = net2.add_node(np.eye(2) * 4)
  net2.add_subnetwork(net1)
  assert a in net2
  assert b in net2
  assert c in net2
  assert a.network is net2
  assert b.network is net2
  assert c.network is net2
  e2 = net2.connect(c[0], a[1])
  e3 = net2.connect(c[1], b[1])
  net2.check_correct()
  for edge in [e1, e2, e3]:
    net2.contract(edge)
  result = net2.get_final_node()
  np.testing.assert_allclose(result.tensor, 48.0)


def test_add_subnetwork_incompatible_backends():
  net1 = tensornetwork.TensorNetwork(backend="numpy")
  net2 = tensornetwork.TensorNetwork(backend="tensorflow")
  with pytest.raises(ValueError):
    net1.add_subnetwork(net2)


def test_merge_networks(backend):
  net1 = tensornetwork.TensorNetwork(backend=backend)
  net2 = tensornetwork.TensorNetwork(backend=backend)
  a = net1.add_node(np.eye(2) * 2)
  b = net1.add_node(np.eye(2) * 3)
  e1 = net1.connect(a[0], b[0])
  c = net2.add_node(np.eye(2) * 4)
  net3 = tensornetwork.TensorNetwork.merge_networks([net1, net2])
  assert a in net3
  assert b in net3
  assert c in net3
  assert a.network is net3
  assert b.network is net3
  assert c.network is net3
  e2 = net3.connect(c[0], a[1])
  e3 = net3.connect(c[1], b[1])
  net3.check_correct()
  for edge in [e1, e2, e3]:
    net3.contract(edge)
  result = net3.get_final_node()
  np.testing.assert_allclose(result.tensor, 48.0)


def test_merge_networks_incompatible_backends():
  net1 = tensornetwork.TensorNetwork(backend="numpy")
  net2 = tensornetwork.TensorNetwork(backend="tensorflow")
  with pytest.raises(ValueError):
    tensornetwork.TensorNetwork.merge_networks([net1, net2])


def test_switch_backend_to_tensorflow():
  net = tensornetwork.TensorNetwork(backend="numpy")
  a = net.add_node(np.eye(2))
  net.switch_backend(new_backend="tensorflow")
  assert isinstance(a.tensor, tf.Tensor)


def test_add_node_sanity_check(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  net.add_node(np.eye(2), "a")
  net.check_correct()


def test_add_node_in_network(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  assert a in net


def test_add_node_twice_raise_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(tensornetwork.CopyNode(3, 3, backend=backend))
  with pytest.raises(ValueError):
    net.add_node(a)


def test_add_copy_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(
      tensornetwork.CopyNode(
          3, 3, name="TestName", axis_names=['a', 'b', 'c'], backend=backend))
  assert a in net


def test_connect_dangling_edges(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((3,)))
  b = net.add_node(np.ones((3,)))
  edge = net.connect(a[0], b[0])
  net.check_correct()
  assert a.edges == b.edges
  assert edge in net


def test_connect_axis_names(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((3,)), name="a", axis_names=["one"])
  b = net.add_node(np.ones((3,)), name="b", axis_names=["one"])
  edge = net.connect(a["one"], b["one"])
  net.check_correct()
  assert a.edges == b.edges
  assert edge in net


def test_connect_twice_edge_axis_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([2.]), name="a")
  b = net.add_node(np.array([2.]), name="b")
  net.connect(a[0], b[0])
  with pytest.raises(ValueError):
    net.connect(a[0], b[0])


def test_connect_same_edge_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  with pytest.raises(ValueError):
    net.connect(a[0], a[0])


def test_disconnect_edge(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([1.0] * 5), "a")
  b = net.add_node(np.array([1.0] * 5), "b")
  e = net.connect(a[0], b[0])
  assert not e.is_dangling()
  dangling_edge_1, dangling_edge_2 = net.disconnect(e)
  net.check_correct(check_connections=False)
  assert dangling_edge_1.is_dangling()
  assert dangling_edge_2.is_dangling()
  assert a.get_edge(0) == dangling_edge_1
  assert b.get_edge(0) == dangling_edge_2


def test_disconnect_edge_not_in_network(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  edge = net.connect(a[0], b[0])
  net.disconnect(edge)
  assert edge not in net


def test_disconnect_dangling_edge_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  with pytest.raises(ValueError):
    net.disconnect(a[0])


def test_contract_trace_single_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones([10, 10]), name="a")
  edge = net.connect(a[0], a[1], "edge")
  net.check_correct()
  result = net._contract_trace(edge)
  net.check_correct()
  np.testing.assert_allclose(result.tensor, 10.0)


def test_contract_trace_different_nodes_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([2.]))
  b = net.add_node(np.array([2.]))
  e = net.connect(a[0], b[0])
  with pytest.raises(ValueError):
    net._contract_trace(e)


def test_contract_trace_dangling_edge_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([1.0]))
  e = a[0]
  with pytest.raises(ValueError):
    net._contract_trace(e)


def test_contract_single_edge(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([1.0] * 5), "a")
  b = net.add_node(np.array([1.0] * 5), "b")
  e = net.connect(a[0], b[0])
  c = net.contract(e)
  net.check_correct()
  val = c.tensor
  np.testing.assert_allclose(val, 5.0)


def test_contract_name_contracted_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  node = net.add_node(np.eye(2), name="Identity Matrix")
  assert node.name == "Identity Matrix"
  edge = net.connect(node[0], node[1], name="Trace Edge")
  assert edge.name == "Trace Edge"
  final_result = net.contract(edge, name="Trace Of Identity")
  assert final_result.name == "Trace Of Identity"


def test_contract_edge_twice_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  e = net.connect(a[0], a[1], name="edge")
  net.contract(e)
  with pytest.raises(ValueError):
    net.contract(e)


def test_contract_dangling_edge_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([1.0]))
  e = a[0]
  with pytest.raises(ValueError):
    net.contract(e)


def test_contract_copy_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([1, 2, 3], dtype=np.float64))
  b = net.add_node(np.array([10, 20, 30], dtype=np.float64))
  c = net.add_node(np.array([5, 6, 7], dtype=np.float64))
  d = net.add_node(np.array([1, -1, 1], dtype=np.float64))
  cn = net.add_copy_node(rank=4, dimension=3)
  net.connect(a[0], cn[0])
  net.connect(b[0], cn[1])
  net.connect(c[0], cn[2])
  net.connect(d[0], cn[3])

  net.contract_copy_node(cn)
  val = net.get_final_node()
  result = val.tensor
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 50 - 240 + 630)


def test_contract_copy_node_dangling_edge_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([1, 2, 3], dtype=np.float64))
  b = net.add_node(np.array([10, 20, 30], dtype=np.float64))
  c = net.add_node(np.array([5, 6, 7], dtype=np.float64))
  cn = net.add_copy_node(rank=4, dimension=3)
  net.connect(a[0], cn[0])
  net.connect(b[0], cn[1])
  net.connect(c[0], cn[2])

  with pytest.raises(ValueError):
    net.contract_copy_node(cn)


def test_outer_product(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 4, 5)), name="A")
  b = net.add_node(np.ones((4, 3, 6)), name="B")
  c = net.add_node(np.ones((3, 2)), name="C")
  net.connect(a[1], b[0])
  net.connect(a[0], c[1])
  net.connect(b[1], c[0])
  # Purposely leave b's 3rd axis undefined.
  d = net.outer_product(a, b, name="D")
  net.check_correct()
  assert d.shape == (2, 4, 5, 4, 3, 6)
  np.testing.assert_allclose(d.tensor, np.ones((2, 4, 5, 4, 3, 6)))
  assert d.name == "D"


def test_get_final_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((3, 3)), "a")
  b = net.add_node(np.ones((3, 3)), "b")
  e1 = net.connect(a[0], b[0])
  e2 = net.connect(a[1], b[1])
  net.contract(e1)
  expected = net.contract(e2)
  assert net.get_final_node() == expected


def test_get_final_node_dangling_edge_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((3, 3)), "a")
  net.connect(a[0], a[1])
  with pytest.raises(ValueError):
    net.get_final_node()


def test_get_final_node_two_nodes_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((3, 3)), "a")
  b = net.add_node(np.ones((3, 3)), "b")
  net.connect(a[0], b[0])
  net.connect(a[1], b[1])
  with pytest.raises(ValueError):
    net.get_final_node()


def test_get_all_nondangling(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  edge1 = net.connect(a[0], b[0])
  c = net.add_node(np.eye(2))
  d = net.add_node(np.eye(2))
  edge2 = net.connect(c[0], d[0])
  edge3 = net.connect(a[1], c[1])
  assert {edge1, edge2, edge3} == net.get_all_nondangling()


def test_get_all_edges(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  assert {a[0], a[1], b[0], b[1]} == net.get_all_edges()


def test_outer_product_final_nodes(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  edges = []
  for i in range(1, 5):
    edges.append(net.add_node(np.ones(i))[0])
  final_node = net.outer_product_final_nodes(edges)
  np.testing.assert_allclose(final_node.tensor, np.ones([1, 2, 3, 4]))
  assert final_node.get_all_edges() == edges


def test_outer_product_final_nodes_not_contracted_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones(2))
  b = net.add_node(np.ones(2))
  e = net.connect(a[0], b[0])
  with pytest.raises(ValueError):
    net.outer_product_final_nodes([e])


def test_check_connected_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([2, 2.]))
  b = net.add_node(np.array([2, 2.]))
  net.connect(a[0], b[0])
  c = net.add_node(np.array([2, 2.]))
  d = net.add_node(np.array([2, 2.]))
  net.connect(c[0], d[0])
  with pytest.raises(ValueError):
    net.check_connected()


def test_flatten_trace_edges(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4, 3, 5, 5)))
  c = net.add_node(np.zeros((2, 4)))
  e1 = net.connect(a[1], a[3])
  e2 = net.connect(a[4], a[5])
  external_1 = net.connect(a[0], c[0])
  external_2 = net.connect(c[1], a[2])
  new_edge = net.flatten_edges([e1, e2], "New Edge")
  net.check_correct()
  assert a.shape == (2, 4, 15, 15)
  assert a.edges == [external_1, external_2, new_edge, new_edge]
  assert new_edge.name == "New Edge"


def test_flatten_edges_standard(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 5)), name="A")
  b = net.add_node(np.zeros((2, 3, 4, 5)), name="B")
  e1 = net.connect(a[0], b[0], "Edge_1_1")
  e2 = net.connect(a[2], b[3], "Edge_2_3")
  edge_a_1 = a[1]
  edge_b_1 = b[1]
  edge_b_2 = b[2]
  new_edge = net.flatten_edges([e1, e2], new_edge_name="New Edge")
  assert a.shape == (3, 10)
  assert b.shape == (3, 4, 10)
  assert a.edges == [edge_a_1, new_edge]
  assert b.edges == [edge_b_1, edge_b_2, new_edge]
  net.check_correct()


def test_flatten_edges_dangling(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.zeros((2, 3, 4, 5)), name="A")
  e1 = a[0]
  e2 = a[1]
  e3 = a[2]
  e4 = a[3]
  flattened_edge = net.flatten_edges([e1, e3], new_edge_name="New Edge")
  assert a.shape == (3, 5, 8)
  assert a.edges == [e2, e4, flattened_edge]
  assert flattened_edge.name == "New Edge"
  net.check_correct()


def test_flatten_edges_empty_list_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  net.connect(a[0], b[0])
  with pytest.raises(ValueError):
    net.flatten_edges([])


def test_flatten_edges_different_nodes_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  c = net.add_node(np.eye(2))
  e1 = net.connect(a[0], b[0])
  e2 = net.connect(a[1], c[0])
  net.connect(b[1], c[1])
  with pytest.raises(ValueError):
    net.flatten_edges([e1, e2])


def test_get_shared_edges(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2)))
  b = net.add_node(np.ones((2, 2, 2)))
  c = net.add_node(np.ones((2, 2, 2)))
  e1 = net.connect(a[0], b[0])
  e2 = net.connect(b[1], c[1])
  e3 = net.connect(a[2], b[2])
  assert net.get_shared_edges(a, b) == {e1, e3}
  assert net.get_shared_edges(b, c) == {e2}


def test_get_parallel_edge(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2,) * 5))
  b = net.add_node(np.ones((2,) * 5))
  edges = set()
  for i in {0, 1, 3}:
    edges.add(net.connect(a[i], b[i]))
  # sort by edge signature
  a = sorted(list(edges))[0]
  assert net.get_parallel_edges(a) == edges


def test_flatten_edges_between(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((3, 4, 5)))
  b = net.add_node(np.ones((5, 4, 3)))
  net.connect(a[0], b[2])
  net.connect(a[1], b[1])
  net.connect(a[2], b[0])
  net.flatten_edges_between(a, b)
  net.check_correct()
  np.testing.assert_allclose(a.tensor, np.ones((60,)))
  np.testing.assert_allclose(b.tensor, np.ones((60,)))


def test_flatten_edges_between_no_edges(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((3)))
  b = net.add_node(np.ones((3)))
  assert net.flatten_edges_between(a, b) is None


def test_flatten_all_edges(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((3, 3, 5, 6, 2, 2)))
  b = net.add_node(np.ones((5, 6, 7)))
  c = net.add_node(np.ones((7,)))
  trace_edge1 = net.connect(a[0], a[1])
  trace_edge2 = net.connect(a[4], a[5])
  split_edge1 = net.connect(a[2], b[0])
  split_edge2 = net.connect(a[3], b[1])
  ok_edge = net.connect(b[2], c[0])
  flat_edges = net.flatten_all_edges()
  net.check_correct()
  assert len(flat_edges) == 3
  assert trace_edge1 not in flat_edges
  assert trace_edge2 not in flat_edges
  assert split_edge1 not in flat_edges
  assert split_edge2 not in flat_edges
  assert ok_edge in flat_edges


def test_contract_between(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((2, 3, 4, 5))
  b_val = np.ones((3, 5, 4, 2))
  a = net.add_node(a_val)
  b = net.add_node(b_val)
  net.connect(a[0], b[3])
  net.connect(b[1], a[3])
  net.connect(a[1], b[0])
  edge_a = a[2]
  edge_b = b[2]
  c = net.contract_between(a, b, name="New Node")
  c.reorder_edges([edge_a, edge_b])
  net.check_correct()
  # Check expected values.
  a_flat = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
  b_flat = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (4, 30))
  final_val = np.matmul(a_flat, b_flat.T)
  assert c.name == "New Node"
  np.testing.assert_allclose(c.tensor, final_val)


def test_contract_between_no_outer_product_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((2, 3, 4))
  b_val = np.ones((5, 6, 7))
  a = net.add_node(a_val)
  b = net.add_node(b_val)
  with pytest.raises(ValueError):
    net.contract_between(a, b)


def test_contract_between_outer_product_no_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a_val = np.ones((2, 3, 4))
  b_val = np.ones((5, 6, 7))
  a = net.add_node(a_val)
  b = net.add_node(b_val)
  c = net.contract_between(a, b, allow_outer_product=True)
  assert c.shape == (2, 3, 4, 5, 6, 7)


def test_contract_parallel(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  edge1 = net.connect(a[0], b[0])
  edge2 = net.connect(a[1], b[1])
  c = net.contract_parallel(edge1)
  assert edge2 not in net
  np.testing.assert_allclose(c.tensor, 2.0)


def test_remove_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  net.connect(a[0], b[0])
  broken_edges_by_name, broken_edges_by_axis = net.remove_node(b)
  assert broken_edges_by_name == {"__Edge_3": a[0]}
  assert broken_edges_by_axis == {0: a[0]}


def test_remove_node_value_error(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  net2 = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2)))
  with pytest.raises(ValueError):
    net2.remove_node(a)
