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


def test_tnwork_copy_conj(backend):
  if backend == "pytorch":
    pytest.skip("Pytorch does not support complex numbers")
  a = tn.Node(np.array([1.0 + 2.0j, 2.0 - 1.0j]))
  nodes, _ = tn.copy({a}, conjugate=True)
  np.testing.assert_allclose(nodes[a].tensor, np.array([1.0 - 2.0j,
                                                        2.0 + 1.0j]))


def test_tnwork_copy(backend):
  a = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  c = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  a[0] ^ b[1]
  a[1] ^ c[2]
  b[2] ^ c[0]
  node_dict, _ = tn.copy({a, b, c})
  tn.check_correct({node_dict[n] for n in {a, b, c}})

  res = a @ b @ c
  res_copy = node_dict[a] @ node_dict[b] @ node_dict[c]
  np.testing.assert_allclose(res.tensor, res_copy.tensor)


def test_tnwork_copy_names(backend):
  a = tn.Node(np.random.rand(3, 3, 3), name='a', backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), name='b', backend=backend)
  c = tn.Node(np.random.rand(3, 3, 3), name='c', backend=backend)
  a[0] ^ b[1]
  b[2] ^ c[0]
  node_dict, edge_dict = tn.copy({a, b, c})
  for node in {a, b, c}:
    assert node_dict[node].name == node.name
  for edge in tn.get_all_edges({a, b, c}):
    assert edge_dict[edge].name == edge.name


def test_tnwork_copy_identities(backend):
  a = tn.Node(np.random.rand(3, 3, 3), name='a', backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), name='b', backend=backend)
  c = tn.Node(np.random.rand(3, 3, 3), name='c', backend=backend)
  a[0] ^ b[1]
  b[2] ^ c[0]
  node_dict, edge_dict = tn.copy({a, b, c})
  for node in {a, b, c}:
    assert not node_dict[node] is node
  for edge in tn.get_all_edges({a, b, c}):
    assert not edge_dict[edge] is edge


def test_tnwork_copy_subgraph(backend):
  a = tn.Node(np.random.rand(3, 3, 3), name='a', backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), name='b', backend=backend)
  c = tn.Node(np.random.rand(3, 3, 3), name='c', backend=backend)
  a[0] ^ b[1]
  edge2 = b[2] ^ c[0]
  node_dict, edge_dict = tn.copy({a, b})
  cut_edge = edge_dict[edge2]
  assert cut_edge.is_dangling()
  assert cut_edge.axis1 == 2
  assert cut_edge.get_nodes() == [node_dict[b], None]
  assert len(a.get_all_nondangling()) == 1


def test_tnwork_copy_subgraph_2(backend):
  a = tn.Node(np.random.rand(3, 3, 3), name='a', backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), name='b', backend=backend)
  c = tn.Node(np.random.rand(3, 3, 3), name='c', backend=backend)
  a[0] ^ b[1]
  edge2 = c[2] ^ b[0]
  node_dict, edge_dict = tn.copy({a, b})
  cut_edge = edge_dict[edge2]
  assert cut_edge.is_dangling()
  assert cut_edge.axis1 == 0
  assert cut_edge.get_nodes() == [node_dict[b], None]
  assert len(a.get_all_nondangling()) == 1


def test_connect_axis_names(backend):
  a = tn.Node(np.ones((3,)), name="a", axis_names=["one"], backend=backend)
  b = tn.Node(np.ones((3,)), name="b", axis_names=["one"], backend=backend)
  tn.connect(a["one"], b["one"])
  assert a.edges == b.edges


def test_connect_twice_edge_axis_value_error(backend):
  a = tn.Node(np.array([2.]), name="a", backend=backend)
  b = tn.Node(np.array([2.]), name="b", backend=backend)
  tn.connect(a[0], b[0])
  with pytest.raises(ValueError):
    tn.connect(a[0], b[0])


def test_connect_same_edge_value_error(backend):
  a = tn.Node(np.eye(2), backend=backend)
  with pytest.raises(ValueError):
    tn.connect(a[0], a[0])


def test_disconnect_edge(backend):
  a = tn.Node(np.array([1.0] * 5), "a", backend=backend)
  b = tn.Node(np.array([1.0] * 5), "b", backend=backend)
  e = tn.connect(a[0], b[0])
  assert not e.is_dangling()
  dangling_edge_1, dangling_edge_2 = tn.disconnect(e)
  assert dangling_edge_1.is_dangling()
  assert dangling_edge_2.is_dangling()
  assert a.get_edge(0) == dangling_edge_1
  assert b.get_edge(0) == dangling_edge_2


def test_disconnect_dangling_edge_value_error(backend):
  a = tn.Node(np.eye(2), backend=backend)
  with pytest.raises(ValueError):
    tn.disconnect(a[0])


def test_contract_trace_single_node(backend):
  a = tn.Node(np.ones([10, 10]), name="a", backend=backend)
  edge = tn.connect(a[0], a[1], "edge")
  result = tn.contract(edge)
  np.testing.assert_allclose(result.tensor, 10.0)


def test_contract_single_edge(backend):
  a = tn.Node(np.array([1.0] * 5), "a", backend=backend)
  b = tn.Node(np.array([1.0] * 5), "b", backend=backend)
  e = tn.connect(a[0], b[0])
  c = tn.contract(e)
  tn.check_correct({c})
  val = c.tensor
  np.testing.assert_allclose(val, 5.0)


def test_contract_name_contracted_node(backend):
  node = tn.Node(np.eye(2), name="Identity Matrix", backend=backend)
  assert node.name == "Identity Matrix"
  edge = tn.connect(node[0], node[1], name="Trace Edge")
  assert edge.name == "Trace Edge"
  final_result = tn.contract(edge, name="Trace Of Identity")
  assert final_result.name == "Trace Of Identity"


def test_contract_edge_twice_value_error(backend):
  a = tn.Node(np.eye(2), backend=backend)
  e = tn.connect(a[0], a[1], name="edge")
  tn.contract(e)
  with pytest.raises(ValueError):
    tn.contract(e)


def test_contract_dangling_edge_value_error(backend):
  a = tn.Node(np.array([1.0]), backend=backend)
  e = a[0]
  with pytest.raises(ValueError):
    tn.contract(e)


def test_contract_copy_node(backend):
  a = tn.Node(np.array([1, 2, 3]), backend=backend)
  b = tn.Node(np.array([10, 20, 30]), backend=backend)
  c = tn.Node(np.array([5, 6, 7]), backend=backend)
  d = tn.Node(np.array([1, -1, 1]), backend=backend)
  cn = tn.CopyNode(rank=4, dimension=3, backend=backend)
  tn.connect(a[0], cn[0])
  tn.connect(b[0], cn[1])
  tn.connect(c[0], cn[2])
  tn.connect(d[0], cn[3])

  val = tn.contract_copy_node(cn)
  result = val.tensor
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 50 - 240 + 630)


def test_contract_copy_node_dangling_edge_value_error(backend):
  a = tn.Node(np.array([1, 2, 3]), backend=backend)
  b = tn.Node(np.array([10, 20, 30]), backend=backend)
  c = tn.Node(np.array([5, 6, 7]), backend=backend)
  cn = tn.CopyNode(rank=4, dimension=3, backend=backend)
  tn.connect(a[0], cn[0])
  tn.connect(b[0], cn[1])
  tn.connect(c[0], cn[2])

  with pytest.raises(ValueError):
    tn.contract_copy_node(cn)


def test_outer_product(backend):
  a = tn.Node(np.ones((2, 4, 5)), name="A", backend=backend)
  b = tn.Node(np.ones((4, 3, 6)), name="B", backend=backend)
  c = tn.Node(np.ones((3, 2)), name="C", backend=backend)
  tn.connect(a[1], b[0])
  tn.connect(a[0], c[1])
  tn.connect(b[1], c[0])
  # Purposely leave b's 3rd axis undefined.
  d = tn.outer_product(a, b, name="D")
  tn.check_correct({c, d})
  assert d.shape == (2, 4, 5, 4, 3, 6)
  np.testing.assert_allclose(d.tensor, np.ones((2, 4, 5, 4, 3, 6)))
  assert d.name == "D"


@pytest.mark.parametrize("a, b, expected_val, expected_shape, expected_name", [
    pytest.param(
        np.ones((2, 4, 5)), np.ones(()), np.ones((2, 4, 5)), (2, 4, 5), "C"),
    pytest.param(
        np.ones(()), np.ones((2, 4, 5)), np.ones((2, 4, 5)), (2, 4, 5), "C"),
])
def test_outer_product_without_legs(a, b, expected_val, expected_shape,
                                    expected_name, backend):
  node1 = tn.Node(a, name="A", backend=backend)
  node2 = tn.Node(b, name="B", backend=backend)

  node3 = tn.outer_product(node1, node2, name=expected_name)
  np.testing.assert_allclose(node3.tensor, expected_val)
  assert node3.shape == expected_shape
  assert node3.name == expected_name


def test_get_all_nondangling(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  edge1 = tn.connect(a[0], b[0])
  c = tn.Node(np.eye(2), backend=backend)
  d = tn.Node(np.eye(2), backend=backend)
  edge2 = tn.connect(c[0], d[0])
  edge3 = tn.connect(a[1], c[1])
  assert {edge1, edge2, edge3} == tn.get_all_nondangling({a, b, c, d})


def test_get_all_edges(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  assert {a[0], a[1], b[0], b[1]} == tn.get_all_edges({a, b})


def test_check_connected_value_error(backend):
  a = tn.Node(np.array([2, 2.]), backend=backend)
  b = tn.Node(np.array([2, 2.]), backend=backend)
  tn.connect(a[0], b[0])
  c = tn.Node(np.array([2, 2.]), backend=backend)
  d = tn.Node(np.array([2, 2.]), backend=backend)
  tn.connect(c[0], d[0])
  with pytest.raises(ValueError):
    tn.check_connected({a, b, c, d})


def test_flatten_trace_edges(backend):
  a = tn.Node(np.zeros((2, 3, 4, 3, 5, 5)), backend=backend)
  c = tn.Node(np.zeros((2, 4)), backend=backend)
  e1 = tn.connect(a[1], a[3])
  e2 = tn.connect(a[4], a[5])
  external_1 = tn.connect(a[0], c[0])
  external_2 = tn.connect(c[1], a[2])
  new_edge = tn.flatten_edges([e1, e2], "New Edge")
  tn.check_correct({a, c})
  assert a.shape == (2, 4, 15, 15)
  assert a.edges == [external_1, external_2, new_edge, new_edge]
  assert new_edge.name == "New Edge"


def test_flatten_edges_standard(backend):
  a = tn.Node(np.zeros((2, 3, 5)), name="A", backend=backend)
  b = tn.Node(np.zeros((2, 3, 4, 5)), name="B", backend=backend)
  e1 = tn.connect(a[0], b[0], "Edge_1_1")
  e2 = tn.connect(a[2], b[3], "Edge_2_3")
  edge_a_1 = a[1]
  edge_b_1 = b[1]
  edge_b_2 = b[2]
  new_edge = tn.flatten_edges([e1, e2], new_edge_name="New Edge")
  assert a.shape == (3, 10)
  assert b.shape == (3, 4, 10)
  assert a.edges == [edge_a_1, new_edge]
  assert b.edges == [edge_b_1, edge_b_2, new_edge]
  tn.check_correct({a, b})


def test_flatten_edges_dangling(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5)), name="A", backend=backend)
  e1 = a[0]
  e2 = a[1]
  e3 = a[2]
  e4 = a[3]
  flattened_edge = tn.flatten_edges([e1, e3], new_edge_name="New Edge")
  assert a.shape == (3, 5, 8)
  assert a.edges == [e2, e4, flattened_edge]
  assert flattened_edge.name == "New Edge"
  tn.check_correct({a})


def test_flatten_edges_empty_list_value_error(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  tn.connect(a[0], b[0])
  with pytest.raises(ValueError):
    tn.flatten_edges([])


def test_flatten_edges_different_nodes_value_error(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  c = tn.Node(np.eye(2), backend=backend)
  e1 = tn.connect(a[0], b[0])
  e2 = tn.connect(a[1], c[0])
  tn.connect(b[1], c[1])
  with pytest.raises(ValueError):
    tn.flatten_edges([e1, e2])


def test_split_trace_edge(backend):
  a = tn.Node(np.zeros((2, 6, 4, 6, 5, 5)), backend=backend)
  c = tn.Node(np.zeros((2, 4)), backend=backend)
  e1 = tn.connect(a[1], a[3])
  e2 = tn.connect(a[4], a[5])
  external_1 = tn.connect(a[0], c[0])
  external_2 = tn.connect(c[1], a[2])
  shape = (2, 1, 3)
  new_edge_names = ["New Edge 2", "New Edge 1", "New Edge 3"]
  new_edges = tn.split_edge(e1, shape, new_edge_names)
  assert a.shape == (2, 4, 5, 5) + shape + shape
  assert a.edges == [external_1, external_2, e2, e2, *new_edges, *new_edges]
  for new_edge, dim in zip(new_edges, shape):
    assert new_edge.dimension == dim
  for new_edge, new_name in zip(new_edges, new_edge_names):
    assert new_edge.name == new_name
  tn.check_correct({a, c})


def test_split_edges_standard(backend):
  a = tn.Node(np.zeros((6, 3, 5)), name="A", backend=backend)
  b = tn.Node(np.zeros((2, 4, 6, 3)), name="B", backend=backend)
  e1 = tn.connect(a[0], b[2], "Edge_1_1")  # to be split
  e2 = tn.connect(a[1], b[3], "Edge_1_2")  # background standard edge
  edge_a_2 = a[2]  # dangling
  edge_b_0 = b[0]  # dangling
  edge_b_1 = b[1]  # dangling
  shape = (2, 1, 3)
  new_edge_names = ["New Edge 2", "New Edge 1", "New Edge 3"]
  new_edges = tn.split_edge(e1, shape, new_edge_names)
  assert a.shape == (3, 5) + shape
  assert b.shape == (2, 4, 3) + shape
  assert a.edges == [e2, edge_a_2, *new_edges]
  assert b.edges == [edge_b_0, edge_b_1, e2, *new_edges]
  for new_edge, dim in zip(new_edges, shape):
    assert new_edge.dimension == dim
  for new_edge, new_name in zip(new_edges, new_edge_names):
    assert new_edge.name == new_name
  tn.check_correct({a, b})


def test_split_edges_standard_contract_between(backend):
  a = tn.Node(np.random.randn(6, 3, 5), name="A", backend=backend)
  b = tn.Node(np.random.randn(2, 4, 6, 3), name="B", backend=backend)
  e1 = tn.connect(a[0], b[2], "Edge_1_1")  # to be split
  tn.connect(a[1], b[3], "Edge_1_2")  # background standard edge
  node_dict, _ = tn.copy({a, b})
  c_prior = node_dict[a] @ node_dict[b]
  shape = (2, 1, 3)
  tn.split_edge(e1, shape)
  tn.check_correct({a, b})
  c_post = tn.contract_between(a, b)
  np.testing.assert_allclose(c_prior.tensor, c_post.tensor)


def test_split_edges_dangling(backend):
  a = tn.Node(np.zeros((2, 10, 4, 5)), name="A", backend=backend)
  e1 = a[0]
  e2 = a[1]
  e3 = a[2]
  e4 = a[3]
  shape = (2, 5)
  new_edge_names = ["New Edge 2", "New Edge 5"]
  new_edges = tn.split_edge(e2, shape, new_edge_names)
  assert a.shape == (2, 4, 5, 2, 5)
  assert a.edges == [e1, e3, e4, *new_edges]
  for new_edge, dim in zip(new_edges, shape):
    assert new_edge.dimension == dim
  for new_edge, new_name in zip(new_edges, new_edge_names):
    assert new_edge.name == new_name
  tn.check_correct({a})


def test_split_edges_dimension_mismatch_value_error(backend):
  a = tn.Node(np.eye(5), backend=backend)
  e1 = tn.connect(a[0], a[1])
  with pytest.raises(ValueError):
    tn.split_edge(e1, (2, 2))


def test_get_shared_edges(backend):
  a = tn.Node(np.ones((2, 2, 2)), backend=backend)
  b = tn.Node(np.ones((2, 2, 2)), backend=backend)
  c = tn.Node(np.ones((2, 2, 2)), backend=backend)
  e1 = tn.connect(a[0], b[0])
  e2 = tn.connect(b[1], c[1])
  e3 = tn.connect(a[2], b[2])
  assert tn.get_shared_edges(a, b) == {e1, e3}
  assert tn.get_shared_edges(b, c) == {e2}


def test_get_parallel_edge(backend):
  a = tn.Node(np.ones((2,) * 5), backend=backend)
  b = tn.Node(np.ones((2,) * 5), backend=backend)
  edges = set()
  for i in {0, 1, 3}:
    edges.add(tn.connect(a[i], b[i]))
  for e in edges:
    assert set(tn.get_parallel_edges(e)) == edges


def test_flatten_edges_between(backend):
  a = tn.Node(np.ones((3, 4, 5)), backend=backend)
  b = tn.Node(np.ones((5, 4, 3)), backend=backend)
  tn.connect(a[0], b[2])
  tn.connect(a[1], b[1])
  tn.connect(a[2], b[0])
  tn.flatten_edges_between(a, b)
  tn.check_correct({a, b})
  np.testing.assert_allclose(a.tensor, np.ones((60,)))
  np.testing.assert_allclose(b.tensor, np.ones((60,)))


def test_flatten_edges_between_no_edges(backend):
  a = tn.Node(np.ones((3)), backend=backend)
  b = tn.Node(np.ones((3)), backend=backend)
  assert tn.flatten_edges_between(a, b) is None


def test_flatten_all_edges(backend):
  a = tn.Node(np.ones((3, 3, 5, 6, 2, 2)), backend=backend)
  b = tn.Node(np.ones((5, 6, 7)), backend=backend)
  c = tn.Node(np.ones((7,)), backend=backend)
  trace_edge1 = tn.connect(a[0], a[1])
  trace_edge2 = tn.connect(a[4], a[5])
  split_edge1 = tn.connect(a[2], b[0])
  split_edge2 = tn.connect(a[3], b[1])
  ok_edge = tn.connect(b[2], c[0])
  flat_edges = tn.flatten_all_edges({a, b, c})
  tn.check_correct({a, b, c})
  assert len(flat_edges) == 3
  assert trace_edge1 not in flat_edges
  assert trace_edge2 not in flat_edges
  assert split_edge1 not in flat_edges
  assert split_edge2 not in flat_edges
  assert ok_edge in flat_edges


def test_contract_between(backend):
  a_val = np.random.rand(2, 3, 4, 5)
  b_val = np.random.rand(3, 5, 6, 2)
  a = tn.Node(a_val, backend=backend)
  b = tn.Node(b_val, backend=backend)
  tn.connect(a[0], b[3])
  tn.connect(b[1], a[3])
  tn.connect(a[1], b[0])
  output_axis_names = ["a2", "b2"]
  c = tn.contract_between(a, b, name="New Node", axis_names=output_axis_names)
  tn.check_correct({c})
  # Check expected values.
  a_flat = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
  b_flat = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (6, 30))
  final_val = np.matmul(a_flat, b_flat.T)
  assert c.name == "New Node"
  assert c.axis_names == output_axis_names
  np.testing.assert_allclose(c.tensor, final_val)


def test_contract_between_output_edge_order(backend):
  a_val = np.random.rand(2, 3, 4, 5)
  b_val = np.random.rand(3, 5, 6, 2)
  a = tn.Node(a_val, backend=backend)
  b = tn.Node(b_val, backend=backend)
  tn.connect(a[0], b[3])
  tn.connect(b[1], a[3])
  tn.connect(a[1], b[0])
  output_axis_names = ["b2", "a2"]
  c = tn.contract_between(
      a,
      b,
      name="New Node",
      axis_names=output_axis_names,
      output_edge_order=[b[2], a[2]])
  # Check expected values.
  a_flat = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
  b_flat = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (6, 30))
  final_val = np.matmul(a_flat, b_flat.T)
  assert c.name == "New Node"
  assert c.axis_names == output_axis_names
  np.testing.assert_allclose(c.tensor, final_val.T)


def test_contract_between_no_outer_product_value_error(backend):
  a_val = np.ones((2, 3, 4))
  b_val = np.ones((5, 6, 7))
  a = tn.Node(a_val, backend=backend)
  b = tn.Node(b_val, backend=backend)
  with pytest.raises(ValueError):
    tn.contract_between(a, b)


def test_contract_between_outer_product_no_value_error(backend):
  a_val = np.ones((2, 3, 4))
  b_val = np.ones((5, 6, 7))
  a = tn.Node(a_val, backend=backend)
  b = tn.Node(b_val, backend=backend)
  output_axis_names = ["a0", "a1", "a2", "b0", "b1", "b2"]
  c = tn.contract_between(
      a, b, allow_outer_product=True, axis_names=output_axis_names)
  assert c.shape == (2, 3, 4, 5, 6, 7)
  assert c.axis_names == output_axis_names


def test_contract_between_outer_product_output_edge_order(backend):
  a_val = np.ones((2, 3, 4))
  b_val = np.ones((5, 6, 7))
  a = tn.Node(a_val, backend=backend)
  b = tn.Node(b_val, backend=backend)
  output_axis_names = ["b0", "b1", "a0", "b2", "a1", "a2"]
  c = tn.contract_between(
      a,
      b,
      allow_outer_product=True,
      output_edge_order=[b[0], b[1], a[0], b[2], a[1], a[2]],
      axis_names=output_axis_names)
  assert c.shape == (5, 6, 2, 7, 3, 4)
  assert c.axis_names == output_axis_names


def test_contract_between_trace(backend):
  a_val = np.ones((2, 3, 2, 4))
  a = tn.Node(a_val, backend=backend)
  tn.connect(a[0], a[2])
  c = tn.contract_between(a, a, axis_names=["1", "3"])
  assert c.shape == (3, 4)
  assert c.axis_names == ["1", "3"]


def test_contract_between_trace_output_edge_order(backend):
  a_val = np.ones((2, 3, 2, 4))
  a = tn.Node(a_val, backend=backend)
  tn.connect(a[0], a[2])
  c = tn.contract_between(
      a, a, output_edge_order=[a[3], a[1]], axis_names=["3", "1"])
  assert c.shape == (4, 3)
  assert c.axis_names == ["3", "1"]


def test_contract_parallel(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  edge1 = tn.connect(a[0], b[0])
  tn.connect(a[1], b[1])
  c = tn.contract_parallel(edge1)
  np.testing.assert_allclose(c.tensor, 2.0)


def test_remove_node(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  tn.connect(a[0], b[0])
  broken_edges_by_name, broken_edges_by_axis = tn.remove_node(b)
  assert broken_edges_by_name == {"0": a[0]}
  assert broken_edges_by_axis == {0: a[0]}

def test_from_topology(backend):
  #pylint: disable=unbalanced-tuple-unpacking
  x, y, z = tn.from_topology(
      "abc,bceg,adef", 
      [np.ones((2,) * n) for n in [3, 4, 4]],
      backend=backend)
  assert x.axis_names == ['a', 'b', 'c']
  assert y.axis_names == ['b', 'c', 'e', 'g']
  assert z.axis_names == ['a', 'd', 'e', 'f']
  assert x['a'] is z['a']
  assert x['b'] is y['b']
  assert x['c'] is y['c']
  assert z['d'].is_dangling()
  assert y['e'] is z['e']
  assert z['f'].is_dangling()
  assert y['g'].is_dangling()
