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
from tensornetwork.backends.base_backend import BaseBackend


def test_replicate_nodes(backend):
  a = tn.Node(np.random.rand(10, 10), backend=backend)
  b = tn.Node(np.random.rand(10, 10), backend=backend)
  c = tn.Node(np.random.rand(10, 10), backend=backend)
  tn.connect(a[1], b[0])
  tn.connect(b[1], c[0])
  [a_copy, b_copy] = tn.replicate_nodes([a, b])
  assert b_copy in tn.reachable([a_copy])
  assert not set([a_copy, b_copy]).issubset(tn.reachable([c]))
  assert len(b_copy.get_all_dangling()) == 1


def test_split_node_full_svd_names(backend):
  a = tn.Node(np.random.rand(10, 10), backend=backend)
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


def test_split_node_relative_tolerance(backend):
  absolute = tn.Node(np.diag([2.0, 1.0, 0.2, 0.1]), backend=backend)
  relative = tn.Node(np.diag([2.0, 1.0, 0.2, 0.1]), backend=backend)
  max_truncation_err = 0.2

  _, _, trunc_sv_absolute, = tn.split_node(
      node=absolute,
      left_edges=[absolute[0]],
      right_edges=[absolute[1]],
      max_truncation_err=max_truncation_err,
      relative=False)
  _, _, trunc_sv_relative, = tn.split_node(
      node=relative,
      left_edges=[relative[0]],
      right_edges=[relative[1]],
      max_truncation_err=max_truncation_err,
      relative=True)
  np.testing.assert_almost_equal(trunc_sv_absolute, [0.1])
  np.testing.assert_almost_equal(trunc_sv_relative, [0.2, 0.1])


def test_split_node_full_svd_relative_tolerance(backend):
  absolute = tn.Node(np.diag([2.0, 1.0, 0.2, 0.1]), backend=backend)
  relative = tn.Node(np.diag([2.0, 1.0, 0.2, 0.1]), backend=backend)
  max_truncation_err = 0.2

  _, _, _, trunc_sv_absolute, = tn.split_node_full_svd(
      node=absolute,
      left_edges=[absolute[0]],
      right_edges=[absolute[1]],
      max_truncation_err=max_truncation_err,
      relative=False)
  _, _, _, trunc_sv_relative, = tn.split_node_full_svd(
      node=relative,
      left_edges=[relative[0]],
      right_edges=[relative[1]],
      max_truncation_err=max_truncation_err,
      relative=True)
  np.testing.assert_almost_equal(trunc_sv_absolute, [0.1])
  np.testing.assert_almost_equal(trunc_sv_relative, [0.2, 0.1])


def test_split_node_rq_names(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)), backend=backend)
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


def test_split_node_qr_names(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)), backend=backend)
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


def test_split_node_names(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)), backend=backend)
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


def test_split_node_rq_unitarity_complex(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")
  if backend == "jax":
    pytest.skip("Complex QR crashes jax")

  a = tn.Node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3), backend=backend)
  r, q = tn.split_node_rq(a, [a[0]], [a[1]])
  r[1] | q[0]
  qbar = tn.conj(q)
  q[1] ^ qbar[1]
  u1 = q @ qbar
  qbar[0] ^ q[0]
  u2 = qbar @ q

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_rq_unitarity_float(backend):
  a = tn.Node(np.random.rand(3, 3), backend=backend)
  r, q = tn.split_node_rq(a, [a[0]], [a[1]])
  r[1] | q[0]
  qbar = tn.conj(q)
  q[1] ^ qbar[1]
  u1 = q @ qbar
  qbar[0] ^ q[0]
  u2 = qbar @ q

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_rq(backend):
  a = tn.Node(np.random.rand(2, 3, 4, 5, 6), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_rq(a, left_edges, right_edges)
  tn.check_correct([left, right])
  np.testing.assert_allclose(a.tensor, tn.contract(left[3]).tensor)


def test_split_node_qr_unitarity_complex(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")
  if backend == "jax":
    pytest.skip("Complex QR crashes jax")

  a = tn.Node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3), backend=backend)
  q, r = tn.split_node_qr(a, [a[0]], [a[1]])
  q[1] | r[0]
  qbar = tn.conj(q)
  q[1] ^ qbar[1]
  u1 = q @ qbar
  qbar[0] ^ q[0]
  u2 = qbar @ q

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_qr_unitarity_float(backend):
  a = tn.Node(np.random.rand(3, 3), backend=backend)
  q, r = tn.split_node_qr(a, [a[0]], [a[1]])
  q[1] | r[0]
  qbar = tn.conj(q)
  q[1] ^ qbar[1]
  u1 = q @ qbar
  qbar[0] ^ q[0]
  u2 = qbar @ q

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_qr(backend):
  a = tn.Node(np.random.rand(2, 3, 4, 5, 6), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_qr(a, left_edges, right_edges)
  tn.check_correct([left, right])
  np.testing.assert_allclose(a.tensor, tn.contract(left[3]).tensor)


def test_conj(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")

  a = tn.Node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3), backend=backend)
  abar = tn.conj(a)
  np.testing.assert_allclose(abar.tensor, a.backend.conj(a.tensor))


def test_transpose(backend):
  a = tn.Node(np.random.rand(1, 2, 3, 4, 5), backend=backend)
  order = [a[n] for n in reversed(range(5))]
  transpa = tn.transpose(a, [4, 3, 2, 1, 0])
  a.reorder_edges(order)
  np.testing.assert_allclose(a.tensor, transpa.tensor)


def test_reachable(backend):
  nodes = [tn.Node(np.random.rand(2, 2, 2), backend=backend) for _ in range(10)]
  _ = [nodes[n][0] ^ nodes[n + 1][1] for n in range(len(nodes) - 1)]
  assert set(nodes) == tn.reachable(nodes[0])


def test_reachable_2(backend):
  a = tn.Node(np.zeros((3, 5)), backend=backend)
  b = tn.Node(np.zeros((3, 4, 5)), backend=backend)
  e1 = tn.connect(a[0], b[0])
  e2 = tn.connect(a[1], b[2])
  nodes = [a, b]
  edges = [e1, e2]
  assert set(nodes) == tn.reachable(edges[0])
  assert set(nodes) == tn.reachable(edges)


def test_reachable_disconnected_1(backend):
  nodes = [tn.Node(np.random.rand(2, 2, 2), backend=backend) for _ in range(4)]
  nodes[0][1] ^ nodes[1][0]
  nodes[2][1] ^ nodes[3][0]
  assert set(tn.reachable([nodes[0], nodes[2]])) == set(nodes)

  assert set(tn.reachable([nodes[0]])) == {nodes[0], nodes[1]}
  assert set(tn.reachable([nodes[1]])) == {nodes[0], nodes[1]}
  assert set(tn.reachable([nodes[0], nodes[1]])) == {nodes[0], nodes[1]}

  assert set(tn.reachable([nodes[2]])) == {nodes[2], nodes[3]}
  assert set(tn.reachable([nodes[3]])) == {nodes[2], nodes[3]}
  assert set(tn.reachable([nodes[2], nodes[3]])) == {nodes[2], nodes[3]}

  assert set(tn.reachable([nodes[0], nodes[1], nodes[2]])) == set(nodes)
  assert set(tn.reachable([nodes[0], nodes[1], nodes[3]])) == set(nodes)
  assert set(tn.reachable([nodes[0], nodes[2], nodes[3]])) == set(nodes)
  assert set(tn.reachable([nodes[1], nodes[2], nodes[3]])) == set(nodes)


def test_reachable_disconnected_2(backend):
  nodes = [tn.Node(np.random.rand(2, 2, 2), backend=backend) for _ in range(4)]
  nodes[1][1] ^ nodes[2][0]  #connect 2nd and third node
  assert set(tn.reachable([nodes[0],
                           nodes[1]])) == {nodes[0], nodes[1], nodes[2]}
  nodes[2][1] ^ nodes[3][0]  #connect third and fourth node
  assert set(tn.reachable([nodes[0], nodes[1]])) == set(nodes)


def test_subgraph_sanity(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  a[0] ^ b[0]
  edges = tn.get_subgraph_dangling({a})
  assert edges == {a[0], a[1]}


def test_subgraph_disconnected_nodes(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  c = tn.Node(np.eye(2), backend=backend)
  a[0] ^ b[0]
  b[1] ^ c[1]
  edges = tn.get_subgraph_dangling({a, c})
  assert edges == {a[0], a[1], c[0], c[1]}


def test_full_graph_subgraph_dangling(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  c = tn.Node(np.eye(2), backend=backend)
  a[0] ^ b[0]
  b[1] ^ c[1]
  edges = tn.get_subgraph_dangling({a, b, c})
  assert edges == {a[1], c[0]}


def test_reduced_density(backend):
  a = tn.Node(np.random.rand(3, 3, 3), name="A", backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), name="B", backend=backend)
  c = tn.Node(np.random.rand(3, 3, 3), name="C", backend=backend)
  edges = tn.get_all_edges({a, b, c})

  node_dict, edge_dict = tn.reduced_density([a[0], b[1], c[2]])

  assert not a[0].is_dangling()
  assert not b[1].is_dangling()
  assert not c[2].is_dangling()
  assert a[1].is_dangling() & a[2].is_dangling()
  assert b[0].is_dangling() & b[2].is_dangling()
  assert c[0].is_dangling() & c[1].is_dangling()

  for node in {a, b, c}:
    assert node_dict[node].name == node.name
  for edge in edges:
    assert edge_dict[edge].name == edge.name


def test_reduced_density_nondangling(backend):
  a = tn.Node(np.random.rand(3, 3, 3), name="A", backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), name="B", backend=backend)
  c = tn.Node(np.random.rand(3, 3, 3), name="C", backend=backend)

  a[0] ^ b[1]
  b[2] ^ c[1]

  err_msg = "traced_out_edges must only include dangling edges!"
  with pytest.raises(ValueError, match=err_msg):
    tn.reduced_density([a[0], b[1], c[1]])


def test_reduced_density_contraction(backend):
  if backend == "pytorch":
    pytest.skip("pytorch doesn't support complex numbers")
  a = tn.Node(
      np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=np.complex64),
      backend=backend)
  tn.reduced_density([a[0]])
  result = tn.contractors.greedy(tn.reachable(a), ignore_edge_order=True)
  np.testing.assert_allclose(result.tensor, np.eye(2))


def test_switch_backend(backend):
  a = tn.Node(np.random.rand(3, 3, 3), name="A", backend="numpy")
  b = tn.Node(np.random.rand(3, 3, 3), name="B", backend="numpy")
  c = tn.Node(np.random.rand(3, 3, 3), name="C", backend="numpy")
  nodes = [a, b, c]
  tn.switch_backend(nodes, backend)
  assert nodes[0].backend.name == backend


def test_norm_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    tn.norm(node)


def test_conj_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    tn.conj(node)


def test_transpose_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    tn.transpose(node, permutation=[])


def test_split_node_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    tn.split_node(node, left_edges=[], right_edges=[])


def test_split_node_qr_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    tn.split_node_qr(node, left_edges=[], right_edges=[])


def test_split_node_rq_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    tn.split_node_rq(node, left_edges=[], right_edges=[])


def test_split_node_full_svd_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    tn.split_node_full_svd(node, left_edges=[], right_edges=[])


def test_reachable_raises_value_error():
  with pytest.raises(ValueError):
    tn.reachable({})


def test_check_correct_raises_value_error_1(backend):
  a = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  edge = a.edges[0]
  edge.node1 = b
  edge.node2 = b
  with pytest.raises(ValueError):
    tn.check_correct({a, b})


def test_check_correct_raises_value_error_2(backend):
  a = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  edge = a.edges[0]
  edge.axis1 = -1
  with pytest.raises(ValueError):
    tn.check_correct({a, b})


def test_get_all_nodes(backend):
  a = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  b = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  edge = tn.connect(a[0], b[0])
  assert tn.get_all_nodes({edge}) == {a, b}


def test_contract_trace_edges(backend):
  a = tn.Node(np.random.rand(3, 3, 3), backend=backend)
  with pytest.raises(ValueError):
    tn.contract_trace_edges(a)


def test_switch_backend_raises_error(backend):
  a = tn.Node(np.random.rand(3, 3, 3))
  a.backend = BaseBackend()
  with pytest.raises(NotImplementedError):
    tn.switch_backend({a}, backend)


def test_split_node_orig_shape(backend):
  n1 = tn.Node(np.random.rand(3, 4, 5), backend=backend)
  tn.split_node(n1, [n1[0], n1[2]], [n1[1]])
  np.testing.assert_allclose(n1.shape, (3, 4, 5))


def test_split_node_full_svd_orig_shape(backend):
  n1 = tn.Node(np.random.rand(3, 4, 5), backend=backend)
  tn.split_node_full_svd(n1, [n1[0], n1[2]], [n1[1]])
  np.testing.assert_allclose(n1.shape, (3, 4, 5))


def test_split_node_rq_orig_shape(backend):
  n1 = tn.Node(np.random.rand(3, 4, 5), backend=backend)
  tn.split_node_rq(n1, [n1[0], n1[2]], [n1[1]])
  np.testing.assert_allclose(n1.shape, (3, 4, 5))


def test_split_node_qr_orig_shape(backend):
  n1 = tn.Node(np.random.rand(3, 4, 5), backend=backend)
  tn.split_node_qr(n1, [n1[0], n1[2]], [n1[1]])
  np.testing.assert_allclose(n1.shape, (3, 4, 5))
