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
import tensornetwork.linalg
import tensornetwork.linalg.node_linalg


def test_split_node(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right, _ = tn.split_node(a, left_edges, right_edges)
  tn.check_correct({left, right})
  np.testing.assert_allclose(left.tensor, np.zeros((2, 3, 4, 24)))
  np.testing.assert_allclose(right.tensor, np.zeros((24, 5, 6)))


def test_split_node_mixed_order(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)), backend=backend)
  left_edges = []
  for i in [0, 2, 4]:
    left_edges.append(a[i])
  right_edges = []
  for i in [1, 3]:
    right_edges.append(a[i])
  left, right, _ = tn.split_node(a, left_edges, right_edges)
  tn.check_correct({left, right})
  np.testing.assert_allclose(left.tensor, np.zeros((2, 4, 6, 15)))
  np.testing.assert_allclose(right.tensor, np.zeros((15, 3, 5)))


def test_split_node_full_svd(backend):
  unitary1 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
  unitary2 = np.array([[0.0, 1.0], [1.0, 0.0]])
  singular_values = np.array([9.1, 7.5], dtype=np.float32)
  val = np.dot(unitary1, np.dot(np.diag(singular_values), (unitary2.T)))
  a = tn.Node(val, backend=backend)
  e1 = a[0]
  e2 = a[1]
  _, s, _, _, = tn.split_node_full_svd(a, [e1], [e2])
  tn.check_correct(tn.reachable(s))
  np.testing.assert_allclose(s.tensor, np.diag([9.1, 7.5]), rtol=1e-5)


def test_svd_consistency(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")

  original_tensor = np.array(
      [[1.0, 2.0j, 3.0, 4.0], [5.0, 6.0 + 1.0j, 3.0j, 2.0 + 1.0j]],
      dtype=np.complex64)
  node = tn.Node(original_tensor, backend=backend)
  u, vh, _ = tn.split_node(node, [node[0]], [node[1]])
  final_node = tn.contract_between(u, vh)
  np.testing.assert_allclose(final_node.tensor, original_tensor, rtol=1e-6)


def test_svd_consistency_symmetric_real_matrix(backend):
  original_tensor = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 3.0, 2.0]],
                             dtype=np.float64)
  node = tn.Node(original_tensor, backend=backend)
  u, vh, _ = tn.split_node(node, [node[0]], [node[1]])
  final_node = tn.contract_between(u, vh)
  np.testing.assert_allclose(final_node.tensor, original_tensor, rtol=1e-6)


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


def test_split_node_rq(backend):
  a = tn.Node(np.random.rand(2, 3, 4, 5, 6), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, _ = tn.split_node_rq(a, left_edges, right_edges)
  tn.check_correct(tn.reachable(left))
  np.testing.assert_allclose(a.tensor, tn.contract(left[3]).tensor)


def test_split_node_qr(backend):
  a = tn.Node(np.random.rand(2, 3, 4, 5, 6), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, _ = tn.split_node_qr(a, left_edges, right_edges)
  tn.check_correct(tn.reachable(left))
  np.testing.assert_allclose(a.tensor, tn.contract(left[3]).tensor)


def test_split_node_rq_unitarity_complex(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")
  if backend == "jax":
    pytest.skip("Complex QR crashes jax")

  a = tn.Node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3), backend=backend)
  _, q = tn.split_node_rq(a, [a[0]], [a[1]])
  n1 = tn.Node(q.tensor, backend=backend)
  n2 = tn.linalg.node_linalg.conj(q)
  n1[1] ^ n2[1]
  u1 = tn.contract_between(n1, n2)
  n1 = tn.Node(q.tensor, backend=backend)
  n2 = tn.linalg.node_linalg.conj(q)
  n2[0] ^ n1[0]
  u2 = tn.contract_between(n1, n2)

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_rq_unitarity_float(backend):
  a = tn.Node(np.random.rand(3, 3), backend=backend)
  _, q = tn.split_node_rq(a, [a[0]], [a[1]])
  n1 = tn.Node(q.tensor, backend=backend)
  n2 = tn.linalg.node_linalg.conj(q)
  n1[1] ^ n2[1]
  u1 = tn.contract_between(n1, n2)
  n1 = tn.Node(q.tensor, backend=backend)
  n2 = tn.Node(q.tensor, backend=backend)
  n2[0] ^ n1[0]
  u2 = tn.contract_between(n1, n2)

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_qr_unitarity_complex(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")
  if backend == "jax":
    pytest.skip("Complex QR crashes jax")

  a = tn.Node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3), backend=backend)
  q, _ = tn.split_node_qr(a, [a[0]], [a[1]])
  n1 = tn.Node(q.tensor, backend=backend)
  n2 = tn.linalg.node_linalg.conj(q)
  n1[1] ^ n2[1]
  u1 = tn.contract_between(n1, n2)

  n1 = tn.Node(q.tensor, backend=backend)
  n2 = tn.linalg.node_linalg.conj(q)
  n2[0] ^ n1[0]
  u2 = tn.contract_between(n1, n2)

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_qr_unitarity_float(backend):
  a = tn.Node(np.random.rand(3, 3), backend=backend)
  q, _ = tn.split_node_qr(a, [a[0]], [a[1]])
  n1 = tn.Node(q.tensor, backend=backend)
  n2 = tn.linalg.node_linalg.conj(q)
  n1[1] ^ n2[1]
  u1 = tn.contract_between(n1, n2)

  n1 = tn.Node(q.tensor, backend=backend)
  n2 = tn.Node(q.tensor, backend=backend)
  n2[0] ^ n1[0]
  u2 = tn.contract_between(n1, n2)

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))
