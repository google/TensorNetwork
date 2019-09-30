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
import tensornetwork as tn
import pytest
import numpy as np
import tensorflow as tf
import torch
import jax
from jax.config import config


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
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)))
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
