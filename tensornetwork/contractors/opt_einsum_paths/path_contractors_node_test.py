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

import numpy as np
import pytest
from tensornetwork import Node
from tensornetwork.contractors import auto
from tensornetwork.contractors.opt_einsum_paths import path_contractors
from tensornetwork.ncon_interface import ncon

@pytest.fixture(
    name="path_algorithm", params=["optimal", "branch", "greedy", "auto"])
def path_algorithm_fixture(request):
  return getattr(path_contractors, request.param)


def test_sanity_check(backend, path_algorithm):
  a = Node(np.eye(2), backend=backend)
  b = Node(np.ones((2, 7, 11)), backend=backend)
  c = Node(np.ones((7, 11, 13, 2)), backend=backend)
  d = Node(np.eye(13), backend=backend)

  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  b[1] ^ c[0]
  b[2] ^ c[1]
  c[2] ^ d[1]
  c[3] ^ a[1]
  nodes = [a, b, c, d]
  final_node = path_algorithm(nodes)
  assert final_node.shape == (13,)


def test_trace_edge(backend, path_algorithm):
  a = Node(np.ones((2, 2, 2, 2, 2)), backend=backend)
  b = Node(np.ones((2, 2, 2)), backend=backend)
  c = Node(np.ones((2, 2, 2)), backend=backend)

  # pylint: disable=pointless-statement
  a[0] ^ a[1]
  a[2] ^ b[0]
  a[3] ^ c[0]
  b[1] ^ c[1]
  b[2] ^ c[2]
  nodes = [a, b, c]
  node = path_algorithm(nodes)
  np.testing.assert_allclose(node.tensor, np.ones(2) * 32.0)


def test_single_node(backend, path_algorithm):
  a = Node(np.ones((2, 2, 2)), backend=backend)
  # pylint: disable=pointless-statement
  a[0] ^ a[1]
  nodes = [a]
  node = path_algorithm(nodes)
  np.testing.assert_allclose(node.tensor, np.ones(2) * 2.0)


def test_custom_sanity_check(backend):
  a = Node(np.ones(2), backend=backend)
  b = Node(np.ones((2, 5)), backend=backend)

  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  nodes = [a, b]

  class PathOptimizer:

    def __call__(self, inputs, output, size_dict, memory_limit=None):
      return [(0, 1)]

  optimizer = PathOptimizer()
  final_node = path_contractors.custom(nodes, optimizer)
  np.testing.assert_allclose(final_node.tensor, np.ones(5) * 2.0)


def test_subgraph_contraction(backend, path_algorithm):
  a_tensor = np.arange(4).reshape((2, 2))
  b_tensor = np.arange(4).reshape((2, 2)) + 10
  c_tensor = np.arange(4).reshape((2, 2)) + 20
  a = Node(a_tensor, backend=backend)
  b = Node(b_tensor, backend=backend)
  c = Node(c_tensor, backend=backend)
  a[0] ^ b[1]
  c[1] ^ b[0]
  remaining_edges = [c[0], a[1]]
  result = path_algorithm({a, b}, [b[0], a[1]])
  np.testing.assert_allclose(result.tensor, b_tensor @ a_tensor)
  final = (c @ result).reorder_edges(remaining_edges)
  np.testing.assert_allclose(final.tensor, c_tensor @ b_tensor @ a_tensor)


def test_multiple_partial_contractions(backend, path_algorithm):
  a_tensor = np.arange(4).reshape((2, 2))
  b_tensor = np.arange(4).reshape((2, 2)) + 10
  c_tensor = np.arange(4).reshape((2, 2)) + 20
  d_tensor = np.arange(4).reshape((2, 2)) + 30
  a = Node(a_tensor, backend=backend)
  b = Node(b_tensor, backend=backend)
  c = Node(c_tensor, backend=backend)
  d = Node(d_tensor, backend=backend)
  a[1] ^ b[0]
  b[1] ^ c[0]
  c[1] ^ d[0]
  d[1] ^ a[0]
  ab = path_algorithm({a, b}, [a[0], b[1]])
  np.testing.assert_allclose(ab.tensor, a_tensor @ b_tensor)
  cd = path_algorithm({c, d}, [c[0], d[1]])
  np.testing.assert_allclose(cd.tensor, c_tensor @ d_tensor)
  result = path_algorithm({ab, cd})
  np.testing.assert_allclose(
      result.tensor, np.trace(a_tensor @ b_tensor @ c_tensor @ d_tensor))


def test_single_node_reorder(backend, path_algorithm):
  a = Node(np.arange(4).reshape((2, 2)), backend=backend)
  expected_edge_order = [a[1], a[0]]
  result = path_algorithm({a}, expected_edge_order)
  assert result.edges == expected_edge_order
  np.testing.assert_allclose(result.tensor, np.arange(4).reshape((2, 2)).T)


def test_ignore_edge_order(backend, path_algorithm):
  a = Node(np.ones((1, 1, 1)), backend=backend)
  b = Node(np.ones((1, 1, 1, 2, 3)), backend=backend)

  a[0] ^ b[0]
  a[1] ^ b[1]
  a[2] ^ b[2]

  e0 = b[3]
  e1 = b[4]

  final_node = path_algorithm({a, b}, ignore_edge_order=True)

  assert set(final_node.edges) == {e0, e1}


def test_ignore_edge_order_with_order(backend, path_algorithm):
  a = Node(np.ones((1, 1, 1)), backend=backend)
  b = Node(np.ones((1, 1, 1, 2, 3)), backend=backend)

  a[0] ^ b[0]
  a[1] ^ b[1]
  a[2] ^ b[2]

  e0 = b[3]
  e1 = b[4]

  final_node = path_algorithm({a, b}, [e1, e0], ignore_edge_order=True)

  assert set(final_node.edges) == {e0, e1}


def test_disconnected_network(backend, path_algorithm):
  a = Node(np.eye(2), backend=backend)
  b = Node(np.eye(2), backend=backend)
  c = Node(np.eye(2), backend=backend)
  d = Node(np.eye(2), backend=backend)
  e = Node(np.eye(2), backend=backend)
  f = Node(np.eye(2), backend=backend)
  g = Node(np.eye(2), backend=backend)
  a[1] ^ b[0]
  c[0] ^ d[1]
  e[0] ^ f[0]
  g[0] ^ f[1]
  final_edges = [a[0], b[1], c[1], d[0], e[1], g[1]]
  result = path_algorithm({a, b, c, d, e, f, g}, final_edges)
  assert result.edges == final_edges


def test_passes_ignore_edge_order_from_auto(backend):
  a = Node(np.eye(2), backend=backend)
  b = Node(np.eye(2), backend=backend)
  c = Node(np.eye(2), backend=backend)
  d = Node(np.eye(2), backend=backend)
  e = Node(np.eye(2), backend=backend)
  # pylint: disable=pointless-statement
  a[1] ^ b[0]
  c[0] ^ d[1]
  c[1] ^ e[0]
  nodes = [a, b, c, d, e]
  try:
    auto(nodes, ignore_edge_order=True)
  except ValueError:
    pytest.fail("auto should pass ignore_edge_order when n >= 5 && n < 7")


def test_path_solver_optimal(backend):
  np.random.seed(10)
  D, d, M = 100, 4, 10

  mps = Node(np.random.rand(D, d, D), backend=backend)
  mpsc = Node(np.random.rand(D, d, D), backend=backend)
  L = Node(np.random.rand(M, D, D), backend=backend)
  mpo = Node(np.random.rand(M, M, d, d), backend=backend)

  L[0] ^ mpo[0]
  L[1] ^ mps[0]
  L[2] ^ mpsc[0]
  mps[1] ^ mpo[2]
  mpsc[1] ^ mpo[3]

  nodes = [mps, mpsc, mpo, L]
  path = path_contractors.path_solver(algorithm="optimal", nodes=nodes)
  assert path == [(1, 3), (1, 2), (0, 1)]


@pytest.mark.parametrize("algorithm", ["optimal", "branch", "greedy", "auto"])
def test_contract_path(backend, algorithm):
  np.random.seed(10)
  D, d, M = 100, 4, 10

  mps = Node(np.random.rand(D, d, D), backend=backend)
  mpsc = Node(np.random.rand(D, d, D), backend=backend)
  L = Node(np.random.rand(M, D, D), backend=backend)
  mpo = Node(np.random.rand(M, M, d, d), backend=backend)

  L[0] ^ mpo[0]
  L[1] ^ mps[0]
  L[2] ^ mpsc[0]
  mps[1] ^ mpo[3]
  mpsc[1] ^ mpo[2]

  nodes = [mps, mpsc, mpo, L]
  path = path_contractors.path_solver(algorithm=algorithm, nodes=nodes)
  order = [mpo[1], mps[2], mpsc[2]]
  res = path_contractors.contract_path(
      path=path, nodes=nodes, output_edge_order=order)
  exp = ncon([mps.tensor, mpsc.tensor, L.tensor, mpo.tensor],
             [[1, 2, -2], [5, 4, -3], [3, 1, 5], [3, -1, 4, 2]],
             backend=backend)
  np.testing.assert_allclose(res.tensor, exp)


def test_contract_path_raises(backend):
  np.random.seed(10)
  D, d, M = 100, 4, 10

  mps = Node(np.random.rand(D, d, D), backend=backend)
  mpsc = Node(np.random.rand(D, d, D), backend=backend)
  L = Node(np.random.rand(M, D, D), backend=backend)
  mpo = Node(np.random.rand(M, M, d, d), backend=backend)

  L[0] ^ mpo[0]
  L[1] ^ mps[0]
  L[2] ^ mpsc[0]
  mps[1] ^ mpo[3]
  mpsc[1] ^ mpo[2]

  nodes = [mps, mpsc, mpo, L]
  with pytest.raises(ValueError, match="algorithm"):
    _ = path_contractors.path_solver(algorithm="no-algorithm", nodes=nodes)
