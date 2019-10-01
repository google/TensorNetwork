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
import tensornetwork as tn
from tensornetwork.contractors.opt_einsum_paths import path_contractors
import tensorflow as tf
tf.enable_v2_behavior()


@pytest.fixture(
    name="path_algorithm", params=["optimal", "branch", "greedy", "auto"])
def path_algorithm_fixture(request):
  return getattr(path_contractors, request.param)


def test_sanity_check(backend, path_algorithm):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.ones((2, 7, 11)), backend=backend)
  c = tn.Node(np.ones((7, 11, 13, 2)), backend=backend)
  d = tn.Node(np.eye(13), backend=backend)

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
  a = tn.Node(np.ones((2, 2, 2, 2, 2)), backend=backend)
  b = tn.Node(np.ones((2, 2, 2)), backend=backend)
  c = tn.Node(np.ones((2, 2, 2)), backend=backend)

  # pylint: disable=pointless-statement
  a[0] ^ a[1]
  a[2] ^ b[0]
  a[3] ^ c[0]
  b[1] ^ c[1]
  b[2] ^ c[2]
  nodes = [a, b, c]
  node = path_algorithm(nodes)
  np.testing.assert_allclose(node.tensor, np.ones(2) * 32.0)


def test_disconnected_network(backend, path_algorithm):
  a = tn.Node(np.array([2, 2]), backend=backend)
  b = tn.Node(np.array([2, 2]), backend=backend)
  c = tn.Node(np.array([2, 2]), backend=backend)
  d = tn.Node(np.array([2, 2]), backend=backend)

  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  c[0] ^ d[0]
  nodes = [a, b, c, d]
  with pytest.raises(ValueError):
    path_algorithm(nodes)


def test_single_node(backend, path_algorithm):
  a = tn.Node(np.ones((2, 2, 2)), backend=backend)
  # pylint: disable=pointless-statement
  a[0] ^ a[1]
  nodes = [a]
  node = path_algorithm(nodes)
  np.testing.assert_allclose(node.tensor, np.ones(2) * 2.0)


def test_custom_sanity_check(backend):
  a = tn.Node(np.ones(2), backend=backend)
  b = tn.Node(np.ones((2, 5)), backend=backend)

  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  nodes = [a, b]

  class PathOptimizer:

    def __call__(self, inputs, output, size_dict, memory_limit=None):
      return [(0, 1)]

  optimizer = PathOptimizer()
  final_node = path_contractors.custom(nodes, optimizer)
  np.testing.assert_allclose(final_node.tensor, np.ones(5) * 2.0)
