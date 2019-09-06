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
"""Tests contraction paths calculated by `utils.gate_path`.

These tests are based on `opt_einsum`s tests from
github.com/dgasmith/opt_einsum/blob/master/opt_einsum/tests/test_paths.py
"""
import numpy as np
import opt_einsum
import pytest
import tensornetwork
from tensornetwork.contractors.opt_einsum_paths import utils


def check_path(calculated_path, correct_path):
  if not isinstance(calculated_path, list):
    return False

  if len(calculated_path) != len(correct_path):
    return False

  ret = True
  for pos in range(len(calculated_path)):
    ret &= isinstance(calculated_path[pos], tuple)
    ret &= calculated_path[pos] == correct_path[pos]
  return ret


# We do not use the backend fixture as this file tests only contraction paths
# that `opt_einsum` returns and not the actual contractions performed by
# `TensorNetwork`.
def gemm_network():
  """Creates 'GEMM1' contraction from `opt_einsum` tests."""
  net = tensornetwork.TensorNetwork()
  x = net.add_node(np.ones([1, 2, 4]))
  y = net.add_node(np.ones([1, 3]))
  z = net.add_node(np.ones([2, 4, 3]))
  # pylint: disable=pointless-statement
  x[0] ^ y[0]
  x[1] ^ z[0]
  x[2] ^ z[1]
  y[1] ^ z[2]
  return net


def inner_network():
  """Creates a (modified) `Inner1` contraction from `opt_einsum` tests."""
  net = tensornetwork.TensorNetwork()
  x = net.add_node(np.ones([5, 2, 3, 4]))
  y = net.add_node(np.ones([5, 3]))
  z = net.add_node(np.ones([2, 4]))
  # pylint: disable=pointless-statement
  x[0] ^ y[0]
  x[1] ^ z[0]
  x[2] ^ y[1]
  x[3] ^ z[1]
  return net


def matrix_chain():
  """Creates a contraction of chain of matrices.

  The `greedy` algorithm does not find the optimal path in this case!
  """
  net = tensornetwork.TensorNetwork()
  d = [10, 8, 6, 4, 2]
  nodes = [net.add_node(np.ones([d1, d2])) for d1, d2 in zip(d[:-1], d[1:])]
  for i in range(len(d) - 2):
    nodes[i][1] ^ nodes[i + 1][0]
  return net


# Parametrize tests by giving:
# (contraction algorithm, network, correct path that is expected)
test_list = [
    ("optimal", "gemm_network", [(0, 2), (0, 1)]),
    ("branch", "gemm_network", [(0, 2), (0, 1)]),
    ("greedy", "gemm_network", [(0, 2), (0, 1)]),
    ("optimal", "inner_network", [(0, 1), (0, 1)]),
    ("branch", "inner_network", [(0, 1), (0, 1)]),
    ("greedy", "inner_network", [(0, 1), (0, 1)]),
    ("optimal", "matrix_chain", [(2, 3), (1, 2), (0, 1)]),
    ("branch", "matrix_chain", [(2, 3), (1, 2), (0, 1)]),
    ("greedy", "matrix_chain", [(0, 1), (0, 2), (0, 1)]),
    ]


@pytest.mark.parametrize("params", test_list)
def test_path_optimal(params):
  algorithm_name, network_name, correct_path = params

  net = globals()[network_name]()
  path_algorithm = getattr(opt_einsum.paths, algorithm_name)

  calculated_path, sorted_nodes = utils.get_path(net, path_algorithm)
  assert sorted_nodes == sorted(net.nodes_set, key = lambda n: n.signature)
  assert check_path(calculated_path, correct_path)