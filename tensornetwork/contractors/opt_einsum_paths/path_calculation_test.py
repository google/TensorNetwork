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


@pytest.fixture(name="path_algorithm",
                params=["optimal", "branch", "greedy"])
def path_algorithm_fixture(request):
  return getattr(opt_einsum.paths, request.param)


def check_path(calculated_path, correct_path, bypass=False):
  if not isinstance(calculated_path, list):
    return False

  if len(calculated_path) != len(correct_path):
    return False

  ret = True
  for pos in range(len(calculated_path)):
    ret &= isinstance(calculated_path[pos], tuple)
    ret &= calculated_path[pos] == correct_path[pos]
  return ret


def create_tensor_network():
  """Creates 'GEMM1' contraction from `opt_einsum` tests in `TensorNetowrk`.

  Note that the optimal contraction order is [(0, 2), (0, 1)].
  """
  net = tensornetwork.TensorNetwork()
  x = net.add_node(np.ones([1, 2, 4]))
  y = net.add_node(np.ones([1, 3]))
  z = net.add_node(np.ones([2, 4, 3]))
  # pylint: disable=pointless-statement
  x[0] ^ y[0]
  x[1] ^ z[0]
  x[2] ^ z[1]
  y[1] ^ z[2]

  # This ordering is compatible with `TensorNetwork` only if we sort
  # according to `node.signature`!
  optimal_order = [(0, 2), (0, 1)]
  return net, optimal_order


def test_path_optimal(path_algorithm):
  net, optimal_order = create_tensor_network()
  calculated_path, sorted_nodes = utils.get_path(net, path_algorithm)
  assert sorted_nodes == sorted(net.nodes_set, key = lambda n: n.signature)
  assert check_path(calculated_path, optimal_order)