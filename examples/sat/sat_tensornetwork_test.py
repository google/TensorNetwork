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

import tensornetwork
from examples.sat import sat_tensornetwork


def test_sanity_check():
  nodes = sat_tensornetwork.sat_count_tn([
      (1, 2, 3),
  ])
  count = tensornetwork.contractors.greedy(nodes).tensor
  assert count == 7


def test_dual_clauses():
  nodes = sat_tensornetwork.sat_count_tn([
      (1, 2, 3),
      (1, -2, 3),
  ])
  count = tensornetwork.contractors.greedy(nodes).tensor
  assert count == 6


def test_many_clauses():
  nodes = sat_tensornetwork.sat_count_tn([
      (1, 2, 3),
      (1, 2, -3),
      (1, -2, 3),
      (1, -2, -3),
      (-1, 2, 3),
      (-1, 2, -3),
      (-1, -2, 3),
      (-1, -2, -3),
  ])
  count = tensornetwork.contractors.greedy(nodes).tensor
  assert count == 0


def test_four_variables():
  nodes = sat_tensornetwork.sat_count_tn([
      (1, 2, 3),
      (1, 2, 4),
  ])
  count = tensornetwork.contractors.greedy(nodes).tensor
  assert count == 13


def test_four_variables_four_clauses():
  nodes = sat_tensornetwork.sat_count_tn([
      (1, 2, 3),
      (1, 2, 4),
      (-3, -4, 2),
      (-1, 3, -2),
  ])
  count = tensornetwork.contractors.greedy(nodes).tensor
  assert count == 9


def test_single_variable():
  nodes = sat_tensornetwork.sat_count_tn([
      (1, 1, 1),
  ])
  count = tensornetwork.contractors.greedy(nodes).tensor
  assert count == 1


def test_solutions():
  edge_order = sat_tensornetwork.sat_tn([
      (1, 2, -3),
  ])
  solutions = tensornetwork.contractors.greedy(
      tensornetwork.reachable(edge_order[0].node1), edge_order).tensor
  assert solutions[0][0][0] == 1
  # Only unaccepted value.
  assert solutions[0][0][1] == 0
  assert solutions[0][1][0] == 1
  assert solutions[0][1][1] == 1
  assert solutions[1][0][0] == 1
  assert solutions[1][0][1] == 1
  assert solutions[1][1][0] == 1
  assert solutions[1][1][1] == 1
