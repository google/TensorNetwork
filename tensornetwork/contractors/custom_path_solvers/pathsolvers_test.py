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
# pylint: disable=line-too-long
from tensornetwork.contractors.custom_path_solvers.pathsolvers import greedy_size_solve, greedy_cost_solve, full_solve_complete


@pytest.mark.parametrize('N', range(2, 20))
def test_greedy_size_solve(N):
  log_adj = (1 + np.sin(range(N**2))).reshape(N, N)
  log_adj += log_adj.T
  order, cost = greedy_size_solve(log_adj)
  assert order.shape == (2, N - 1)
  assert isinstance(cost, float)


@pytest.mark.parametrize('d1', np.linspace(1, 6, 10))
@pytest.mark.parametrize('d2', np.linspace(1, 6, 10))
def test_greedy_size_solve2(d1, d2):
  N = 3
  log_adj = np.zeros([N, N])
  log_adj[0, 1] = d1
  log_adj[1, 2] = d2
  log_adj += log_adj.T
  order, cost = greedy_size_solve(log_adj)
  if d1 >= d2:
    ex_order = np.array([[0, 0], [1, 1]])
    ex_cost = d2 + np.log10(10**d1 + 1)
  else:
    ex_order = np.array([[1, 0], [2, 1]])
    ex_cost = d1 + np.log10(10**d2 + 1)
  assert np.array_equal(order, ex_order)
  assert np.allclose(ex_cost, cost)


@pytest.mark.parametrize('N', range(2, 20))
def test_greedy_cost_solve(N):
  log_adj = (1 + np.sin(range(N**2))).reshape(N, N)
  log_adj += log_adj.T
  order, cost = greedy_cost_solve(log_adj)
  assert order.shape == (2, N - 1)
  assert isinstance(cost, float)


@pytest.mark.parametrize('d1', np.linspace(1, 6, 5))
@pytest.mark.parametrize('d2', np.linspace(1, 6, 5))
@pytest.mark.parametrize('d3', np.linspace(1, 6, 5))
def test_greedy_cost_solve2(d1, d2, d3):
  N = 3
  log_adj = np.zeros([N, N])
  log_adj[0, 1] = d1
  log_adj[1, 2] = d2
  log_adj += log_adj.T
  log_adj[2, 2] = d3
  order, cost = greedy_cost_solve(log_adj)
  ex_order = np.array([[0, 0], [1, 1]])
  ex_cost = d1 + d2 + np.log10(1 + 10**(d3 - d1))
  assert np.array_equal(order, ex_order)
  assert np.allclose(ex_cost, cost)


@pytest.mark.parametrize('N', range(2, 8))
def test_full_solve_complete(N):
  log_adj = (1 + np.sin(range(N**2))).reshape(N, N)
  log_adj += log_adj.T
  order, cost, _ = full_solve_complete(log_adj)
  assert order.shape == (2, N - 1)
  assert isinstance(cost, float)


@pytest.mark.parametrize('d1', np.linspace(1, 6, 5))
def test_full_solve_complete2(d1):
  N = 7
  log_adj = np.zeros([N, N])
  log_adj[:(N - 1), 1:] = np.diag(d1 * np.ones(N - 1))
  log_adj += log_adj.T
  log_adj[0, 0] = d1
  log_adj[-1, -1] = d1
  _, cost, is_optimal = full_solve_complete(log_adj)
  ex_cost = np.log10((N - 1) * 10**(3 * d1))
  assert np.allclose(ex_cost, cost)
  assert is_optimal


@pytest.mark.parametrize('cost_bound', range(1, 50, 5))
@pytest.mark.parametrize('max_branch', range(1, 1000, 100))
def test_full_solve_complete3(cost_bound, max_branch):
  N = 7
  log_adj = (1 + np.sin(range(N**2))).reshape(N, N)
  log_adj += log_adj.T
  order, cost, _ = full_solve_complete(
      log_adj, cost_bound=cost_bound, max_branch=max_branch)
  assert order.shape == (2, N - 1)
  assert isinstance(cost, float)
