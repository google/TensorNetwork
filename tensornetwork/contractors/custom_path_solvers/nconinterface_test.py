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
from tensornetwork.contractors.custom_path_solvers.nconinterface import ncon_solver, ncon_to_adj, ord_to_ncon, ncon_cost_check


@pytest.mark.parametrize('chi', range(2, 6))
def test_ncon_solver(chi):
  # test against network with known cost
  chi = np.random.randint(2, 10)
  u = np.random.rand(chi, chi, chi, chi)
  w = np.random.rand(chi, chi, chi)
  ham = np.random.rand(chi, chi, chi, chi, chi, chi)
  tensors = [u, u, w, w, w, ham, u, u, w, w, w]
  connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, -4], [11, 12, -5],
              [13, 14, -6], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], [5, 6, 16, 15],
              [8, 9, -1], [17, 16, -2], [15, 14, -3]]
  con_order, costs, is_optimal = ncon_solver(tensors, connects, max_branch=None)
  flat_connects = np.concatenate(connects)
  inds = np.sort(np.unique(flat_connects[flat_connects > 0]))
  ex_cost = np.log10(2 * chi**9 + 4 * chi**8 + 2 * chi**6 + 2 * chi**5)
  assert np.allclose(costs, ex_cost)
  assert is_optimal
  assert np.array_equal(inds, np.sort(con_order))


@pytest.mark.parametrize('num_closed', range(1, 20))
def test_ncon_solver2(num_closed):
  chi = 4
  N = 10
  A = np.zeros([chi, chi, chi, chi, chi, chi])
  tensors = [A] * N
  num_open = 4 * N - 2 * num_closed
  cl_inds = 1 + np.arange(num_closed)
  op_inds = -1 - np.arange(num_open)
  connects = [0] * N
  perm = np.argsort(np.sin(range(4 * N)))
  comb_inds = np.concatenate((op_inds, cl_inds, cl_inds))[perm]
  for k in range(N):
    if k < (N - 1):
      connect_temp = np.concatenate((comb_inds[4 * k:4 * (k + 1)],
                                     [num_closed + k + 1, num_closed + k + 2]))
    else:
      connect_temp = np.concatenate(
          (comb_inds[4 * k:4 * (k + 1)], [num_closed + k + 1, num_closed + 1]))
    connects[k] = list(connect_temp[np.argsort(np.random.rand(6))])
  max_branch = 1000
  con_order, costs, _ = ncon_solver(tensors, connects, max_branch=max_branch)
  ex_cost = ncon_cost_check(tensors, connects, con_order)
  assert np.allclose(costs, ex_cost)
  assert np.array_equal(np.arange(num_closed + N) + 1, np.sort(con_order))


@pytest.mark.parametrize('chi', range(2, 6))
@pytest.mark.parametrize('N', range(2, 7))
def test_ncon_to_adj(chi, N):
  A = np.zeros([chi, chi])
  tensors = [A] * N
  connects = [0] * N
  for k in range(N):
    if k == 0:
      connects[k] = [-1, 1]
    elif k == (N - 1):
      connects[k] = [k, -2]
    else:
      connects[k] = [k, k + 1]
  log_adj = ncon_to_adj(tensors, connects)
  ex_log_adj = np.zeros([N, N])
  ex_log_adj[:(N - 1), 1:] = np.diag(np.log10(chi) * np.ones([N - 1]))
  ex_log_adj += ex_log_adj.T
  ex_log_adj[0, 0] = np.log10(chi)
  ex_log_adj[-1, -1] = np.log10(chi)
  assert np.allclose(log_adj, ex_log_adj)


@pytest.mark.parametrize('num_closed', range(1, 16))
def test_ord_to_ncon(num_closed):
  N = 8
  num_open = 4 * N - 2 * num_closed
  cl_inds = 1 + np.arange(num_closed)
  op_inds = -1 - np.arange(num_open)
  connects = [0] * N
  perm = np.argsort(np.random.rand(4 * N))
  comb_inds = np.concatenate((op_inds, cl_inds, cl_inds))[perm]
  for k in range(N):
    if k < (N - 1):
      connect_temp = np.concatenate((comb_inds[4 * k:4 * (k + 1)],
                                     [num_closed + k + 1, num_closed + k + 2]))
    else:
      connect_temp = np.concatenate(
          (comb_inds[4 * k:4 * (k + 1)], [num_closed + k + 1, num_closed + 1]))
    connects[k] = list(connect_temp[np.argsort(np.random.rand(6))])
  order = np.zeros([2, N - 1], dtype=int)
  for k in range(N - 1):
    temp_loc = np.random.randint(0, N - k - 1)
    order[0, k] = temp_loc
    order[1, k] = np.random.randint(temp_loc + 1, N - k)
  con_order = ord_to_ncon(connects, order)
  assert np.array_equal(np.sort(con_order), np.arange(num_closed + N) + 1)


@pytest.mark.parametrize('chi', range(2, 6))
def test_ncon_cost_check(chi):
  # test against network with known cost
  u = np.random.rand(chi, chi, chi, chi)
  w = np.random.rand(chi, chi, chi)
  ham = np.random.rand(chi, chi, chi, chi, chi, chi)
  tensors = [u, u, w, w, w, ham, u, u, w, w, w]
  connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, -4], [11, 12, -5],
              [13, 14, -6], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], [5, 6, 16, 15],
              [8, 9, -1], [17, 16, -2], [15, 14, -3]]
  con_order = [4, 7, 17, 5, 6, 11, 3, 12, 14, 1, 2, 16, 8, 9, 10, 13, 15]
  cost = ncon_cost_check(tensors, connects, con_order)
  ex_cost = np.log10(2 * chi**9 + 4 * chi**8 + 2 * chi**6 + 2 * chi**5)
  assert np.allclose(cost, ex_cost)


@pytest.mark.parametrize('chi', range(2, 6))
def test_ncon_cost_check2(chi):
  # test against network with known (includes traces and inner products
  A = np.zeros([chi, chi, chi, chi])
  B = np.zeros([chi, chi, chi, chi, chi, chi])
  C = np.zeros([chi, chi, chi])
  D = np.zeros([chi, chi])
  tensors = [A, B, C, D]
  connects = [[1, 2, 3, 1], [2, 4, 4, 5, 6, 6], [3, 5, -1], [-2, -3]]
  con_order = [1, 2, 3, 4, 5, 6]
  cost = ncon_cost_check(tensors, connects, con_order)
  ex_cost = np.log10(3 * chi**3)
  assert np.allclose(cost, ex_cost)
