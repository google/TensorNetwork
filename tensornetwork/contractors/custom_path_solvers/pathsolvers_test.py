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

def test_greedy_size_solve():
  N = np.random.randint(2,20)
  log_adj = 6.0*np.random.randint(0,2,[N,N])*np.random.rand(N,N)
  log_adj += log_adj.T
  order, cost = greedy_size_solve(log_adj)
  assert order.shape == (2,N-1)
  assert isinstance(cost,float)

def test_greedy_size_solve2():
  N = 3
  log_adj = np.zeros([N,N])
  d1 = round(1e5*3*(1+np.random.rand())) / 1e5
  d2 = round(1e5*3*(1+np.random.rand())) / 1e5
  log_adj[0,1] = d1
  log_adj[1,2] = d2
  log_adj += log_adj.T
  order, cost = greedy_size_solve(log_adj)
  if (d1 >= d2):
    ex_order = np.array([[0,0],[1,1]])
    ex_cost = d2 + np.log10(10**d1+1)
  else:
    ex_order = np.array([[1,0],[2,1]])
    ex_cost = d1 + np.log10(10**d2+1)
  assert np.array_equal(order,ex_order)
  assert np.allclose(ex_cost,cost)
  
def test_greedy_cost_solve():
  N = np.random.randint(2,20)
  log_adj = 6.0*np.random.randint(0,2,[N,N])*np.random.rand(N,N)
  log_adj += log_adj.T
  order, cost = greedy_cost_solve(log_adj)
  assert order.shape == (2,N-1)
  assert isinstance(cost,float)

def test_greedy_cost_solve2():
  N = 3
  log_adj = np.zeros([N,N])
  d1 = 3*(1+np.random.rand())
  d2 = 3*(1+np.random.rand())
  d3 = 3*(1+np.random.rand())
  log_adj[0,1] = d1
  log_adj[1,2] = d2
  log_adj += log_adj.T
  log_adj[2,2] = d3
  order, cost = greedy_cost_solve(log_adj)
  ex_order = np.array([[0,0],[1,1]])
  ex_cost = d1 + d2 + np.log10(1+10**(d3-d1))
  assert np.array_equal(order,ex_order)
  assert np.allclose(ex_cost,cost)

def test_full_solve_complete():
  N = np.random.randint(2,7)
  log_adj = 4.0*np.random.randint(0,2,[N,N])*np.random.rand(N,N)
  log_adj += log_adj.T
  order, cost, is_optimal = full_solve_complete(log_adj)
  assert order.shape == (2,N-1)
  assert isinstance(cost,float)
  
def test_full_solve_complete2():
  N = 7
  d1 = 3*(1+np.random.rand())
  log_adj = np.zeros([N,N])
  log_adj[:(N-1),1:] = np.diag(d1*np.ones(N-1))
  log_adj += log_adj.T
  log_adj[0,0] = d1
  log_adj[-1,-1] = d1
  order, cost, is_optimal = full_solve_complete(log_adj)
  ex_cost = np.log10((N-1)*10**(3*d1))
  assert np.allclose(ex_cost,cost)
  assert is_optimal == True
  
def test_full_solve_complete3():
  N = np.random.randint(2,7)
  log_adj = 4.0*np.random.randint(0,2,[N,N])*np.random.rand(N,N)
  log_adj += log_adj.T
  cost_bound = 10*np.random.rand()
  max_branch= np.random.randint(1,1000)
  order, cost, is_optimal = full_solve_complete(log_adj,
                                                cost_bound=cost_bound,
                                                max_branch=max_branch)
  assert order.shape == (2,N-1)
  assert isinstance(cost,float)

# for k in range(1000):
#   test_greedy_size_solve()
#   test_greedy_size_solve2()
#   test_greedy_cost_solve()
#   test_greedy_cost_solve2()
#   test_full_solve_complete()
#   test_full_solve_complete2()
#   test_full_solve_complete3()
#   print(k)

