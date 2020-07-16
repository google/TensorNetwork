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
from typing import Optional


def greedy_size_solve(log_adj_in: np.ndarray):
  """
  Solve for the contraction order of a tensor network (encoded as a
  log-adjacency matrix) using a greedy algorithm that minimizes the
  intermediate tensor sizes.
  Args:
    log_adj_in: matrix where element [i,j] is the log10 of the total dimension
      of the indices connecting ith and jth tensors.
  Returns:
    np.ndarray: cheapest contraction order found, specified as a sequence of
      binary contractions.
    float: the cost of the network contraction, given as log10(total_FLOPS).
  """
  tol = 1e-6  # tolerance for float comparison
  N0 = log_adj_in.shape[0]
  log_adj = log_adj_in.copy().reshape(N0, N0)
  orders = np.zeros([2, 0], dtype=int)
  costs = None

  for _ in range(N0 - 1):
    # compute tensor dims
    N = log_adj.shape[0]
    dims = np.sum(log_adj, axis=0).reshape(N)
    comb_dims = np.add.outer(dims, dims)

    # compute contraction costs and new dims
    single_cost = comb_dims - log_adj
    new_dims = comb_dims - 2 * log_adj
    new_dims = new_dims + np.max(new_dims.flatten()) * np.eye(N)

    # compute maximum dim of tensor in contraction
    temp_mat = np.kron(dims, np.ones([N, 1]))
    max_dim = np.maximum(temp_mat, temp_mat.T)
    dim_change = ((1 / tol) * (new_dims - max_dim)).astype(int)

    # compute coords of minimal dim increase
    xcoord, ycoord = np.where(dim_change == np.min(dim_change.flatten()))
    upper_tri = (xcoord < ycoord)
    xcoord = xcoord[upper_tri]
    ycoord = ycoord[upper_tri]

    # find contraction with minimal cost
    all_costs = np.array(
        [single_cost[xcoord[i], ycoord[i]] for i in range(len(xcoord))])
    cont_dims = np.array(
        [log_adj[xcoord[i], ycoord[i]] for i in range(len(xcoord))])
    if max(cont_dims) > 0:  # prioritise non-trivial contractions
      all_costs[cont_dims == 0] += max(all_costs) + 1

    cheapest_pos = np.argmin(all_costs)
    i = ycoord[cheapest_pos]
    j = xcoord[cheapest_pos]

    # build new log adjacency
    log_adj[j, j] = log_adj[j, j] - 2 * log_adj[j, i]
    log_adj[j, :] = log_adj[j, :] + log_adj[i, :]
    log_adj[:, j] = log_adj[:, j] + log_adj[:, i]
    log_adj = np.delete(log_adj, i, axis=0)
    log_adj = np.delete(log_adj, i, axis=1)

    # build new orders
    orders = np.hstack((orders, np.asarray([j, i]).reshape(2, 1)))

    # tally the cost
    if costs is None:
      costs = single_cost[i, j]
    else:
      costs = costs + np.log10(1 + 10**(single_cost[i, j] - costs))

  return orders, costs


def greedy_cost_solve(log_adj_in: np.ndarray):
  """
  Solve for the contraction order of a tensor network (encoded as a
  log-adjacency matrix) using a greedy algorithm that minimizes the
  contraction cost at each step.
  Args:
    log_adj_in: matrix where element [i,j] is the log10 of the total dimension
      of the indices connecting ith and jth tensors.
  Returns:
    np.ndarray: cheapest contraction order found, specified as a sequence of
      binary contractions.
    float: the cost of the network contraction, given as log10(total_FLOPS).
  """
  tol = 1e-6  # tolerance for float comparison
  N = log_adj_in.shape[0]
  log_adj = log_adj_in.copy().reshape(N, N)
  orders = np.zeros([2, 0], dtype=int)
  costs = None

  for _ in range(N - 1):
    # compute tensor dims and costs
    N = log_adj.shape[0]
    dims = np.sum(log_adj, axis=0).reshape(N)
    comb_dims = np.add.outer(dims, dims)
    single_cost = comb_dims - log_adj

    # penalize trivial contractions and self-contractions
    triv_conts = (log_adj < tol)
    trimmed_costs = single_cost + np.max(single_cost.flatten()) * triv_conts
    trimmed_costs = trimmed_costs + np.max(trimmed_costs.flatten()) * np.eye(N)

    # find best contraction
    tensors_to_contract = np.divmod(np.argmin(trimmed_costs), N)
    i = max(tensors_to_contract)
    j = min(tensors_to_contract)

    # build new log adjacency
    log_adj[j, j] = log_adj[j, j] - 2 * log_adj[j, i]
    log_adj[j, :] = log_adj[j, :] + log_adj[i, :]
    log_adj[:, j] = log_adj[:, j] + log_adj[:, i]
    log_adj = np.delete(log_adj, i, axis=0)
    log_adj = np.delete(log_adj, i, axis=1)

    # build new orders
    orders = np.hstack((orders, np.asarray(tensors_to_contract).reshape(2, 1)))

    # tally the cost
    if costs is None:
      costs = single_cost[i, j]
    else:
      costs = costs + np.log10(1 + 10**(single_cost[i, j] - costs))

  return orders, costs


def full_solve_complete(log_adj: np.ndarray,
                        cost_bound: Optional[int] = None,
                        max_branch: Optional[int] = None):
  """
  Solve for optimal contraction path of a network encoded as a log-adjacency
  matrix via a full search.
  Args:
    log_adj: matrix where element [i,j] is the log10 of the total dimension
      of the indices connecting ith and jth tensors.
    cost_bound: upper cost threshold for discarding paths, in log10(FLOPS).
    max_branch: bound for the total number of paths to retain.
  Returns:
    np.ndarray: the cheapest contraction order found.
    float: the cost of the network contraction, given as log10(total_FLOPS).
    bool: specifies if contraction order is guaranteed optimal.
  """
  tol = 1e-6  # tolerance for float comparison
  # start by trying both greedy algorithms
  order0, cost0 = greedy_size_solve(log_adj)
  order1, cost1 = greedy_cost_solve(log_adj)
  if cost0 < cost1:
    order_greedy = order0
    cost_greedy = cost0
  else:
    order_greedy = order1
    cost_greedy = cost1

  if max_branch == 1:
    # return results from greedy
    order_was_found = False
  else:
    # initialize arrays
    N = log_adj.shape[0]
    costs = np.zeros([1, 0])
    groups = np.array(2**np.arange(N), dtype=np.uint64).reshape(N, 1)
    orders = np.zeros([2, 0, 1], dtype=int)

    # try full algorithm (using cost_bound from greedy)
    cost_bound = cost_greedy + tol
    total_truncated = 0
    order_was_found = True
    for _ in range(N - 1):
      log_adj, costs, groups, orders, num_truncated = _full_solve_single(
          log_adj,
          costs,
          groups,
          orders,
          cost_bound=cost_bound,
          max_branch=max_branch)
      if log_adj.size == 0:
        # no paths found within the cost-bound
        order_was_found = False
        break
      total_truncated = total_truncated + num_truncated

  if order_was_found:
    # return result from full algorithm
    is_optimal = (total_truncated == 0)
    return orders.reshape(2, N - 1), costs.item(), is_optimal

  # return result from greedy algorithm
  is_optimal = False
  return order_greedy, cost_greedy, is_optimal


def _full_solve_single(log_adj: np.ndarray,
                       costs: np.ndarray,
                       groups: np.ndarray,
                       orders: np.ndarray,
                       cost_bound: Optional[int] = None,
                       max_branch: Optional[int] = None,
                       allow_outer: Optional[bool] = False):
  """
  Solve for the most-likely contraction step given a set of networks encoded
  as log-adjacency matrices. Uses an algorithm that searches multiple (or,
  potentially, all viable paths) as to minimize the total contraction cost.
  Args:
    log_adj: an np.ndarray of log-adjacency matrices of dim (N,N,m), with `N`
      the number of tensors and `m` the number of (intermediate) networks.
    costs: np.ndarray of length `m` detailing to prior cost of each network.
    groups: np.ndarray of dim (N,m) providing an id-tag for each network,
      based on a power-2 encoding.
    orders: np.ndarray of dim (2,t,m) detailing the pairwise contraction
      history of each network from the previous `t` contraction steps.
    cost_bound: upper cost threshold for discarding paths, in log10(FLOPS).
    max_branch: bound for the total number of paths to retain.
    allow_outer: sets whether outer products are allowed.
  Returns:
    np.ndarray: new set of `log_adj` matrices.
    np.ndarray: new set of `costs`.
    np.ndarray: new set of `groups`.
    np.ndarray: new set of `orders`.
    int: total number of potentially viable paths that were trimmed.
  """
  tol = 1e-6  # tolerance for float comparison

  # set threshold required to trigger compression routine
  if max_branch is None:
    mid_kept = 10000
  else:
    mid_kept = max_branch

  # initialize outputs
  N = log_adj.shape[0]
  if log_adj.ndim == 2:
    log_adj = log_adj.reshape(N, N, 1)
  final_adj = np.zeros([N - 1, N - 1, 0])
  final_costs = np.zeros([1, 0])
  final_groups = np.zeros([N - 1, 0], dtype=np.uint64)
  final_orders = np.zeros([2, orders.shape[1] + 1, 0], dtype=int)
  final_stable = np.zeros([1, 0], dtype=bool)
  total_truncated = 0

  only_outer_exist = not allow_outer
  none_inbounds = True

  # try to contract j-th tensor with i-th tensor (j<i)
  for i in range(1, N):
    for j in range(i):

      if not allow_outer:
        # only attempt non-trivial contractions
        new_pos = np.flatnonzero(log_adj[j, i, :] > 0)
        num_kept = len(new_pos)
      else:
        new_pos = np.arange(log_adj.shape[2])
        num_kept = len(new_pos)

      if num_kept > 0:
        only_outer_exist = False

        # dims of tensors and cost of contraction
        dims = np.sum(log_adj[:, :, new_pos], axis=0).reshape(N, num_kept)
        comb_dims = dims[j, :] + dims[i, :]
        single_cost = np.reshape(comb_dims - log_adj[j, i, new_pos],
                                 [1, num_kept])
        if costs.size == 0:
          new_costs = single_cost
        else:
          prev_cost = costs[0, new_pos]
          new_costs = prev_cost + np.log10(1 + 10**(single_cost - prev_cost))

        if cost_bound is not None:
          # only keep contractions under the cost bound
          pos_under_bound = new_costs.flatten() < cost_bound
          new_pos = new_pos[pos_under_bound]
          num_kept = len(new_pos)

          new_costs = new_costs[0, pos_under_bound].reshape(1, num_kept)

      if num_kept > 0:
        none_inbounds = False

        # order the costs
        cost_order = np.argsort(new_costs).flatten()
        sorted_pos = new_pos[cost_order]

        # identify identical networks
        new_groups = groups[:, sorted_pos]
        new_groups[j, :] = new_groups[j, :] + new_groups[i, :]
        new_groups = np.delete(new_groups, i, axis=0)
        new_groups, temp_pos = np.unique(new_groups, return_index=True, axis=1)

        new_costs = new_costs[:, cost_order[temp_pos]]
        new_pos = sorted_pos[temp_pos]
        num_kept = len(new_pos)

        # new log adjacency
        new_adj = log_adj[:, :, new_pos]
        new_adj[j, j, :] = new_adj[j, j, :] - 2 * new_adj[j, i, :]
        new_adj[j, :, :] = new_adj[j, :, :] + new_adj[i, :, :]
        new_adj[:, j, :] = new_adj[:, j, :] + new_adj[:, i, :]
        new_adj = np.delete(new_adj, i, axis=0)
        new_adj = np.delete(new_adj, i, axis=1)

        # new orders
        prev_orders = orders[:, :, new_pos]
        next_orders = np.vstack([
            j * np.ones(len(new_pos), dtype=int),
            i * np.ones(len(new_pos), dtype=int)
        ]).reshape(2, 1, len(new_pos))
        new_orders = np.concatenate((prev_orders, next_orders), axis=1)

        # new_stable
        dims = np.sum(log_adj[:, :, new_pos], axis=0).reshape(N, num_kept)
        comb_dims = dims[j, :] + dims[i, :]
        final_dims = np.reshape(comb_dims - 2 * log_adj[j, i, new_pos],
                                [1, num_kept])

        # include a fudge factor to avoid rounding errors
        stable_pos = final_dims < (np.maximum(dims[j, :], dims[i, :]) + tol)

        final_adj = np.concatenate((final_adj, new_adj), axis=2)
        final_costs = np.concatenate((final_costs, new_costs), axis=1)
        final_groups = np.concatenate((final_groups, new_groups), axis=1)
        final_orders = np.concatenate((final_orders, new_orders), axis=2)
        final_stable = np.concatenate((final_stable, stable_pos), axis=1)

        # if number of intermediates too large then trigger compression routine
        if final_costs.size > mid_kept:
          temp_pos, num_truncated = _reduce_nets(
              final_costs, final_groups, final_stable, max_branch=max_branch)
          final_adj = final_adj[:, :, temp_pos]
          final_costs = final_costs[:, temp_pos]
          final_groups = final_groups[:, temp_pos]
          final_orders = final_orders[:, :, temp_pos]
          final_stable = final_stable[:, temp_pos]
          total_truncated = total_truncated + num_truncated

  if not only_outer_exist:
    if none_inbounds:
      # no orders found under the cost bound; return trivial
      return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0

  if only_outer_exist:  # network contains only outer products
    # re-solve with outer products enabled
    return _full_solve_single(
        log_adj,
        costs,
        groups,
        orders,
        cost_bound=cost_bound,
        max_branch=max_branch,
        allow_outer=True)

  # compress outputs
  temp_pos = _reduce_nets(final_costs, final_groups, final_stable)[0]
  final_adj = final_adj[:, :, temp_pos]
  final_costs = final_costs[:, temp_pos]
  final_groups = final_groups[:, temp_pos]
  final_orders = final_orders[:, :, temp_pos]
  final_stable = final_stable[:, temp_pos]
  return final_adj, final_costs, final_groups, final_orders, total_truncated


def _reduce_nets(costs: np.ndarray,
                 groups: np.ndarray,
                 stable: np.ndarray,
                 max_branch: Optional[int] = None):
  """
  Reduce from `m` starting paths smaller number of paths by first (i)
  identifying any equivalent networks then (ii) trimming the most expensive
  paths.
  Args:
    costs: np.ndarray of length `m` detailing to prior cost of each network.
    groups: np.ndarray of dim (N,m) providing an id-tag for each network,
      based on a power-2 encoding.
    stable: np.ndarray of dim (m) denoting which paths were size-stable.
    max_branch: bound for the total number of paths to retain.
  Returns:
    np.ndarray: index positions of the kept paths.
    int: total number of potentially viable paths that were trimmed.
  """

  # sort according to the costs
  new_pos = np.argsort(costs).flatten()

  # identify and remove identical networks
  temp_pos = np.unique(groups[:, new_pos], return_index=True, axis=1)[1]
  orig_kept = len(temp_pos)
  new_pos = new_pos[temp_pos]
  num_truncated = 0

  if max_branch is not None:
    if orig_kept > max_branch:
      # re-sort according to the costs
      cost_order = np.argsort(costs[:, new_pos]).flatten()
      new_pos = new_pos[cost_order]

      # reserve some percertage for size-stable contractions
      preserve_ratio = 0.2
      num_stable = int(np.ceil(max_branch * preserve_ratio))
      num_cheapest = int(np.ceil(max_branch * (1 - preserve_ratio)))

      stable_pos = np.flatnonzero(stable[0, new_pos[num_cheapest:]])
      temp_pos = np.concatenate(
          (np.arange(num_cheapest),
           stable_pos[:min(len(stable_pos), num_stable)] + num_cheapest),
          axis=0)
      new_pos = new_pos[temp_pos]
      num_truncated = orig_kept - len(new_pos)

  return new_pos, num_truncated
