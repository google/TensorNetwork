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
from typing import List, Union, Tuple, Optional
# pylint: disable=line-too-long
from tensornetwork.contractors.custom_path_solvers.pathsolvers import full_solve_complete


def ncon_solver(tensors: List[np.ndarray],
                labels: List[List[int]],
                max_branch: Optional[int] = None):
  """
  Solve for the contraction order of a tensor network (encoded in the `ncon`
  syntax) that minimizes the computational cost.
  Args:
    tensors: list of the tensors in the network.
    labels: list of the tensor connections (in standard `ncon` format).
    max_branch: maximum number of contraction paths to search at each step.
  Returns:
    np.ndarray: the cheapest contraction order found (in ncon format).
    float: the cost of the network contraction, given as log10(total_FLOPS).
    bool: specifies if contraction order is guaranteed optimal.
  """
  # build log-adjacency matrix
  log_adj = ncon_to_adj(tensors, labels)

  # run search algorithm
  order, costs, is_optimal = full_solve_complete(log_adj, max_branch=max_branch)

  # put contraction order back into ncon format
  con_order = ord_to_ncon(labels, order)

  return con_order, costs, is_optimal


def ncon_to_adj(tensors: List[np.ndarray], labels: List[List[int]]):
  """
  Create a log-adjacency matrix, where element [i,j] is the log10 of the total
  dimension of the indices connecting ith and jth tensors, for a network
  defined in the `ncon` syntax.
  Args:
    tensors: list of the tensors in the network.
    labels: list of the tensor connections (in standard `ncon` format).
  Returns:
    np.ndarray: the log-adjacency matrix.
  """
  # process inputs
  N = len(labels)
  ranks = [len(labels[i]) for i in range(N)]
  flat_labels = np.hstack([labels[i] for i in range(N)])
  tensor_counter = np.hstack(
      [i * np.ones(ranks[i], dtype=int) for i in range(N)])
  index_counter = np.hstack([np.arange(ranks[i]) for i in range(N)])

  # build log-adjacency index-by-index
  log_adj = np.zeros([N, N])
  unique_labels = np.unique(flat_labels)
  for ele in unique_labels:
    # identify tensor/index location of each edge
    tnr = tensor_counter[flat_labels == ele]
    ind = index_counter[flat_labels == ele]
    if len(ind) == 1:  # external index
      log_adj[tnr[0], tnr[0]] += np.log10(tensors[tnr[0]].shape[ind[0]])
    elif len(ind) == 2:  # internal index
      if tnr[0] != tnr[1]:  # ignore partial traces
        log_adj[tnr[0], tnr[1]] += np.log10(tensors[tnr[0]].shape[ind[0]])
        log_adj[tnr[1], tnr[0]] += np.log10(tensors[tnr[0]].shape[ind[0]])

  return log_adj


def ord_to_ncon(labels: List[List[int]], orders: np.ndarray):
  """
  Produces a `ncon` compatible index contraction order from the sequence of
  pairwise contractions.
  Args:
    labels: list of the tensor connections (in standard `ncon` format).
    orders: array of dim (2,N-1) specifying the set of N-1 pairwise
      tensor contractions.
  Returns:
    np.ndarray: the contraction order (in `ncon` format).
  """

  N = len(labels)
  orders = orders.reshape(2, N - 1)
  new_labels = [np.array(labels[i]) for i in range(N)]
  con_order = np.zeros([0], dtype=int)

  # remove all partial trace indices
  for counter, temp_label in enumerate(new_labels):
    uni_inds, counts = np.unique(temp_label, return_counts=True)
    tr_inds = uni_inds[np.flatnonzero(counts == 2)]
    con_order = np.concatenate((con_order, tr_inds))
    new_labels[counter] = temp_label[np.isin(temp_label, uni_inds[counts == 1])]

  for i in range(N - 1):
    # find common indices between tensor pair
    cont_many, A_cont, B_cont = np.intersect1d(
        new_labels[orders[0, i]], new_labels[orders[1, i]], return_indices=True)
    temp_labels = np.append(
        np.delete(new_labels[orders[0, i]], A_cont),
        np.delete(new_labels[orders[1, i]], B_cont))
    con_order = list(np.concatenate((con_order, cont_many), axis=0))

    # build new set of labels
    new_labels[orders[0, i]] = temp_labels
    del new_labels[orders[1, i]]

  return con_order


def ncon_cost_check(tensors: List[np.ndarray],
                    labels: List[Union[List[int], Tuple[int]]],
                    con_order: Optional[Union[List[int], str]] = None):
  """
  Checks the computational cost of an `ncon` contraction (without actually
  doing the contraction). Ignore the cost contributions from partial traces
  (which are always sub-leading).
  Args:
    tensors: list of the tensors in the network.
    labels: length-N list of lists (or tuples) specifying the network
      connections. The jth entry of the ith list in labels labels the edge
      connected to the jth index of the ith tensor. Labels should be positive
      integers for internal indices and negative integers for free indices.
    con_order: optional argument to specify the order for contracting the
      positive indices. Defaults to ascending order if omitted.
  Returns:
    float: the cost of the network contraction, given as log10(total_FLOPS).
  """

  total_cost = np.float('-inf')
  N = len(tensors)
  tensor_dims = [np.array(np.log10(ele.shape)) for ele in tensors]
  connect_list = [np.array(ele) for ele in labels]

  # generate contraction order if necessary
  flat_connect = np.concatenate(connect_list)
  if con_order is None:
    con_order = np.unique(flat_connect[flat_connect > 0])
  else:
    con_order = np.array(con_order)

  # do all partial traces
  for counter, temp_connect in enumerate(connect_list):
    uni_inds, counts = np.unique(temp_connect, return_counts=True)
    tr_inds = np.isin(temp_connect, uni_inds[counts == 1])
    tensor_dims[counter] = tensor_dims[counter][tr_inds]
    connect_list[counter] = temp_connect[tr_inds]
    con_order = con_order[np.logical_not(
        np.isin(con_order, uni_inds[counts == 2]))]

  # do all binary contractions
  while len(con_order) > 0:
    # identify tensors to be contracted
    cont_ind = con_order[0]
    locs = [
        ele for ele in range(len(connect_list))
        if sum(connect_list[ele] == cont_ind) > 0
    ]

    # identify indices to be contracted
    c1 = connect_list.pop(locs[1])
    c0 = connect_list.pop(locs[0])
    cont_many, A_cont, B_cont = np.intersect1d(
        c0, c1, assume_unique=True, return_indices=True)

    # identify dimensions of contracted
    d1 = tensor_dims.pop(locs[1])
    d0 = tensor_dims.pop(locs[0])
    single_cost = np.sum(d0) + np.sum(d1) - np.sum(d0[A_cont])
    total_cost = single_cost + np.log10(1 + 10**(total_cost - single_cost))

    # update lists
    tensor_dims.append(np.append(np.delete(d0, A_cont), np.delete(d1, B_cont)))
    connect_list.append(np.append(np.delete(c0, A_cont), np.delete(c1, B_cont)))
    con_order = con_order[np.logical_not(np.isin(con_order, cont_many))]

  # do all outer products
  N = len(tensor_dims)
  if N > 1:
    tensor_sizes = np.sort([np.sum(tensor_dims[ele]) for ele in range(N)])
    for _ in range(N - 1):
      single_cost = tensor_sizes[0] + tensor_sizes[1]
      tensor_sizes[0] += tensor_sizes[1]
      tensor_sizes = np.sort(np.delete(tensor_sizes, 1))
      total_cost = single_cost + np.log10(1 + 10**(total_cost - single_cost))

  return total_cost
