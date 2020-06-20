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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from typing import List, Union, Any, Tuple, Type, Optional, Sequence

from tensornetwork.contractors.custom_path_solvers.pathsolvers import full_solve_complete


#------------------------------------------------------------------------#
def ncon_solver(tensors: List[np.ndarray],
                connects: List[List[int]],
                max_branch: Optional[int] = None):
  """
  Solve for the contraction order of a tensor network (encoded in the `ncon` 
  syntax) that minimizes the computational cost. 
  Args:
    tensors: list of the tensors in the network.
    connects: list of the tensor connections (in standard `ncon` format).
    max_branch: maximum number of contraction paths to search at each step. 
  Returns:
    np.ndarray: the cheapest contraction order found (in ncon format).
    float: the cost of the network contraction, given as log10(total_FLOPS).
    bool: specifies if contraction order is guaranteed optimal.
  """
  # build log-adjacency matrix
  log_adj = ncon_to_adj(tensors,connects)
  
  # run search algorithm
  order, costs, is_optimal = full_solve_complete(log_adj,max_branch=max_branch)
  
  # put contraction order back into ncon format
  cont_order = ord_to_ncon(connects,order)
  
  return cont_order, costs, is_optimal

#------------------------------------------------------------------------#
def ncon_to_adj(tensors: List[np.ndarray],
                connects: List[List[int]]):
  """
  Create a log-adjacency matrix for a network from the `ncon` syntax. 
  Args:
    tensors: list of the tensors in the network.
    connects: list of the tensor connections (in standard `ncon` format).
  Returns:
    np.ndarray: the log-adjacency matrix.
  """
  # process inputs  
  N = len(connects)
  ranks = [len(connects[i]) for i in range(N)]
  flat_connects = np.hstack([connects[i] for i in range(N)])
  tensor_counter = np.hstack([i*np.ones(ranks[i],dtype=int) for i in range(N)])
  index_counter = np.hstack([np.arange(ranks[i]) for i in range(N)])
  
  # build log-adjacency index-by-index  
  log_adj = np.zeros([N,N]);
  unique_connects = np.unique(flat_connects)
  for i in range(len(unique_connects)):
      # identify tensor/index location of each edge
      tnr = tensor_counter[flat_connects == unique_connects[i]]
      ind = index_counter[flat_connects == unique_connects[i]]
      if len(ind) == 1: # external index
        log_adj[tnr[0],tnr[0]] += np.log10(tensors[tnr[0]].shape[ind[0]])
      elif len(ind) == 2: # internal index
        if (tnr[0] != tnr[1]): # ignore partial traces
          log_adj[tnr[0],tnr[1]] += np.log10(tensors[tnr[0]].shape[ind[0]])
          log_adj[tnr[1],tnr[0]] += np.log10(tensors[tnr[0]].shape[ind[0]])
   
  return log_adj

#------------------------------------------------------------------------#
def ord_to_ncon(connects: List[List[int]],
                orders: np.ndarray):
  """
  Produces a `ncon` compatible index contraction order from the sequence of 
  pairwise contractions. 
  Args:
    connects: list of the tensor connections (in standard `ncon` format).
    orders: array of dim (2,N-1) specifying the set of N-1 pairwise 
      tensor contractions.
  Returns:
    np.ndarray: the contraction order (in `ncon` format).
  """
  
  N = 1 + (orders.size//2)
  orders = orders.reshape(2,N-1)
  new_connects = [connects[i] for i in range(N)]
  cont_order = np.zeros([0],dtype=int)
  
  for i in range(N): # always do partial traces first
    uni_inds, counts = np.unique(new_connects[i],return_counts=True)
    tr_inds = uni_inds[np.flatnonzero(counts == 2)]
    cont_order = np.concatenate((cont_order,tr_inds))
    temp_connect = np.array(new_connects[i])
    for ind in tr_inds:
      temp_connect = temp_connect[temp_connect != ind]
    new_connects[i] = temp_connect 
  
  for i in range(N-1):  
    # find common indices between tensor pair
    cont_many, A_cont, B_cont = np.intersect1d(new_connects[orders[0,i]],
                                               new_connects[orders[1,i]],  
                                               return_indices=True)
    temp_connects = np.append(np.delete(new_connects[orders[0,i]], A_cont), 
                              np.delete(new_connects[orders[1,i]], B_cont))
    cont_order = list(np.concatenate((cont_order,cont_many),axis=0))
    
    # build new set of connects
    new_connects[orders[0,i]] = temp_connects
    del new_connects[orders[1,i]]
    
  return cont_order

#------------------------------------------------------------------------#
def ncon_cost_check(tensor_list_in: List[np.ndarray], 
                    connect_list_in: List[Union[List[int], Tuple[int]]], 
                    cont_order: Optional[Union[List[int], str]] = None):
  """
  Checks the computational cost of an `ncon` contraction (without actually 
  doing the contraction). Ignore the cost contributions from partial traces 
  (which are always sub-leading).
  Args:
    tensors: list of the tensors in the network.
    connects: length-N list of lists (or tuples) specifying the network 
      connections. The jth entry of the ith list in connects labels the edge 
      connected to the jth index of the ith tensor. Labels should be positive 
      integers for internal indices and negative integers for free indices.
    cont_order: optional argument to specify the order for contracting the 
      positive indices. Defaults to ascending order if omitted. 
  Returns:
    float: the cost of the network contraction, given as log10(total_FLOPS).
  """
  
  total_cost = None
  num_tensors = len(tensor_list_in)
  tensor_dims = [np.array(np.log10(tensor_list_in[ele].shape)) 
                 for ele in range(num_tensors)]
  connect_list = [np.array(connect_list_in[ele]) for ele in range(num_tensors)]
  
  # generate contraction order if necessary
  flat_connect = np.concatenate(connect_list)
  if cont_order is None:
    cont_order = np.unique(flat_connect[flat_connect > 0])
  else:
    cont_order = np.array(cont_order)
  
  # do all partial traces
  for ele in range(num_tensors):
    uni_inds, counts = np.unique(connect_list[ele],return_counts=True)
    tr_inds = uni_inds[counts==2]
    temp_connect = connect_list[ele]
    temp_dims = tensor_dims[ele]
    for ind in tr_inds:
      temp_dims = temp_dims[temp_connect != ind]
      temp_connect = temp_connect[temp_connect != ind]
      cont_order = np.delete(cont_order, np.flatnonzero(cont_order == ind))
      
    tensor_dims[ele] = temp_dims
    connect_list[ele] = temp_connect
  
  # do all binary contractions
  while len(cont_order) > 0:
    # identify tensors to be contracted
    cont_ind = cont_order[0]
    locs = [ele for ele in range(len(connect_list)) 
            if sum(connect_list[ele] == cont_ind) > 0]
  
    # do binary contraction
    cont_many, A_cont, B_cont = np.intersect1d(connect_list[locs[0]], 
                                               connect_list[locs[1]], 
                                               assume_unique=True, 
                                               return_indices=True)
    A_dim = np.sum(tensor_dims[locs[0]])
    B_dim = np.sum(tensor_dims[locs[1]]) 
    cont_dim = np.sum((tensor_dims[locs[0]])[A_cont])
    single_cost = A_dim + B_dim - cont_dim
    if total_cost is None:
      total_cost =  single_cost
    else:
      total_cost =  total_cost + np.log10(1+10**(single_cost-total_cost))
    
    tensor_dims.append(np.append(np.delete(tensor_dims[locs[0]], A_cont), 
                                 np.delete(tensor_dims[locs[1]], B_cont)))
    connect_list.append(np.append(np.delete(connect_list[locs[0]], A_cont), 
                                  np.delete(connect_list[locs[1]], B_cont)))
  
    # remove contracted tensors from list and update cont_order
    del tensor_dims[locs[1]]
    del tensor_dims[locs[0]]
    del connect_list[locs[1]]
    del connect_list[locs[0]]
    cont_order = np.delete(cont_order,np.intersect1d(cont_order,cont_many, 
                                                     assume_unique=True, 
                                                     return_indices=True)[1])
  
  # do all outer products
  num_tensors = len(tensor_dims)
  if (num_tensors > 1):
    tensor_sizes = np.sort([np.sum(tensor_dims[ele]) 
                            for ele in range(num_tensors)])
    for i in range(num_tensors-1):
      single_cost = tensor_sizes[0] + tensor_sizes[1]
      tensor_sizes[0] += tensor_sizes[1]
      tensor_sizes = np.sort(np.delete(tensor_sizes,1))
      if total_cost is None:
        total_cost =  single_cost
      else:
        total_cost =  total_cost + np.log10(1+10**(single_cost-total_cost))
      
  return total_cost
    
  

