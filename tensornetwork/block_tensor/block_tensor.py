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
#from tensornetwork.block_tensor.lookup import lookup
from tensornetwork.backends import backend_factory
# pylint: disable=line-too-long
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index
# pylint: disable=line-too-long
from tensornetwork.block_tensor.charge import fuse_degeneracies, fuse_charges, fuse_degeneracies, BaseCharge, ChargeCollection
import numpy as np
import scipy as sp
import itertools
import time
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable, Sequence
Tensor = Any


def _compute_sparse_lookups(row_charges: Union[BaseCharge, ChargeCollection],
                            row_flows, column_charges, column_flows):
  """
  Compute lookup tables for looking up how dense index positions map 
  to sparse index positions for the diagonal blocks a symmetric matrix.
  Args:
    row_charges:

  """
  column_flows = list(-np.asarray(column_flows))
  fused_column_charges = fuse_charges(column_charges, column_flows)
  fused_row_charges = fuse_charges(row_charges, row_flows)
  unique_column_charges, column_inverse = fused_column_charges.unique(
      return_inverse=True)
  unique_row_charges, row_inverse = fused_row_charges.unique(
      return_inverse=True)
  common_charges, comm_row, comm_col = unique_row_charges.intersect(
      unique_column_charges, return_indices=True)

  col_ind_sort = np.argsort(column_inverse, kind='stable')
  row_ind_sort = np.argsort(row_inverse, kind='stable')
  _, col_charge_degeneracies = compute_fused_charge_degeneracies(
      column_charges, column_flows)
  _, row_charge_degeneracies = compute_fused_charge_degeneracies(
      row_charges, row_flows)
  # labelsorted_indices = column_inverse[col_ind_sort]
  # tmp = np.nonzero(
  #     np.append(labelsorted_indices, unique_column_charges.charges.shape[0] + 1) -
  #     np.append(labelsorted_indices[0], labelsorted_indices))[0]
  #charge_degeneracies = tmp - np.append(0, tmp[0:-1])

  col_start_positions = np.cumsum(np.append(0, col_charge_degeneracies))
  row_start_positions = np.cumsum(np.append(0, row_charge_degeneracies))
  column_lookup = np.empty(len(fused_column_charges), dtype=np.int64)
  row_lookup = np.zeros(len(fused_row_charges), dtype=np.int64)
  for n in range(len(common_charges)):
    column_lookup[col_ind_sort[col_start_positions[
        comm_col[n]]:col_start_positions[comm_col[n] + 1]]] = np.arange(
            col_charge_degeneracies[comm_col[n]])
    # row_start_positions[comm_row[n]]
    # row_start_positions[comm_row[n] + 1]
    row_lookup[
        row_ind_sort[row_start_positions[comm_row[n]]:row_start_positions[
            comm_row[n] + 1]]] = col_charge_degeneracies[comm_col[n]]

  return np.append(0, np.cumsum(row_lookup[0:-1])), column_lookup


def _get_strides(dims):
  return np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))


def _get_stride_arrays(dims):
  strides = np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))
  return [np.arange(dims[n]) * strides[n] for n in range(len(dims))]


def _find_values_in_fused(indices: np.ndarray, left: np.ndarray,
                          right: np.ndarray) -> np.ndarray:
  """
  Returns fuse(left,right)[indices], i.e. the elements
  in the fusion of `left` and `right` at positions `indices'.
  """
  left_inds, right_inds = np.divmod(indices, len(right))
  return left[left_inds] + right[right_inds]


def fuse_ndarray_pair(array1: Union[List, np.ndarray],
                      array2: Union[List, np.ndarray]) -> np.ndarray:
  """
  Fuse ndarrays `array1` and `array2` by kronecker-addition. 
  Given `array1 = [0,1,2]` and `array2 = [10,100]`, this returns
  `[10, 100, 11, 101, 12, 102]`.

  Args:
    array1: np.ndarray
    array2: np.ndarray
  Returns:
    np.ndarray: The result of adding `array1` and `array2`
  """
  return np.reshape(
      np.asarray(array1)[:, None] + np.asarray(array2)[None, :],
      len(array1) * len(array2))


def fuse_ndarrays(arrays: List[Union[List, np.ndarray]]) -> np.ndarray:
  """
  Fuse all `arrays` by simple kronecker addition.
  Arrays are fused from "right to left", 
  Args:
    arrays: A list of arrays to be fused.
  Returns:
    np.ndarray: The result of fusing `charges`.
  """
  if len(arrays) == 1:
    return arrays[0]
  fused_arrays = arrays[0]
  for n in range(1, len(arrays)):
    fused_arrays = fuse_ndarray_pair(array1=fused_arrays, array2=arrays[n])
  return fused_arrays


def _check_flows(flows: List[int]) -> None:
  if (set(flows) != {1}) and (set(flows) != {-1}) and (set(flows) != {-1, 1}):
    raise ValueError(
        "flows = {} contains values different from 1 and -1".format(flows))


def _find_best_partition(charges: List[Union[BaseCharge, ChargeCollection]],
                         flows: List[int],
                         return_charges: Optional[bool] = True
                        ) -> Tuple[Union[BaseCharge, ChargeCollection],
                                   Union[BaseCharge, ChargeCollection], int]:
  """
  compute the best partition for fusing `charges`, i.e. the integer `p`
  such that fusing `len(fuse_charges(charges[0:p],flows[0:p]))` is
  and `len(fuse_charges(charges[p::],flows[p::]))` are as close as possible.
  Returns:
    fused_left_charges, fused_right_charges, p
  
  """
  #FIXME: fusing charges with dims (N,M) with M>~N is faster than fusing charges
  # with dims (M,N). Thus, it is not always best to fuse at the minimum cut.
  #for example, for dims (1000, 4, 1002), its better to fuse at the cut
  #(1000, 4008) than at (4000, 1002), even though the difference between the
  #dimensions is minimal for the latter case. We should implement some heuristic
  #to find these cuts.
  if len(charges) == 1:
    raise ValueError(
        '_expecting `charges` with a length of at least 2, got `len(charges)={}`'
        .format(len(charges)))
  dims = np.asarray([len(c) for c in charges])
  diffs = [
      np.abs(np.prod(dims[0:n]) - np.prod(dims[n::]))
      for n in range(1, len(charges))
  ]
  min_inds = np.nonzero(diffs == np.min(diffs))[0]
  if len(min_inds) > 1:
    right_dims = [np.prod(len(charges[min_ind + 1::])) for min_ind in min_inds]
    min_ind = min_inds[np.argmax(right_dims)]
  else:
    min_ind = min_inds[0]
  if return_charges:
    fused_left_charges = fuse_charges(charges[0:min_ind + 1],
                                      flows[0:min_ind + 1])
    fused_right_charges = fuse_charges(charges[min_ind + 1::],
                                       flows[min_ind + 1::])

    return fused_left_charges, fused_right_charges, min_ind + 1
  return min_ind + 1


def compute_fused_charge_degeneracies(
    charges: List[Union[BaseCharge, ChargeCollection]],
    flows: List[Union[bool, int]]
) -> Tuple[Union[BaseCharge, ChargeCollection], np.ndarray]:
  """
  For a list of charges, compute all possible fused charges resulting
  from fusing `charges`, together with their respective degeneracies
  Args:
    charges: List of np.ndarray of int, one for each leg of the 
      underlying tensor. Each np.ndarray `charges[leg]` 
      is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
  Returns:
    Union[BaseCharge, ChargeCollection]:  The unique fused charges.
    np.ndarray of integers: The degeneracies of each unqiue fused charge.
  """
  if len(charges) == 1:
    return (charges[0] * flows[0]).unique(return_counts=True)

  # get unique charges and their degeneracies on the first leg.
  # We are fusing from "left" to "right".
  accumulated_charges, accumulated_degeneracies = (
      charges[0] * flows[0]).unique(return_counts=True)
  for n in range(1, len(charges)):
    #list of unique charges and list of their degeneracies
    #on the next unfused leg of the tensor
    leg_charges, leg_degeneracies = charges[n].unique(return_counts=True)
    #fuse the unique charges
    #Note: entries in `fused_charges` are not unique anymore.
    #flow1 = 1 because the flow of leg 0 has already been
    #mulitplied above
    fused_charges = accumulated_charges + leg_charges * flows[n]
    #compute the degeneracies of `fused_charges` charges
    #`fused_degeneracies` is a list of degeneracies such that
    # `fused_degeneracies[n]` is the degeneracy of of
    # charge `c = fused_charges[n]`.
    fused_degeneracies = fuse_degeneracies(accumulated_degeneracies,
                                           leg_degeneracies)
    accumulated_charges = fused_charges.unique()
    accumulated_degeneracies = np.empty(
        len(accumulated_charges), dtype=np.int64)

    for n in range(len(accumulated_charges)):
      accumulated_degeneracies[n] = np.sum(
          fused_degeneracies[fused_charges == accumulated_charges[n]])

  return accumulated_charges, accumulated_degeneracies


def compute_unique_fused_charges(
    charges: List[Union[BaseCharge, ChargeCollection]],
    flows: List[Union[bool, int]]
) -> Tuple[Union[BaseCharge, ChargeCollection], np.ndarray]:
  """
  For a list of charges, compute all possible fused charges resulting
  from fusing `charges`.
  Args:
    charges: List of np.ndarray of int, one for each leg of the 
      underlying tensor. Each np.ndarray `charges[leg]` 
      is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
  Returns:
    Union[BaseCharge, ChargeCollection]:  The unique fused charges.
    np.ndarray of integers: The degeneracies of each unqiue fused charge.
  """
  if len(charges) == 1:
    return (charges[0] * flows[0]).unique()

  # get unique charges and their degeneracies on the first leg.
  # We are fusing from "left" to "right".
  accumulated_charges = (charges[0] * flows[0]).unique()
  for n in range(1, len(charges)):
    #list of unique charges and list of their degeneracies
    #on the next unfused leg of the tensor
    leg_charges = charges[n].unique()
    #fuse the unique charges
    #Note: entries in `fused_charges` are not unique anymore.
    #flow1 = 1 because the flow of leg 0 has already been
    #mulitplied above
    fused_charges = accumulated_charges + leg_charges * flows[n]
    #compute the degeneracies of `fused_charges` charges
    #`fused_degeneracies` is a list of degeneracies such that
    # `fused_degeneracies[n]` is the degeneracy of of
    # charge `c = fused_charges[n]`.
    accumulated_charges = fused_charges.unique()
  return accumulated_charges


def compute_num_nonzero(charges: List[np.ndarray],
                        flows: List[Union[bool, int]]) -> int:
  """
  Compute the number of non-zero elements, given the meta-data of 
  a symmetric tensor.
  Args:
    charges: List of np.ndarray of int, one for each leg of the 
      underlying tensor. Each np.ndarray `charges[leg]` 
      is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
  Returns:
    int: The number of non-zero elements.
  """
  accumulated_charges, accumulated_degeneracies = compute_fused_charge_degeneracies(
      charges, flows)
  res = accumulated_charges == accumulated_charges.zero_charge

  if len(np.nonzero(res)[0]) == 0:
    raise ValueError(
        "given leg-charges `charges` and flows `flows` are incompatible "
        "with a symmetric tensor")
  return accumulated_degeneracies[res][0]


def _find_diagonal_sparse_blocks(
    data: np.ndarray,
    row_charges: List[Union[BaseCharge, ChargeCollection]],
    column_charges: List[Union[BaseCharge, ChargeCollection]],
    row_flows: List[Union[bool, int]],
    column_flows: List[Union[bool, int]],
    return_data: Optional[bool] = False
) -> Tuple[Union[BaseCharge, ChargeCollection], List, np.ndarray, Dict, Dict]:
  """
  Given the meta data and underlying data of a symmetric matrix, compute 
  all diagonal blocks and return them in a dict.
  `row_charges` and `column_charges` are lists of np.ndarray. The tensor
  is viewed as a matrix with rows given by fusing `row_charges` and 
  columns given by fusing `column_charges`. 

  Args: 
    data: An np.ndarray of the data. The number of elements in `data`
      has to match the number of non-zero elements defined by `charges` 
      and `flows`
    row_charges: List of np.ndarray, one for each leg of the row-indices.
      Each np.ndarray `row_charges[leg]` is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    column_charges: List of np.ndarray, one for each leg of the column-indices.
      Each np.ndarray `row_charges[leg]` is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    row_flows: A list of integers, one for each entry in `row_charges`.
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
    column_flows: A list of integers, one for each entry in `column_charges`.
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
    return_data: If `True`, the return dictionary maps quantum numbers `q` to 
      actual `np.ndarray` with the data. This involves a copy of data.
      If `False`, the returned dict maps quantum numbers of a list 
      [locations, shape], where `locations` is an np.ndarray of type np.int64
      containing the sparse locations of the tensor elements within A.data, i.e.
      `A.data[locations]` contains the elements belonging to the tensor with 
      quantum numbers `(q,q). `shape` is the shape of the corresponding array.

  Returns:
  return common_charges, blocks, start_positions, row_locations, column_degeneracies
    List[Union[BaseCharge, ChargeCollection]]: A list of unique charges, one per block.
    List[np.ndarray]: A list containing the blocks.
    np.ndarray: The start position within the sparse data array of each row with non-zero 
      elements.
    Dict: Dict mapping row-charges of each block to an np.ndarray of sparse positions 
      along the rows 
    Dict: Dict mapping row-charges of each block to its column-degeneracy
  """
  flows = row_flows.copy()
  flows.extend(column_flows)
  _check_flows(flows)
  if len(flows) != (len(row_charges) + len(column_charges)):
    raise ValueError(
        "`len(flows)` is different from `len(row_charges) + len(column_charges)`"
    )

  #get the unique column-charges
  #we only care about their degeneracies, not their order; that's much faster
  #to compute since we don't have to fuse all charges explicitly
  #`compute_fused_charge_degeneracies` multiplies flows into the column_charges
  unique_column_charges, column_dims = compute_fused_charge_degeneracies(
      column_charges, column_flows)
  unique_row_charges = compute_unique_fused_charges(row_charges, row_flows)
  #get the charges common to rows and columns (only those matter)
  common_charges = unique_row_charges.intersect(unique_column_charges * (-1))

  #convenience container for storing the degeneracies of each
  #column charge
  #column_degeneracies = dict(zip(unique_column_charges, column_dims))
  column_degeneracies = dict(zip(unique_column_charges * (-1), column_dims))
  row_locations = find_sparse_positions(
      charges=row_charges, flows=row_flows, target_charges=common_charges)

  degeneracy_vector = np.empty(
      np.sum([len(v) for v in row_locations.values()]), dtype=np.int64)
  #for each charge `c` in `common_charges` we generate a boolean mask
  #for indexing the positions where `relevant_column_charges` has a value of `c`.
  for c in common_charges:
    degeneracy_vector[row_locations[c]] = column_degeneracies[c]

  # the result of the cumulative sum is a vector containing
  # the stop positions of the non-zero values of each row
  # within the data vector.
  # E.g. for `relevant_row_charges` = [0,1,0,0,3],  and
  # column_degeneracies[0] = 10
  # column_degeneracies[1] = 20
  # column_degeneracies[3] = 30
  # we have
  # `stop_positions` = [10, 10+20, 10+20+10, 10+20+10+10, 10+20+10+10+30]
  # The starting positions of consecutive elements (in row-major order) in
  # each row with charge `c=0` within the data vector are then simply obtained using
  # masks[0] = [True, False, True, True, False]
  # and `stop_positions[masks[0]] - column_degeneracies[0]`
  start_positions = np.cumsum(degeneracy_vector) - degeneracy_vector
  blocks = []

  for c in common_charges:
    #numpy broadcasting is substantially faster than kron!
    rlocs = row_locations[c]
    rlocs.sort()  #sort in place (we need it again later)
    cdegs = column_degeneracies[c]
    a = np.expand_dims(start_positions[rlocs], 1)
    b = np.expand_dims(np.arange(cdegs), 0)
    inds = np.reshape(a + b, len(rlocs) * cdegs)
    if not return_data:
      blocks.append([inds, (len(rlocs), cdegs)])
    else:
      blocks.append(np.reshape(data[inds], (len(rlocs), cdegs)))
  return common_charges, blocks, start_positions, row_locations, column_degeneracies


def _find_diagonal_dense_blocks(
    row_charges: List[Union[BaseCharge, ChargeCollection]],
    column_charges: List[Union[BaseCharge, ChargeCollection]],
    row_flows: List[Union[bool, int]],
    column_flows: List[Union[bool, int]],
    row_strides: Optional[np.ndarray] = None,
    column_strides: Optional[np.ndarray] = None,
) -> Tuple[Union[BaseCharge, ChargeCollection], List[np.ndarray]]:
  """
  Given the meta data and underlying data of a symmetric matrix, compute the 
  dense positions of all diagonal blocks and return them in a dict.
  `row_charges` and `column_charges` are lists of np.ndarray. The tensor
  is viewed as a matrix with rows given by fusing `row_charges` and 
  columns given by fusing `column_charges`.

  Args: 
    data: An np.ndarray of the data. The number of elements in `data`
      has to match the number of non-zero elements defined by `charges` 
      and `flows`
    row_charges: List of np.ndarray, one for each leg of the row-indices.
      Each np.ndarray `row_charges[leg]` is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    column_charges: List of np.ndarray, one for each leg of the column-indices.
      Each np.ndarray `row_charges[leg]` is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    row_flows: A list of integers, one for each entry in `row_charges`.
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
    column_flows: A list of integers, one for each entry in `column_charges`.
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
    row_strides: An optional np.ndarray denoting the strides of `row_charges`.
      If `None`, natural stride ordering is assumed.
    column_strides: An optional np.ndarray denoting the strides of 
      `column_charges`. If `None`, natural stride ordering is assumed.

  Returns:
    List[Union[BaseCharge, ChargeCollection]]: A list of unique charges, one per block.
    List[List]: A list containing the blocks information. 
      For each element `e` in the list `e[0]` is an `np.ndarray` of ints 
      denoting the dense positions of the non-zero elements and `e[1]` 
      is a tuple corresponding to the blocks' matrix shape

  """
  flows = list(row_flows).copy()
  flows.extend(column_flows)
  _check_flows(flows)
  if len(flows) != (len(row_charges) + len(column_charges)):
    raise ValueError(
        "`len(flows)` is different from `len(row_charges) + len(column_charges)`"
    )
  #get the unique column-charges
  #we only care about their degeneracies, not their order; that's much faster
  #to compute since we don't have to fuse all charges explicitly
  #`compute_fused_charge_degeneracies` multiplies flows into the column_charges
  unique_column_charges = compute_unique_fused_charges(column_charges,
                                                       column_flows)

  unique_row_charges = compute_unique_fused_charges(row_charges, row_flows)
  #get the charges common to rows and columns (only those matter)
  fused = unique_row_charges + unique_column_charges
  li, ri = np.divmod(
      np.nonzero(fused == unique_column_charges.zero_charge)[0],
      len(unique_column_charges))
  common_charges = unique_row_charges.intersect(unique_column_charges * (-1))
  #print('_find_diagonal_sparse_blocks: unique charges ', time.time() - t1)
  if ((row_strides is None) and
      (column_strides is not None)) or ((row_strides is not None) and
                                        (column_strides is None)):
    raise ValueError("`row_strides` and `column_strides` "
                     "have to be passed simultaneously."
                     " Found `row_strides={}` and "
                     "`column_strides={}`".format(row_strides, column_strides))
  if row_strides is not None:
    row_locations = find_dense_positions(
        charges=row_charges,
        flows=row_flows,
        target_charges=unique_row_charges[li],
        strides=row_strides)

  else:
    column_dim = np.prod([len(c) for c in column_charges])
    row_locations = find_dense_positions(
        charges=row_charges,
        flows=row_flows,
        target_charges=unique_row_charges[li])
    for v in row_locations.values():
      v *= column_dim
  if column_strides is not None:
    column_locations = find_dense_positions(
        charges=column_charges,
        flows=column_flows,
        target_charges=unique_column_charges[ri],
        strides=column_strides,
        store_dual=True)

  else:
    column_locations = find_dense_positions(
        charges=column_charges,
        flows=column_flows,
        target_charges=unique_column_charges[ri],
        store_dual=True)
  blocks = []
  for c in unique_row_charges[li]:
    #numpy broadcasting is substantially faster than kron!
    rlocs = np.expand_dims(row_locations[c], 1)
    clocs = np.expand_dims(column_locations[c], 0)
    inds = np.reshape(rlocs + clocs, rlocs.shape[0] * clocs.shape[1])
    blocks.append([inds, (rlocs.shape[0], clocs.shape[1])])
  return unique_row_charges[li], blocks


def find_dense_positions(charges: List[Union[BaseCharge, ChargeCollection]],
                         flows: List[Union[int, bool]],
                         target_charges: Union[BaseCharge, ChargeCollection],
                         strides: Optional[np.ndarray] = None,
                         store_dual: Optional[bool] = False) -> np.ndarray:
  """
  Find the dense locations of elements (i.e. the index-values within the DENSE tensor)
  in the vector of `fused_charges` resulting from fusing all elements of `charges`
  that have a value of `target_charge`.
  For example, given 
  ```
  charges = [[-2,0,1,0,0],[-1,0,2,1]]
  target_charge = 0
  fused_charges = fuse_charges(charges,[1,1]) 
  print(fused_charges) # [-3,-2,0,-1,-1,0,2,1,0,1,3,2,-1,0,2,1,-1,0,2,1]
  ```
  we want to find the index-positions of charges
  that fuse to `target_charge=0`, i.e. where `fused_charges==0`,
  within the dense array. As one additional wrinkle, `charges` 
  is a subset of the permuted charges of a tensor with rank R > len(charges), 
  and `stride_arrays` are their corresponding range of strides, i.e.

  ```
  R=5
  D = [2,3,4,5,6]
  tensor_flows = np.random.randint(-1,2,R)
  tensor_charges = [np.random.randing(-5,5,D[n]) for n in range(R)]
  order = np.arange(R)
  np.random.shuffle(order)
  tensor_strides = [360, 120,  30,   6,   1]
  
  charges = [tensor_charges[order[n]] for n in range(3)]
  flows = [tensor_flows[order[n]] for n in range(len(3))]
  strides = [tensor_stride[order[n]] for n in range(3)]
  _ = _find_transposed_dense_positions(charges, flows, 0, strides)
  
  ```
  `_find_transposed_dense_blocks` returns an np.ndarray containing the 
  index-positions of  these elements calculated using `stride_arrays`. 
  The result only makes sense in conjuction with the complementary 
  data computed from the complementary 
  elements in`tensor_charges`,
  `tensor_strides` and `tensor_flows`.
  This routine is mainly used in `_find_diagonal_dense_blocks`.

  Args:
    charges: A list of BaseCharge or ChargeCollection.
    flows: The flow directions of the `charges`.
    target_charge: The target charge.
    strides: The strides for the `charges` subset.
      if `None`, natural stride ordering is assumed.

  Returns:
    np.ndarray: The index-positions within the dense data array 
      of the elements fusing to `target_charge`.
  """

  _check_flows(flows)
  out = {}
  if store_dual:
    store_charges = target_charges * (-1)
  else:
    store_charges = target_charges

  if len(charges) == 1:
    fused_charges = charges[0] * flows[0]
    inds = np.nonzero(fused_charges == target_charges)
    if len(target_charges) > 1:
      for n in range(len(target_charges)):
        i = inds[0][inds[1] == n]
        if len(i) == 0:
          continue
        if strides is not None:
          permuted_inds = strides[0] * np.arange(len(charges[0]))
          out[store_charges.get_item(n)] = permuted_inds[i]
        else:
          out[store_charges.get_item(n)] = i
      return out
    else:
      if strides is not None:
        permuted_inds = strides[0] * np.arange(len(charges[0]))
        out[store_charges.get_item(n)] = permuted_inds[inds[0]]
      else:
        out[store_charges.get_item(n)] = inds[0]
      return out

  left_charges, right_charges, partition = _find_best_partition(charges, flows)
  if strides is not None:
    stride_arrays = [
        np.arange(len(charges[n])) * strides[n] for n in range(len(charges))
    ]
    permuted_left_inds = fuse_ndarrays(stride_arrays[0:partition])
    permuted_right_inds = fuse_ndarrays(stride_arrays[partition:])

  # unique_target_charges, inds = target_charges.unique(return_index=True)
  # target_charges = target_charges[np.sort(inds)]
  unique_left, left_inverse = left_charges.unique(return_inverse=True)
  unique_right, right_inverse = right_charges.unique(return_inverse=True)

  fused_unique = unique_left + unique_right
  unique_inds = np.nonzero(fused_unique == target_charges)

  relevant_positions = unique_inds[0]
  tmp_inds_left, tmp_inds_right = np.divmod(relevant_positions,
                                            len(unique_right))

  relevant_unique_left_inds = np.unique(tmp_inds_left)
  left_lookup = np.empty(np.max(relevant_unique_left_inds) + 1, dtype=np.int64)
  left_lookup[relevant_unique_left_inds] = np.arange(
      len(relevant_unique_left_inds))
  relevant_unique_right_inds = np.unique(tmp_inds_right)
  right_lookup = np.empty(
      np.max(relevant_unique_right_inds) + 1, dtype=np.int64)
  right_lookup[relevant_unique_right_inds] = np.arange(
      len(relevant_unique_right_inds))

  left_charge_labels = np.nonzero(
      np.expand_dims(left_inverse, 1) == np.expand_dims(
          relevant_unique_left_inds, 0))
  right_charge_labels = np.nonzero(
      np.expand_dims(right_inverse, 1) == np.expand_dims(
          relevant_unique_right_inds, 0))

  len_right = len(right_charges)

  for n in range(len(target_charges)):
    if len(unique_inds) > 1:
      lis, ris = np.divmod(unique_inds[0][unique_inds[1] == n],
                           len(unique_right))
    else:
      lis, ris = np.divmod(unique_inds[0], len(unique_right))
    dense_positions = []
    left_positions = []
    lookup = []
    for m in range(len(lis)):
      li = lis[m]
      ri = ris[m]
      dense_left_positions = left_charge_labels[0][left_charge_labels[1] ==
                                                   left_lookup[li]]
      dense_right_positions = right_charge_labels[0][right_charge_labels[1] ==
                                                     right_lookup[ri]]
      if strides is None:
        positions = np.expand_dims(dense_left_positions * len_right,
                                   1) + np.expand_dims(dense_right_positions, 0)
      else:
        positions = np.expand_dims(
            permuted_left_inds[dense_left_positions], 1) + np.expand_dims(
                permuted_right_inds[dense_right_positions], 0)

      dense_positions.append(positions)
      left_positions.append(dense_left_positions)
      lookup.append(
          np.stack([
              np.arange(len(dense_left_positions)),
              np.full(len(dense_left_positions), fill_value=m, dtype=np.int64)
          ],
                   axis=1))

    if len(lookup) > 0:
      ind_sort = np.argsort(np.concatenate(left_positions))
      it = np.concatenate(lookup, axis=0)
      table = it[ind_sort, :]
      out[store_charges.get_item(n)] = np.concatenate([
          dense_positions[table[n, 1]][table[n, 0], :]
          for n in range(table.shape[0])
      ])
    else:
      out[store_charges.get_item(n)] = np.array([])

  return out


def find_sparse_positions(
    charges: List[Union[BaseCharge, ChargeCollection]],
    flows: List[Union[int, bool]],
    target_charges: Union[BaseCharge, ChargeCollection]) -> Dict:
  """
  Find the sparse locations of elements (i.e. the index-values within 
  the SPARSE tensor) in the vector `fused_charges` (resulting from 
  fusing `left_charges` and `right_charges`) 
  that have a value of `target_charges`, assuming that all elements 
  different from `target_charges` are `0`.
  For example, given 
  ```
  left_charges = [-2,0,1,0,0]
  right_charges = [-1,0,2,1]
  target_charges = [0,1]
  fused_charges = fuse_charges([left_charges, right_charges],[1,1]) 
  print(fused_charges) # [-3,-2,0,-1,-1,0,2,1,0,1,3,2,-1,0,2,1,-1,0,2,1]
  ```                           0       1   2 3 4        5   6    7   8 
  we want to find the all different blocks 
  that fuse to `target_charges=[0,1]`, i.e. where `fused_charges==0` or `1`, 
  together with their corresponding sparse index-values of the data in the sparse array, 
  assuming that all elements in `fused_charges` different from `target_charges` are 0.
  
  `find_sparse_blocks` returns a dict mapping integers `target_charge`
  to an array of integers denoting the sparse locations of elements within 
  `fused_charges`.
  For the above example, we get:
  * `target_charge=0`: [0,1,3,5,7]
  * `target_charge=1`: [2,4,6,8]
  Args:
    left_charges: An np.ndarray of integer charges.
    left_flow: The flow direction of the left charges.
    right_charges: An np.ndarray of integer charges.
    right_flow: The flow direction of the right charges.
    target_charge: The target charge.
  Returns:
    dict: Mapping integers to np.ndarray of integers.
  """
  #FIXME: this is probably still not optimal

  _check_flows(flows)
  if len(charges) == 1:
    fused_charges = charges[0] * flows[0]
    unique_charges = fused_charges.unique()
    target_charges = target_charges.unique()
    relevant_target_charges = unique_charges.intersect(target_charges)
    relevant_fused_charges = fused_charges[fused_charges.isin(
        relevant_target_charges)]
    return {
        c: np.nonzero(relevant_fused_charges == c)[0]
        for c in relevant_target_charges
    }

  left_charges, right_charges, partition = _find_best_partition(charges, flows)

  # unique_target_charges, inds = target_charges.unique(return_index=True)
  # target_charges = target_charges[np.sort(inds)]
  unique_left, left_inverse = left_charges.unique(return_inverse=True)
  unique_right, right_inverse, right_dims = right_charges.unique(
      return_inverse=True, return_counts=True)

  fused_unique = unique_left + unique_right
  unique_inds = np.nonzero(fused_unique == target_charges)
  relevant_positions = unique_inds[0]
  tmp_inds_left, tmp_inds_right = np.divmod(relevant_positions,
                                            len(unique_right))

  relevant_unique_left_inds = np.unique(tmp_inds_left)
  left_lookup = np.empty(np.max(relevant_unique_left_inds) + 1, dtype=np.int64)
  left_lookup[relevant_unique_left_inds] = np.arange(
      len(relevant_unique_left_inds))
  relevant_unique_right_inds = np.unique(tmp_inds_right)
  right_lookup = np.empty(
      np.max(relevant_unique_right_inds) + 1, dtype=np.int64)
  right_lookup[relevant_unique_right_inds] = np.arange(
      len(relevant_unique_right_inds))

  left_charge_labels = np.nonzero(
      np.expand_dims(left_inverse, 1) == np.expand_dims(
          relevant_unique_left_inds, 0))
  relevant_left_inverse = np.arange(len(left_charge_labels[0]))

  right_charge_labels = np.expand_dims(right_inverse, 1) == np.expand_dims(
      relevant_unique_right_inds, 0)
  right_block_information = {}
  for n in relevant_unique_left_inds:
    ri = np.nonzero((unique_left[n] + unique_right).isin(target_charges))[0]
    tmp_inds = np.nonzero(right_charge_labels[:, right_lookup[ri]])
    right_block_information[n] = [ri, np.arange(len(tmp_inds[0])), tmp_inds[1]]

  relevant_right_inverse = np.arange(len(right_charge_labels[0]))

  #generate a degeneracy vector which for each value r in relevant_right_charges
  #holds the corresponding number of non-zero elements `relevant_right_charges`
  #that can add up to `target_charges`.
  degeneracy_vector = np.empty(len(left_charge_labels[0]), dtype=np.int64)
  for n in range(len(relevant_unique_left_inds)):
    degeneracy_vector[relevant_left_inverse[
        left_charge_labels[1] == n]] = np.sum(right_dims[tmp_inds_right[
            tmp_inds_left == relevant_unique_left_inds[n]]])
  start_positions = np.cumsum(degeneracy_vector) - degeneracy_vector
  out = {}
  for n in range(len(target_charges)):
    block = []
    if len(unique_inds) > 1:
      lis, ris = np.divmod(unique_inds[0][unique_inds[1] == n],
                           len(unique_right))
    else:
      lis, ris = np.divmod(unique_inds[0], len(unique_right))

    for m in range(len(lis)):
      a = np.expand_dims(
          start_positions[relevant_left_inverse[left_charge_labels[1] ==
                                                left_lookup[lis[m]]]], 0)

      ri_tmp, arange, tmp_inds = right_block_information[lis[m]]
      b = np.expand_dims(arange[tmp_inds == np.nonzero(ri_tmp == ris[m])[0]], 1)
      inds = a + b
      block.append(np.reshape(inds, np.prod(inds.shape)))
    out[target_charges.get_item(n)] = np.concatenate(block)
  return out


# def find_sparse_positions_2(
#     charges: List[Union[BaseCharge, ChargeCollection]],
#     flows: List[Union[int, bool]],
#     target_charges: Union[BaseCharge, ChargeCollection]) -> Dict:
#   """
#   Find the sparse locations of elements (i.e. the index-values within
#   the SPARSE tensor) in the vector `fused_charges` (resulting from
#   fusing `left_charges` and `right_charges`)
#   that have a value of `target_charges`, assuming that all elements
#   different from `target_charges` are `0`.
#   For example, given
#   ```
#   left_charges = [-2,0,1,0,0]
#   right_charges = [-1,0,2,1]
#   target_charges = [0,1]
#   fused_charges = fuse_charges([left_charges, right_charges],[1,1])
#   print(fused_charges) # [-3,-2,0,-1,-1,0,2,1,0,1,3,2,-1,0,2,1,-1,0,2,1]
#   ```                           0       1   2 3 4        5   6    7   8
#   we want to find the all different blocks
#   that fuse to `target_charges=[0,1]`, i.e. where `fused_charges==0` or `1`,
#   together with their corresponding sparse index-values of the data in the sparse array,
#   assuming that all elements in `fused_charges` different from `target_charges` are 0.

#   `find_sparse_blocks` returns a dict mapping integers `target_charge`
#   to an array of integers denoting the sparse locations of elements within
#   `fused_charges`.
#   For the above example, we get:
#   * `target_charge=0`: [0,1,3,5,7]
#   * `target_charge=1`: [2,4,6,8]
#   Args:
#     left_charges: An np.ndarray of integer charges.
#     left_flow: The flow direction of the left charges.
#     right_charges: An np.ndarray of integer charges.
#     right_flow: The flow direction of the right charges.
#     target_charge: The target charge.
#   Returns:
#     dict: Mapping integers to np.ndarray of integers.
#   """
#   #FIXME: this is probably still not optimal

#   _check_flows(flows)
#   if len(charges) == 1:
#     fused_charges = charges[0] * flows[0]
#     unique_charges = fused_charges.unique()
#     target_charges = target_charges.unique()
#     relevant_target_charges = unique_charges.intersect(target_charges)
#     relevant_fused_charges = fused_charges[fused_charges.isin(
#         relevant_target_charges)]
#     return {
#         c: np.nonzero(relevant_fused_charges == c)[0]
#         for c in relevant_target_charges
#     }

#   left_charges, right_charges, partition = _find_best_partition(charges, flows)

#   unique_target_charges, inds = target_charges.unique(return_index=True)
#   target_charges = target_charges[np.sort(inds)]

#   unique_left = left_charges.unique()
#   unique_right = right_charges.unique()
#   fused = unique_left + unique_right

#   #compute all unique charges that can add up to
#   #target_charges
#   left_inds, right_inds = [], []
#   for target_charge in target_charges:
#     li, ri = np.divmod(np.nonzero(fused == target_charge)[0], len(unique_right))
#     left_inds.append(li)
#     right_inds.append(ri)

#   #now compute the relevant unique left and right charges
#   unique_left_charges = unique_left[np.unique(np.concatenate(left_inds))]
#   unique_right_charges = unique_right[np.unique(np.concatenate(right_inds))]

#   #only keep those charges that are relevant
#   relevant_left_charges = left_charges[left_charges.isin(unique_left_charges)]
#   relevant_right_charges = right_charges[right_charges.isin(
#       unique_right_charges)]

#   unique_right_charges, right_dims = relevant_right_charges.unique(
#       return_counts=True)
#   right_degeneracies = dict(zip(unique_right_charges, right_dims))
#   #generate a degeneracy vector which for each value r in relevant_right_charges
#   #holds the corresponding number of non-zero elements `relevant_right_charges`
#   #that can add up to `target_charges`.
#   degeneracy_vector = np.empty(len(relevant_left_charges), dtype=np.int64)
#   right_indices = {}

#   for n in range(len(unique_left_charges)):
#     left_charge = unique_left_charges[n]
#     total_charge = left_charge + unique_right_charges
#     total_degeneracy = np.sum(right_dims[total_charge.isin(target_charges)])
#     tmp_relevant_right_charges = relevant_right_charges[
#         relevant_right_charges.isin((target_charges + left_charge * (-1)))]

#     for n in range(len(target_charges)):
#       target_charge = target_charges[n]
#       right_indices[(left_charge.get_item(0),
#                      target_charge.get_item(0))] = np.nonzero(
#                          tmp_relevant_right_charges == (
#                              target_charge + left_charge * (-1)))[0]

#     degeneracy_vector[relevant_left_charges == left_charge] = total_degeneracy

#   start_positions = np.cumsum(degeneracy_vector) - degeneracy_vector
#   blocks = {t: [] for t in target_charges}
#   # iterator returns tuple of `int` for ChargeCollection objects
#   # and `int` for Ba seCharge objects (both hashable)
#   for left_charge in unique_left_charges:
#     a = np.expand_dims(start_positions[relevant_left_charges == left_charge], 0)
#     for target_charge in target_charges:
#       ri = right_indices[(left_charge, target_charge)]
#       if len(ri) != 0:
#         b = np.expand_dims(ri, 1)
#         tmp = a + b
#         blocks[target_charge].append(np.reshape(tmp, np.prod(tmp.shape)))
#   out = {}
#   for target_charge in target_charges:
#     out[target_charge] = np.concatenate(blocks[target_charge])
#   return out


def compute_dense_to_sparse_mapping(
    charges: List[Union[BaseCharge, ChargeCollection]],
    flows: List[Union[bool, int]],
    target_charge: Union[BaseCharge, ChargeCollection]) -> List[np.ndarray]:
  """
  Compute the mapping from multi-index positions to the linear positions
  within the sparse data container, given the meta-data of a symmetric tensor.
  This function returns a list of np.ndarray `index_positions`, with
  `len(index_positions)=len(charges)` (equal to the rank of the tensor).
  When stacked into a `(N,r)` np.ndarray `multi_indices`, i.e.
  `
  multi_indices = np.stack(index_positions, axis=1) #np.ndarray of shape (N,r)
  `
  with `r` the rank of the tensor and `N` the number of non-zero elements of
  the symmetric tensor, then the element at position `n` within the linear
  data-array `data` of the tensor have multi-indices given by `multi_indices[n,:],
  i.e. `data[n]` has the multi-index `multi_indices[n,:]`, and the total charges
  can for example be obtained using
  ```
  index_positions = compute_dense_to_sparse_mapping(charges, flows, target_charge=0)
  total_charges = np.zeros(len(index_positions[0]), dtype=np.int16)
  for n in range(len(charges)):
    total_charges += flows[n]*charges[n][index_positions[n]]
  np.testing.assert_allclose(total_charges, 0)
  ```
  Args:
    charges: List of np.ndarray of int, one for each leg of the
      underlying tensor. Each np.ndarray `charges[leg]`
      is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
    target_charge: The total target charge of the blocks to be calculated.
  Returns:
    list of np.ndarray: A list of length `r`, with `r` the rank of the tensor.
      Each element in the list is an N-dimensional np.ndarray of int, 
       with `N` the number of non-zero elements.
  """
  #find the best partition (the one where left and right dimensions are
  #closest
  dims = np.asarray([len(c) for c in charges])
  #note: left_charges and right_charges have been fused from RIGHT to LEFT
  left_charges, right_charges, partition = _find_best_partition(charges, flows)
  nz_indices = find_dense_positions([left_charges], [1], [right_charges], [1],
                                    target_charges=target_charge)

  if len(nz_indices) == 0:
    raise ValueError(
        "`charges` do not add up to a total charge {}".format(target_charge))
  return np.unravel_index(nz_indices, dims)


class BlockSparseTensor:
  """
  Minimal class implementation of block sparsity.
  The class design follows Glen's proposal (Design 0).
  The class currently only supports a single U(1) symmetry
  and only numpy.ndarray.

  Attributes:
    * self.data: A 1d np.ndarray storing the underlying 
      data of the tensor
    * self.charges: A list of `np.ndarray` of shape
      (D,), where D is the bond dimension. Once we go beyond
      a single U(1) symmetry, this has to be updated.

    * self.flows: A list of integers of length `k`.
        `self.flows` determines the flows direction of charges
        on each leg of the tensor. A value of `-1` denotes 
        outflowing charge, a value of `1` denotes inflowing
        charge.

  The tensor data is stored in self.data, a 1d np.ndarray.
  """

  def copy(self):
    return BlockSparseTensor(self.data.copy(), [i.copy() for i in self.indices])

  def __init__(self, data: np.ndarray, indices: List[Index]) -> None:
    """
    Args: 
      data: An np.ndarray of the data. The number of elements in `data`
        has to match the number of non-zero elements defined by `charges` 
        and `flows`
      indices: List of `Index` objecst, one for each leg. 
    """
    for n, i in enumerate(indices):
      if i is None:
        i.name = 'index_{}'.format(n)

    index_names = [
        i.name if i.name else 'index_{}'.format(n)
        for n, i in enumerate(indices)
    ]
    unique, cnts = np.unique(index_names, return_counts=True)
    if np.any(cnts > 1):
      raise ValueError("Index names {} appeared multiple times. "
                       "Please rename indices uniquely.".format(
                           unique[cnts > 1]))

    self.indices = indices
    _check_flows(self.flows)
    num_non_zero_elements = compute_num_nonzero(self.charges, self.flows)

    if num_non_zero_elements != len(data.flat):
      raise ValueError("number of tensor elements {} defined "
                       "by `charges` is different from"
                       " len(data)={}".format(num_non_zero_elements,
                                              len(data.flat)))

    self.data = np.asarray(data.flat)  #do not copy data

  def todense(self) -> np.ndarray:
    """
    Map the sparse tensor to dense storage.
    
    """
    out = np.asarray(np.zeros(self.dense_shape, dtype=self.dtype).flat)

    charges = self.charges
    out[np.nonzero(fuse_charges(charges, self.flows) == charges[0].zero_charge)
        [0]] = self.data
    return np.reshape(out, self.dense_shape)

  @classmethod
  def randn(cls, indices: List[Index],
            dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a random symmetric tensor from random normal distribution.
    Args:
      indices: List of `Index` objecst, one for each leg. 
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    charges = [i.charges for i in indices]
    flows = [i.flow for i in indices]
    num_non_zero_elements = compute_num_nonzero(charges, flows)
    backend = backend_factory.get_backend('numpy')
    data = backend.randn((num_non_zero_elements,), dtype=dtype)
    return cls(data=data, indices=indices)

  @classmethod
  def ones(cls, indices: List[Index],
           dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a symmetric tensor with ones.
    Args:
      indices: List of `Index` objecst, one for each leg. 
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    charges = [i.charges for i in indices]
    flows = [i.flow for i in indices]
    num_non_zero_elements = compute_num_nonzero(charges, flows)
    backend = backend_factory.get_backend('numpy')
    data = backend.ones((num_non_zero_elements,), dtype=dtype)
    return cls(data=data, indices=indices)

  @classmethod
  def zeros(cls, indices: List[Index],
            dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a symmetric tensor with zeros.
    Args:
      indices: List of `Index` objecst, one for each leg. 
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    charges = [i.charges for i in indices]
    flows = [i.flow for i in indices]
    num_non_zero_elements = compute_num_nonzero(charges, flows)
    backend = backend_factory.get_backend('numpy')
    data = backend.zeros((num_non_zero_elements,), dtype=dtype)
    return cls(data=data, indices=indices)

  @classmethod
  def random(cls, indices: List[Index],
             dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a random symmetric tensor from random normal distribution.
    Args:
      indices: List of `Index` objecst, one for each leg. 
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    charges = [i.charges for i in indices]
    flows = [i.flow for i in indices]

    num_non_zero_elements = compute_num_nonzero(charges, flows)
    dtype = dtype if dtype is not None else np.float64

    def init_random():
      if ((np.dtype(dtype) is np.dtype(np.complex128)) or
          (np.dtype(dtype) is np.dtype(np.complex64))):
        return np.random.rand(num_non_zero_elements).astype(
            dtype) - 0.5 + 1j * (
                np.random.rand(num_non_zero_elements).astype(dtype) - 0.5)
      return np.random.randn(num_non_zero_elements).astype(dtype) - 0.5

    return cls(data=init_random(), indices=indices)

  @property
  def index_names(self):
    return [i.name for i in self.indices]

  @property
  def rank(self):
    return len(self.indices)

  @property
  def dense_shape(self) -> Tuple:
    """
    The dense shape of the tensor.
    Returns:
      Tuple: A tuple of `int`.
    """
    return tuple([i.dimension for i in self.indices])

  @property
  def shape(self) -> Tuple:
    """
    The sparse shape of the tensor.
    Returns:
      Tuple: A tuple of `Index` objects.
    """
    return tuple(self.indices)

  @property
  def dtype(self) -> Type[np.number]:
    return self.data.dtype

  @property
  def flows(self):
    return [i.flow for i in self.indices]

  @property
  def charges(self):
    return [i.charges for i in self.indices]

  def transpose(
      self,
      order: Union[List[int], np.ndarray],
  ) -> "BlockSparseTensor":
    """
    Transpose the tensor in place into the new order `order`. 
    Args:
      order: The new order of indices.
    Returns:
      BlockSparseTensor: The transposed tensor.
    """

    if len(order) != self.rank:
      raise ValueError(
          "`len(order)={}` is different form `self.rank={}`".format(
              len(order), self.rank))

    #check for trivial permutation
    if np.all(order == np.arange(len(order))):
      return self
    _, tr_data, tr_partition = _compute_transposition_data(self, order)
    flat_charges, flat_flows, _, flat_order = flatten_meta_data(
        self.indices, order)

    cs, sparse_blocks, _, _, _ = _find_diagonal_sparse_blocks(
        [], [flat_charges[n] for n in flat_order[0:tr_partition]],
        [flat_charges[n] for n in flat_order[tr_partition:]],
        [flat_flows[n] for n in flat_order[0:tr_partition]],
        [flat_flows[n] for n in flat_order[tr_partition:]],
        return_data=False)
    for n in range(len(sparse_blocks)):
      sparse_block = sparse_blocks[n]
      self.data[sparse_block[0]] = self.data[tr_data[cs.get_item(n)][0]]

    return self

  def reset_shape(self) -> None:
    """
    Bring the tensor back into its elementary shape.
    """
    self.indices = self.get_elementary_indices()

  def get_elementary_indices(self) -> List:
    """
    Compute the elementary indices of the array.
    """
    elementary_indices = []
    for i in self.indices:
      elementary_indices.extend(i.get_elementary_indices())

    return elementary_indices

  def reshape(self, shape: Union[Iterable[Index], Iterable[int]]) -> None:
    """
    Reshape `tensor` into `shape` in place.
    `BlockSparseTensor.reshape` works essentially the same as the dense 
    version, with the notable exception that the tensor can only be 
    reshaped into a form compatible with its elementary indices. 
    The elementary indices are the indices at the leaves of the `Index` 
    objects `tensors.indices`.
    For example, while the following reshaping is possible for regular 
    dense numpy tensor,
    ```
    A = np.random.rand(6,6,6)
    np.reshape(A, (2,3,6,6))
    ```
    the same code for BlockSparseTensor
    ```
    q1 = np.random.randint(0,10,6)
    q2 = np.random.randint(0,10,6)
    q3 = np.random.randint(0,10,6)
    i1 = Index(charges=q1,flow=1)
    i2 = Index(charges=q2,flow=-1)
    i3 = Index(charges=q3,flow=1)
    A=BlockSparseTensor.randn(indices=[i1,i2,i3])
    print(A.shape) #prints (6,6,6)
    A.reshape((2,3,6,6)) #raises ValueError
    ```
    raises a `ValueError` since (2,3,6,6)
    is incompatible with the elementary shape (6,6,6) of the tensor.
    
    Args:
      tensor: A symmetric tensor.
      shape: The new shape. Can either be a list of `Index` 
        or a list of `int`.
    Returns:
      BlockSparseTensor: A new tensor reshaped into `shape`
    """
    dense_shape = []
    for s in shape:
      if isinstance(s, Index):
        dense_shape.append(s.dimension)
      else:
        dense_shape.append(s)
    # a few simple checks
    if np.prod(dense_shape) != np.prod(self.dense_shape):
      raise ValueError("A tensor with {} elements cannot be "
                       "reshaped into a tensor with {} elements".format(
                           np.prod(self.shape), np.prod(dense_shape)))

    #keep a copy of the old indices for the case where reshaping fails
    #FIXME: this is pretty hacky!
    index_copy = [i.copy() for i in self.indices]

    def raise_error():
      #if this error is raised then `shape` is incompatible
      #with the elementary indices. We then reset the shape
      #to what is was before the call to `reshape`.
      self.indices = index_copy
      elementary_indices = []
      for i in self.indices:
        elementary_indices.extend(i.get_elementary_indices())
      raise ValueError("The shape {} is incompatible with the "
                       "elementary shape {} of the tensor.".format(
                           dense_shape,
                           tuple([e.dimension for e in elementary_indices])))

    self.reset_shape()  #bring tensor back into its elementary shape
    for n in range(len(dense_shape)):
      if dense_shape[n] > self.dense_shape[n]:
        while dense_shape[n] > self.dense_shape[n]:
          #fuse indices
          i1, i2 = self.indices.pop(n), self.indices.pop(n)
          #note: the resulting flow is set to one since the flow
          #is multiplied into the charges. As a result the tensor
          #will then be invariant in any case.
          self.indices.insert(n, fuse_index_pair(i1, i2))
        if self.dense_shape[n] > dense_shape[n]:
          raise_error()
      elif dense_shape[n] < self.dense_shape[n]:
        raise_error()
    #at this point the first len(dense_shape) indices of the tensor
    #match the `dense_shape`.
    while len(dense_shape) < len(self.indices):
      i2, i1 = self.indices.pop(), self.indices.pop()
      self.indices.append(fuse_index_pair(i1, i2))

  def _get_diagonal_blocks(self, return_data: Optional[bool] = False) -> Dict:
    """
    Obtain the diagonal blocks of a symmetric matrix.
    BlockSparseTensor has to be a matrix.
    This routine avoids explicit fusion of row or column charges.

    Args:
      return_data: If `True`, the returned dictionary maps quantum numbers `q` to 
        an actual `np.ndarray` containing the data of block `q`. 
        If `False`, the returned dict maps quantum numbers `q` to a list 
        `[locations, shape]`, where `locations` is an np.ndarray of type np.int64
        containing the locations of the tensor elements within `self.data`, i.e.
        `self.data[locations]` contains the elements belonging to the tensor with 
        quantum numbers `(q,q). `shape` is the shape of the corresponding array.
    Returns:
      dict: If `return_data=True`: Dictionary mapping charge `q` to an 
        np.ndarray of rank 2 (a matrix).
        If `return_data=False`: Dictionary mapping charge `q` to a
        list `[locations, shape]`, where `locations` is an np.ndarray of type 
        np.int64 containing the locations of the tensor elements within `self.data`

    
    """
    if self.rank != 2:
      raise ValueError(
          "`get_diagonal_blocks` can only be called on a matrix, but found rank={}"
          .format(self.rank))

    row_indices = self.indices[0].get_elementary_indices()
    column_indices = self.indices[1].get_elementary_indices()

    return _find_diagonal_sparse_blocks(
        data=self.data,
        row_charges=[i.charges for i in row_indices],
        column_charges=[i.charges for i in column_indices],
        row_flows=[i.flow for i in row_indices],
        column_flows=[i.flow for i in column_indices],
        return_data=return_data)


def reshape(tensor: BlockSparseTensor,
            shape: Union[Iterable[Index], Iterable[int]]) -> BlockSparseTensor:
  """
  Reshape `tensor` into `shape`.
  `reshape` works essentially the same as the dense version, with the
  notable exception that the tensor can only be reshaped into a form
  compatible with its elementary indices. The elementary indices are 
  the indices at the leaves of the `Index` objects `tensors.indices`.
  For example, while the following reshaping is possible for regular 
  dense numpy tensor,
  ```
  A = np.random.rand(6,6,6)
  np.reshape(A, (2,3,6,6))
  ```
  the same code for BlockSparseTensor
  ```
  q1 = np.random.randint(0,10,6)
  q2 = np.random.randint(0,10,6)
  q3 = np.random.randint(0,10,6)
  i1 = Index(charges=q1,flow=1)
  i2 = Index(charges=q2,flow=-1)
  i3 = Index(charges=q3,flow=1)
  A=BlockSparseTensor.randn(indices=[i1,i2,i3])
  print(nA.shape) #prints (6,6,6)
  reshape(A, (2,3,6,6)) #raises ValueError
  ```
  raises a `ValueError` since (2,3,6,6)
  is incompatible with the elementary shape (6,6,6) of the tensor.

  Args:
    tensopr: A symmetric tensor.
    shape: The new shape. Can either be a list of `Index` 
      or a list of `int`.
  Returns:
    BlockSparseTensor: A new tensor reshaped into `shape`
  """
  result = BlockSparseTensor(
      data=tensor.data.copy(), indices=[i.copy() for i in tensor.indices])
  result.reshape(shape)
  return result


def transpose(
    tensor: BlockSparseTensor,
    order: Union[List[int], np.ndarray],
    permutation: Optional[np.ndarray] = None,
    return_permutation: Optional[bool] = False) -> "BlockSparseTensor":
  """
  Transpose `tensor` into the new order `order`. This routine currently shuffles
  data.
  Args: 
    tensor: The tensor to be transposed.
    order: The new order of indices.
    permutation: An np.ndarray of int for reshuffling the data,
      typically the output of a prior call to `transpose`. Passing `permutation`
      can greatly speed up the transposition.
    return_permutation: If `True`, return the the permutation data.
  Returns:
    if `return_permutation == False`:
      BlockSparseTensor: The transposed tensor.
    if `return_permutation == True`:
      BlockSparseTensor, permutation: The transposed tensor 
      and the permutation data

  """
  if (permutation is not None) and (len(permutation) != len(tensor.data)):
    raise ValueError("len(permutation) != len(tensor.data).")
  result = tensor.copy()
  inds = result.transpose(order, permutation, return_permutation)
  if return_permutation:
    return result, inds
  return result


def tensordot(
    tensor1: BlockSparseTensor,
    tensor2: BlockSparseTensor,
    axes: Sequence[Sequence[int]],
    final_order: Optional[Union[List, np.ndarray]] = None) -> BlockSparseTensor:
  """
  Contract two `BlockSparseTensor`s along `axes`.
  Args:
    tensor1: First tensor.
    tensor2: Second tensor.
    axes: The axes to contract.
    permutation1: Permutation data for `tensor1`.
    permutation2: Permutation data for `tensor2`.
    return_permutation: If `True`, return the the permutation data.
  Returns:
    if `return_permutation == False`:
      BlockSparseTensor: The result of contracting `tensor1` and `tensor2`.
    if `return_permutation == True`:
      BlockSparseTensor, np.ndarrays, np.ndarray:  The result of 
      contracting `tensor1` and `tensor2`, together with their respective 
      permutation data.

  """
  axes1 = axes[0]
  axes2 = axes[1]
  if not np.all(np.unique(axes1) == np.sort(axes1)):
    raise ValueError(
        "Some values in axes[0] = {} appear more than once!".format(axes1))
  if not np.all(np.unique(axes2) == np.sort(axes2)):
    raise ValueError(
        "Some values in axes[1] = {} appear more than once!".format(axes2n))

  if max(axes1) >= len(tensor1.shape):
    raise ValueError(
        "rank of `tensor1` is smaller than `max(axes1) = {}.`".format(
            max(axes1)))

  if max(axes2) >= len(tensor2.shape):
    raise ValueError(
        "rank of `tensor2` is smaller than `max(axes2) = {}`".format(
            max(axes1)))
  elementary_1, elementary_2 = [], []
  for a in axes1:
    elementary_1.extend(tensor1.indices[a].get_elementary_indices())
  for a in axes2:
    elementary_2.extend(tensor2.indices[a].get_elementary_indices())

  if len(elementary_2) != len(elementary_1):
    raise ValueError("axes1 and axes2 have incompatible elementary"
                     " shapes {} and {}".format(elementary_1, elementary_2))
  if not np.all(
      np.array([i.flow for i in elementary_1]) ==
      (-1) * np.array([i.flow for i in elementary_2])):
    raise ValueError("axes1 and axes2 have incompatible elementary"
                     " flows {} and {}".format(
                         np.array([i.flow for i in elementary_1]),
                         np.array([i.flow for i in elementary_2])))

  free_axes1 = sorted(set(np.arange(len(tensor1.shape))) - set(axes1))
  free_axes2 = sorted(set(np.arange(len(tensor2.shape))) - set(axes2))
  if (final_order is not None) and (len(final_order) !=
                                    len(free_axes1) + len(free_axes2)):
    raise ValueError("`final_order = {}` is not a valid order for "
                     "a final tensor of rank {}".format(
                         final_order,
                         len(free_axes1) + len(free_axes2)))

  if (final_order is not None) and not np.all(
      np.sort(final_order) == np.arange(len(final_order))):
    raise ValueError(
        "`final_order = {}` is not a valid permutation of {} ".format(
            final_order, np.arange(len(final_order))))

  new_order1 = free_axes1 + list(axes1)
  new_order2 = list(axes2) + free_axes2
  #t1 = time.time()
  charges1, tr_data_1, tr_partition1 = _compute_transposition_data(
      tensor1, new_order1, len(free_axes1))
  charges2, tr_data_2, tr_partition2 = _compute_transposition_data(
      tensor2, new_order2, len(axes2))
  #print('compute transposition data', time.time() - t1)
  common_charges = charges1.intersect(charges2)

  #get the flattened indices for the output tensor
  left_indices = []
  right_indices = []
  for n in free_axes1:
    left_indices.extend(tensor1.indices[n].get_elementary_indices())
  for n in free_axes2:
    right_indices.extend(tensor2.indices[n].get_elementary_indices())
  indices = left_indices + right_indices
  if final_order is not None:
    indices = [indices[n] for n in final_order]

  for n, i in enumerate(indices):
    i.name = 'index_{}'.format(n) if i.name is None else i.name
  index_names = [i.name for i in indices]
  unique = np.unique(index_names)
  #rename indices if they are not unique
  if len(unique) < len(index_names):
    for n, i in enumerate(indices):
      i.name = 'index_{}'.format(n)

  #initialize the data-vector of the output with zeros;
  #Note that empty is not a viable choice here.
  #ts = []
  #t1 = time.time()
  cs, sparse_blocks, _, _, _ = _find_diagonal_sparse_blocks(
      [], [i.charges for i in left_indices], [i.charges for i in right_indices],
      [i.flow for i in left_indices], [i.flow for i in right_indices],
      return_data=False)
  #print('finding sparse positions', time.time() - t1)
  num_nonzero_elements = np.sum([len(v[0]) for v in sparse_blocks])
  data = np.zeros(
      num_nonzero_elements, dtype=np.result_type(tensor1.dtype, tensor2.dtype))
  for n in range(len(common_charges)):
    c = common_charges.get_item(n)
    permutation1 = tr_data_1[c]
    permutation2 = tr_data_2[c]
    sparse_block = sparse_blocks[np.nonzero(cs == c)[0][0]]
    b1 = np.reshape(tensor1.data[permutation1[0]], permutation1[1])
    b2 = np.reshape(tensor2.data[permutation2[0]], permutation2[1])
    res = np.matmul(b1, b2)
    data[sparse_block[0]] = res.flat
  #print('tensordot', time.time() - t1)
  return BlockSparseTensor(data=data, indices=indices)


def flatten_meta_data(indices, order):
  elementary_indices = {}
  flat_elementary_indices = []
  for n in range(len(indices)):
    elementary_indices[n] = indices[n].get_elementary_indices()
    flat_elementary_indices.extend(elementary_indices[n])
  flat_index_list = np.arange(len(flat_elementary_indices))
  cum_num_legs = np.append(
      0, np.cumsum([len(elementary_indices[n]) for n in range(len(indices))]))

  flat_charges = [i.charges for i in flat_elementary_indices]
  flat_flows = [i.flow for i in flat_elementary_indices]
  flat_dims = [len(c) for c in flat_charges]
  flat_strides = _get_strides(flat_dims)
  flat_order = np.concatenate(
      [flat_index_list[cum_num_legs[n]:cum_num_legs[n + 1]] for n in order])
  return flat_charges, flat_flows, flat_strides, flat_order


def _compute_transposition_data(
    tensor: BlockSparseTensor,
    order: Union[List[int], np.ndarray],
    transposed_partition: Optional[int] = None
) -> Tuple[Union[BaseCharge, ChargeCollection], Dict, int]:
  """
  Args:
    tensor: A symmetric tensor.
    order: The new order of indices.
    permutation: An np.ndarray of int for reshuffling the data,
      typically the output of a prior call to `transpose`. Passing `permutation`
      can greatly speed up the transposition.
    return_permutation: If `True`, return the the permutation data.
  Returns:
    BlockSparseTensor: The transposed tensor.
  """
  if len(order) != tensor.rank:
    raise ValueError(
        "`len(order)={}` is different form `tensor.rank={}`".format(
            len(order), tensor.rank))

  #we use flat meta data because it is
  #more efficient to get the fused charges using
  #the best partition
  flat_charges, flat_flows, flat_strides, flat_order = flatten_meta_data(
      tensor.indices, order)
  #t0 = time.time()
  partition = _find_best_partition(
      flat_charges, flat_flows, return_charges=False)
  # ts = []
  # t1 = time.time()

  # ts.append(t1 - t0)
  # print('in _compute_transposition_data: finding best partition', ts[-1])
  if transposed_partition is None:
    transposed_partition = _find_best_partition(
        [flat_charges[n] for n in flat_order],
        [flat_flows[n] for n in flat_order],
        return_charges=False)
  row_lookup, column_lookup = _compute_sparse_lookups(
      flat_charges[0:partition], flat_flows[0:partition],
      flat_charges[partition:], flat_flows[partition:])
  # t2 = time.time()
  # ts.append(t2 - t1)
  # print('in _compute_transposition_data: computing lookup tables', ts[-1])
  cs, dense_blocks = _find_diagonal_dense_blocks(
      [flat_charges[n] for n in flat_order[0:transposed_partition]],
      [flat_charges[n] for n in flat_order[transposed_partition:]],
      [flat_flows[n] for n in flat_order[0:transposed_partition]],
      [flat_flows[n] for n in flat_order[transposed_partition:]],
      row_strides=flat_strides[flat_order[0:transposed_partition]],
      column_strides=flat_strides[flat_order[transposed_partition:]])
  # t3 = time.time()
  # ts.append(t3 - t2)
  # print('in _compute_transposition_data: finding dense blocks', ts[-1])
  column_dim = np.prod(
      [len(flat_charges[n]) for n in range(partition, len(flat_charges))])
  transposed_positions = {}

  for n in range(len(dense_blocks)):
    b = dense_blocks[n]
    rinds, cinds = np.divmod(b[0], column_dim)
    start_pos = row_lookup[rinds]
    transposed_positions[cs.get_item(n)] = [
        row_lookup[rinds] + column_lookup[cinds], b[1]
    ]
  # t4 = time.time()
  # ts.append(t4 - t3)

  # print('in _compute_transposition_data: computing the new positions', ts[-1])
  return cs, transposed_positions, transposed_partition
