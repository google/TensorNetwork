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
# pylint: disable=line-too-long
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index
from tensornetwork.block_tensor.charge import fuse_degeneracies, fuse_charges, fuse_degeneracies, BaseCharge, ChargeCollection
import numpy as np
import scipy as sp
import itertools
import time
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable, Sequence
Tensor = Any


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


def _find_best_partition(
    charges: List[Union[BaseCharge, ChargeCollection]],
    flows: List[int]) -> Tuple[Union[BaseCharge, ChargeCollection],
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
  fused_left_charges = fuse_charges(charges[0:min_ind + 1],
                                    flows[0:min_ind + 1])
  fused_right_charges = fuse_charges(charges[min_ind + 1::],
                                     flows[min_ind + 1::])

  return fused_left_charges, fused_right_charges, min_ind + 1


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
  columns given by fusing `column_charges`. Note that `column_charges`
  are never explicitly fused (`row_charges` are).
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
    List[Union[BaseCharge, ChargeCollection]]: A list of unique charges, one per block.
    List[np.ndarray]: A list containing the blocks.

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
  #convenience container for storing the degeneracies of each
  #column charge
  #column_degeneracies = dict(zip(unique_column_charges, column_dims))
  column_degeneracies = dict(zip(unique_column_charges * (-1), column_dims))
  if len(row_charges) > 1:
    left_row_charges, right_row_charges, _ = _find_best_partition(
        row_charges, row_flows)
    unique_left = left_row_charges.unique()
    unique_right = right_row_charges.unique()
    unique_row_charges = (unique_left + unique_right).unique()

    #get the charges common to rows and columns (only those matter)
    concatenated = unique_row_charges.concatenate(unique_column_charges * (-1))
    tmp_unique, counts = concatenated.unique(return_counts=True)
    common_charges = tmp_unique[
        counts == 2]  #common_charges is a BaseCharge or ChargeCollection

    row_locations = find_sparse_positions(
        left_charges=left_row_charges,
        left_flow=1,
        right_charges=right_row_charges,
        right_flow=1,
        target_charges=common_charges)

  elif len(row_charges) == 1:
    fused_row_charges = fuse_charges(row_charges, row_flows)

    #get the unique row-charges
    unique_row_charges, row_dims = fused_row_charges.unique(return_counts=True)
    #get the charges common to rows and columns (only those matter)
    #get the charges common to rows and columns (only those matter)
    concatenated = unique_row_charges.concatenate(unique_column_charges * (-1))
    tmp_unique, counts = concatenated.unique(return_counts=True)
    common_charges = tmp_unique[
        counts == 2]  #common_charges is a BaseCharge or ChargeCollection

    relevant_fused_row_charges = fused_row_charges[fused_row_charges.isin(
        common_charges)]
    row_locations = {}
    for c in common_charges:
      #c = common_charges.get_item(n)
      row_locations[c] = np.nonzero(relevant_fused_row_charges == c)[0]
  else:
    raise ValueError('Found an empty sequence for `row_charges`')

  degeneracy_vector = np.empty(
      np.sum([len(v) for v in row_locations.values()]), dtype=np.int64)
  #for each charge `c` in `common_charges` we generate a boolean mask
  #for indexing the positions where `relevant_column_charges` has a value of `c`.
  masks = {}
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
  stop_positions = np.cumsum(degeneracy_vector)
  start_positions = stop_positions - degeneracy_vector
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


def find_dense_positions(
    left_charges: Union[BaseCharge, ChargeCollection], left_flow: int,
    right_charges: Union[BaseCharge, ChargeCollection], right_flow: int,
    target_charge: Union[BaseCharge, ChargeCollection]) -> np.ndarray:
  """
  Find the dense locations of elements (i.e. the index-values within the DENSE tensor)
  in the vector `fused_charges` (resulting from fusing np.ndarrays 
  `left_charges` and `right_charges`) that have a value of `target_charge`.
  For example, given 
  ```
  left_charges = [-2,0,1,0,0]
  right_charges = [-1,0,2,1]
  target_charge = 0
  fused_charges = fuse_charges([left_charges, right_charges],[1,1]) 
  print(fused_charges) # [-3,-2,0,-1,-1,0,2,1,0,1,3,2,-1,0,2,1,-1,0,2,1]
  ```
  we want to find the all different blocks 
  that fuse to `target_charge=0`, i.e. where `fused_charges==0`, 
  together with their corresponding index-values of the data in the dense array.
  `find_dense_blocks` returns a dict mapping tuples `(left_charge, right_charge)`
  to an array of integers.
  For the above example, we get:
  * for `left_charge` = -2 and `right_charge` = 2 we get an array [2]. Thus, `fused_charges[2]`
    was obtained from fusing -2 and 2.
  * for `left_charge` = 0 and `right_charge` = 0 we get an array [5, 13, 17]. Thus, 
    `fused_charges[5,13,17]` were obtained from fusing 0 and 0.
  * for `left_charge` = 1 and `right_charge` = -1 we get an array [8]. Thus, `fused_charges[8]`
    was obtained from fusing 1 and -1.
  Args:
    left_charges: An np.ndarray of integer charges.
    left_flow: The flow direction of the left charges.
    right_charges: An np.ndarray of integer charges.
    right_flow: The flow direction of the right charges.
    target_charge: The target charge.
  Returns:
    np.ndarray: The indices of the elements fusing to `target_charge`.
  """
  _check_flows([left_flow, right_flow])
  unique_left, left_degeneracies = left_charges.unique(return_counts=True)
  unique_right, right_degeneracies = right_charges.unique(return_counts=True)

  tmp_charges = (target_charge + (unique_right * right_flow * (-1))) * left_flow
  concatenated = unique_left.concatenate(tmp_charges)
  tmp_unique, counts = concatenated.unique(return_counts=True)
  common_charges = tmp_unique[
      counts == 2]  #common_charges is a BaseCharge or ChargeCollection
  right_locations = {}
  for n in range(len(common_charges)):
    c = common_charges[n]

    right_charge = (target_charge + (c * left_flow * (-1))) * right_flow
    right_locations[right_charge.get_item(0)] = np.nonzero(
        right_charges == right_charge)[0]

  len_right_charges = len(right_charges)
  indices = []
  for n in range(len(left_charges)):
    c = left_charges[n]
    right_charge = (target_charge + (c * left_flow * (-1))) * right_flow

    if c not in common_charges:
      continue
    indices.append(n * len_right_charges +
                   right_locations[right_charge.get_item(0)])

  return np.concatenate(indices)


def find_sparse_positions(
    left_charges: Union[BaseCharge, ChargeCollection], left_flow: int,
    right_charges: Union[BaseCharge, ChargeCollection], right_flow: int,
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

  _check_flows([left_flow, right_flow])
  target_charges = target_charges.unique()
  unique_left = left_charges.unique()
  unique_right = right_charges.unique()
  fused = unique_left * left_flow + unique_right * right_flow

  #compute all unique charges that can add up to
  #target_charges
  left_inds, right_inds = [], []
  for target_charge in target_charges:
    li, ri = np.divmod(np.nonzero(fused == target_charge)[0], len(unique_right))
    left_inds.append(li)
    right_inds.append(ri)

  #now compute the relevant unique left and right charges
  unique_left_charges = unique_left[np.unique(np.concatenate(left_inds))]
  unique_right_charges = unique_right[np.unique(np.concatenate(right_inds))]

  #only keep those charges that are relevant
  relevant_left_charges = left_charges[left_charges.isin(unique_left_charges)]
  relevant_right_charges = right_charges[right_charges.isin(
      unique_right_charges)]

  unique_right_charges, right_dims = relevant_right_charges.unique(
      return_counts=True)
  right_degeneracies = dict(zip(unique_right_charges, right_dims))
  #generate a degeneracy vector which for each value r in relevant_right_charges
  #holds the corresponding number of non-zero elements `relevant_right_charges`
  #that can add up to `target_charges`.
  degeneracy_vector = np.empty(len(relevant_left_charges), dtype=np.int64)
  right_indices = {}

  for n in range(len(unique_left_charges)):
    left_charge = unique_left_charges[n]
    total_charge = left_charge * left_flow + unique_right_charges * right_flow
    total_degeneracy = np.sum(right_dims[total_charge.isin(target_charges)])
    tmp_relevant_right_charges = relevant_right_charges[
        relevant_right_charges.isin(
            (target_charges + left_charge * ((-1) * left_flow)) * right_flow)]

    for n in range(len(target_charges)):
      target_charge = target_charges[n]
      right_indices[(
          left_charge.get_item(0), target_charge.get_item(0))] = np.nonzero(
              tmp_relevant_right_charges == (target_charge + left_charge * (
                  (-1) * left_flow)) * right_flow)[0]

    degeneracy_vector[relevant_left_charges == left_charge] = total_degeneracy

  stop_positions = np.cumsum(degeneracy_vector)
  start_positions = stop_positions - degeneracy_vector
  blocks = {t: [] for t in target_charges}
  # iterator returns tuple of `int` for ChargeCollection objects
  # and `int` for Ba seCharge objects (both hashable)
  for left_charge in unique_left_charges:
    a = np.expand_dims(start_positions[relevant_left_charges == left_charge], 0)
    for target_charge in target_charges:
      ri = right_indices[(left_charge, target_charge)]
      if len(ri) != 0:
        b = np.expand_dims(ri, 1)
        tmp = a + b
        blocks[target_charge].append(np.reshape(tmp, np.prod(tmp.shape)))
  out = {}
  for target_charge in target_charges:
    out[target_charge] = np.concatenate(blocks[target_charge])
  return out


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
  nz_indices = find_dense_positions(
      left_charges, 1, right_charges, 1, target_charge=target_charge)

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
    dtype = dtype if dtype is not None else self.np.float64

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
      permutation: Optional[np.ndarray] = None,
      return_permutation: Optional[bool] = False) -> "BlockSparseTensor":
    """
    Transpose the tensor into the new order `order`. This routine currently shuffles
    data.
    Args:
      order: The new order of indices.
      permutation: An np.ndarray of int for reshuffling the data,
        typically the output of a prior call to `transpose`. Passing `permutation`
        can greatly speed up the transposition.
      return_permutation: If `True`, return the the permutation data.
    Returns:
      BlockSparseTensor: The transposed tensor.
    """
    if (permutation is not None) and (len(permutation) != len(self.data)):
      raise ValueError("len(permutation) != len(tensor.data).")

    if len(order) != self.rank:
      raise ValueError(
          "`len(order)={}` is different form `self.rank={}`".format(
              len(order), self.rank))

    #check for trivial permutation
    if np.all(order == np.arange(len(order))):
      if return_permutation:
        return np.arange(len(self.data))
      return

    #we use elementary indices here because it is
    #more efficient to get the fused charges using
    #the best partition
    if permutation is None:
      elementary_indices = {}
      flat_elementary_indices = []
      for n in range(len(self.indices)):
        elementary_indices[n] = self.indices[n].get_elementary_indices()
        flat_elementary_indices.extend(elementary_indices[n])
      flat_index_list = np.arange(len(flat_elementary_indices))
      cum_num_legs = np.append(
          0,
          np.cumsum(
              [len(elementary_indices[n]) for n in range(len(self.indices))]))

      flat_charges = [i.charges for i in flat_elementary_indices]
      flat_flows = [i.flow for i in flat_elementary_indices]
      flat_dims = [len(c) for c in flat_charges]
      flat_strides = np.flip(np.append(1, np.cumprod(np.flip(flat_dims[1::]))))
      flat_order = np.concatenate(
          [flat_index_list[cum_num_legs[n]:cum_num_legs[n + 1]] for n in order])
      #find the best partition into left and right charges
      left_charges, right_charges, _ = _find_best_partition(
          flat_charges, flat_flows)
      #find the index-positions of the elements in the fusion
      #of `left_charges` and `right_charges` that have `0`
      #total charge (those are the only non-zero elements).
      linear_positions = find_dense_positions(
          left_charges,
          1,
          right_charges,
          1,
          target_charge=flat_charges[0].zero_charge)
      flat_tr_charges = [flat_charges[n] for n in flat_order]
      flat_tr_flows = [flat_flows[n] for n in flat_order]
      flat_tr_strides = [flat_strides[n] for n in flat_order]
      flat_tr_dims = [flat_dims[n] for n in flat_order]

      tr_left_charges, tr_right_charges, partition = _find_best_partition(
          flat_tr_charges, flat_tr_flows)
      tr_linear_positions = find_dense_positions(
          tr_left_charges, 1, tr_right_charges, 1, tr_left_charges.zero_charge)
      stride_arrays = [
          np.arange(flat_tr_dims[n]) * flat_tr_strides[n]
          for n in range(len(flat_tr_dims))
      ]

      dense_permutation = _find_values_in_fused(
          tr_linear_positions, fuse_ndarrays(stride_arrays[0:partition]),
          fuse_ndarrays(stride_arrays[partition::]))
      assert np.all(np.sort(dense_permutation) == linear_positions)
      permutation = np.searchsorted(linear_positions, dense_permutation)

    self.indices = [self.indices[n] for n in order]
    self.data = self.data[permutation]
    if return_permutation:
      return permutation

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

  def _get_diagonal_blocks(self, return_data: Optional[bool] = True) -> Dict:
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


def tensordot(tensor1: BlockSparseTensor,
              tensor2: BlockSparseTensor,
              axes: Sequence[Sequence[int]],
              permutation1: Optional[np.ndarray] = None,
              permutation2: Optional[np.ndarray] = None,
              return_permutation: Optional[bool] = False):
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

  if max(axes2) >= len(tensor2.shape):
    raise ValueError(
        "rank of `tensor2` is smaller than `max(axes2) = {}`".format(
            max(axes1)))
  free_axes1 = sorted(set(np.arange(len(tensor1.shape))) - set(axes1))
  free_axes2 = sorted(set(np.arange(len(tensor2.shape))) - set(axes2))
  new_order1 = free_axes1 + list(axes1)
  new_order2 = list(axes2) + free_axes2

  tr1 = transpose(
      tensor=tensor1,
      order=new_order1,
      permutation=permutation1,
      return_permutation=return_permutation)
  if return_permutation:
    permutation1 = tr1[1]
    tr1 = tr1[1]

  trshape1 = tr1.dense_shape
  Dl1 = np.prod([trshape1[n] for n in range(len(free_axes1))])
  Dr1 = np.prod([trshape1[n] for n in range(len(free_axes1), len(trshape1))])

  tmp1 = reshape(tr1, (Dl1, Dr1))

  tr2 = transpose(
      tensor=tensor2,
      order=new_order2,
      permutation=permutation2,
      return_permutation=return_permutation)
  if return_permutation:
    permutation2 = tr2[1]
    tr2 = tr2[1]
  trshape2 = tr2.dense_shape
  Dl2 = np.prod([trshape2[n] for n in range(len(axes2))])
  Dr2 = np.prod([trshape2[n] for n in range(len(axes2), len(trshape2))])

  tmp2 = reshape(tr2, (Dl2, Dr2))

  #avoid data-copying here by setting `return_data=False`
  column_charges1, data1, start_positions, row_locations, _ = tmp1._get_diagonal_blocks(
      return_data=False)
  row_charges2, data2, _, _, column_degeneracies = tmp2._get_diagonal_blocks(
      return_data=False)

  #get common charges between rows and columns
  tmp_charges, cnts = column_charges1.concatenate(row_charges2).unique(
      return_counts=True)
  common_charges = tmp_charges[cnts == 2]

  #get the flattened indices for the output tensor
  indices = []
  indices.extend(tmp1.indices[0].get_elementary_indices())
  indices.extend(tmp2.indices[1].get_elementary_indices())
  index_names = [i.name for i in indices]
  unique = np.unique(index_names)
  #rename indices if they are not unique
  if len(unique) < len(index_names):
    for n, i in enumerate(indices):
      i.name = 'index_{}'.format(n)

  #initialize the data-vector of the output with zeros
  num_nonzero_elements = compute_num_nonzero([i.charges for i in indices],
                                             [i.flow for i in indices])
  data = np.zeros(
      num_nonzero_elements, dtype=np.result_type(tensor1.dtype, tensor2.dtype))

  for c in common_charges:
    rlocs = row_locations[c]
    cdegs = column_degeneracies[c]
    a = np.expand_dims(start_positions[rlocs], 1)
    b = np.expand_dims(np.arange(cdegs), 0)
    new_locations = np.reshape(a + b, len(rlocs) * cdegs)
    i1 = np.nonzero(column_charges1 == c)[0][0]
    i2 = np.nonzero(row_charges2 == c)[0][0]
    try:
      #place the result of the block-matrix multiplication
      #into the new data-vector
      data[new_locations] = np.matmul(
          np.reshape(tensor1.data[data1[i1][0]], data1[i1][1]),
          np.reshape(tensor2.data[data2[i2][0]], data2[i2][1])).flat
    except ValueError:
      raise ValueError("for quantum number {}, shapes {} and {} "
                       "of left and right blocks have "
                       "incompatible shapes".format(c, data1[i1].shape,
                                                    data2[i2].shape))

  out = BlockSparseTensor(data=data, indices=indices)
  resulting_shape = [trshape1[n] for n in range(len(free_axes1))
                    ] + [trshape2[n] for n in range(len(axes2), len(trshape2))]
  out.reshape(resulting_shape)
  if return_permutation:
    return out, permutation1, permutation2
  return out
