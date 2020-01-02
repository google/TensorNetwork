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
from tensornetwork.network_components import Node, contract, contract_between
from tensornetwork.backends import backend_factory
# pylint: disable=line-too-long
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index, fuse_charge_pair, fuse_degeneracies, fuse_charges, unfuse
import numpy as np
import scipy as sp
import itertools
import time
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable
Tensor = Any


def _check_flows(flows) -> None:
  if (set(flows) != {1}) and (set(flows) != {-1}) and (set(flows) != {-1, 1}):
    raise ValueError(
        "flows = {} contains values different from 1 and -1".format(flows))


def _find_best_partition(charges, flows):
  if len(charges) == 1:
    raise ValueError(
        '_expecting `charges` with a length of at least 2, got `len(charges)={}`'
        .format(len(charges)))
  dims = np.asarray([len(c) for c in charges])
  min_ind = np.argmin([
      np.abs(np.prod(dims[0:n]) - np.prod(dims[n::]))
      for n in range(1, len(charges))
  ])
  fused_left_charges = fuse_charges(charges[0:min_ind + 1],
                                    flows[0:min_ind + 1])
  fused_right_charges = fuse_charges(charges[min_ind + 1::],
                                     flows[min_ind + 1::])

  return fused_left_charges, fused_right_charges, min_ind + 1


def map_to_integer(dims: Union[List, np.ndarray],
                   table: np.ndarray,
                   dtype: Optional[Type[np.number]] = np.int64):
  """
  Map a `table` of integers of shape (N, r) bijectively into 
  an np.ndarray `integers` of length N of unique numbers.
  The mapping is done using
  ```
  `integers[n] = table[n,0] * np.prod(dims[1::]) + table[n,1] * np.prod(dims[2::]) + ... + table[n,r-1] * 1`
  
  Args:
    dims: An iterable of integers.
    table: An array of shape (N,r) of integers.
    dtype: An optional dtype used for the conversion.
      Care should be taken when choosing this to avoid overflow issues.
  Returns:
    np.ndarray: An array of integers.
  """
  converter_table = np.expand_dims(
      np.flip(np.append(1, np.cumprod(np.flip(dims[1::])))), 0)
  tmp = table * converter_table
  integers = np.sum(tmp, axis=1)
  return integers


def compute_fused_charge_degeneracies(charges: List[np.ndarray],
                                      flows: List[Union[bool, int]]) -> Dict:
  """
  For a list of charges, compute all possible fused charges resulting
  from fusing `charges`, together with their respective degeneracyn
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
    dict: Mapping fused charges (int) to degeneracies (int)
  """
  if len(charges) == 1:
    return np.unique(charges[0], return_counts=True)

  # get unique charges and their degeneracies on the first leg.
  # We are fusing from "left" to "right".
  accumulated_charges, accumulated_degeneracies = np.unique(
      charges[0], return_counts=True)
  #multiply the flow into the charges of first leg
  accumulated_charges *= flows[0]
  for n in range(1, len(charges)):
    #list of unique charges and list of their degeneracies
    #on the next unfused leg of the tensor
    leg_charges, leg_degeneracies = np.unique(charges[n], return_counts=True)

    #fuse the unique charges
    #Note: entries in `fused_charges` are not unique anymore.
    #flow1 = 1 because the flow of leg 0 has already been
    #mulitplied above
    fused_charges = fuse_charge_pair(
        q1=accumulated_charges, flow1=1, q2=leg_charges, flow2=flows[n])
    #compute the degeneracies of `fused_charges` charges
    #`fused_degeneracies` is a list of degeneracies such that
    # `fused_degeneracies[n]` is the degeneracy of of
    # charge `c = fused_charges[n]`.
    fused_degeneracies = fuse_degeneracies(accumulated_degeneracies,
                                           leg_degeneracies)
    #compute the new degeneracies resulting from fusing
    #`accumulated_charges` and `leg_charges_2`
    accumulated_charges = np.unique(fused_charges)
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
  if len(np.nonzero(accumulated_charges == 0)[0]) == 0:
    raise ValueError(
        "given leg-charges `charges` and flows `flows` are incompatible "
        "with a symmetric tensor")
  return accumulated_degeneracies[accumulated_charges == 0][0]


def compute_nonzero_block_shapes(charges: List[np.ndarray],
                                 flows: List[Union[bool, int]]) -> Dict:
  """
  Compute the blocks and their respective shapes of a symmetric tensor,
  given its meta-data.
  Args:
    charges: List of np.ndarray, one for each leg. 
      Each np.ndarray `charges[leg]` is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
  Returns:
    dict: Dictionary mapping a tuple of charges to a shape tuple.
      Each element corresponds to a non-zero valued block of the tensor.
  """
  #FIXME: this routine is slow
  _check_flows(flows)
  degeneracies = []
  unique_charges = []
  rank = len(charges)
  #find the unique quantum numbers and their degeneracy on each leg
  for leg in range(rank):
    c, d = np.unique(charges[leg], return_counts=True)
    unique_charges.append(c)
    degeneracies.append(dict(zip(c, d)))

  #find all possible combination of leg charges c0, c1, ...
  #(with one charge per leg 0, 1, ...)
  #such that sum([flows[0] * c0, flows[1] * c1, ...]) = 0
  charge_combinations = list(
      itertools.product(*[
          unique_charges[leg] * flows[leg]
          for leg in range(len(unique_charges))
      ]))
  net_charges = np.array([np.sum(c) for c in charge_combinations])
  zero_idxs = np.nonzero(net_charges == 0)[0]
  charge_shape_dict = {}
  for idx in zero_idxs:
    c = charge_combinations[idx]
    shapes = [degeneracies[leg][flows[leg] * c[leg]] for leg in range(rank)]
    charge_shape_dict[c] = shapes
  return charge_shape_dict


def find_diagonal_sparse_blocks(data: np.ndarray,
                                row_charges: List[Union[List, np.ndarray]],
                                column_charges: List[Union[List, np.ndarray]],
                                row_flows: List[Union[bool, int]],
                                column_flows: List[Union[bool, int]],
                                return_data: Optional[bool] = True) -> Dict:
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
    dict: Dictionary mapping quantum numbers (integers) to either an np.ndarray 
      or a python list of locations and shapes, depending on the value of `return_data`.
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
  unique_column_charges, column_dims = compute_fused_charge_degeneracies(
      column_charges, column_flows)
  #convenience container for storing the degeneracies of each
  #column charge
  column_degeneracies = dict(zip(unique_column_charges, column_dims))

  if len(row_charges) > 1:
    left_row_charges, right_row_charges, _ = _find_best_partition(
        row_charges, row_flows)
    unique_left = np.unique(left_row_charges)
    unique_right = np.unique(right_row_charges)
    unique_row_charges = np.unique(
        fuse_charges(charges=[unique_left, unique_right], flows=[1, 1]))

    #get the charges common to rows and columns (only those matter)
    common_charges = np.intersect1d(
        unique_row_charges, -unique_column_charges, assume_unique=True)

    row_locations = find_sparse_positions(
        left_charges=left_row_charges,
        left_flow=1,
        right_charges=right_row_charges,
        right_flow=1,
        target_charges=common_charges)
  elif len(row_charges) == 1:
    fused_row_charges = fuse_charges(row_charges, row_flows)

    #get the unique row-charges
    unique_row_charges, row_dims = np.unique(
        fused_row_charges, return_counts=True)
    #get the charges common to rows and columns (only those matter)
    common_charges = np.intersect1d(
        unique_row_charges, -unique_column_charges, assume_unique=True)
    relevant_fused_row_charges = fused_row_charges[np.isin(
        fused_row_charges, common_charges)]
    row_locations = {}
    for c in common_charges:
      row_locations[c] = np.nonzero(relevant_fused_row_charges == c)[0]
  else:
    raise ValueError('Found an empty sequence for `row_charges`')
  #some numpy magic to get the index locations of the blocks
  degeneracy_vector = np.empty(
      np.sum([len(v) for v in row_locations.values()]), dtype=np.int64)
  #for each charge `c` in `common_charges` we generate a boolean mask
  #for indexing the positions where `relevant_column_charges` has a value of `c`.
  masks = {}
  for c in common_charges:
    degeneracy_vector[row_locations[c]] = column_degeneracies[-c]

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
  blocks = {}

  for c in common_charges:
    #numpy broadcasting is substantially faster than kron!
    a = np.expand_dims(start_positions[np.sort(row_locations[c])], 1)
    b = np.expand_dims(np.arange(column_degeneracies[-c]), 0)
    inds = np.reshape(a + b, len(row_locations[c]) * column_degeneracies[-c])
    if not return_data:
      blocks[c] = [inds, (len(row_locations[c]), column_degeneracies[-c])]
    else:
      blocks[c] = np.reshape(data[inds],
                             (len(row_locations[c]), column_degeneracies[-c]))
  return blocks


def find_diagonal_sparse_blocks_depreacated_1(
    data: np.ndarray,
    row_charges: List[Union[List, np.ndarray]],
    column_charges: List[Union[List, np.ndarray]],
    row_flows: List[Union[bool, int]],
    column_flows: List[Union[bool, int]],
    return_data: Optional[bool] = True) -> Dict:
  """
  Deprecated

  This version is slow for matrices with shape[0] >> shape[1], but fast otherwise.
  
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
    dict: Dictionary mapping quantum numbers (integers) to either an np.ndarray 
      or a python list of locations and shapes, depending on the value of `return_data`.
  """
  flows = row_flows.copy()
  flows.extend(column_flows)
  _check_flows(flows)
  if len(flows) != (len(row_charges) + len(column_charges)):
    raise ValueError(
        "`len(flows)` is different from `len(row_charges) + len(column_charges)`"
    )

  #since we are using row-major we have to fuse the row charges anyway.
  fused_row_charges = fuse_charges(row_charges, row_flows)
  #get the unique row-charges
  unique_row_charges, row_dims = np.unique(
      fused_row_charges, return_counts=True)

  #get the unique column-charges
  #we only care about their degeneracies, not their order; that's much faster
  #to compute since we don't have to fuse all charges explicitly
  unique_column_charges, column_dims = compute_fused_charge_degeneracies(
      column_charges, column_flows)
  #get the charges common to rows and columns (only those matter)
  common_charges = np.intersect1d(
      unique_row_charges, -unique_column_charges, assume_unique=True)

  #convenience container for storing the degeneracies of each
  #row and column charge
  row_degeneracies = dict(zip(unique_row_charges, row_dims))
  column_degeneracies = dict(zip(unique_column_charges, column_dims))

  # we only care about charges common to row and columns
  mask = np.isin(fused_row_charges, common_charges)
  relevant_row_charges = fused_row_charges[mask]
  #some numpy magic to get the index locations of the blocks
  #we generate a vector of `len(relevant_row_charges) which,
  #for each charge `c` in `relevant_row_charges` holds the
  #column-degeneracy of charge `c`
  degeneracy_vector = np.empty(len(relevant_row_charges), dtype=np.int64)
  #for each charge `c` in `common_charges` we generate a boolean mask
  #for indexing the positions where `relevant_column_charges` has a value of `c`.
  masks = {}
  for c in common_charges:
    mask = relevant_row_charges == c
    masks[c] = mask
    degeneracy_vector[mask] = column_degeneracies[-c]

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
  blocks = {}

  for c in common_charges:
    #numpy broadcasting is substantially faster than kron!
    a = np.expand_dims(start_positions[masks[c]], 1)
    b = np.expand_dims(np.arange(column_degeneracies[-c]), 0)
    if not return_data:
      blocks[c] = [
          np.reshape(a + b, row_degeneracies[c] * column_degeneracies[-c]),
          (row_degeneracies[c], column_degeneracies[-c])
      ]
    else:
      blocks[c] = np.reshape(
          data[np.reshape(a + b,
                          row_degeneracies[c] * column_degeneracies[-c])],
          (row_degeneracies[c], column_degeneracies[-c]))
  return blocks


def find_diagonal_sparse_blocks_deprecated_0(data: np.ndarray,
                                             charges: List[np.ndarray],
                                             flows: List[Union[bool, int]],
                                             return_data: Optional[bool] = True
                                            ) -> Dict:
  """
  Deprecated: this version is about 2 times slower (worst case) than the current used
  implementation
  Given the meta data and underlying data of a symmetric matrix, compute 
  all diagonal blocks and return them in a dict.
  Args: 
    data: An np.ndarray of the data. The number of elements in `data`
      has to match the number of non-zero elements defined by `charges` 
      and `flows`
    charges: List of np.ndarray, one for each leg. 
      Each np.ndarray `charges[leg]` is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
    return_data: If `True`, the return dictionary maps quantum numbers `q` to 
      actual `np.ndarray` with the data. This involves a copy of data.
      If `False`, the returned dict maps quantum numbers of a list 
      [locations, shape], where `locations` is an np.ndarray of type np.int64
      containing the locations of the tensor elements within A.data, i.e.
      `A.data[locations]` contains the elements belonging to the tensor with 
      quantum numbers `(q,q). `shape` is the shape of the corresponding array.

  Returns:
    dict: Dictionary mapping quantum numbers (integers) to either an np.ndarray 
      or a python list of locations and shapes, depending on the value of `return_data`.
  """
  if len(charges) != 2:
    raise ValueError("input has to be a two-dimensional symmetric matrix")
  _check_flows(flows)
  if len(flows) != len(charges):
    raise ValueError("`len(flows)` is different from `len(charges)`")

  #we multiply the flows into the charges
  row_charges = flows[0] * charges[0]  # a list of charges on each row
  column_charges = flows[1] * charges[1]  # a list of charges on each column

  #get the unique charges
  unique_row_charges, row_dims = np.unique(row_charges, return_counts=True)
  unique_column_charges, column_dims = np.unique(
      column_charges, return_counts=True)
  #get the charges common to rows and columns (only those matter)
  common_charges = np.intersect1d(
      unique_row_charges, -unique_column_charges, assume_unique=True)

  #convenience container for storing the degeneracies of each
  #row and column charge
  row_degeneracies = dict(zip(unique_row_charges, row_dims))
  column_degeneracies = dict(zip(unique_column_charges, column_dims))

  # we only care about charges common to row and columns
  mask = np.isin(row_charges, common_charges)
  relevant_row_charges = row_charges[mask]

  #some numpy magic to get the index locations of the blocks
  #we generate a vector of `len(relevant_row_charges) which,
  #for each charge `c` in `relevant_row_charges` holds the
  #column-degeneracy of charge `c`
  degeneracy_vector = np.empty(len(relevant_row_charges), dtype=np.int64)
  #for each charge `c` in `common_charges` we generate a boolean mask
  #for indexing the positions where `relevant_column_charges` has a value of `c`.
  masks = {}
  for c in common_charges:
    mask = relevant_row_charges == c
    masks[c] = mask
    degeneracy_vector[mask] = column_degeneracies[-c]

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
  blocks = {}

  for c in common_charges:
    #numpy broadcasting is substantially faster than kron!
    a = np.expand_dims(stop_positions[masks[c]] - column_degeneracies[-c], 0)
    b = np.expand_dims(np.arange(column_degeneracies[-c]), 1)
    if not return_data:
      blocks[c] = [
          np.reshape(a + b, row_degeneracies[c] * column_degeneracies[-c]),
          (row_degeneracies[c], column_degeneracies[-c])
      ]
    else:
      blocks[c] = np.reshape(
          data[np.reshape(a + b,
                          row_degeneracies[c] * column_degeneracies[-c])],
          (row_degeneracies[c], column_degeneracies[-c]))
  return blocks


def find_diagonal_sparse_blocks_column_major(data: np.ndarray,
                                             charges: List[np.ndarray],
                                             flows: List[Union[bool, int]],
                                             return_data: Optional[bool] = True
                                            ) -> Dict:
  """
  Deprecated

  Given the meta data and underlying data of a symmetric matrix, compute 
  all diagonal blocks and return them in a dict, assuming column-major 
  ordering.
  Args: 
    data: An np.ndarray of the data. The number of elements in `data`
      has to match the number of non-zero elements defined by `charges` 
      and `flows`
    charges: List of np.ndarray, one for each leg. 
      Each np.ndarray `charges[leg]` is of shape `(D[leg],)`.
      The bond dimension `D[leg]` can vary on each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
    return_data: If `True`, the return dictionary maps quantum numbers `q` to 
      actual `np.ndarray` with the data. This involves a copy of data.
      If `False`, the returned dict maps quantum numbers of a list 
      [locations, shape], where `locations` is an np.ndarray of type np.int64
      containing the locations of the tensor elements within A.data, i.e.
      `A.data[locations]` contains the elements belonging to the tensor with 
      quantum numbers `(q,q). `shape` is the shape of the corresponding array.

  Returns:
    dict: Dictionary mapping quantum numbers (integers) to either an np.ndarray 
      or a python list of locations and shapes, depending on the value of `return_data`.
  """
  if len(charges) != 2:
    raise ValueError("input has to be a two-dimensional symmetric matrix")
  _check_flows(flows)
  if len(flows) != len(charges):
    raise ValueError("`len(flows)` is different from `len(charges)`")

  #we multiply the flows into the charges
  row_charges = flows[0] * charges[0]  # a list of charges on each row
  column_charges = flows[1] * charges[1]  # a list of charges on each column

  #get the unique charges
  unique_row_charges, row_dims = np.unique(row_charges, return_counts=True)
  unique_column_charges, column_dims = np.unique(
      column_charges, return_counts=True)
  #get the charges common to rows and columns (only those matter)
  common_charges = np.intersect1d(
      unique_row_charges, -unique_column_charges, assume_unique=True)

  #convenience container for storing the degeneracies of each
  #row and column charge
  row_degeneracies = dict(zip(unique_row_charges, row_dims))
  column_degeneracies = dict(zip(unique_column_charges, column_dims))

  # we only care about charges common to row and columns
  mask = np.isin(column_charges, -common_charges)
  relevant_column_charges = column_charges[mask]

  #some numpy magic to get the index locations of the blocks
  #we generate a vector of `len(relevant_column_charges) which,
  #for each charge `c` in `relevant_column_charges` holds the
  #row-degeneracy of charge `c`
  degeneracy_vector = np.empty(len(relevant_column_charges), dtype=np.int64)
  #for each charge `c` in `common_charges` we generate a boolean mask
  #for indexing the positions where `relevant_column_charges` has a value of `c`.
  masks = {}
  for c in common_charges:
    mask = relevant_column_charges == -c
    masks[c] = mask
    degeneracy_vector[mask] = row_degeneracies[c]

  # the result of the cumulative sum is a vector containing
  # the stop positions of the non-zero values of each column
  # within the data vector.
  # E.g. for `relevant_column_charges` = [0,1,0,0,3],  and
  # row_degeneracies[0] = 10
  # row_degeneracies[1] = 20
  # row_degeneracies[3] = 30
  # we have
  # `stop_positions` = [10, 10+20, 10+20+10, 10+20+10+10, 10+20+10+10+30]
  # The starting positions of consecutive elements (in column-major order) in
  # each column with charge `c=0` within the data vector are then simply obtained using
  # masks[0] = [True, False, True, True, False]
  # and `stop_positions[masks[0]] - row_degeneracies[0]`
  stop_positions = np.cumsum(degeneracy_vector)
  blocks = {}

  for c in common_charges:
    #numpy broadcasting is substantially faster than kron!
    a = np.expand_dims(stop_positions[masks[c]] - row_degeneracies[c], 0)
    b = np.expand_dims(np.arange(row_degeneracies[c]), 1)
    if not return_data:
      blocks[c] = [
          np.reshape(a + b, row_degeneracies[c] * column_degeneracies[-c]),
          (row_degeneracies[c], column_degeneracies[-c])
      ]
    else:
      blocks[c] = np.reshape(
          data[np.reshape(a + b,
                          row_degeneracies[c] * column_degeneracies[-c])],
          (row_degeneracies[c], column_degeneracies[-c]))
  return blocks


def find_dense_positions_deprecated(left_charges: np.ndarray, left_flow: int,
                                    right_charges: np.ndarray, right_flow: int,
                                    target_charge: int) -> Dict:
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
    dict: Mapping tuples of integers to np.ndarray of integers.
  """
  _check_flows([left_flow, right_flow])
  unique_left = np.unique(left_charges)
  unique_right = np.unique(right_charges)
  fused = fuse_charges([unique_left, unique_right], [left_flow, right_flow])
  left_inds, right_inds = unfuse(
      np.nonzero(fused == target_charge)[0], len(unique_left),
      len(unique_right))
  left_c = unique_left[left_inds]
  right_c = unique_right[right_inds]
  len_right_charges = len(right_charges)
  linear_positions = {}
  for left_charge, right_charge in zip(left_c, right_c):
    left_positions = np.nonzero(left_charges == left_charge)[0]
    left_offsets = np.expand_dims(left_positions * len_right_charges, 1)
    right_offsets = np.expand_dims(
        np.nonzero(right_charges == right_charge)[0], 0)
    linear_positions[(left_charge, right_charge)] = np.reshape(
        left_offsets + right_offsets,
        left_offsets.shape[0] * right_offsets.shape[1])
  return np.sort(np.concatenate(list(linear_positions.values())))


def find_dense_positions(left_charges: np.ndarray, left_flow: int,
                         right_charges: np.ndarray, right_flow: int,
                         target_charge: int) -> Dict:
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
    dict: Mapping tuples of integers to np.ndarray of integers.
  """
  _check_flows([left_flow, right_flow])
  unique_left, left_degeneracies = np.unique(left_charges, return_counts=True)
  unique_right, right_degeneracies = np.unique(
      right_charges, return_counts=True)

  common_charges = np.intersect1d(
      unique_left, (target_charge - right_flow * unique_right) * left_flow,
      assume_unique=True)

  right_locations = {}
  for c in common_charges:
    right_locations[(target_charge - left_flow * c) * right_flow] = np.nonzero(
        right_charges == (target_charge - left_flow * c) * right_flow)[0]

  len_right_charges = len(right_charges)
  indices = []
  for n in range(len(left_charges)):
    c = left_charges[n]
    indices.append(n * len_right_charges +
                   right_locations[(target_charge - left_flow * c) *
                                   right_flow])
  return np.concatenate(indices)


def find_sparse_positions(left_charges: np.ndarray, left_flow: int,
                          right_charges: np.ndarray, right_flow: int,
                          target_charges: Union[List[int], np.ndarray]) -> Dict:
  """
  Find the sparse locations of elements (i.e. the index-values within the SPARSE tensor)
  in the vector `fused_charges` (resulting from fusing np.ndarrays 
  `left_charges` and `right_charges`) that have a value of `target_charge`,
  assuming that all elements different from `target_charges` are `0`.
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
  target_charges = np.unique(target_charges)
  unique_left = np.unique(left_charges)
  unique_right = np.unique(right_charges)
  fused = fuse_charges([unique_left, unique_right], [left_flow, right_flow])

  #compute all unique charges that can add up to
  #target_charges
  left_inds, right_inds = [], []
  for target_charge in target_charges:
    li, ri = unfuse(
        np.nonzero(fused == target_charge)[0], len(unique_left),
        len(unique_right))
    left_inds.append(li)
    right_inds.append(ri)

  #now compute the relevant unique left and right charges
  unique_left_charges = unique_left[np.unique(np.concatenate(left_inds))]
  unique_right_charges = unique_right[np.unique(np.concatenate(right_inds))]

  #only keep those charges that are relevant
  relevant_left_charges = left_charges[np.isin(left_charges,
                                               unique_left_charges)]
  relevant_right_charges = right_charges[np.isin(right_charges,
                                                 unique_right_charges)]

  unique_right_charges, right_dims = np.unique(
      relevant_right_charges, return_counts=True)
  right_degeneracies = dict(zip(unique_right_charges, right_dims))
  #generate a degeneracy vector which for each value r in relevant_right_charges
  #holds the corresponding number of non-zero elements `relevant_right_charges`
  #that can add up to `target_charges`.
  degeneracy_vector = np.empty(len(relevant_left_charges), dtype=np.int64)
  right_indices = {}
  for left_charge in unique_left_charges:
    total_degeneracy = np.sum(right_dims[np.isin(
        left_flow * left_charge + right_flow * unique_right_charges,
        target_charges)])
    tmp_relevant_right_charges = relevant_right_charges[np.isin(
        relevant_right_charges,
        (target_charges - left_flow * left_charge) * right_flow)]

    for target_charge in target_charges:
      right_indices[(left_charge, target_charge)] = np.nonzero(
          tmp_relevant_right_charges == (target_charge -
                                         left_flow * left_charge) *
          right_flow)[0]

    degeneracy_vector[relevant_left_charges == left_charge] = total_degeneracy

  stop_positions = np.cumsum(degeneracy_vector)
  start_positions = stop_positions - degeneracy_vector
  blocks = {t: [] for t in target_charges}
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


def compute_dense_to_sparse_mapping_deprecated(charges: List[np.ndarray],
                                               flows: List[Union[bool, int]],
                                               target_charge: int) -> int:
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
    np.ndarray: An (N, r) np.ndarray of dtype np.int16,
      with `N` the number of non-zero elements, and `r`
      the rank of the tensor.
  """
  #find the best partition (the one where left and right dimensions are
  #closest
  dims = np.asarray([len(c) for c in charges])
  t1 = time.time()
  fused_charges = fuse_charges(charges, flows)
  nz_indices = np.nonzero(fused_charges == target_charge)[0]
  if len(nz_indices) == 0:
    raise ValueError(
        "`charges` do not add up to a total charge {}".format(target_charge))

  index_locations = []
  for n in reversed(range(len(charges))):
    t1 = time.time()
    nz_indices, right_indices = unfuse(nz_indices, np.prod(dims[0:n]), dims[n])
    index_locations.insert(0, right_indices)
    print(time.time() - t1)
  return index_locations


def compute_dense_to_sparse_mapping_2(charges: List[np.ndarray],
                                      flows: List[Union[bool, int]],
                                      target_charge: int) -> int:
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
    np.ndarray: An (N, r) np.ndarray of dtype np.int16,
      with `N` the number of non-zero elements, and `r`
      the rank of the tensor.
  """
  #find the best partition (the one where left and right dimensions are
  #closest
  dims = np.asarray([len(c) for c in charges])

  #note: left_charges and right_charges have been fused from RIGHT to LEFT
  left_charges, right_charges, partition = _find_best_partition(charges, flows)
  t1 = time.time()
  nz_indices = find_dense_positions(
      left_charges, 1, right_charges, 1, target_charge=target_charge)
  print(time.time() - t1)
  if len(nz_indices) == 0:
    raise ValueError(
        "`charges` do not add up to a total charge {}".format(target_charge))
  t1 = time.time()
  nz_left_indices, nz_right_indices = unfuse(nz_indices, len(left_charges),
                                             len(right_charges))
  print(time.time() - t1)
  index_locations = []
  #first unfuse left charges
  for n in range(partition):
    t1 = time.time()
    indices, nz_left_indices = unfuse(nz_left_indices, dims[n],
                                      np.prod(dims[n + 1:partition]))
    index_locations.append(indices)
    print(time.time() - t1)
  for n in range(partition, len(dims)):
    t1 = time.time()
    indices, nz_right_indices = unfuse(nz_right_indices, dims[n],
                                       np.prod(dims[n + 1::]))
    index_locations.append(indices)
    print(time.time() - t1)

  return index_locations


def compute_dense_to_sparse_mapping(charges: List[np.ndarray],
                                    flows: List[Union[bool, int]],
                                    target_charge: int) -> int:
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

  def __init__(self, data: np.ndarray, indices: List[Index]) -> None:
    """
    Args: 
      data: An np.ndarray of the data. The number of elements in `data`
        has to match the number of non-zero elements defined by `charges` 
        and `flows`
      indices: List of `Index` objecst, one for each leg. 
    """
    self.indices = indices
    _check_flows(self.flows)
    num_non_zero_elements = compute_num_nonzero(self.charges, self.flows)

    if num_non_zero_elements != len(data.flat):
      raise ValueError("number of tensor elements defined "
                       "by `charges` is different from"
                       " len(data)={}".format(len(data.flat)))

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

  def transpose(self,
                order: Union[List[int], np.ndarray],
                transposed_linear_positions: Optional[np.ndarray] = None
               ) -> "BlockSparseTensor":
    """
    Transpose the tensor into the new order `order`. This routine currently shuffles
    data.
    Args: 
      order: The new order of indices.
      transposed_linear_positions: An np.ndarray of int for reshuffling the data,
        typically the output of a prior call to `transpose`. Passing `transposed_linear_positions`
        can greatly speed up the transposition.
    Returns:
      BlockSparseTensor: The transposed tensor.
    """
    #FIXME: this implementation uses scipy.sparse.csr_matrix to generate the
    #lookup-table from dense to sparse indices. According to some quick
    #testing, the final lookup is currently the bottleneck.
    #FIXME: transpose currently shuffles data. This can in principle be postponed
    #until `tensordot` or `find_diagonal_sparsenn_blocks`, at the cost of
    #maintaining two lookup tables for sparse-to-dense positions and dense-to-sparse
    #positions
    if len(order) != self.rank:
      raise ValueError(
          "`len(order)={}` is different form `self.rank={}`".format(
              len(order), self.rank))
    #transpose is the only function using self.dense_to_sparse_table
    #so we can initialize it here. This will change if we are implementing
    #lazy shuffling of data. In this case, `find_diagonal_sparse_blocks`
    #also needs

    #we use elementary indices here because it is
    #more efficient to get the fused charges using
    #the best partition
    if transposed_linear_positions is None:
      elementary_indices = {}
      flat_elementary_indices = []

      for n in range(self.rank):
        elementary_indices[n] = self.indices[n].get_elementary_indices()
        flat_elementary_indices.extend(elementary_indices[n])
      flat_index_list = np.arange(len(flat_elementary_indices))
      cum_num_legs = np.append(
          0, np.cumsum([len(elementary_indices[n]) for n in range(self.rank)]))
      flat_order = np.concatenate(
          [flat_index_list[cum_num_legs[n]:cum_num_legs[n + 1]] for n in order])

      flat_charges = [i.charges for i in flat_elementary_indices]
      flat_flows = [i.flow for i in flat_elementary_indices]
      flat_dims = [len(c) for c in flat_charges]
      flat_strides = np.flip(np.append(1, np.cumprod(np.flip(flat_dims[1::]))))
      if not hasattr(self, 'dense_to_sparse_table'):
        #find the best partition into left and right charges
        left_charges, right_charges, _ = _find_best_partition(
            flat_charges, flat_flows)
        #find the index-positions of the elements in the fusion
        #of `left_charges` and `right_charges` that have `0`
        #total charge (those are the only non-zero elements).
        linear_positions = find_dense_positions(
            left_charges, 1, right_charges, 1, target_charge=0)

        self.dense_to_sparse_table = sp.sparse.csr_matrix(
            (np.arange(len(self.data)),
             (linear_positions, np.zeros(len(self.data), dtype=np.int64))))

      flat_tr_charges = [flat_charges[n] for n in flat_order]
      flat_tr_flows = [flat_flows[n] for n in flat_order]
      flat_tr_strides = [flat_strides[n] for n in flat_order]
      flat_tr_dims = [flat_dims[n] for n in flat_order]

      tr_left_charges, tr_right_charges, _ = _find_best_partition(
          flat_tr_charges, flat_tr_flows)
      #FIXME: this should be done without fully fusing  the strides
      tr_dense_linear_positions = fuse_charges([
          np.arange(flat_tr_dims[n]) * flat_tr_strides[n]
          for n in range(len(flat_tr_dims))
      ],
                                               flows=[1] * len(flat_tr_dims))
      tr_linear_positions = find_dense_positions(tr_left_charges, 1,
                                                 tr_right_charges, 1, 0)

      inds = np.squeeze(self.dense_to_sparse_table[
          tr_dense_linear_positions[tr_linear_positions], 0].toarray())
    else:
      inds = transposed_linear_positions
    self.data = self.data[inds]
    return inds

  def transpose_intersect1d(self, order: Union[List[int], np.ndarray]
                           ) -> "BlockSparseTensor":
    """
    Transpose the tensor into the new order `order`
    Args: pp
      order: The new order of indices.
    Returns:
      BlockSparseTensor: The transposed tensor.
    """
    #FIXME: this implementation uses scipy.sparse.csr_matrix to generate the
    #lookup-table from dense to sparse indices. According to some quick
    #testing, the final lookup is currently the bottleneck.
    #FIXME: transpose currently shuffles data. This can in principle be postponed
    #until `tensordot` or `find_diagonal_sparse_blocks`
    if len(order) != self.rank:
      raise ValueError(len(order), self.rank)
    charges = self.charges  #call only once in case some of the indices are merged indices
    dims = [len(c) for c in charges]

    strides = np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))
    #find the best partition into left and right charges
    left_charges, right_charges, _ = _find_best_partition(charges, self.flows)
    #find the index-positions of the elements in the fusion
    #of `left_charges` and `right_charges` that have `0`
    #total charge (those are the only non-zero elements).
    linear_positions = find_dense_positions(
        left_charges, 1, right_charges, 1, target_charge=0)

    tr_charges = [charges[n] for n in order]
    tr_flows = [self.flows[n] for n in order]
    tr_strides = [strides[n] for n in order]
    tr_dims = [dims[n] for n in order]
    tr_left_charges, tr_right_charges, _ = _find_best_partition(
        tr_charges, tr_flows)

    tr_dense_linear_positions = fuse_charges(
        [np.arange(tr_dims[n]) * tr_strides[n] for n in range(len(tr_dims))],
        flows=[1] * len(tr_dims))
    tr_linear_positions = find_dense_positions(tr_left_charges, 1,
                                               tr_right_charges, 1, 0)
    new_linear_positions = tr_dense_linear_positions[tr_linear_positions]
    _, _, inds = np.intersect1d(
        linear_positions,
        new_linear_positions,
        return_indices=True,
        assume_unique=True)
    self.data = self.data[inds]

  # def transpose_lookup(self, order: Union[List[int], np.ndarray]
  #                     ) -> "BlockSparseTensor":
  #   """
  #   Deprecated

  #   Transpose the tensor into the new order `order`. Uses a simple cython std::map
  #   for the lookup
  #   Args:
  #     order: The new order of indices.
  #   Returns:
  #     BlockSparseTensor: The transposed tensor.
  #   """
  #   if len(order) != self.rank:
  #     raise ValueError(
  #         "`len(order)={}` is different form `self.rank={}`".format(
  #             len(order), self.rank))
  #   charges = self.charges  #call only once in case some of the indices are merged indices
  #   dims = [len(c) for c in charges]

  #   strides = np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))
  #   #find the best partition into left and right charges
  #   left_charges, right_charges, _ = _find_best_partition(charges, self.flows)
  #   #find the index-positions of the elements in the fusion
  #   #of `left_charges` and `right_charges` that have `0`
  #   #total charge (those are the only non-zero elements).
  #   linear_positions = find_dense_positions(
  #       left_charges, 1, right_charges, 1, target_charge=0)

  #   tr_charges = [charges[n] for n in order]
  #   tr_flows = [self.flows[n] for n in order]
  #   tr_strides = [strides[n] for n in order]
  #   tr_dims = [dims[n] for n in order]
  #   tr_left_charges, tr_right_charges, _ = _find_best_partition(
  #       tr_charges, tr_flows)
  #   #FIXME: this should be done without fully fusing  the strides
  #   tr_dense_linear_positions = fuse_charges(
  #       [np.arange(tr_dims[n]) * tr_strides[n] for n in range(len(tr_dims))],
  #       flows=[1] * len(tr_dims))
  #   tr_linear_positions = find_dense_positions(tr_left_charges, 1,
  #                                              tr_right_charges, 1, 0)
  #   inds = lookup(linear_positions,
  #                 tr_dense_linear_positions[tr_linear_positions])
  #   self.data = self.data[inds]

  def transpose_searchsorted(self, order: Union[List[int], np.ndarray]
                            ) -> "BlockSparseTensor":
    """
    Deprecated:
    
    Transpose the tensor into the new order `order`. Uses `np.searchsorted`
    for the lookup.
    Args: 
      order: The new order of indices.
    Returns:
      BlockSparseTensor: The transposed tensor.
    """
    if len(order) != self.rank:
      raise ValueError(
          "`len(order)={}` is different form `self.rank={}`".format(
              len(order), self.rank))
    charges = self.charges  #call only once in case some of the indices are merged indices
    dims = [len(c) for c in charges]

    strides = np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))
    #find the best partition into left and right charges
    left_charges, right_charges, _ = _find_best_partition(charges, self.flows)
    #find the index-positions of the elements in the fusion
    #of `left_charges` and `right_charges` that have `0`
    #total charge (those are the only non-zero elements).
    linear_positions = find_dense_positions(
        left_charges, 1, right_charges, 1, target_charge=0)

    tr_charges = [charges[n] for n in order]
    tr_flows = [self.flows[n] for n in order]
    tr_strides = [strides[n] for n in order]
    tr_dims = [dims[n] for n in order]
    tr_left_charges, tr_right_charges, _ = _find_best_partition(
        tr_charges, tr_flows)
    #FIXME: this should be done without fully fusing  the strides
    tr_dense_linear_positions = fuse_charges(
        [np.arange(tr_dims[n]) * tr_strides[n] for n in range(len(tr_dims))],
        flows=[1] * len(tr_dims))
    tr_linear_positions = find_dense_positions(tr_left_charges, 1,
                                               tr_right_charges, 1, 0)

    inds = np.searchsorted(linear_positions,
                           tr_dense_linear_positions[tr_linear_positions])
    self.data = self.data[inds]

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

  def get_diagonal_blocks(self, return_data: Optional[bool] = True) -> Dict:
    """
    Obtain the diagonal blocks of symmetric matrix.
    BlockSparseTensor has to be a matrix.
    For matrices with shape[0] << shape[1], this routine avoids explicit fusion
    of column charges.

    Args:
      return_data: If `True`, the return dictionary maps quantum numbers `q` to 
        actual `np.ndarray` with the data. This involves a copy of data.
        If `False`, the returned dict maps quantum numbers of a list 
        [locations, shape], where `locations` is an np.ndarray of type np.int64
        containing the locations of the tensor elements within A.data, i.e.
        `A.data[locations]` contains the elements belonging to the tensor with 
        quantum numbers `(q,q). `shape` is the shape of the corresponding array.
    Returns:
      dict: Dictionary mapping charge to np.ndarray of rank 2 (a matrix)
    
    """
    if self.rank != 2:
      raise ValueError(
          "`get_diagonal_blocks` can only be called on a matrix, but found rank={}"
          .format(self.rank))

    row_indices = self.indices[0].get_elementary_indices()
    column_indices = self.indices[1].get_elementary_indices()

    return find_diagonal_sparse_blocks(
        data=self.data,
        row_charges=[i.charges for i in row_indices],
        column_charges=[i.charges for i in column_indices],
        row_flows=[i.flow for i in row_indices],
        column_flows=[i.flow for i in column_indices],
        return_data=return_data)

  def get_diagonal_blocks_deprecated_1(self, return_data: Optional[bool] = True
                                      ) -> Dict:
    """
    Obtain the diagonal blocks of symmetric matrix.
    BlockSparseTensor has to be a matrix.
    For matrices with shape[0] << shape[1], this routine avoids explicit fusion
    of column charges.

    Args:
      return_data: If `True`, the return dictionary maps quantum numbers `q` to 
        actual `np.ndarray` with the data. This involves a copy of data.
        If `False`, the returned dict maps quantum numbers of a list 
        [locations, shape], where `locations` is an np.ndarray of type np.int64
        containing the locations of the tensor elements within A.data, i.e.
        `A.data[locations]` contains the elements belonging to the tensor with 
        quantum numbers `(q,q). `shape` is the shape of the corresponding array.
    Returns:
      dict: Dictionary mapping charge to np.ndarray of rank 2 (a matrix)
    
    """
    if self.rank != 2:
      raise ValueError(
          "`get_diagonal_blocks` can only be called on a matrix, but found rank={}"
          .format(self.rank))

    row_indices = self.indices[0].get_elementary_indices()
    column_indices = self.indices[1].get_elementary_indices()

    return find_diagonal_sparse_blocks_deprecated_1(
        data=self.data,
        row_charges=[i.charges for i in row_indices],
        column_charges=[i.charges for i in column_indices],
        row_flows=[i.flow for i in row_indices],
        column_flows=[i.flow for i in column_indices],
        return_data=return_data)

  def get_diagonal_blocks_deprecated_0(self, return_data: Optional[bool] = True
                                      ) -> Dict:
    """
    Deprecated

    Obtain the diagonal blocks of symmetric matrix.
    BlockSparseTensor has to be a matrix.
    Args:
      return_data: If `True`, the return dictionary maps quantum numbers `q` to 
        actual `np.ndarray` with the data. This involves a copy of data.
        If `False`, the returned dict maps quantum numbers of a list 
        [locations, shape], where `locations` is an np.ndarray of type np.int64
        containing the locations of the tensor elements within A.data, i.e.
        `A.data[locations]` contains the elements belonging to the tensor with 
        quantum numbers `(q,q). `shape` is the shape of the corresponding array.
    Returns:
      dict: Dictionary mapping charge to np.ndarray of rank 2 (a matrix)
    
    """
    if self.rank != 2:
      raise ValueError(
          "`get_diagonal_blocks` can only be called on a matrix, but found rank={}"
          .format(self.rank))

    return find_diagonal_sparse_blocks_deprecated_0(
        data=self.data,
        charges=self.charges,
        flows=self.flows,
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
  print(A.shape) #prints (6,6,6)
  reshape(A, (2,3,6,6)) #raises ValueError
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
  result = BlockSparseTensor(
      data=tensor.data.copy(), indices=[i.copy() for i in tensor.indices])
  result.reshape(shape)
  return result
