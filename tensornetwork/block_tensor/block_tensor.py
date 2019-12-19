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
# pylint: disable=line-too-long
from tensornetwork.network_components import Node, contract, contract_between
from tensornetwork.backends import backend_factory
# pylint: disable=line-too-long
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index, fuse_charge_pair, fuse_degeneracies, fuse_charges
import numpy as np
import itertools
import time
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable
Tensor = Any


def check_flows(flows) -> None:
  if (set(flows) != {1}) and (set(flows) != {-1}) and (set(flows) != {-1, 1}):
    raise ValueError(
        "flows = {} contains values different from 1 and -1".format(flows))


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
  check_flows(flows)
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


def retrieve_non_zero_diagonal_blocks(
    data: np.ndarray,
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
      containing the locations of the tensor elements within A.data, i.e.
      `A.data[locations]` contains the elements belonging to the tensor with 
      quantum numbers `(q,q). `shape` is the shape of the corresponding array.

  Returns:
    dict: Dictionary mapping quantum numbers (integers) to either an np.ndarray 
      or a python list of locations and shapes, depending on the value of `return_data`.
  """
  flows = row_flows.copy()
  flows.extend(column_flows)
  check_flows(flows)
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
  blocks = {}

  for c in common_charges:
    #numpy broadcasting is substantially faster than kron!
    a = np.expand_dims(stop_positions[masks[c]] - column_degeneracies[-c], 0)
    b = np.expand_dims(np.arange(column_degeneracies[-c]), 1)
    if not return_data:
      blocks[c] = [a + b, (row_degeneracies[c], column_degeneracies[-c])]
    else:
      blocks[c] = np.reshape(data[a + b],
                             (row_degeneracies[c], column_degeneracies[-c]))
  return blocks


def retrieve_non_zero_diagonal_blocks_old_version(
    data: np.ndarray,
    charges: List[np.ndarray],
    flows: List[Union[bool, int]],
    return_data: Optional[bool] = True) -> Dict:
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
  check_flows(flows)
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
      blocks[c] = [a + b, (row_degeneracies[c], column_degeneracies[-c])]
    else:
      blocks[c] = np.reshape(data[a + b],
                             (row_degeneracies[c], column_degeneracies[-c]))
  return blocks


def retrieve_non_zero_diagonal_blocks_column_major(
    data: np.ndarray,
    charges: List[np.ndarray],
    flows: List[Union[bool, int]],
    return_data: Optional[bool] = True) -> Dict:
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
  check_flows(flows)
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
      blocks[c] = [a + b, (row_degeneracies[c], column_degeneracies[-c])]
    else:
      blocks[c] = np.reshape(data[a + b],
                             (row_degeneracies[c], column_degeneracies[-c]))
  return blocks


def retrieve_non_zero_diagonal_blocks_deprecated(
    data: np.ndarray,
    charges: List[np.ndarray],
    flows: List[Union[bool, int]],
    return_data: Optional[bool] = False) -> Dict:
  """
  Deprecated

  Given the meta data and underlying data of a symmetric matrix, compute 
  all diagonal blocks and return them in a dict.
  This is a deprecated version which in general performs worse than the
  current main implementation.

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
  check_flows(flows)
  if len(flows) != len(charges):
    raise ValueError("`len(flows)` is different from `len(charges)`")

  #we multiply the flows into the charges
  row_charges = flows[0] * charges[0]  # a list of charges on each row
  column_charges = flows[1] * charges[1]  # a list of charges on each column

  # we only care about charges common to rows and columns
  common_charges = np.unique(np.intersect1d(row_charges, -column_charges))
  row_charges = row_charges[np.isin(row_charges, common_charges)]
  column_charges = column_charges[np.isin(column_charges, -common_charges)]

  #get the unique charges
  unique_row_charges, row_locations, row_dims = np.unique(
      row_charges, return_inverse=True, return_counts=True)
  unique_column_charges, column_locations, column_dims = np.unique(
      column_charges, return_inverse=True, return_counts=True)
  #convenience container for storing the degeneracies of each
  #row and column charge
  row_degeneracies = dict(zip(unique_row_charges, row_dims))
  column_degeneracies = dict(zip(unique_column_charges, column_dims))

  #some numpy magic to get the index locations of the blocks
  #we generate a vector of `len(relevant_column_charges) which,
  #for each charge `c` in `relevant_column_charges` holds the
  #row-degeneracy of charge `c`

  degeneracy_vector = column_dims[row_locations]
  stop_positions = np.cumsum(degeneracy_vector)
  blocks = {}
  for c in common_charges:
    #numpy broadcasting is substantially faster than kron!
    a = np.expand_dims(
        stop_positions[row_charges == c] - column_degeneracies[-c], 0)
    b = np.expand_dims(np.arange(column_degeneracies[-c]), 1)
    if not return_data:
      blocks[c] = [a + b, (row_degeneracies[c], column_degeneracies[-c])]
    else:
      blocks[c] = np.reshape(data[a + b],
                             (row_degeneracies[c], column_degeneracies[-c]))
  return blocks


def compute_mapping_table(charges: List[np.ndarray],
                          flows: List[Union[bool, int]]) -> int:
  """
  Compute a mapping table mapping the linear positions of the non-zero 
  elements to their multi-index label.
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
    np.ndarray: An (N, r) np.ndarray of dtype np.int16, 
      with `N` the number of non-zero elements, and `r` 
      the rank of the tensor.
  """
  # we are using row-major encoding, meaning that the last index
  # is moving quickest when iterating through the linear data
  # transposing is done taking, for each value of the indices i_0 to i_N-2
  # the junk i_N-1 that gives non-zero

  #for example
  raise NotImplementedError()


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
    check_flows(self.flows)
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

  def transpose(self, order):
    """
    Transpose the tensor into the new order `order`
    """
    raise NotImplementedError('transpose is not implemented!!')

  def reset_shape(self) -> None:
    """
    Bring the tensor back into its elementary shape.
    """
    elementary_indices = []
    for i in self.indices:
      elementary_indices.extend(i.get_elementary_indices())

    self.indices = elementary_indices

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

    return retrieve_non_zero_diagonal_blocks(
        data=self.data,
        row_charges=[i.charges for i in row_indices],
        column_charges=[i.charges for i in column_indices],
        row_flows=[i.flow for i in row_indices],
        column_flows=[i.flow for i in column_indices],
        return_data=return_data)

  def get_diagonal_blocks_old_version(
      self, return_data: Optional[bool] = True) -> Dict:
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

    return retrieve_non_zero_diagonal_blocks_old_version(
        data=self.data,
        charges=self.charges,
        flows=self.flows,
        return_data=return_data)

  def get_diagonal_blocks_deprecated(
      self, return_data: Optional[bool] = True) -> Dict:
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
    return retrieve_non_zero_diagonal_blocks_deprecated(
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
