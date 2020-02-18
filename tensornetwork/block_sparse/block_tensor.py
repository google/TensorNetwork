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
from tensornetwork.block_sparse.index import Index, fuse_index_pair
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import fuse_charges, fuse_degeneracies, BaseCharge, fuse_ndarray_charges, intersect, charge_equal
import scipy as sp
import copy
import time
# pylint: disable=line-too-long
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable, Sequence, Text
Tensor = Any


def flatten(list_of_list: List[List]) -> List:
  """
  Flatten a list of lists into a single list.
  Args:
    list_of_lists: A list of lists.
  Returns:
    list: The flattened input.
  """
  res = []
  for l in list_of_list:
    res.extend(l)
  return np.array(res)


def get_flat_meta_data(indices):
  """
  Return charges and flows of flattened `indices`.
  Args:
    indices: A list of `Index` objects.
  Returns:
    List[BaseCharge]: The flattened charges.
    List[bool]: The flattened flows.
  """
  charges = []
  flows = []
  for i in indices:
    flows.extend(i.flat_flows)
    charges.extend(i.flat_charges)
  return charges, flows


def fuse_stride_arrays(dims: np.ndarray, strides: np.ndarray) -> np.ndarray:
  """
  Compute linear positions of tensor elements 
  of a tensor with dimensions `dimsd` according to `strides`.
  Args: 
    dims: An np.ndarray of (original) tensor dimensions.
    strides: An np.ndarray of (possibly permituted) strides.
  Returns:
    np.ndarray: Linear positions of tensor elements according to `strides`.
  """
  return fuse_ndarrays([
      np.arange(0, strides[n] * dims[n], strides[n], dtype=np.uint32)
      for n in range(len(dims))
  ])


def compute_sparse_lookup(
    charges: List[BaseCharge], flows: List[bool],
    target_charges: BaseCharge) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Compute lookup table for looking up how dense index positions map 
  to sparse index positions.
  Args:
    charges: List of `BaseCharge` objects.
    flows: A list of `bool`; the flow directions.
    target_charges: A `BaseCharge`; the target charges for which 
      the fusion of `charges` is non-zero.
  Returns:
    lookup: An np.ndarray of positive numbers between `0` and
      `len(unique_charges)`. The position of values `n` in `lookup` are positions
       with charge values `unique_values[n]`.
    unique_charges: The unique charges of fusion of `charges`
    lable_to_unique: The  integer labels of the unique charges.
  """

  fused_charges = fuse_charges(charges, flows)
  unique_charges, inverse = fused_charges.unique(return_inverse=True)
  _, label_to_unique, _ = unique_charges.intersect(
      target_charges, return_indices=True)

  tmp = np.full(len(unique_charges), fill_value=-1, dtype=np.int16)
  tmp[label_to_unique] = label_to_unique
  lookup = tmp[inverse]
  lookup = lookup[lookup >= 0]

  return lookup, unique_charges, label_to_unique


def _get_strides(dims):
  """
  compute strides of `dims`.
  """
  return np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))


def fuse_ndarrays(arrays: List[Union[List, np.ndarray]]) -> np.ndarray:
  """
  Fuse all `arrays` by simple kronecker addition.
  Arrays are fused from "right to left", 
  Args:
    arrays: A list of arrays to be fused.
  Returns:
    np.ndarray: The result of fusing `arrays`.
  """
  if len(arrays) == 1:
    return arrays[0]
  fused_arrays = arrays[0]
  for n in range(1, len(arrays)):
    fused_arrays = np.ravel(np.add.outer(fused_arrays, arrays[n]))
  return fused_arrays


def _find_best_partition(dims: List[int]) -> int:
  """
  Find the most-levelled partition of `dims`.
  A levelled partitioning is a partitioning such that
  np.prod(dim[:partition]) and np.prod(dim[partition:])
  are as close as possible.
  Args:
    dims: A list or np.ndarray of integers.
  Returns:
    int: The best partitioning.
  """
  if len(dims) == 1:
    raise ValueError(
        'expecting `dims` with a length of at least 2, got `len(dims ) =1`')
  diffs = [
      np.abs(np.prod(dims[0:n]) - np.prod(dims[n::]))
      for n in range(1, len(dims))
  ]
  min_inds = np.nonzero(diffs == np.min(diffs))[0]
  if len(min_inds) > 1:
    right_dims = [np.prod(dims[min_ind + 1:]) for min_ind in min_inds]
    min_ind = min_inds[np.argmax(right_dims)]
  else:
    min_ind = min_inds[0]
  return min_ind + 1


def compute_fused_charge_degeneracies(
    charges: List[BaseCharge],
    flows: List[bool]) -> Tuple[BaseCharge, np.ndarray]:
  """
  For a list of charges, computes all possible fused charges resulting
  from fusing `charges` and their respective degeneracies
  Args:
    charges: List of `BaseCharge`, one for each leg of a 
      tensor. 
    flows: A list of bool, one for each leg of a tensor.
      with values `False` or `True` denoting inflowing and 
      outflowing charge direction, respectively.
  Returns:
    BaseCharge: The unique fused charges.
    np.ndarray: The degeneracies of each unqiue fused charge.
  """
  if len(charges) == 1:
    return (charges[0] * flows[0]).unique(return_counts=True)

  # get unique charges and their degeneracies on the first leg.
  # We are fusing from "left" to "right".
  accumulated_charges, accumulated_degeneracies = (
      charges[0] * flows[0]).unique(return_counts=True)
  for n in range(1, len(charges)):
    leg_charges, leg_degeneracies = charges[n].unique(return_counts=True)
    # print(accumulated_charges.unique_charges)
    # print(accumulated_charges.charge_labels)
    fused_charges = accumulated_charges + leg_charges * flows[n]
    fused_degeneracies = fuse_degeneracies(accumulated_degeneracies,
                                           leg_degeneracies)
    accumulated_charges = fused_charges.unique()
    accumulated_degeneracies = np.empty(
        len(accumulated_charges), dtype=np.uint32)

    accumulated_degeneracies = np.array([
        np.sum(fused_degeneracies[fused_charges.charge_labels ==
                                  accumulated_charges.charge_labels[m]])
        for m in range(len(accumulated_charges))
    ])

  return accumulated_charges, accumulated_degeneracies


def compute_unique_fused_charges(charges: List[BaseCharge],
                                 flows: List[Union[bool, int]]) -> BaseCharge:
  """
  For a list of charges, compute all possible fused charges resulting
  from fusing `charges`.
  Args:
    charges: List of `BaseCharge`, one for each leg of a 
      tensor. 
    flows: A list of bool, one for each leg of a tensor.
      with values `False` or `True` denoting inflowing and 
      outflowing charge direction, respectively.
  Returns:
    BaseCharge: The unique fused charges.

  """
  if len(charges) == 1:
    return (charges[0] * flows[0]).unique()

  accumulated_charges = (charges[0] * flows[0]).unique()
  for n in range(1, len(charges)):
    leg_charges = charges[n].unique()
    fused_charges = accumulated_charges + leg_charges * flows[n]
    accumulated_charges = fused_charges.unique()
  return accumulated_charges


def compute_num_nonzero(charges: List[BaseCharge], flows: List[bool]) -> int:
  """
  Compute the number of non-zero elements, given the meta-data of 
  a symmetric tensor.
  Args:
    charges: List of `BaseCharge`, one for each leg of a 
      tensor. 
    flows: A list of bool, one for each leg of a tensor.
      with values `False` or `True` denoting inflowing and 
      outflowing charge direction, respectively.
  Returns:
    int: The number of non-zero elements.
  """
  if np.any([len(c) == 0 for c in charges]):
    return 0
  #pylint: disable=line-too-long
  accumulated_charges, accumulated_degeneracies = compute_fused_charge_degeneracies(
      charges, flows)
  res = accumulated_charges == accumulated_charges.identity_charges
  nz_inds = np.nonzero(res)[0]

  if len(nz_inds) > 0:
    return np.squeeze(accumulated_degeneracies[nz_inds][0])
  return 0


def reduce_charges(charges: List[BaseCharge],
                   flows: List[bool],
                   target_charges: np.ndarray,
                   return_locations: Optional[bool] = False,
                   strides: Optional[np.ndarray] = None) -> Any:
  """
  Add quantum numbers arising from combining two or more charges into a
  single index, keeping only the quantum numbers that appear in 'target_charges'.
  Equilvalent to using "combine_charges" followed by "reduce", but is
  generally much more efficient.
  Args:
    charges: List of `BaseCharge`, one for each leg of a 
      tensor. 
    flows: A list of bool, one for each leg of a tensor.
      with values `False` or `True` denoting inflowing and 
      outflowing charge direction, respectively.
    target_charges: n-by-D array of charges which should be kept,
      with `n` the number of symmetries.
    return_locations: If `True` return the location of the kept
      values of the fused charges
    strides: Index strides with which to compute the
      retured locations of the kept elements. Defaults to trivial strides (based on
      row major order).
  Returns:
    BaseCharge: the fused index after reduction.
    np.ndarray: Locations of the fused BaseCharge charges that were kept.
  """

  tensor_dims = [len(c) for c in charges]

  if len(charges) == 1:
    # reduce single index
    if strides is None:
      strides = np.array([1], dtype=np.uint32)
    return charges[0].dual(flows[0]).reduce(
        target_charges, return_locations=return_locations, strides=strides[0])

  # find size-balanced partition of charges
  partition = _find_best_partition(tensor_dims)

  # compute quantum numbers for each partition
  left_ind = fuse_charges(charges[:partition], flows[:partition])
  right_ind = fuse_charges(charges[partition:], flows[partition:])

  # compute combined qnums
  comb_qnums = fuse_ndarray_charges(left_ind.unique_charges,
                                    right_ind.unique_charges,
                                    charges[0].charge_types)
  [unique_comb_qnums, comb_labels] = np.unique(
      comb_qnums, return_inverse=True, axis=1)
  num_unique = unique_comb_qnums.shape[1]

  # intersect combined qnums and target_charges
  reduced_qnums, label_to_unique, _ = intersect(
      unique_comb_qnums, target_charges, axis=1, return_indices=True)
  map_to_kept = -np.ones(num_unique, dtype=np.int16)
  map_to_kept[label_to_unique] = np.arange(len(label_to_unique))
  new_comb_labels = map_to_kept[comb_labels].reshape(
      [left_ind.num_unique, right_ind.num_unique])
  if return_locations:
    if strides is not None:
      # computed locations based on non-trivial strides
      row_pos = fuse_stride_arrays(tensor_dims[:partition], strides[:partition])
      col_pos = fuse_stride_arrays(tensor_dims[partition:], strides[partition:])

      # reduce combined qnums to include only those in target_charges
      reduced_rows = [0] * left_ind.num_unique
      row_locs = [0] * left_ind.num_unique
      for n in range(left_ind.num_unique):
        temp_label = new_comb_labels[n, right_ind.charge_labels]
        temp_keep = temp_label >= 0
        reduced_rows[n] = temp_label[temp_keep]
        row_locs[n] = col_pos[temp_keep]

      reduced_labels = np.concatenate(
          [reduced_rows[n] for n in left_ind.charge_labels])
      reduced_locs = np.concatenate([
          row_pos[n] + row_locs[left_ind.charge_labels[n]]
          for n in range(left_ind.dim)
      ])
      obj = charges[0].__new__(type(charges[0]))
      obj.__init__(reduced_qnums, reduced_labels, charges[0].charge_types)
      return obj, reduced_locs
    # reduce combined qnums to include only those in target_charges
    reduced_rows = [0] * left_ind.num_unique
    row_locs = [0] * left_ind.num_unique
    for n in range(left_ind.num_unique):
      temp_label = new_comb_labels[n, right_ind.charge_labels]
      temp_keep = temp_label >= 0
      reduced_rows[n] = temp_label[temp_keep]
      row_locs[n] = np.where(temp_keep)[0]

    reduced_labels = np.concatenate(
        [reduced_rows[n] for n in left_ind.charge_labels])
    reduced_locs = np.concatenate([
        n * right_ind.dim + row_locs[left_ind.charge_labels[n]]
        for n in range(left_ind.dim)
    ])
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(reduced_qnums, reduced_labels, charges[0].charge_types)

    return obj, reduced_locs
  # reduce combined qnums to include only those in target_charges
  reduced_rows = [0] * left_ind.num_unique
  for n in range(left_ind.num_unique):
    temp_label = new_comb_labels[n, right_ind.charge_labels]
    reduced_rows[n] = temp_label[temp_label >= 0]

  reduced_labels = np.concatenate(
      [reduced_rows[n] for n in left_ind.charge_labels])
  obj = charges[0].__new__(type(charges[0]))
  obj.__init__(reduced_qnums, reduced_labels, charges[0].charge_types)

  return obj


def _find_diagonal_sparse_blocks(
    charges: List[BaseCharge], flows: np.ndarray,
    partition: int) -> Tuple[List, BaseCharge, np.ndarray]:
  """
  Find the location of all non-trivial symmetry blocks from the data vector of
  of BlockSparseTensor (when viewed as a matrix across some prescribed index 
  bi-partition).
  Args:
    charges: List of `BaseCharge`, one for each leg of a tensor. 
    flows: A list of bool, one for each leg of a tensor.
      with values `False` or `True` denoting inflowing and 
      outflowing charge direction, respectively.
    partition: location of tensor partition (i.e. such that the 
      tensor is viewed as a matrix between `charges[:partition]` and 
      the remaining charges).
  Returns:
    block_maps (List[np.ndarray]): list of integer arrays, which each 
      containing the location of a symmetry block in the data vector.
    block_qnums (BaseCharge): The charges of the corresponding blocks.n
      block, with 'n' the number of symmetries and 'm' the number of blocks.
    block_dims (np.ndarray): 2-by-m array of matrix dimensions of each block.
  """
  num_inds = len(charges)
  num_syms = charges[0].num_symmetries

  if partition in (0, num_inds):
    # special cases (matrix of trivial height or width)
    num_nonzero = compute_num_nonzero(charges, flows)
    block_maps = [np.arange(0, num_nonzero, dtype=np.uint64).ravel()]
    block_qnums = np.zeros([num_syms, 1], dtype=np.int16)
    block_dims = np.array([[1], [num_nonzero]])

    if partition == len(flows):
      block_dims = np.flipud(block_dims)

    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(block_qnums, np.arange(0, dtype=np.int16),
                 charges[0].charge_types)

    return block_maps, obj, block_dims

  unique_row_qnums, row_degen = compute_fused_charge_degeneracies(
      charges[:partition], flows[:partition])
  unique_col_qnums, col_degen = compute_fused_charge_degeneracies(
      charges[partition:], np.logical_not(flows[partition:]))

  block_qnums, row_to_block, col_to_block = intersect(
      unique_row_qnums.unique_charges,
      unique_col_qnums.unique_charges,
      axis=1,
      return_indices=True)
  num_blocks = block_qnums.shape[1]
  if num_blocks == 0:
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(
        np.zeros(0, dtype=np.int16), np.arange(0, dtype=np.int16),
        charges[0].charge_types)

    return [], obj, []

  # calculate number of non-zero elements in each row of the matrix
  row_ind = reduce_charges(charges[:partition], flows[:partition], block_qnums)
  row_num_nz = col_degen[col_to_block[row_ind.charge_labels]]
  cumulate_num_nz = np.insert(np.cumsum(row_num_nz[0:-1]), 0,
                              0).astype(np.uint32)

  # calculate mappings for the position in datavector of each block
  if num_blocks < 15:
    # faster method for small number of blocks
    row_locs = np.concatenate(
        [(row_ind.charge_labels == n) for n in range(num_blocks)]).reshape(
            num_blocks, row_ind.dim)
  else:
    # faster method for large number of blocks
    row_locs = np.zeros([num_blocks, row_ind.dim], dtype=bool)
    row_locs[row_ind
             .charge_labels, np.arange(row_ind.dim)] = np.ones(
                 row_ind.dim, dtype=bool)

  # block_dims = np.array([row_degen[row_to_block],col_degen[col_to_block]], dtype=np.uint32)
  block_dims = np.array(
      [[row_degen[row_to_block[n]], col_degen[col_to_block[n]]]
       for n in range(num_blocks)],
      dtype=np.uint32).T
  #pylint: disable=unsubscriptable-object
  block_maps = [
      np.ravel(cumulate_num_nz[row_locs[n, :]][:, None] +
               np.arange(block_dims[1, n])[None, :]) for n in range(num_blocks)
  ]
  obj = charges[0].__new__(type(charges[0]))
  obj.__init__(block_qnums, np.arange(block_qnums.shape[1], dtype=np.int16),
               charges[0].charge_types)

  return block_maps, obj, block_dims


def _find_transposed_diagonal_sparse_blocks(
    charges: List[BaseCharge],
    flows: np.ndarray,
    tr_partition: int,
    order: Optional[np.ndarray] = None) -> Tuple[List, BaseCharge, np.ndarray]:
  """
  Find the diagonal blocks of a transposed tensor with 
  meta-data `charges` and `flows`. `charges` and `flows` 
  are the charges and flows of the untransposed tensor, 
  `order` is the final transposition, and `tr_partition`
  is the partition of the transposed tensor according to 
  which the diagonal blocks shoul be found.
  Args:
    charges: List of `BaseCharge`, one for each leg of a tensor. 
    flows: A list of bool, one for each leg of a tensor.
      with values `False` or `True` denoting inflowing and 
      outflowing charge direction, respectively.
    tr_partition: Location of the transposed tensor partition (i.e. such that the 
      tensor is viewed as a matrix between `charges[order[:partition]]` and 
      the remaining charges).
    order: Order with which to permute the tensor axes. 
  Returns:
    block_maps (List[np.ndarray]): list of integer arrays, which each 
      containing the location of a symmetry block in the data vector.
    block_qnums (BaseCharge): The charges of the corresponding blocks.n
      block, with 'n' the number of symmetries and 'm' the number of blocks.
    block_dims (np.ndarray): 2-by-m array of matrix dimensions of each block.
  """
  flows = np.asarray(flows)
  if np.array_equal(order, None) or (np.array_equal(
      np.array(order), np.arange(len(charges)))):
    # no transpose order
    return _find_diagonal_sparse_blocks(charges, flows, tr_partition)

  # general case: non-trivial transposition is required
  num_inds = len(charges)
  tensor_dims = np.array([charges[n].dim for n in range(num_inds)], dtype=int)
  strides = np.append(np.flip(np.cumprod(np.flip(tensor_dims[1:]))), 1)

  # compute qnums of row/cols in original tensor
  orig_partition = _find_best_partition(tensor_dims)
  orig_width = np.prod(tensor_dims[orig_partition:])

  orig_unique_row_qnums = compute_unique_fused_charges(charges[:orig_partition],
                                                       flows[:orig_partition])
  orig_unique_col_qnums, orig_col_degen = compute_fused_charge_degeneracies(
      charges[orig_partition:], np.logical_not(flows[orig_partition:]))

  orig_block_qnums, row_map, col_map = intersect(
      orig_unique_row_qnums.unique_charges,
      orig_unique_col_qnums.unique_charges,
      axis=1,
      return_indices=True)
  orig_num_blocks = orig_block_qnums.shape[1]
  if orig_num_blocks == 0:
    # special case: trivial number of non-zero elements
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(
        np.empty(0, dtype=np.int16), np.arange(0, dtype=np.int16),
        charges[0].charge_types)

    return [], obj, np.array([], dtype=np.uint32)

  orig_row_ind = fuse_charges(charges[:orig_partition], flows[:orig_partition])
  orig_col_ind = fuse_charges(charges[orig_partition:],
                              np.logical_not(flows[orig_partition:]))

  inv_row_map = -np.ones(
      orig_unique_row_qnums.unique_charges.shape[1], dtype=np.int16)
  inv_row_map[row_map] = np.arange(len(row_map), dtype=np.int16)

  all_degens = np.append(orig_col_degen[col_map],
                         0)[inv_row_map[orig_row_ind.charge_labels]]
  all_cumul_degens = np.cumsum(np.insert(all_degens[:-1], 0,
                                         0)).astype(np.uint32)
  dense_to_sparse = np.zeros(orig_width, dtype=np.uint32)
  for n in range(orig_num_blocks):
    dense_to_sparse[orig_col_ind.charge_labels == col_map[n]] = np.arange(
        orig_col_degen[col_map[n]], dtype=np.uint32)

  # define properties of new tensor resulting from transposition
  new_strides = strides[order]
  new_row_charges = [charges[n] for n in order[:tr_partition]]
  new_col_charges = [charges[n] for n in order[tr_partition:]]
  new_row_flows = flows[order[:tr_partition]]
  new_col_flows = flows[order[tr_partition:]]

  if tr_partition == 0:
    # special case: reshape into row vector

    # compute qnums of row/cols in transposed tensor
    unique_col_qnums, new_col_degen = compute_fused_charge_degeneracies(
        new_col_charges, np.logical_not(new_col_flows))
    identity_charges = charges[0].identity_charges
    block_qnums, new_row_map, new_col_map = intersect(
        identity_charges.unique_charges,
        unique_col_qnums.unique_charges,
        axis=1,
        return_indices=True)
    block_dims = np.array([[1], new_col_degen[new_col_map]], dtype=np.uint32)
    num_blocks = 1
    col_ind, col_locs = reduce_charges(
        new_col_charges,
        np.logical_not(new_col_flows),
        block_qnums,
        return_locations=True,
        strides=new_strides[tr_partition:])

    # find location of blocks in transposed tensor (w.r.t positions in original)
    #pylint: disable=no-member
    orig_row_posR, orig_col_posR = np.divmod(
        col_locs[col_ind.charge_labels == 0], orig_width)
    block_maps = [(all_cumul_degens[orig_row_posR] +
                   dense_to_sparse[orig_col_posR]).ravel()]
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(block_qnums, np.arange(block_qnums.shape[1], dtype=np.int16),
                 charges[0].charge_types)

  elif tr_partition == len(charges):
    # special case: reshape into col vector

    # compute qnums of row/cols in transposed tensor
    unique_row_qnums, new_row_degen = compute_fused_charge_degeneracies(
        new_row_charges, new_row_flows)
    identity_charges = charges[0].identity_charges
    block_qnums, new_row_map, new_col_map = intersect(
        unique_row_qnums.unique_charges,
        identity_charges.unique_charges,
        axis=1,
        return_indices=True)
    block_dims = np.array([new_row_degen[new_row_map], [1]], dtype=np.uint32)
    num_blocks = 1
    row_ind, row_locs = reduce_charges(
        new_row_charges,
        new_row_flows,
        block_qnums,
        return_locations=True,
        strides=new_strides[:tr_partition])

    # find location of blocks in transposed tensor (w.r.t positions in original)
    #pylint: disable=no-member
    orig_row_posL, orig_col_posL = np.divmod(
        row_locs[row_ind.charge_labels == 0], orig_width)
    block_maps = [(all_cumul_degens[orig_row_posL] +
                   dense_to_sparse[orig_col_posL]).ravel()]
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(block_qnums, np.arange(block_qnums.shape[1], dtype=np.int16),
                 charges[0].charge_types)
  else:

    unique_row_qnums, new_row_degen = compute_fused_charge_degeneracies(
        new_row_charges, new_row_flows)

    unique_col_qnums, new_col_degen = compute_fused_charge_degeneracies(
        new_col_charges, np.logical_not(new_col_flows))
    block_qnums, new_row_map, new_col_map = intersect(
        unique_row_qnums.unique_charges,
        unique_col_qnums.unique_charges,
        axis=1,
        return_indices=True)
    block_dims = np.array(
        [new_row_degen[new_row_map], new_col_degen[new_col_map]],
        dtype=np.uint32)
    num_blocks = len(new_row_map)

    row_ind, row_locs = reduce_charges(
        new_row_charges,
        new_row_flows,
        block_qnums,
        return_locations=True,
        strides=new_strides[:tr_partition])

    col_ind, col_locs = reduce_charges(
        new_col_charges,
        np.logical_not(new_col_flows),
        block_qnums,
        return_locations=True,
        strides=new_strides[tr_partition:])

    block_maps = [0] * num_blocks
    for n in range(num_blocks):
      #pylint: disable=no-member
      orig_row_posL, orig_col_posL = np.divmod(
          row_locs[row_ind.charge_labels == n], orig_width)
      #pylint: disable=no-member
      orig_row_posR, orig_col_posR = np.divmod(
          col_locs[col_ind.charge_labels == n], orig_width)
      block_maps[n] = (
          all_cumul_degens[np.add.outer(orig_row_posL, orig_row_posR)] +
          dense_to_sparse[np.add.outer(orig_col_posL, orig_col_posR)]).ravel()
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(block_qnums, np.arange(block_qnums.shape[1], dtype=np.int16),
                 charges[0].charge_types)

  return block_maps, obj, block_dims


class ChargeArray:
  """
  Base class for BlockSparseTensor.
  Stores a dense tensor together with its charge data.
  """

  def __init__(self,
               data: np.ndarray,
               charges: List[BaseCharge],
               flows: List[bool],
               order: Optional[List[List[int]]] = None,
               check_consistency: Optional[bool] = False) -> None:
    """
    Initialize a `ChargeArray` object. `len(data)` has to 
    be equal to `np.prod([c.dim for c in charges])`.
    
    Args: 
      data: An np.ndarray of the data. 
      charges: A list of `BaseCharge` objects.
      flows: The flows of the tensor indices, `False` for inflowing, `True`
        for outflowing.
      order: An optional order argument, determining the shape and order of the
        tensor.
      check_consistency: No effect. Needed for consistency with derived class constructors.
    """
    self._charges = charges
    self._flows = np.asarray(flows)

    self.data = np.asarray(data.flat)  #do not copy data

    if order is None:
      self._order = [[n] for n in range(len(self._charges))]
    else:
      flat_order = []
      for o in order:
        flat_order.extend(o)
      if not np.array_equal(np.sort(flat_order), np.arange(len(self._charges))):
        raise ValueError("flat_order = {} is not a permutation of {}".format(
            flat_order, np.arange(len(self._charges))))

      self._order = order

  @classmethod
  def randn(cls, indices: List[Index],
            dtype: Optional[Type[np.number]] = None) -> "ChargeArray":
    """
    Initialize a random ChargeArray object.
    Args:
      indices: List of `Index` objects.
      dtype: An optional numpy dtype. The dtype of the ChargeArray
    Returns:
      ChargeArray
    """
    charges, flows = get_flat_meta_data(indices)
    num_non_zero_elements = np.prod([c.dim for c in charges])

    def init_random():
      if ((np.dtype(dtype) is np.dtype(np.complex128)) or
          (np.dtype(dtype) is np.dtype(np.complex64))):
        return np.random.randn(num_non_zero_elements).astype(
            dtype) + 1j * np.random.randn(num_non_zero_elements).astype(dtype)
      return np.random.randn(num_non_zero_elements).astype(dtype)

    return cls(data=init_random(), charges=charges, flows=flows)

  @classmethod
  def random(cls,
             indices: List[Index],
             boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
             dtype: Optional[Type[np.number]] = None) -> "ChargeArray":
    """
    Initialize a random ChargeArray object.
    Args:
      indices: List of `Index` objects.
      boundaries: Tuple of interval boundaries for the random uniform 
        distribution.
      dtype: An optional numpy dtype. The dtype of the ChargeArray
    Returns:
      ChargeArray
    """
    num_non_zero_elements = np.prod([len(i) for i in indices])
    charges, flows = get_flat_meta_data(indices)

    def init_random():
      if ((np.dtype(dtype) is np.dtype(np.complex128)) or
          (np.dtype(dtype) is np.dtype(np.complex64))):
        return np.random.uniform(
            boundaries[0], boundaries[1], num_non_zero_elements
        ).astype(dtype) + 1j * np.random.uniform(
            boundaries[0], boundaries[1], num_non_zero_elements).astype(dtype)
      return np.random.uniform(boundaries[0], boundaries[1],
                               num_non_zero_elements).astype(dtype)

    return cls(data=init_random(), charges=charges, flows=flows)

  @property
  def ndim(self):
    """
    The number of tensor dimensions.
    """
    return len(self._order)

  @property
  def dtype(self) -> Type[np.number]:
    """
    The dtype of `ChargeArray`.
    """
    return self.data.dtype

  @property
  def shape(self) -> Tuple:
    """
    The dense shape of the tensor.
    Returns:
      Tuple: A tuple of `int`.
    """
    return tuple(
        [np.prod([self._charges[n].dim for n in s]) for s in self._order])

  @property
  def charges(self) -> List[List[BaseCharge]]:
    """
    A list of list of `BaseCharge`.
    The charges, in the current shape and index order as determined by `ChargeArray._order`.
    Returns:
      List of List of BaseCharge
    """
    return [[self._charges[n] for n in o] for o in self._order]

  @property
  def flows(self):
    """
    A list of list of `bool`.
    The flows, in the current shape and index order as determined by `ChargeArray._order`.
    Returns:
      List of List of bool
    """

    return [[self._flows[n] for n in o] for o in self._order]

  @property
  def flat_charges(self):
    return self._charges

  @property
  def flat_flows(self):
    return list(self._flows)

  @property
  def flat_order(self) -> Tuple:
    """
    The flattened `ChargeArray._oder`.
    """
    return flatten(self._order)

  @property
  def sparse_shape(self) -> Tuple:
    """
    The sparse shape of the tensor.
    Returns:
      Tuple: A tuple of `Index` objects.
    """

    indices = [
        Index([self._charges[n]
               for n in s], [self._flows[n]
                             for n in s])
        for s in self._order
    ]
    return tuple(indices)

  def todense(self) -> np.ndarray:
    """
    Map the sparse tensor to dense storage.
    
    """
    return np.reshape(self.data, self.shape)


class BlockSparseTensor(ChargeArray):
  """
  A block-sparse tensor class. This class stores non-zero
  elements of a symmetric tensor using an element wise
  encoding.
  The tensor data is stored in self.data, a 1d np.ndarray.
  Attributes:
    * _data: np.ndarray; the data of the tensor.
    * _charges: A list of `BaseCharge` objects, one for each 
        elementary leg of the tensor.
    * _flows: A list of bool, denoting the flow direction of
        each elementary leg.
    * _order: A list of list of int: Used to implement `reshape` and 
        `transpose` operations. Both operations act entirely
        on meta-data of the tensor. `_order` determines which elemetary 
        legs of the tensor are combined, and where they go. 
        E.g. a tensor of rank 4 is initialized with 
        `_order=[[0],[1],[2],[3]]`. Fusing legs 1 and 2
        results in `_order=[[0],[1,2],[3]]`, transposing with 
        `(1,2,0)` results in `_order=[[1,2],[3],[0]]`.
        No data is shuffled during these operations.
  """

  def __init__(self,
               data: np.ndarray,
               charges: List[BaseCharge],
               flows: List[bool],
               order: Optional[List[Union[List, np.ndarray]]] = None,
               check_consistency: Optional[bool] = True) -> None:
    """
    Args: 
      data: An np.ndarray of the data. 
      charges: A list of `BaseCharge` objects.
      flows: The flows of the tensor indices, `False` for inflowing, `True`
        for outflowing.
      order: An optional order argument, determining the shape and order of the
        tensor.
      check_consistency: If `True`, check if `len(data)` is consistent with 
        number of non-zero elements given by the charges. This usually causes
        significant overhead.
    """
    super().__init__(data=data, charges=charges, flows=flows, order=order)

    if check_consistency and (len(self._charges) > 0):
      num_non_zero_elements = compute_num_nonzero(self._charges, self._flows)
      if num_non_zero_elements != len(data.flat):
        raise ValueError("number of tensor elements {} defined "
                         "by `charges` is different from"
                         " len(data)={}".format(num_non_zero_elements,
                                                len(data.flat)))

  def copy(self) -> "BlockSparseTensor":
    """
    Return a copy of the tensor.
    """
    return BlockSparseTensor(self.data.copy(),
                             [c.copy() for c in self._charges], self._flows,
                             self._order, False)

  def todense(self) -> np.ndarray:
    """
    Map the sparse tensor to dense storage.
    
    """
    if len(self.shape) == 0:
      return self.data
    out = np.asarray(np.zeros(self.shape, dtype=self.dtype).flat)
    out[np.nonzero(
        fuse_charges(self._charges, self._flows) ==
        self._charges[0].identity_charges)[0]] = self.data
    result = np.reshape(out, [c.dim for c in self._charges])
    flat_order = flatten(self._order)
    return result.transpose(flat_order).reshape(self.shape)

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
    charges, flows = get_flat_meta_data(indices)
    num_non_zero_elements = compute_num_nonzero(charges, flows)
    tmp = np.append(0, np.cumsum([len(i.flat_charges) for i in indices]))
    order = [list(np.arange(tmp[n], tmp[n + 1])) for n in range(len(tmp) - 1)]

    def init_random():
      if ((np.dtype(dtype) is np.dtype(np.complex128)) or
          (np.dtype(dtype) is np.dtype(np.complex64))):
        return np.random.randn(num_non_zero_elements).astype(
            dtype) + 1j * np.random.randn(num_non_zero_elements).astype(dtype)
      return np.random.randn(num_non_zero_elements).astype(dtype)

    return cls(
        data=init_random(),
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

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
    charges, flows = get_flat_meta_data(indices)
    num_non_zero_elements = compute_num_nonzero(charges, flows)
    tmp = np.append(0, np.cumsum([len(i.flat_charges) for i in indices]))
    order = [list(np.arange(tmp[n], tmp[n + 1])) for n in range(len(tmp) - 1)]

    return cls(
        data=np.ones((num_non_zero_elements,), dtype=dtype),
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

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
    charges, flows = get_flat_meta_data(indices)
    num_non_zero_elements = compute_num_nonzero(charges, flows)
    tmp = np.append(0, np.cumsum([len(i.flat_charges) for i in indices]))
    order = [list(np.arange(tmp[n], tmp[n + 1])) for n in range(len(tmp) - 1)]

    return cls(
        data=np.zeros((num_non_zero_elements,), dtype=dtype),
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

  @classmethod
  def random(cls,
             indices: List[Index],
             boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
             dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a random symmetric tensor from random uniform distribution.
    Args:
      indices: List of `Index` objecst, one for each leg. 
      boundaries: Tuple of interval boundaries for the random uniform 
        distribution.
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    charges, flows = get_flat_meta_data(indices)
    num_non_zero_elements = compute_num_nonzero(charges, flows)
    dtype = dtype if dtype is not None else np.float64

    def init_random():
      if ((np.dtype(dtype) is np.dtype(np.complex128)) or
          (np.dtype(dtype) is np.dtype(np.complex64))):
        return np.random.uniform(
            boundaries[0], boundaries[1], num_non_zero_elements
        ).astype(dtype) + 1j * np.random.uniform(
            boundaries[0], boundaries[1], num_non_zero_elements).astype(dtype)
      return np.random.uniform(boundaries[0], boundaries[1],
                               num_non_zero_elements).astype(dtype)

    tmp = np.append(0, np.cumsum([len(i.flat_charges) for i in indices]))
    order = [list(np.arange(tmp[n], tmp[n + 1])) for n in range(len(tmp) - 1)]

    return cls(
        data=init_random(),
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

  def _sub_add_protection(self, other):
    if not isinstance(other, type(self)):
      raise TypeError(
          "Can only add or subtract BlockSparseTensor from BlockSparseTensor. "
          "Found type {}".format(type(other)))

    if self.shape != other.shape:
      raise ValueError(
          "cannot add or subtract tensors with shapes {}and {}".format(
              self.shape, other.shape))
    if len(self._charges) != len(other._charges):
      raise ValueError(
          "cannot add or subtract tensors with different charge lengths {} and {}"
          .format(len(self._charges), len(other._charges)))
    if not np.all([
        self.sparse_shape[n] == other.sparse_shape[n]
        for n in range(len(self.sparse_shape))
    ]):
      raise ValueError(
          "cannot add or subtract tensors non-matching sparse shapes")

  def __sub__(self, other: "BlockSparseTensor"):
    self._sub_add_protection(other)
    _, index_other = np.unique(other.flat_order, return_index=True)
    #bring self into the same storage layout as other
    out = self.transpose_data(self.flat_order[index_other])
    #now subtraction is save
    return BlockSparseTensor(
        data=out.data - other.data,
        charges=out._charges,
        flows=out._flows,
        order=out._order,
        check_consistency=False)

  def __add__(self, other: "BlockSparseTensor"):
    self._sub_add_protection(other)
    #bring self into the same storage layout as other
    _, index_other = np.unique(other.flat_order, return_index=True)
    out = self.transpose_data(self.flat_order[index_other])
    #now subtraction is save
    return BlockSparseTensor(
        data=out.data + other.data,
        charges=out._charges,
        flows=out._flows,
        order=out._order,
        check_consistency=False)

  def __mul__(self, number: np.number):
    if not np.isscalar(number):
      raise TypeError(
          "Can only multiply BlockSparseTensor by a number. Found type {}"
          .format(type(number)))
    return BlockSparseTensor(
        data=self.data * number,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)

  def __rmul__(self, number: np.number):
    if not np.isscalar(number):
      raise TypeError(
          "Can only right-multiply BlockSparseTensor by a number. Found type {}"
          .format(type(number)))
    return BlockSparseTensor(
        data=self.data * number,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)

  def __truediv__(self, number: np.number):
    if not np.isscalar(number):
      raise TypeError(
          "Can only divide BlockSparseTensor by a number. Found type {}".format(
              type(number)))

    return BlockSparseTensor(
        data=self.data / number,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)

  def __matmul__(self, other):

    if (self.ndim != 2) or (other.ndim != 2):
      raise ValueError("__matmul__ only implemented for matrices."
                       " Found ndims =  {} and {}".format(
                           self.ndim, other.ndim))
    return tensordot(self, other, ([1], [0]))

  def conj(self):
    """
    Complex conjugate operation.
    Returns:
      BlockSparseTensor: The conjugated tensor
    """
    return BlockSparseTensor(
        data=np.conj(self.data),
        charges=self._charges,
        flows=list(np.logical_not(self._flows)),
        order=self._order,
        check_consistency=False)

  @property
  def T(self):
    return self.transpose()

  def transpose_data(self,
                     flat_order: Optional[Union[List, np.ndarray]] = None):
    """
    Transpose the tensor data in place such that the linear order 
    of the elements in `BlockSparseTensor.data` corresponds to the 
    current order of tensor indices. 
    Consider a tensor with current order given by `_order=[[1,2],[3],[0]]`,
    i.e. `data` was initialized according to order [0,1,2,3], and the tensor
    has since been reshaped and transposed. The linear oder of `data` does not
    match the desired order [1,2,3,0] of the tensor. `transpose_data` fixes this
    by permuting `data` into this order, transposing `_charges` and `_flows`,
    and changing `_order` to `[[0,1],[2],[3]]`.
    Args:
      flat_order: An optional alternative order to be used to transposed the 
        tensor. If `None` defaults to `BlockSparseTensor.flat_order`.
    """
    flat_charges = self.flat_charges
    flat_flows = self.flat_flows
    if flat_order is None:
      flat_order = self.flat_order

    if np.array_equal(flat_order, np.arange(len(flat_order))):
      return self

    tr_partition = _find_best_partition(
        [flat_charges[n].dim for n in flat_order])

    tr_sparse_blocks, tr_charges, _ = _find_transposed_diagonal_sparse_blocks(
        flat_charges, flat_flows, tr_partition, flat_order)

    sparse_blocks, charges, _ = _find_diagonal_sparse_blocks(
        [flat_charges[n] for n in flat_order],
        [flat_flows[n] for n in flat_order], tr_partition)
    data = np.empty(len(self.data), dtype=self.dtype)
    for n, sparse_block in enumerate(sparse_blocks):
      ind = np.nonzero(tr_charges == charges[n])[0][0]
      permutation = tr_sparse_blocks[ind]
      data[sparse_block] = self.data[permutation]
    tmp = np.append(0, np.cumsum([len(o) for o in self._order]))
    order = [list(np.arange(tmp[n], tmp[n + 1])) for n in range(len(tmp) - 1)]
    return BlockSparseTensor(
        data,
        charges=[self._charges[o] for o in flat_order],
        flows=[self._flows[o] for o in flat_order],
        order=order,
        check_consistency=False)

  def transpose(self,
                order: Optional[Union[List[int], np.ndarray]] = np.asarray(
                    [1, 0]),
                shuffle: Optional[bool] = False) -> "BlockSparseTensor":
    """
    Transpose the tensor into the new order `order`. If `shuffle=False`
    no data-reshuffling is done.
    Args:
      order: The new order of indices.
      shuffle: If `True`, reshuffle data.
    Returns:
      BlockSparseTensor: The transposed tensor.
    """
    if len(order) != self.ndim:
      raise ValueError(
          "`len(order)={}` is different form `self.ndim={}`".format(
              len(order), self.ndim))

    order = [self._order[o] for o in order]

    tensor = BlockSparseTensor(
        data=self.data,
        charges=self._charges,
        flows=self._flows,
        order=order,
        check_consistency=False)
    if shuffle:
      return tensor.transpose_data()
    return tensor

  def reshape(
      self,
      shape: Union[List[Index], Tuple[Index, ...], List[int], Tuple[int, ...]]
  ) -> "BlockSparseTensor":
    """
    Reshape `tensor` into `shape.
    `BlockSparseTensor.reshape` works the same as the dense 
    version, with the notable exception that the tensor can only be 
    reshaped into a form compatible with its elementary shape. 
    The elementary shape is the shape determined by BlockSparseTensor._charges.
    For example, while the following reshaping is possible for regular 
    dense numpy tensor,
    ```
    A = np.random.rand(6,6,6)
    np.reshape(A, (2,3,6,6))
    ```
    the same code for BlockSparseTensor
    ```
    q1 = U1Charge(np.random.randint(0,10,6))
    q2 = U1Charge(np.random.randint(0,10,6))
    q3 = U1Charge(np.random.randint(0,10,6))
    i1 = Index(charges=q1,flow=False)
    i2 = Index(charges=q2,flow=True)
    i3 = Index(charges=q3,flow=False)
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
    new_shape = []
    for s in shape:
      if isinstance(s, Index):
        new_shape.append(s.dim)
      else:
        new_shape.append(s)

    # a few simple checks
    if np.prod(new_shape) != np.prod(self.shape):
      raise ValueError("A tensor with {} elements cannot be "
                       "reshaped into a tensor with {} elements".format(
                           np.prod(self.shape), np.prod(new_shape)))

    flat_dims = flatten(
        [[self._charges[n].dim for n in o] for o in self._order])

    partitions = [0]
    for n, ns in enumerate(new_shape):
      tmp = np.nonzero(np.cumprod(flat_dims) == ns)[0]
      if len(tmp) == 0:
        raise ValueError("The shape {} is incompatible with the "
                         "elementary shape {} of the tensor.".format(
                             new_shape, tuple(flat_dims)))

      partitions.append(tmp[0] + 1)
      flat_dims = flat_dims[partitions[-1]:]
    for d in flat_dims:
      if d != 1:
        raise ValueError("The shape {} is incompatible with the "
                         "elementary shape {} of the tensor.".format(
                             new_shape, tuple(flat_dims)))
      partitions[-1] += 1

    partitions = np.cumsum(partitions)

    flat_order = self.flat_order
    new_order = []
    for n in range(1, len(partitions)):
      new_order.append(list(flat_order[partitions[n - 1]:partitions[n]]))
    result = BlockSparseTensor(
        data=self.data,
        charges=self._charges,
        flows=self._flows,
        order=new_order,
        check_consistency=False)
    return result


def norm(tensor: BlockSparseTensor) -> float:
  """
  The norm of the tensor.
  """
  return np.linalg.norm(tensor.data)


def diag(tensor: ChargeArray) -> Any:
  """
  Return a diagonal BlockSparseTensor from a ChargeArray, or 
  return the diagonal of a BlockSparseTensor as a ChargeArray
  
  """
  if tensor.ndim > 2:
    raise TypeError("`diag` currently only implemented for matrices, "
                    "found `ndim={}".format(tensor.ndim))
  if not isinstance(tensor, BlockSparseTensor):
    if tensor.ndim > 1:
      raise TypeError(
          "`diag` currently only implemented for `ChargeArray` with ndim=1, "
          "found `ndim={}`".format(tensor.ndim))
    flat_charges = tensor._charges + tensor._charges
    flat_flows = list(tensor.flat_flows) + list(
        np.logical_not(tensor.flat_flows))
    flat_order = list(tensor.flat_order) + list(
        np.asarray(tensor.flat_order) + len(tensor._charges))
    tr_partition = len(tensor._order[0])
    blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
        flat_charges, flat_flows, tr_partition, flat_order)
    data = np.zeros(np.sum(np.prod(shapes, axis=0)), dtype=tensor.dtype)
    lookup, unique, labels = compute_sparse_lookup(tensor._charges,
                                                   tensor._flows, charges)
    for n, block in enumerate(blocks):
      label = labels[np.nonzero(unique == charges[n])[0][0]]
      data[block] = np.ravel(
          np.diag(tensor.data[np.nonzero(lookup == label)[0]]))

    order = [
        tensor._order[0],
        list(np.asarray(tensor._order[0]) + len(tensor._charges))
    ]
    new_charges = [tensor._charges[0].copy(), tensor._charges[0].copy()]
    return BlockSparseTensor(
        data,
        charges=new_charges,
        flows=list(tensor._flows) + list(np.logical_not(tensor._flows)),
        order=order,
        check_consistency=False)

  flat_charges = tensor._charges
  flat_flows = tensor.flat_flows
  flat_order = tensor.flat_order
  tr_partition = len(tensor._order[0])
  sparse_blocks, charges, block_shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  shapes = np.min(block_shapes, axis=0)
  data = np.concatenate([
      np.diag(np.reshape(tensor.data[sparse_blocks[n]], block_shapes[:, n]))
      for n in range(len(sparse_blocks))
  ])
  charge_labels = np.concatenate([
      np.full(shapes[n], fill_value=n, dtype=np.int16)
      for n in range(len(sparse_blocks))
  ])
  newcharges = [charges[charge_labels]]
  flows = [False]
  return ChargeArray(data, newcharges, flows)


def reshape(
    tensor: BlockSparseTensor,
    shape: Union[List[Index], Tuple[Index, ...], List[int], Tuple[int, ...]]
) -> BlockSparseTensor:
  """
  Reshape `tensor` into `shape.
  `BlockSparseTensor.reshape` works the same as the dense 
  version, with the notable exception that the tensor can only be 
  reshaped into a form compatible with its elementary shape. 
  The elementary shape is the shape determined by BlockSparseTensor._charges.
  For example, while the following reshaping is possible for regular 
  dense numpy tensor,
  ```
  A = np.random.rand(6,6,6)
  np.reshape(A, (2,3,6,6))
  ```
  the same code for BlockSparseTensor
  ```
  q1 = U1Charge(np.random.randint(0,10,6))
  q2 = U1Charge(np.random.randint(0,10,6))
  q3 = U1Charge(np.random.randint(0,10,6))
  i1 = Index(charges=q1,flow=False)
  i2 = Index(charges=q2,flow=True)
  i3 = Index(charges=q3,flow=False)
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

  return tensor.reshape(shape)


def conj(tensor: BlockSparseTensor):
  return tensor.conj()


def transpose(tensor: BlockSparseTensor,
              order: Optional[Union[List[int], np.ndarray]] = np.asarray(
                  [1, 0])) -> "BlockSparseTensor":
  """
  Transpose the tensor into the new order `order`. If `shuffle=False`
  no data-reshuffling is done.
  Args:
    order: The new order of indices.
    shuffle: If `True`, reshuffle data.
  Returns:
    BlockSparseTensor: The transposed tensor.
  """
  return tensor.transpose(order)


def outerproduct(tensor1: BlockSparseTensor,
                 tensor2: BlockSparseTensor) -> BlockSparseTensor:
  """
  Compute the outer product of two BlockSparseTensor.
  Args:
    tensor1: A tensor.
    tensor2: A tensor.
  Returns:
    BlockSparseTensor: The result of taking the outer product.
  """

  final_charges = tensor1._charges + tensor2._charges
  final_flows = tensor1.flat_flows + tensor2.flat_flows
  order2 = [list(np.asarray(s) + len(tensor1._charges)) for s in tensor2._order]

  data = np.zeros(
      compute_num_nonzero(final_charges, final_flows), dtype=tensor1.dtype)
  if ((len(tensor1.data) > 0) and (len(tensor2.data) > 0)) and (len(data) > 0):
    # find the location of the zero block in the output
    final_block_maps, final_block_charges, _ = _find_diagonal_sparse_blocks(
        final_charges, final_flows, len(tensor1._charges))
    index = np.nonzero(
        final_block_charges == final_block_charges.identity_charges)[0][0]
    data[final_block_maps[index].ravel()] = np.outer(tensor1.data,
                                                     tensor2.data).ravel()

  return BlockSparseTensor(
      data,
      charges=final_charges,
      flows=final_flows,
      order=tensor1._order + order2,
      check_consistency=False)


def tensordot(tensor1: BlockSparseTensor,
              tensor2: BlockSparseTensor,
              axes: Optional[Union[Sequence[Sequence[int]], int]] = 2
             ) -> BlockSparseTensor:
  """
  Contract two `BlockSparseTensor`s along `axes`.
  Args:
    tensor1: First tensor.
    tensor2: Second tensor.
    axes: The axes to contract.
  Returns:
      BlockSparseTensor: The result of the tensor contraction.
  """

  if isinstance(axes, (np.integer, int)):
    axes = [
        np.arange(tensor1.ndim - axes, tensor1.ndim, dtype=np.int16),
        np.arange(0, axes, dtype=np.int16)
    ]
  elif isinstance(axes[0], (np.integer, int)):
    axes = [np.array(axes, dtype=np.int16), np.array(axes, dtype=np.int16)]
  axes1 = axes[0]
  axes2 = axes[1]
  if not np.all(np.unique(axes1) == np.sort(axes1)):
    raise ValueError(
        "Some values in axes[0] = {} appear more than once!".format(axes1))
  if not np.all(np.unique(axes2) == np.sort(axes2)):
    raise ValueError(
        "Some values in axes[1] = {} appear more than once!".format(axes2))

  if len(axes1) == 0:
    return outerproduct(tensor1, tensor2)

  if (len(axes1) == tensor1.ndim) and (len(axes2) == tensor2.ndim):
    isort = np.argsort(axes1)
    data = np.dot(tensor1.data,
                  tensor2.transpose(np.asarray(axes2)[isort]).data)
    return BlockSparseTensor(
        data=data, charges=[], flows=[], order=[],
        check_consistency=False)  #Index(identity_charges, flow=False)])

  if max(axes1) >= len(tensor1.shape):
    raise ValueError(
        "rank of `tensor1` is smaller than `max(axes1) = {}.`".format(
            max(axes1)))

  if max(axes2) >= len(tensor2.shape):
    raise ValueError(
        "rank of `tensor2` is smaller than `max(axes2) = {}`".format(
            max(axes1)))

  contr_flows_1 = []
  contr_flows_2 = []
  contr_charges_1 = []
  contr_charges_2 = []
  for a in axes1:
    contr_flows_1.extend(tensor1._flows[tensor1._order[a]])
    contr_charges_1.extend([tensor1._charges[n] for n in tensor1._order[a]])
  for a in axes2:
    contr_flows_2.extend(tensor2._flows[tensor2._order[a]])
    contr_charges_2.extend([tensor2._charges[n] for n in tensor2._order[a]])

  if len(contr_charges_2) != len(contr_charges_1):
    raise ValueError(
        "axes1 and axes2 have incompatible elementary"
        " shapes {} and {}".format([e.dim for e in contr_charges_1],
                                   [e.dim for e in contr_charges_2]))
  if not np.all(
      np.asarray(contr_flows_1) == np.logical_not(np.asarray(contr_flows_2))):

    raise ValueError("axes1 and axes2 have incompatible elementary"
                     " flows {} and {}".format(contr_flows_1, contr_flows_2))

  free_axes1 = sorted(set(np.arange(tensor1.ndim)) - set(axes1))
  free_axes2 = sorted(set(np.arange(tensor2.ndim)) - set(axes2))

  new_order1 = [tensor1._order[n] for n in free_axes1
               ] + [tensor1._order[n] for n in axes1]
  new_order2 = [tensor2._order[n] for n in axes2
               ] + [tensor2._order[n] for n in free_axes2]

  flat_order_1 = flatten(new_order1)
  flat_order_2 = flatten(new_order2)

  flat_charges_1, flat_flows_1 = tensor1._charges, tensor1.flat_flows
  flat_charges_2, flat_flows_2 = tensor2._charges, tensor2.flat_flows

  left_charges = []
  right_charges = []
  left_flows = []
  right_flows = []
  left_order = []
  right_order = []
  s = 0
  for n in free_axes1:
    left_charges.extend([tensor1._charges[o] for o in tensor1._order[n]])
    left_order.append(list(np.arange(s, s + len(tensor1._order[n]))))
    s += len(tensor1._order[n])
    left_flows.extend([tensor1._flows[o] for o in tensor1._order[n]])

  s = 0
  for n in free_axes2:
    right_charges.extend([tensor2._charges[o] for o in tensor2._order[n]])
    right_order.append(
        list(len(left_charges) + np.arange(s, s + len(tensor2._order[n]))))
    s += len(tensor2._order[n])
    right_flows.extend([tensor2._flows[o] for o in tensor2._order[n]])

  tr_sparse_blocks_1, charges1, shapes_1 = _find_transposed_diagonal_sparse_blocks(
      flat_charges_1, flat_flows_1, len(left_charges), flat_order_1)

  tr_sparse_blocks_2, charges2, shapes_2 = _find_transposed_diagonal_sparse_blocks(
      flat_charges_2, flat_flows_2, len(contr_charges_2), flat_order_2)

  common_charges, label_to_common_1, label_to_common_2 = intersect(
      charges1.unique_charges,
      charges2.unique_charges,
      axis=1,
      return_indices=True)

  #Note: `cs` may contain charges that are not present in `common_charges`
  charges = left_charges + right_charges
  flows = left_flows + right_flows
  sparse_blocks, cs, _ = _find_diagonal_sparse_blocks(charges, flows,
                                                      len(left_charges))
  num_nonzero_elements = np.sum([len(v) for v in sparse_blocks])
  #Note that empty is not a viable choice here.
  data = np.zeros(
      num_nonzero_elements, dtype=np.result_type(tensor1.dtype, tensor2.dtype))

  label_to_common_final = intersect(
      cs.unique_charges, common_charges, axis=1, return_indices=True)[1]

  for n in range(common_charges.shape[1]):
    n1 = label_to_common_1[n]
    n2 = label_to_common_2[n]
    nf = label_to_common_final[n]

    data[sparse_blocks[nf].ravel()] = np.ravel(
        np.matmul(tensor1.data[tr_sparse_blocks_1[n1].reshape(shapes_1[:, n1])],
                  tensor2.data[tr_sparse_blocks_2[n2].reshape(
                      shapes_2[:, n2])]))
  res = BlockSparseTensor(
      data=data,
      charges=charges,
      flows=flows,
      order=left_order + right_order,
      check_consistency=False)
  return res


def svd(matrix: BlockSparseTensor,
        full_matrices: Optional[bool] = True,
        compute_uv: Optional[bool] = True,
        hermitian: Optional[bool] = False) -> Any:
  """
  Compute the singular value decomposition of `matrix`.
  The matrix if factorized into `u * s * vh`, with 
  `u` and `vh` the left and right eigenvectors of `matrix`,
  and `s` its singular values.
  Args:
    matrix: A matrix (i.e. a rank-2 tensor) of type  `BlockSparseTensor`
    full_matrices: If `True`, expand `u` and `v` to square matrices
      If `False` return the "economic" svd, i.e. `u.shape[1]=s.shape[0]`
      and `v.shape[0]=s.shape[1]`
    compute_uv: If `True`, return `u` and `v`.
    hermitian: If `True`, assume hermiticity of `matrix`.
  Returns:
    If `compute_uv` is `True`: Three BlockSparseTensors `U,S,V`.
    If `compute_uv` is `False`: A BlockSparseTensors `S` containing the 
      singular values.
  """

  if matrix.ndim != 2:
    raise NotImplementedError("svd currently supports only rank-2 tensors.")

  flat_charges = matrix._charges
  flat_flows = matrix.flat_flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  u_blocks = []
  singvals = []
  v_blocks = []
  for n, block in enumerate(blocks):
    out = np.linalg.svd(
        np.reshape(matrix.data[block], shapes[:, n]), full_matrices, compute_uv,
        hermitian)
    if compute_uv:
      u_blocks.append(out[0])
      singvals.append(out[1])
      v_blocks.append(out[2])

    else:
      singvals.append(out)

  left_singval_charge_labels = np.concatenate([
      np.full(len(singvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(singvals))
  ])
  left_singval_charge = charges[left_singval_charge_labels]
  S = ChargeArray(np.concatenate(singvals), [left_singval_charge], [False])

  if compute_uv:
    #define the new charges on the two central bonds
    left_charge_labels = np.concatenate([
        np.full(u_blocks[n].shape[1], fill_value=n, dtype=np.int16)
        for n in range(len(u_blocks))
    ])
    right_charge_labels = np.concatenate([
        np.full(v_blocks[n].shape[0], fill_value=n, dtype=np.int16)
        for n in range(len(v_blocks))
    ])
    new_left_charge = charges[left_charge_labels]
    new_right_charge = charges[right_charge_labels]

    charges_u = [new_left_charge
                ] + [matrix._charges[o] for o in matrix._order[0]]
    order_u = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
    flows_u = [True] + [matrix._flows[o] for o in matrix._order[0]]
    charges_v = [new_right_charge
                ] + [matrix._charges[o] for o in matrix._order[1]]
    flows_v = [False] + [matrix._flows[o] for o in matrix._order[1]]
    order_v = [[0]] + [list(np.arange(1, len(matrix._order[1]) + 1))]
    # We fill in data into the transposed U
    # note that transposing is essentially free
    return BlockSparseTensor(
        np.concatenate([np.ravel(u.T) for u in u_blocks]),
        charges=charges_u,
        flows=flows_u,
        order=order_u,
        check_consistency=False).transpose((1, 0)), S, BlockSparseTensor(
            np.concatenate([np.ravel(v) for v in v_blocks]),
            charges=charges_v,
            flows=flows_v,
            order=order_v,
            check_consistency=False)

  return S


def qr(matrix: BlockSparseTensor, mode: Optional[Text] = 'reduced') -> Any:
  """
  Compute the qr decomposition of an `M` by `N` matrix `matrix`.
  The matrix is factorized into `q*r`, with 
  `q` an orthogonal matrix and `r` an upper triangular matrix.
  Args:
    matrix: A matrix (i.e. a rank-2 tensor) of type  `BlockSparseTensor`
    mode : Can take values {'reduced', 'complete', 'r', 'raw'}.
    If K = min(M, N), then

    * 'reduced'  : returns q, r with dimensions (M, K), (K, N) (default)
    * 'complete' : returns q, r with dimensions (M, M), (M, N)
    * 'r'        : returns r only with dimensions (K, N)

  Returns:
    (BlockSparseTensor,BlockSparseTensor): If mode = `reduced` or `complete`
    BlockSparseTensor: If mode = `r`.
  """
  if mode == 'raw':
    raise NotImplementedError('mode `raw` currenntly not supported')
  if matrix.ndim != 2:
    raise NotImplementedError("qr currently supports only rank-2 tensors.")

  flat_charges = matrix._charges
  flat_flows = matrix.flat_flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  q_blocks = []
  r_blocks = []
  for n, block in enumerate(blocks):
    out = np.linalg.qr(np.reshape(matrix.data[block], shapes[:, n]), mode)
    if mode in ('reduced', 'complete'):
      q_blocks.append(out[0])
      r_blocks.append(out[1])
    elif mode == 'r':
      r_blocks.append(out)
    else:
      raise ValueError('unknown value {} for input `mode`'.format(mode))

  left_r_charge_labels = np.concatenate([
      np.full(r_blocks[n].shape[0], fill_value=n, dtype=np.int16)
      for n in range(len(r_blocks))
  ])

  left_r_charge = charges[left_r_charge_labels]
  charges_r = [left_r_charge] + [matrix._charges[o] for o in matrix._order[1]]
  flows_r = [False] + [matrix._flows[o] for o in matrix._order[1]]
  order_r = [[0]] + [list(np.arange(1, len(matrix._order[1]) + 1))]

  R = BlockSparseTensor(
      np.concatenate([np.ravel(r) for r in r_blocks]),
      charges=charges_r,
      flows=flows_r,
      order=order_r,
      check_consistency=False)
  if mode in ('reduced', 'complete'):
    right_q_charge_labels = np.concatenate([
        np.full(q_blocks[n].shape[1], fill_value=n, dtype=np.int16)
        for n in range(len(q_blocks))
    ])
    right_q_charge = charges[right_q_charge_labels]
    charges_q = [
        right_q_charge,
    ] + [matrix._charges[o] for o in matrix._order[0]]
    order_q = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
    flows_q = [True] + [matrix._flows[o] for o in matrix._order[0]]
    return BlockSparseTensor(
        np.concatenate([np.ravel(q.T) for q in q_blocks]),
        charges=charges_q,
        flows=flows_q,
        order=order_q,
        check_consistency=False).transpose((1, 0)), R

  return R


def eigh(matrix: BlockSparseTensor,
         UPLO: Optional[Text] = 'L') -> Tuple[ChargeArray, BlockSparseTensor]:
  """
  Compute the eigen decomposition of a hermitian `M` by `M` matrix `matrix`.
  Args:
    matrix: A matrix (i.e. a rank-2 tensor) of type  `BlockSparseTensor`

  Returns:
    (ChargeArray,BlockSparseTensor): The eigenvalues and eigenvectors

  """
  if matrix.ndim != 2:
    raise NotImplementedError("eigh currently supports only rank-2 tensors.")

  flat_charges = matrix._charges
  flat_flows = matrix.flat_flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  eigvals = []
  v_blocks = []
  for n, block in enumerate(blocks):
    e, v = np.linalg.eigh(np.reshape(matrix.data[block], shapes[:, n]), UPLO)
    eigvals.append(e)
    v_blocks.append(v)

  eigvalscharge_labels = np.concatenate([
      np.full(len(eigvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(eigvals))
  ])
  eigvalscharge = charges[eigvalscharge_labels]
  E = ChargeArray(np.concatenate(eigvals), [eigvalscharge], [False])
  charges_v = [eigvalscharge] + [matrix._charges[o] for o in matrix._order[0]]
  order_v = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
  flows_v = [True] + [matrix._flows[o] for o in matrix._order[0]]

  V = BlockSparseTensor(
      np.concatenate([np.ravel(v.T) for v in v_blocks]),
      charges=charges_v,
      flows=flows_v,
      order=order_v,
      check_consistency=False).transpose()

  return E, V


def eig(matrix: BlockSparseTensor) -> Tuple[ChargeArray, BlockSparseTensor]:
  """
  Compute the eigen decomposition of an `M` by `M` matrix `matrix`.
  Args:
    matrix: A matrix (i.e. a rank-2 tensor) of type  `BlockSparseTensor`

  Returns:
    (ChargeArray,BlockSparseTensor): The eigenvalues and eigenvectors

  """
  if matrix.ndim != 2:
    raise NotImplementedError("eig currently supports only rank-2 tensors.")

  flat_charges = matrix._charges
  flat_flows = matrix.flat_flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  eigvals = []
  v_blocks = []
  for n, block in enumerate(blocks):
    e, v = np.linalg.eig(np.reshape(matrix.data[block], shapes[:, n]))
    eigvals.append(e)
    v_blocks.append(v)

  eigvalscharge_labels = np.concatenate([
      np.full(len(eigvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(eigvals))
  ])
  eigvalscharge = charges[eigvalscharge_labels]
  E = ChargeArray(np.concatenate(eigvals), [eigvalscharge], [False])
  charges_v = [eigvalscharge] + [matrix._charges[o] for o in matrix._order[0]]
  order_v = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
  flows_v = [True] + [matrix._flows[o] for o in matrix._order[0]]

  V = BlockSparseTensor(
      np.concatenate([np.ravel(v.T) for v in v_blocks]),
      charges=charges_v,
      flows=flows_v,
      order=order_v,
      check_consistency=False).transpose()

  return E, V


def sqrt(tensor: Union[BlockSparseTensor, ChargeArray]) -> Any:
  obj = tensor.__new__(type(tensor))
  obj.__init__(
      np.sqrt(tensor.data),
      charges=tensor._charges,
      flows=tensor._flows,
      order=tensor._order,
      check_consistency=False)
  return obj


def trace(tensor: BlockSparseTensor,
          axes: Optional[Tuple[int, ...]] = None) -> BlockSparseTensor:
  """
  Compute the trace of a matrix or tensor.
  Args:
    The axes over which the trace should be computed.
      Defaults to the last two indices of the tensor.
  Returns:
    BlockSparseTensor: The result of taking the trace.
  """
  if tensor.ndim > 1:
    if axes is None:
      axes = (tensor.ndim - 2, tensor.ndim - 1)
    if len(axes) != 2:
      raise ValueError(
          "`len(axes)` has to be 2, found `axes = {}`".format(axes))
    if not np.array_equal(tensor.flows[axes[0]],
                          np.logical_not(tensor.flows[axes[1]])):
      raise ValueError("trace indices have non-matching flows.")
    charges0 = tensor.charges[axes[0]]
    flows0 = tensor.flows[axes[0]]
    identities = [eye(Index([c], [not f])) for c, f in zip(charges0, flows0)]
    #flatten the shape of `tensor`
    out = tensor.reshape(
        flatten([[tensor._charges[n].dim for n in o] for o in tensor._order]))
    _, _, labels0 = np.intersect1d(
        tensor._order[axes[0]], flatten(out._order), return_indices=True)
    _, _, labels1 = np.intersect1d(
        tensor._order[axes[1]], flatten(out._order), return_indices=True)

    a0 = list(labels0[np.argsort(tensor._order[axes[0]])])
    a1 = list(labels1[np.argsort(tensor._order[axes[1]])])

    while len(a0) > 0:
      i = a0.pop(0)
      j = a1.pop(0)
      identity = eye(
          Index([out._charges[out._order[i][0]]],
                [not out._flows[out._order[i][0]]]))
      out = tensordot(out, identity, ([i, j], [0, 1]))
      a0ar = np.asarray(a0)

      mask_min = a0ar > np.min([i, j])
      mask_max = a0ar > np.max([i, j])
      a0ar[np.logical_and(mask_min, mask_max)] -= 2
      a0ar[np.logical_xor(mask_min, mask_max)] -= 1

      a1ar = np.asarray(a1)
      mask_min = a1ar > np.min([i, j])
      mask_max = a1ar > np.max([i, j])
      a1ar[np.logical_and(mask_min, mask_max)] -= 2
      a1ar[np.logical_xor(mask_min, mask_max)] -= 1
      a0 = list(a0ar)
      a1 = list(a1ar)
    return out
  raise ValueError("trace can only be taken for tensors with ndim>1")


def eye(column_index: Index,
        row_index: Optional[Index] = None,
        dtype: Optional[Type[np.number]] = None) -> BlockSparseTensor:
  """
  Return an identity matrix.
  Args:
    column_index: The column index of the matrix.
    row_index: The row index of the matrix.
    dtype: The dtype of the matrix.
  Returns:
    BlockSparseTensor
  """
  if row_index is None:
    row_index = column_index.copy().flip_flow()
  if dtype is None:
    dtype = np.float64

  blocks, _, shapes = _find_diagonal_sparse_blocks(
      column_index.flat_charges + row_index.flat_charges,
      column_index.flat_flows + row_index.flat_flows,
      len(column_index.flat_charges))
  data = np.empty(np.sum(np.prod(shapes, axis=0)), dtype=dtype)
  for n, block in enumerate(blocks):
    data[block] = np.ravel(np.eye(shapes[0, n], shapes[1, n], dtype=dtype))
  order = [list(np.arange(0, len(column_index.flat_charges)))] + [
      list(
          np.arange(
              len(column_index.flat_charges),
              len(column_index.flat_charges) + len(row_index.flat_charges)))
  ]
  return BlockSparseTensor(
      data=data,
      charges=column_index.flat_charges + row_index.flat_charges,
      flows=column_index.flat_flows + row_index.flat_flows,
      order=order,
      check_consistency=False)


def ones(indices: List[Index],
         dtype: Optional[Type[np.number]] = None) -> BlockSparseTensor:
  """
  Initialize a symmetric tensor with ones.
  Args:
    indices: List of `Index` objecst, one for each leg. 
    dtype: An optional numpy dtype. The dtype of the tensor
  Returns:
    BlockSparseTensor
  """

  return BlockSparseTensor.ones(indices, dtype)


def zeros(indices: List[Index],
          dtype: Optional[Type[np.number]] = None) -> BlockSparseTensor:
  """
  Initialize a symmetric tensor with zeros.
  Args:
    indices: List of `Index` objecst, one for each leg. 
    dtype: An optional numpy dtype. The dtype of the tensor
  Returns:
    BlockSparseTensor
  """

  return BlockSparseTensor.zeros(indices, dtype)


def randn(indices: List[Index],
          dtype: Optional[Type[np.number]] = None) -> BlockSparseTensor:
  """
  Initialize a random symmetric tensor from random normal distribution.
  Args:
    indices: List of `Index` objecst, one for each leg. 
    dtype: An optional numpy dtype. The dtype of the tensor
  Returns:
    BlockSparseTensor
  """

  return BlockSparseTensor.randn(indices, dtype)


def rand(indices: List[Index],
         boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
         dtype: Optional[Type[np.number]] = None) -> BlockSparseTensor:
  """
  Initialize a random symmetric tensor from random uniform distribution.
  Args:
    indices: List of `Index` objecst, one for each leg. 
    boundaries: Tuple of interval boundaries for the random uniform 
      distribution.
    dtype: An optional numpy dtype. The dtype of the tensor
  Returns:
    BlockSparseTensor
  """
  return BlockSparseTensor.random(indices, boundaries, dtype)


def inv(matrix: BlockSparseTensor) -> BlockSparseTensor:
  """
  Compute the matrix inverse of `matrix`.
  Returns:
    BlockSparseTensor: The inverse of `matrix`.
  """
  if matrix.ndim != 2:
    raise ValueError("`inv` can only be taken for matrices, "
                     "found tensor.ndim={}".format(matrix.ndim))
  flat_charges = matrix._charges
  flat_flows = matrix.flat_flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, _, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  data = np.empty(np.sum(np.prod(shapes, axis=0)), dtype=matrix.dtype)
  for n, block in enumerate(blocks):
    data[block] = np.ravel(
        np.linalg.inv(np.reshape(matrix.data[block], shapes[:, n])).T)

  return BlockSparseTensor(
      data=data,
      charges=matrix._charges,
      flows=np.logical_not(matrix._flows),
      order=matrix._order,
      check_consistency=False).transpose((1, 0))


def pinv(matrix: BlockSparseTensor,
         rcond: Optional[float] = 1E-15,
         hermitian: Optional[bool] = False) -> BlockSparseTensor:
  """
  Compute the Moore-Penrose pseudo inverse of `matrix`.
  Args:
    rcond: Pseudo inverse cutoff.
  Returns:
    BlockSparseTensor: The pseudo inverse of `matrix`.
  """

  flat_charges = matrix._charges
  flat_flows = matrix.flat_flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, _, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  data = np.empty(np.sum(np.prod(shapes, axis=0)), dtype=matrix.dtype)
  for n, block in enumerate(blocks):
    data[block] = np.ravel(
        np.linalg.pinv(
            np.reshape(matrix.data[block], shapes[:, n]),
            rcond=rcond,
            hermitian=hermitian).T)

  return BlockSparseTensor(
      data=data,
      charges=matrix._charges,
      flows=np.logical_not(matrix._flows),
      order=matrix._order,
      check_consistency=False).transpose((1, 0))
