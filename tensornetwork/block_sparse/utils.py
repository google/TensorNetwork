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
from functools import reduce
from operator import mul
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.charge import (fuse_charges, fuse_degeneracies,
                                               BaseCharge, fuse_ndarray_charges,
                                               charge_equal, fuse_ndarrays)

from typing import (List, Union, Any, Tuple, Optional, Sequence)
Tensor = Any

SIZE_T = np.int64  #the size-type of index-arrays
_CACHED_BLOCKS = {}

def compute_hash(charges, flows, tr_partition, order):
  return hash(
      tuple([np.stack(c.charges).tostring() for c in charges] + [
          np.array(flows).tostring(), tr_partition,
          np.array(order, dtype=np.int16).tostring()
      ]))

def get_real_dtype(dtype):
  if dtype == np.complex128:
    return np.float64
  if dtype == np.complex64:
    return np.float32
  return dtype

def flatten(list_of_list: List[List]) -> np.ndarray:
  """
  Flatten a list of lists into a single list.
  Args:
    list_of_lists: A list of lists.
  Returns:
    list: The flattened input.
  """
  return np.array([l for sublist in list_of_list for l in sublist])


def get_flat_meta_data(indices: Sequence[Index]) -> Tuple[List, List]:
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


def fuse_stride_arrays(dims: Union[List[int], np.ndarray],
                       strides: Union[List[int], np.ndarray]) -> np.ndarray:
  """
  Compute linear positions of tensor elements 
  of a tensor with dimensions `dims` according to `strides`.
  Args: 
    dims: An np.ndarray of (original) tensor dimensions.
    strides: An np.ndarray of (possibly permuted) strides.
  Returns:
    np.ndarray: Linear positions of tensor elements according to `strides`.
  """
  return fuse_ndarrays([
      np.arange(0, strides[n] * dims[n], strides[n], dtype=SIZE_T)
      for n in range(len(dims))
  ])


def compute_sparse_lookup(
    charges: List[BaseCharge], flows: Union[np.ndarray, List[bool]],
    target_charges: BaseCharge) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Compute lookup table for how dense index positions map 
  to sparse index positions, treating only those elements as non-zero
  whose charges fuse to `target_charges`.

  Args:
    charges: List of `BaseCharge` objects.
    flows: A list of `bool`; the flow directions.
    target_charges: A `BaseCharge`; the target charges for which 
      the fusion of `charges` is non-zero.
  Returns:
    lookup: An np.ndarray of positive numbers between `0` and
      `len(unique_charges)`. The position of values `n` in `lookup` 
      are positions with charge values `unique_charges[n]`.
    unique_charges: The unique charges of fusion of `charges`
    label_to_unique: The integer labels of the unique charges.
  """
  #Note (mganahl): check if target_charges can be permanently collapsed
  fused_charges = fuse_charges(charges, flows)
  # collapse now to avoid unneccesary expand in unique
  #fused_charges.collapse()
  unique_charges, labels = fused_charges.unique(return_inverse=True)
  _, labels_of_unique, _ = unique_charges.intersect(
      target_charges, return_indices=True)
  tmp = np.full(
      len(unique_charges), fill_value=-1, dtype=labels_of_unique.dtype)

  tmp[labels_of_unique] = labels_of_unique
  lookup = tmp[labels]
  lookup = lookup[lookup >= 0]

  return lookup, unique_charges, np.sort(labels_of_unique)


def _get_strides(dims: Union[List[int], np.ndarray]) -> np.ndarray:
  """
  compute strides of `dims`.
  """
  return np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))


def _find_best_partition(dims: Union[List[int], np.ndarray]) -> int:
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
    flows: Union[np.ndarray, List[bool]]) -> Tuple[BaseCharge, np.ndarray]:
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
  dims = [c.dim for c in charges]
  # for small dims is faster to fuse all and use unique
  # directly
  if reduce(mul, dims, 1) < 20000:
    fused = fuse_charges(charges, flows)
    return fused.unique(return_counts=True)
  # get unique charges and their degeneracies on the first leg.
  # We are fusing from "left" to "right".
  accumulated_charges, accumulated_degeneracies = charges[0].dual(
      flows[0]).unique(return_counts=True)
  num_symmetries = charges[0].num_symmetries
  for n in range(1, len(charges)):
    leg_charges, leg_degeneracies = charges[n].unique(return_counts=True)
    fused_charges = accumulated_charges + leg_charges.dual(flows[n])
    fused_degeneracies = fuse_degeneracies(accumulated_degeneracies,
                                           leg_degeneracies)
    if num_symmetries >= 2:
      accumulated_charges, fused_charge_labels = fused_charges.unique(
          return_inverse=True)
      # new_order = np.argsort(fused_charge_labels)
      # all_degens = np.cumsum(fused_degeneracies[new_order])
      # cum_degens = all_degens[np.flatnonzero(np.diff(fused_charge_labels[new_order]))]
      # accumulated_degeneracies = np.append(cum_degens,all_degens[-1]) - np.append(0,cum_degens)
      accumulated_degeneracies = np.array([
          np.sum(fused_degeneracies[fused_charge_labels == m])
          for m in range(len(accumulated_charges))
      ])
    else:
      accumulated_charges = fused_charges.unique()
      mask = fused_charges == accumulated_charges
      accumulated_degeneracies = np.sum( #pylint: disable=unexpected-keyword-arg
          np.repeat(
              fused_degeneracies[:, None], accumulated_charges.dim, axis=1),
          axis=0,
          where=mask)

  return accumulated_charges, accumulated_degeneracies


def compute_unique_fused_charges(
    charges: List[BaseCharge], flows: Union[np.ndarray,
                                            List[bool]]) -> BaseCharge:
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


def compute_num_nonzero(charges: List[BaseCharge],
                        flows: Union[np.ndarray, List[bool]]) -> int:
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
  res = accumulated_charges == accumulated_charges.identity_charges()
  if np.any(res) > 0:
    return np.squeeze(accumulated_degeneracies[res][0])
  return 0


def reduce_charges(charges: List[BaseCharge],
                   flows: Union[np.ndarray, List[bool]],
                   target_charges: BaseCharge,
                   return_locations: Optional[bool] = False,
                   return_unique: bool = False,
                   return_type: str = 'labels',
                   strides: Optional[np.ndarray] = None) -> Any:
  """
  Add quantum numbers arising from combining two or more charges into 
  a single index, keeping only the quantum numbers that appear in 
  `target_charges`. If return_type == 'labels', function returns
  charge-labels in an array `returned_labels` instead of the charges 
  themselves. The labels are the integer numbers that indicate the 
  charge in the an array of sorted unique charges.
  For example, if `fused_charges` is an np.ndarray of single integer 
  charges, and `unique, labels = np.unique(fused_charges, return_inverse=True)`, 
  the values in the returned array correspond to charges in `unique`, i.e.
  `actual_charges = unique[returned_labels]`
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
      retured locations of the kept elements. Defaults to trivial strides 
      (based on row major order).
  Returns:
    Union[BaseCharge, np.ndarray]: If `return_type=='labels' 
      the integer labels of the fused charges. If `return_type=='charges' 
      the reduced charges.
    np.ndarray: Locations of the fused BaseCharge charges that were kept.
  """
  tensor_dims = [len(c) for c in charges]

  if len(charges) == 1:
    # reduce single index
    if strides is None:
      strides = np.array([1], dtype=SIZE_T)
    return charges[0].dual(flows[0]).reduce(
        target_charges,
        return_locations=return_locations,
        return_type=return_type,
        return_unique=return_unique,
        strides=strides[0])
  # find size-balanced partition of charges
  partition = _find_best_partition(tensor_dims)

  # compute quantum numbers for each partition
  left_charges = fuse_charges(charges[:partition], flows[:partition])
  right_charges = fuse_charges(charges[partition:], flows[partition:])
  left_unique_charges, left_charge_labels = left_charges.unique(
      return_inverse=True)
  right_unique_charges, right_charge_labels = right_charges.unique(
      return_inverse=True)
  combined_charges = fuse_charges([left_unique_charges, right_unique_charges],
                                  [False, False])

  #special case of empty charges
  #pylint: disable=unsubscriptable-object
  if len(combined_charges) == 0 or len(left_charges) == 0 or len(
      right_charges) == 0:
    if return_type == 'labels':
      res = np.empty(0, dtype=charges[0].label_dtype)
    elif return_type == 'charges':
      res = charges[0].__new__(type(charges[0]))
      newcharges = [
          np.empty(0, dtype=combined_charges.charges[n].dtype)
          for n in range(len(combined_charges.charges))
      ]
      res.__init__(
          charges=newcharges,
          charge_types=combined_charges.charge_types,
          original_dtypes=combined_charges.original_dtypes,
          charge_indices=combined_charges.charge_indices)
    else:
      raise ValueError(f"unrecognized value {return_type} for return_type."
                       f" Allowed values are 'labels' anbd 'charges'")

    if return_locations:
      if return_unique:
        out = res, np.empty(0, dtype=SIZE_T), res
      else:
        out = res, np.empty(0, dtype=SIZE_T)
    else:
      if return_unique:
        out = res, res
      else:
        out = res
    return out

  unique_combined_charges, comb_labels = combined_charges.unique(
      return_inverse=True)

  num_unique = len(unique_combined_charges)
  num_left_unique = len(left_unique_charges)
  num_right_unique = len(right_unique_charges)
  # intersect combined charges and target charges
  reduced_unique_charges, label_to_unique, _ = unique_combined_charges.intersect(# pylint: disable=line-too-long
      target_charges, return_indices=True)

  map_to_kept = -np.ones(num_unique, dtype=charges[0].label_dtype)
  map_to_kept[label_to_unique] = np.arange(
      len(label_to_unique), dtype=charges[0].label_dtype)

  # new_comb_labels is a matrix of shape
  # (left_charges.num_unique, right_charges.num_unique)
  # each row new_comb_labels[n,:] contains integers values.
  # Positions where values > 0
  # denote labels of right-charges that are kept.
  label_table = map_to_kept[comb_labels].reshape(
      [num_left_unique, num_right_unique])
  reduced_rows = [0] * num_left_unique

  for n in range(num_left_unique):
    temp_label = label_table[n, right_charge_labels]
    reduced_rows[n] = temp_label[temp_label >= 0]

  reduced_labels = np.concatenate([reduced_rows[n] for n in left_charge_labels])
  if return_type == 'labels':
    res = reduced_labels
  elif return_type == 'charges':
    res = reduced_unique_charges[reduced_labels]
  else:
    raise ValueError(f"unrecognized value {return_type} for return_type."
                     f" Allowed values are 'labels' anbd 'charges'")

  if return_locations:
    row_locs = [0] * num_left_unique
    if strides is not None:
      # computed locations based on non-trivial strides
      row_pos = fuse_stride_arrays(tensor_dims[:partition], strides[:partition])
      col_pos = fuse_stride_arrays(tensor_dims[partition:], strides[partition:])
    for n in range(num_left_unique):
      temp_label = label_table[n, right_charge_labels]
      temp_keep = temp_label >= 0
      if strides is not None:
        row_locs[n] = col_pos[temp_keep]
      else:
        row_locs[n] = np.where(temp_keep)[0]

    if strides is not None:
      reduced_locs = np.concatenate([
          row_pos[n] + row_locs[left_charge_labels[n]]
          for n in range(left_charges.dim)
      ])
    else:
      reduced_locs = np.concatenate([
          n * right_charges.dim + row_locs[left_charge_labels[n]]
          for n in range(left_charges.dim)
      ])
    if return_unique:
      return res, reduced_locs, reduced_unique_charges
    return res, reduced_locs
  if return_unique:
    return res, reduced_unique_charges

  return res

def _find_diagonal_sparse_blocks(
    charges: List[BaseCharge],
    flows: Union[np.ndarray, List[bool]],
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
  if partition in (0, len(charges)):
    # special cases (matrix of trivial height or width)
    num_nonzero = compute_num_nonzero(charges, flows)
    block_maps = [np.arange(0, num_nonzero, dtype=SIZE_T).ravel()]
    block_charges = charges[0].identity_charges(dim=num_nonzero)
    block_dims = np.array([[1], [num_nonzero]])

    if partition == len(flows):
      block_dims = np.flipud(block_dims)

    return block_maps, block_charges, block_dims
  unique_row_charges, row_degen = compute_fused_charge_degeneracies(
      charges[:partition], flows[:partition])
  unique_col_charges, col_degen = compute_fused_charge_degeneracies(
      charges[partition:], np.logical_not(flows[partition:]))

  block_charges, row_to_block, col_to_block = unique_row_charges.intersect(
      unique_col_charges, return_indices=True)
  num_blocks = len(block_charges)
  if num_blocks == 0:
    return [], charges[0][np.empty(0, dtype=SIZE_T)], np.empty((2, 0),
                                                               dtype=SIZE_T)

  # calculate number of non-zero elements in each row of the matrix
  row_charge_labels, block_charges = reduce_charges(
      charges[:partition],
      flows[:partition],
      block_charges,
      return_type='labels',
      return_unique=True)

  row_num_nz = col_degen[col_to_block[row_charge_labels]]
  cumulate_num_nz = np.insert(np.cumsum(row_num_nz[0:-1]), 0, 0).astype(SIZE_T)
  # calculate mappings for the position in datavector of each block
  if num_blocks < 15:
    # faster method for small number of blocks
    row_locs = np.concatenate([
        (row_charge_labels == n) for n in range(num_blocks)
    ]).reshape(num_blocks, len(row_charge_labels))
  else:
    # faster method for large number of blocks
    row_locs = np.zeros([num_blocks, len(row_charge_labels)], dtype=bool)
    row_locs[row_charge_labels,
             np.arange(len(row_charge_labels))] = np.ones(
                 len(row_charge_labels), dtype=bool)
  block_dims = np.array(
      [[row_degen[row_to_block[n]], col_degen[col_to_block[n]]]
       for n in range(num_blocks)],
      dtype=SIZE_T).T

  #pylint: disable=unsubscriptable-object
  block_maps = [
      np.ravel(cumulate_num_nz[row_locs[n, :]][:, None] +
               np.arange(block_dims[1, n])[None, :]) for n in range(num_blocks)
  ]

  return block_maps, block_charges, block_dims


def _find_transposed_diagonal_sparse_blocks(
    charges: List[BaseCharge],
    flows: Union[np.ndarray, List[bool]],
    tr_partition: int,
    order: Optional[Union[List, np.ndarray]] = None
) -> Tuple[List, BaseCharge, np.ndarray]:
  """
  Find the diagonal blocks of a transposed tensor with 
  meta-data `charges` and `flows`. `charges` and `flows` 
  are the charges and flows of the untransposed tensor, 
  `order` is the final transposition, and `tr_partition`
  is the partition of the transposed tensor according to 
  which the diagonal blocks should be found.
  Args:
    charges: List of `BaseCharge`, one for each leg of a tensor. 
    flows: A list of bool, one for each leg of a tensor.
      with values `False` or `True` denoting inflowing and 
      outflowing charge direction, respectively.
    tr_partition: Location of the transposed tensor partition 
      (i.e. such that the tensor is viewed as a matrix between 
      `charges[order[:partition]]` and `charges[order[partition:]]`).
    order: Order with which to permute the tensor axes. 
  Returns:
    block_maps (List[np.ndarray]): list of integer arrays, which each 
      containing the location of a symmetry block in the data vector.
    block_qnums (BaseCharge): The charges of the corresponding blocks.
    block_dims (np.ndarray): 2-by-m array of matrix dimensions of each block.
  """
  flows = np.asarray(flows)
  # hash_val = compute_hash(charges, flows, tr_partition, order)
  # if hash_val in _CACHED_BLOCKS:
  #   return _CACHED_BLOCKS[hash_val]
  
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

  orig_row_charges = fuse_charges(charges[:orig_partition],
                                  flows[:orig_partition])
  orig_col_charges = fuse_charges(charges[orig_partition:],
                                  np.logical_not(flows[orig_partition:]))

  orig_unique_row_charges, row_labels = orig_row_charges.unique(
      return_inverse=True)
  orig_unique_col_charges, col_labels, orig_col_degen = orig_col_charges.unique(
      return_inverse=True, return_counts=True)


  orig_block_charges, common_row_labels, common_col_labels = orig_unique_row_charges.intersect( # pylint: disable=line-too-long
      orig_unique_col_charges, return_indices=True)
  orig_num_blocks = len(orig_block_charges)
  if orig_num_blocks == 0:
    # special case: trivial number of non-zero elements
    return [], charges[0][np.empty(0, dtype=np.int16)], np.empty((2, 0),
                                                                 dtype=SIZE_T)
  # inv_row_map is positive for the row labels that are kept, and -1 else.
  # the -1 is used to index the last value of an array below
  inv_row_map = np.full(
      len(orig_unique_row_charges), fill_value=-1, dtype=charges[0].label_dtype)
  inv_row_map[common_row_labels] = np.arange(
      len(common_row_labels), dtype=charges[0].label_dtype)

  # all_degens is an array of len(row_labels) that holds the degeneracies
  # (i.e. the number of non-zero elements) in that row
  all_degens = np.append(orig_col_degen[common_col_labels],
                         0)[inv_row_map[row_labels]]
  all_cumul_degens = np.cumsum(np.insert(all_degens[:-1], 0, 0)).astype(SIZE_T)

  # dense_to_sparse is an array of length of the column-size
  # for each kept charge `c` in the column-charges it labels the
  # positions where `col_labels==label-of(c)` from 0 to the number of times
  # that `c` appears in the column charges.
  dense_to_sparse = np.empty(orig_width, dtype=SIZE_T)
  for n in range(orig_num_blocks):
    dense_to_sparse[col_labels == common_col_labels[n]] = np.arange(
        orig_col_degen[common_col_labels[n]], dtype=SIZE_T)

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
    identity_charges = charges[0].identity_charges(dim=1)
    block_charges, new_row_map, new_col_map = identity_charges.intersect(
        unique_col_qnums, return_indices=True)
    block_dims = np.array([[1], new_col_degen[new_col_map]], dtype=SIZE_T)
    col_charge_labels, col_locs = reduce_charges(
        new_col_charges,
        np.logical_not(new_col_flows),
        block_charges,
        return_type='labels',
        return_locations=True,
        strides=new_strides[tr_partition:])
    #return_unique=True)
    # find location of blocks in transposed tensor (w.r.t positions in original)
    #pylint: disable=no-member
    orig_row_posR, orig_col_posR = np.divmod(col_locs[col_charge_labels == 0],
                                             orig_width)
    block_maps = [(all_cumul_degens[orig_row_posR] +
                   dense_to_sparse[orig_col_posR]).ravel()]

  elif tr_partition == len(charges):
    # special case: reshape into col vector

    # compute qnums of row/cols in transposed tensor
    unique_row_qnums, new_row_degen = compute_fused_charge_degeneracies(
        new_row_charges, new_row_flows)
    identity_charges = charges[0].identity_charges(dim=1)
    block_charges, new_row_map, new_col_map = unique_row_qnums.intersect(
        identity_charges, return_indices=True)
    # block_qnums, new_row_map, new_col_map = intersect(
    #     unique_row_qnums.unique_charges,
    #     identity_charges.unique_charges,
    #     axis=1,
    #     return_indices=True)
    block_dims = np.array([new_row_degen[new_row_map], [1]], dtype=SIZE_T)
    row_charge_labels, row_locs = reduce_charges(
        new_row_charges,
        new_row_flows,
        block_charges,
        return_type='labels',
        return_locations=True,
        strides=new_strides[:tr_partition])

    # find location of blocks in transposed tensor (w.r.t positions in original)
    #pylint: disable=no-member
    orig_row_posL, orig_col_posL = np.divmod(row_locs[row_charge_labels == 0],
                                             orig_width)
    block_maps = [(all_cumul_degens[orig_row_posL] +
                   dense_to_sparse[orig_col_posL]).ravel()]

  else:
    unique_row_qnums, new_row_degen = compute_fused_charge_degeneracies(
        new_row_charges, new_row_flows)
    unique_col_qnums, new_col_degen = compute_fused_charge_degeneracies(
        new_col_charges, np.logical_not(new_col_flows))
    block_charges, new_row_map, new_col_map = unique_row_qnums.intersect(
        unique_col_qnums, return_indices=True)

    block_dims = np.array(
        [new_row_degen[new_row_map], new_col_degen[new_col_map]], dtype=SIZE_T)
    num_blocks = len(new_row_map)
    row_charge_labels, row_locs = reduce_charges(
        new_row_charges,
        new_row_flows,
        block_charges,
        return_type='labels',
        return_locations=True,
        strides=new_strides[:tr_partition])
    col_charge_labels, col_locs = reduce_charges(
        new_col_charges,
        np.logical_not(new_col_flows),
        block_charges,
        return_type='labels',
        return_locations=True,
        strides=new_strides[tr_partition:])

    block_maps = [0] * num_blocks
    for n in range(num_blocks):
      #pylint: disable=no-member
      orig_row_posL, orig_col_posL = np.divmod(row_locs[row_charge_labels == n],
                                               orig_width)
      #pylint: disable=no-member
      orig_row_posR, orig_col_posR = np.divmod(col_locs[col_charge_labels == n],
                                               orig_width)
      block_maps[n] = (
          all_cumul_degens[np.add.outer(orig_row_posL, orig_row_posR)] +
          dense_to_sparse[np.add.outer(orig_col_posL, orig_col_posR)]).ravel()
  return block_maps, block_charges, block_dims
  #_CACHED_BLOCKS[hash_val] = (block_maps, block_charges, block_dims)
  #return _CACHED_BLOCKS[hash_val]

