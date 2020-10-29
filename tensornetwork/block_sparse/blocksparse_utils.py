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
from tensornetwork.block_sparse.charge import (fuse_charges, BaseCharge,
                                               fuse_ndarray_charges,
                                               charge_equal)
from tensornetwork.block_sparse.utils import (fuse_stride_arrays, unique,
                                              fuse_degeneracies, intersect,
                                              _find_best_partition,
                                              fuse_ndarrays)
from tensornetwork.block_sparse.caching import get_cacher
from typing import List, Union, Any, Tuple, Optional, Sequence, Callable
from tensornetwork.block_sparse.sizetypes import SIZE_T

Tensor = Any


def _data_initializer(
    numpy_initializer: Callable, comp_num_elements: Callable,
    indices: Sequence[Index], *args, **kwargs
) -> Tuple[np.ndarray, List[BaseCharge], List[bool], List[List[int]]]:
  """
  Initialize a 1d np.ndarray using `numpy_initializer` function.

  Args:
    numpy_initializer: Callable, should return a 1d np.ndarray.
      Function call signature: `numpy_initializer(*args, **kwargs)`.
    comp_num_elements: Callable, computes the number of elements of
      the returned 1d np.ndarray, using  `numel = comp_num_elements(indices)`.
    indices: List if `Index` objects.
    *args, **kwargs: Arguments to `numpy_initializer`.

  Returns:
    np.ndarray: An initialized numpy array.
    List[BaseCharge]: A list containing the flattened charges in `indices`
    List[bool]: The flattened flows of `indices`.
    List[List]: A list of list of int, the order information needed to
      initialize a BlockSparseTensor.
  """
  charges, flows = get_flat_meta_data(indices)
  num_elements = comp_num_elements(charges, flows)
  tmp = np.append(0, np.cumsum([len(i.flat_charges) for i in indices]))
  order = [list(np.arange(tmp[n], tmp[n + 1])) for n in range(len(tmp) - 1)]
  data = numpy_initializer(num_elements, *args, **kwargs)
  return data, charges, flows, order


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


def compute_sparse_lookup(
    charges: List[BaseCharge], flows: Union[np.ndarray, List[bool]],
    target_charges: BaseCharge) -> Tuple[np.ndarray, BaseCharge, np.ndarray]:
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

  fused_charges = fuse_charges(charges, flows)
  unique_charges, inverse = unique(fused_charges.charges, return_inverse=True)
  _, label_to_unique, _ = intersect(
      unique_charges, target_charges.charges, return_indices=True)
  # _, label_to_unique, _ = unique_charges.intersect(
  #     target_charges, return_indices=True)
  tmp = np.full(
      unique_charges.shape[0], fill_value=-1, dtype=charges[0].label_dtype)
  obj = charges[0].__new__(type(charges[0]))
  obj.__init__(
      charges=unique_charges,
      charge_labels=None,
      charge_types=charges[0].charge_types)

  tmp[label_to_unique] = label_to_unique
  lookup = tmp[inverse]
  lookup = lookup[lookup >= 0]

  return lookup, obj, np.sort(label_to_unique)


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

  partition = _find_best_partition(dims)
  fused_left = fuse_charges(charges[:partition], flows[:partition])
  fused_right = fuse_charges(charges[partition:], flows[partition:])
  left_unique, left_degens = fused_left.unique(return_counts=True)
  right_unique, right_degens = fused_right.unique(return_counts=True)
  fused = left_unique + right_unique
  unique_charges, charge_labels = fused.unique(return_inverse=True)
  fused_degeneracies = fuse_degeneracies(left_degens, right_degens)
  new_ord = np.argsort(charge_labels)
  all_degens = np.cumsum(fused_degeneracies[new_ord])
  cum_degens = all_degens[np.flatnonzero(np.diff(charge_labels[new_ord]))]
  final_degeneracies = np.append(cum_degens, all_degens[-1]) - np.append(
      0, cum_degens)
  return unique_charges, final_degeneracies

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
  res = accumulated_charges == accumulated_charges.identity_charges(dim=1)
  nz_inds = np.nonzero(res)[0]

  if len(nz_inds) > 0:
    return np.squeeze(accumulated_degeneracies[nz_inds][0])
  return 0


def reduce_charges(charges: List[BaseCharge],
                   flows: Union[np.ndarray, List[bool]],
                   target_charges: np.ndarray,
                   return_locations: Optional[bool] = False,
                   strides: Optional[np.ndarray] = None) -> Any:
  """
  Add quantum numbers arising from combining two or more charges into a
  single index, keeping only the quantum numbers that appear in 
  `target_charges`. Equilvalent to using "combine_charges" followed 
  by "reduce", but is generally much more efficient.
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
    BaseCharge: the fused index after reduction.
    np.ndarray: Locations of the fused BaseCharge charges that were kept.
  """

  tensor_dims = [len(c) for c in charges]

  if len(charges) == 1:
    # reduce single index
    if strides is None:
      strides = np.array([1], dtype=SIZE_T)
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
  #special case of empty charges
  #pylint: disable=unsubscriptable-object
  if (comb_qnums.shape[0] == 0) or (len(left_ind.charge_labels) == 0) or (len(
      right_ind.charge_labels) == 0):
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(
        np.empty((0, charges[0].num_symmetries), dtype=charges[0].dtype),
        np.empty(0, dtype=charges[0].label_dtype), charges[0].charge_types)
    if return_locations:
      return obj, np.empty(0, dtype=SIZE_T)
    return obj

  unique_comb_qnums, comb_labels = unique(comb_qnums, return_inverse=True)
  num_unique = unique_comb_qnums.shape[0]

  # intersect combined qnums and target_charges
  reduced_qnums, label_to_unique, _ = intersect(
      unique_comb_qnums, target_charges, axis=0, return_indices=True)
  map_to_kept = -np.ones(num_unique, dtype=charges[0].label_dtype)
  map_to_kept[label_to_unique] = np.arange(len(label_to_unique))
  # new_comb_labels is a matrix of shape
  # (left_ind.num_unique, right_ind.num_unique)
  # each row new_comb_labels[n,:] contains integers values.
  # Positions where values > 0
  # denote labels of right-charges that are kept.
  new_comb_labels = map_to_kept[comb_labels].reshape(
      [left_ind.num_unique, right_ind.num_unique])
  reduced_rows = [0] * left_ind.num_unique

  for n in range(left_ind.num_unique):
    temp_label = new_comb_labels[n, right_ind.charge_labels]
    reduced_rows[n] = temp_label[temp_label >= 0]

  reduced_labels = np.concatenate(
      [reduced_rows[n] for n in left_ind.charge_labels])
  obj = charges[0].__new__(type(charges[0]))
  obj.__init__(reduced_qnums, reduced_labels, charges[0].charge_types)

  if return_locations:
    row_locs = [0] * left_ind.num_unique
    if strides is not None:
      # computed locations based on non-trivial strides
      row_pos = fuse_stride_arrays(tensor_dims[:partition], strides[:partition])
      col_pos = fuse_stride_arrays(tensor_dims[partition:], strides[partition:])
    for n in range(left_ind.num_unique):
      temp_label = new_comb_labels[n, right_ind.charge_labels]
      temp_keep = temp_label >= 0
      if strides is not None:
        row_locs[n] = col_pos[temp_keep]
      else:
        row_locs[n] = np.where(temp_keep)[0]

    if strides is not None:
      reduced_locs = np.concatenate([
          row_pos[n] + row_locs[left_ind.charge_labels[n]]
          for n in range(left_ind.dim)
      ])
    else:
      reduced_locs = np.concatenate([
          n * right_ind.dim + row_locs[left_ind.charge_labels[n]]
          for n in range(left_ind.dim)
      ])
    return obj, reduced_locs

  return obj


def _find_diagonal_sparse_blocks(
    charges: List[BaseCharge], flows: Union[np.ndarray, List[bool]],
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
  cacher = get_cacher()
  if cacher.do_caching:
    hash_val = _to_string(charges, flows, partition, list(range(len(charges))))
    if hash_val in cacher.cache:
      return cacher.cache[hash_val]

  num_inds = len(charges)
  if partition in (0, num_inds):
    # special cases (matrix of trivial height or width)
    num_nonzero = compute_num_nonzero(charges, flows)
    block_maps = [np.arange(0, num_nonzero, dtype=SIZE_T).ravel()]
    block_qnums = charges[0].identity_charges(dim=1).charges
    block_dims = np.array([[1], [num_nonzero]])

    if partition == len(flows):
      block_dims = np.flipud(block_dims)

    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(block_qnums, np.arange(1, dtype=charges[0].label_dtype),
                 charges[0].charge_types)

    return block_maps, obj, block_dims

  unique_row_qnums, row_degen = compute_fused_charge_degeneracies(
      charges[:partition], flows[:partition])
  unique_col_qnums, col_degen = compute_fused_charge_degeneracies(
      charges[partition:], np.logical_not(flows[partition:]))

  block_qnums, row_to_block, col_to_block = intersect(
      unique_row_qnums.unique_charges,
      unique_col_qnums.unique_charges,
      axis=0,
      return_indices=True)

  num_blocks = block_qnums.shape[0]
  if num_blocks == 0:
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(
        np.zeros((0, charges[0].num_symmetries), dtype=charges[0].dtype),
        np.arange(0, dtype=charges[0].label_dtype), charges[0].charge_types)

    return [], obj, np.empty((2, 0), dtype=SIZE_T)

  # calculate number of non-zero elements in each row of the matrix
  row_ind = reduce_charges(charges[:partition], flows[:partition], block_qnums)
  row_num_nz = col_degen[col_to_block[row_ind.charge_labels]]
  cumulate_num_nz = np.insert(np.cumsum(row_num_nz[0:-1]), 0, 0).astype(SIZE_T)
  # calculate mappings for the position in datavector of each block
  if num_blocks < 15:
    # faster method for small number of blocks
    row_locs = np.concatenate([
        (row_ind.charge_labels == n) for n in range(num_blocks)
    ]).reshape(num_blocks, row_ind.dim)
  else:
    # faster method for large number of blocks
    row_locs = np.zeros([num_blocks, row_ind.dim], dtype=bool)
    row_locs[row_ind.charge_labels,
             np.arange(row_ind.dim)] = np.ones(
                 row_ind.dim, dtype=bool)
  block_dims = np.array(
      [[row_degen[row_to_block[n]], col_degen[col_to_block[n]]]
       for n in range(num_blocks)],
      dtype=SIZE_T).T
  #pylint: disable=unsubscriptable-object
  block_maps = [
      np.ravel(cumulate_num_nz[row_locs[n, :]][:, None] +
               np.arange(block_dims[1, n])[None, :]) for n in range(num_blocks)
  ]
  obj = charges[0].__new__(type(charges[0]))
  obj.__init__(block_qnums,
               np.arange(block_qnums.shape[0], dtype=charges[0].label_dtype),
               charges[0].charge_types)
  if cacher.do_caching:
    cacher.cache[hash_val] = (block_maps, obj, block_dims)
    return cacher.cache[hash_val]
  return block_maps, obj, block_dims


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
  cacher = get_cacher()
  if cacher.do_caching:
    hash_val = _to_string(charges, flows, tr_partition, order)
    if hash_val in cacher.cache:
      return cacher.cache[hash_val]

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
      axis=0,
      return_indices=True)
  orig_num_blocks = orig_block_qnums.shape[0]
  if orig_num_blocks == 0:
    # special case: trivial number of non-zero elements
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(
        np.empty((0, charges[0].num_symmetries), dtype=charges[0].dtype),
        np.arange(0, dtype=charges[0].label_dtype), charges[0].charge_types)

    return [], obj, np.empty((2, 0), dtype=SIZE_T)

  orig_row_ind = fuse_charges(charges[:orig_partition], flows[:orig_partition])
  orig_col_ind = fuse_charges(charges[orig_partition:],
                              np.logical_not(flows[orig_partition:]))

  inv_row_map = -np.ones(
      orig_unique_row_qnums.unique_charges.shape[0],
      dtype=charges[0].label_dtype)
  inv_row_map[row_map] = np.arange(len(row_map), dtype=charges[0].label_dtype)

  all_degens = np.append(orig_col_degen[col_map],
                         0)[inv_row_map[orig_row_ind.charge_labels]]
  all_cumul_degens = np.cumsum(np.insert(all_degens[:-1], 0, 0)).astype(SIZE_T)
  dense_to_sparse = np.empty(orig_width, dtype=SIZE_T)
  for n in range(orig_num_blocks):
    dense_to_sparse[orig_col_ind.charge_labels == col_map[n]] = np.arange(
        orig_col_degen[col_map[n]], dtype=SIZE_T)

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
    block_qnums, new_row_map, new_col_map = intersect(
        identity_charges.unique_charges,
        unique_col_qnums.unique_charges,
        axis=0,
        return_indices=True)
    block_dims = np.array([[1], new_col_degen[new_col_map]], dtype=SIZE_T)
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
    obj.__init__(block_qnums,
                 np.arange(block_qnums.shape[0], dtype=charges[0].label_dtype),
                 charges[0].charge_types)

  elif tr_partition == len(charges):
    # special case: reshape into col vector

    # compute qnums of row/cols in transposed tensor
    unique_row_qnums, new_row_degen = compute_fused_charge_degeneracies(
        new_row_charges, new_row_flows)
    identity_charges = charges[0].identity_charges(dim=1)
    block_qnums, new_row_map, new_col_map = intersect(
        unique_row_qnums.unique_charges,
        identity_charges.unique_charges,
        axis=0,
        return_indices=True)
    block_dims = np.array([new_row_degen[new_row_map], [1]], dtype=SIZE_T)
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
    obj.__init__(block_qnums,
                 np.arange(block_qnums.shape[0], dtype=charges[0].label_dtype),
                 charges[0].charge_types)
  else:

    unique_row_qnums, new_row_degen = compute_fused_charge_degeneracies(
        new_row_charges, new_row_flows)

    unique_col_qnums, new_col_degen = compute_fused_charge_degeneracies(
        new_col_charges, np.logical_not(new_col_flows))
    block_qnums, new_row_map, new_col_map = intersect(
        unique_row_qnums.unique_charges,
        unique_col_qnums.unique_charges,
        axis=0,
        return_indices=True)
    block_dims = np.array(
        [new_row_degen[new_row_map], new_col_degen[new_col_map]], dtype=SIZE_T)
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
    obj.__init__(block_qnums,
                 np.arange(block_qnums.shape[0], dtype=charges[0].label_dtype),
                 charges[0].charge_types)

  if cacher.do_caching:
    cacher.cache[hash_val] = (block_maps, obj, block_dims)
    return cacher.cache[hash_val]
  return block_maps, obj, block_dims

def _to_string(charges: List[BaseCharge], flows: Union[np.ndarray, List],
               tr_partition: int, order: List[int]) -> str:
  """
  map the input arguments of _find_transposed_diagonal_sparse_blocks 
  to a string.
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
    str: The string representation of the input
  """
  return ''.join([str(c.charges.tostring()) for c in charges] + [
      str(np.array(flows).tostring()),
      str(tr_partition),
      str(np.array(order, dtype=np.int16).tostring())
  ])
