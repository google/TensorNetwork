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
from tensornetwork.backends import backend_factory
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index
from tensornetwork.block_tensor.charge import fuse_degeneracies, fuse_charges, fuse_degeneracies, BaseCharge, fuse_ndarray_charges, intersect
import numpy as np
import scipy as sp
import itertools
import time
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable, Sequence
Tensor = Any


def get_flat_order(indices, order):
  flat_charges, _ = get_flat_meta_data(indices)
  flat_labels = np.arange(len(flat_charges))
  cum_num_legs = np.append(0, np.cumsum([len(i.flat_charges) for i in indices]))
  flat_order = np.concatenate(
      [flat_labels[cum_num_legs[n]:cum_num_legs[n + 1]] for n in order])

  return flat_order


def get_flat_meta_data(indices):
  charges = []
  flows = []
  for i in indices:
    flows.extend(i.flat_flows)
    charges.extend(i.flat_charges)
  return charges, flows


def fuse_stride_arrays(dims: np.ndarray, strides: np.ndarray) -> np.ndarray:
  return fuse_ndarrays([
      np.arange(0, strides[n] * dims[n], strides[n], dtype=np.uint32)
      for n in range(len(dims))
  ])


def compute_sparse_lookup(charges: List[BaseCharge], flows: Iterable[bool],
                          target_charges: BaseCharge) -> np.ndarray:
  """
  Compute lookup table for looking up how dense index positions map 
  to sparse index positions for the diagonal blocks a symmetric matrix.
  Args:
    charges:
    flows:
    target_charges:

  """
  fused_charges = fuse_charges(charges, flows)
  unique_charges, inverse, degens = fused_charges.unique(
      return_inverse=True, return_counts=True)
  common_charges, label_to_unique, label_to_target = unique_charges.intersect(
      target_charges, return_indices=True)

  tmp = np.full(len(unique_charges), fill_value=-1, dtype=np.int16)
  tmp[label_to_unique] = label_to_unique
  lookup = tmp[inverse]
  vec = np.empty(len(fused_charges), dtype=np.uint32)
  for n in label_to_unique:
    vec[lookup == n] = np.arange(degens[n])
  return vec


def _get_strides(dims):
  return np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))


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
    fused_arrays = np.ravel(np.add.outer(fused_arrays, arrays[n]))
  return fused_arrays


def _find_best_partition(dims: Iterable[int]) -> int:
  """

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
    Union[BaseCharge, BaseCharge]:  The unique fused charges.
    np.ndarray of integers: The degeneracies of each unqiue fused charge.
  """
  if len(charges) == 1:
    return (charges[0] * flows[0]).unique(return_counts=True)

  # get unique charges and their degeneracies on the first leg.
  # We are fusing from "left" to "right".
  accumulated_charges, accumulated_degeneracies = (
      charges[0] * flows[0]).unique(return_counts=True)
  for n in range(1, len(charges)):
    leg_charges, leg_degeneracies = charges[n].unique(return_counts=True)
    fused_charges = accumulated_charges + leg_charges * flows[n]
    fused_degeneracies = fuse_degeneracies(accumulated_degeneracies,
                                           leg_degeneracies)
    accumulated_charges = fused_charges.unique()
    accumulated_degeneracies = np.empty(
        len(accumulated_charges), dtype=np.uint32)

    for n in range(len(accumulated_charges)):
      accumulated_degeneracies[n] = np.sum(fused_degeneracies[
          fused_charges.charge_labels == accumulated_charges.charge_labels[n]])

  return accumulated_charges, accumulated_degeneracies


def compute_unique_fused_charges(
    charges: List[BaseCharge],
    flows: List[Union[bool, int]]) -> Tuple[BaseCharge, np.ndarray]:
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
  res = accumulated_charges == accumulated_charges.identity_charges
  nz_inds = np.nonzero(res)[0]
  if len(nz_inds) > 0:
    return np.squeeze(accumulated_degeneracies[nz_inds][0])
  return 0


def reduce_charges(
    charges: List[BaseCharge],
    flows: Iterable[bool],
    target_charges: np.ndarray,
    return_locations: Optional[bool] = False,
    strides: Optional[np.ndarray] = None) -> Tuple[BaseCharge, np.ndarray]:
  """
  Add quantum numbers arising from combining two or more charges into a
  single index, keeping only the quantum numbers that appear in 'target_charges'.
  Equilvalent to using "combine_charges" followed by "reduce", but is
  generally much more efficient.
  Args:
    charges (List[SymIndex]): list of SymIndex.
    flows (np.ndarray): vector of bools describing index orientations.
    target_charges (np.ndarray): n-by-m array describing qauntum numbers of the
      qnums which should be kept with 'n' the number of symmetries.
    return_locations (bool, optional): if True then return the location of the kept
      values of the fused charges
    strides (np.ndarray, optional): index strides with which to compute the
      return_locations of the kept elements. Defaults to trivial strides (based on
      row major order) if ommitted.
  Returns:
    SymIndex: the fused index after reduction.
    np.ndarray: locations of the fused SymIndex qnums that were kept.
  """

  num_inds = len(charges)
  tensor_dims = [len(c) for c in charges]

  if len(charges) == 1:
    # reduce single index
    if strides is None:
      strides = np.array([1], dtype=np.uint32)
    return charges[0].dual(flows[0]).reduce(
        target_charges, return_locations=return_locations, strides=strides[0])

  else:
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
    reduced_qnums, label_to_unique, label_to_kept = intersect(
        unique_comb_qnums, target_charges, axis=1, return_indices=True)
    map_to_kept = -np.ones(num_unique, dtype=np.int16)
    for n in range(len(label_to_unique)):
      map_to_kept[label_to_unique[n]] = n
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

    else:  # trivial strides
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

  else:
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
    partition: int) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Find the location of all non-trivial symmetry blocks from the data vector of
  of BlockSparseTensor (when viewed as a matrix across some prescribed index 
  bi-partition).
  Args:
    charges (List[SymIndex]): list of SymIndex.
    flows (np.ndarray): vector of bools describing index orientations.
    partition_loc (int): location of tensor partition (i.e. such that the 
      tensor is viewed as a matrix between first partition_loc charges and 
      the remaining charges).
  Returns:
    block_maps (List[np.ndarray]): list of integer arrays, which each 
      containing the location of a symmetry block in the data vector.
    block_qnums (np.ndarray): n-by-m array describing qauntum numbers of each 
      block, with 'n' the number of symmetries and 'm' the number of blocks.
    block_dims (np.ndarray): 2-by-m array describing the dims each block, 
      with 'm' the number of blocks).
  """
  num_inds = len(charges)
  num_syms = charges[0].num_symmetries

  if (partition == 0) or (partition == num_inds):
    # special cases (matrix of trivial height or width)
    num_nonzero = compute_num_nonzero(charges, flows)
    block_maps = [np.arange(0, num_nonzero, dtype=np.uint64).ravel()]
    block_qnums = np.zeros([num_syms, 1], dtype=np.int16)
    block_dims = np.array([[1], [num_nonzero]])

    if partition == len(flows):
      block_dims = np.flipud(block_dims)

    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(block_qnums, np.arange(block_qnums.shape[1], dtype=np.int16),
                 charges[0].charge_types)

    return block_maps, obj, block_dims

  else:
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

    else:
      # calculate number of non-zero elements in each row of the matrix
      row_ind = reduce_charges(charges[:partition], flows[:partition],
                               block_qnums)
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
      block_maps = [(cumulate_num_nz[row_locs[n, :]][:, None] + np.arange(
          block_dims[1, n])[None, :]).ravel() for n in range(num_blocks)]
      obj = charges[0].__new__(type(charges[0]))
      obj.__init__(block_qnums, np.arange(block_qnums.shape[1], dtype=np.int16),
                   charges[0].charge_types)

      return block_maps, obj, block_dims


def _find_transposed_diagonal_sparse_blocks(
    charges: List[BaseCharge],
    flows: np.ndarray,
    tr_partition: int,
    order: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  
  Args:
    charges (List[BaseCharge]): List of charges.
    flows (np.ndarray): vector of bools describing index orientations.
    tr_partition (int): location of tensor partition (i.e. such that the 
      tensor is viewed as a matrix between first partition charges and 
      the remaining charges).
    order (np.ndarray): order with which to permute the tensor axes. 
  Returns:
    block_maps (List[np.ndarray]): list of integer arrays, which each 
      containing the location of a symmetry block in the data vector.
    block_qnums (np.ndarray): n-by-m array describing qauntum numbers of each 
      block, with 'n' the number of symmetries and 'm' the number of blocks.
    block_dims (np.ndarray): 2-by-m array describing the dims each block, 
      with 'm' the number of blocks).
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
    return [], np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)

  orig_row_ind = fuse_charges(charges[:orig_partition], flows[:orig_partition])
  orig_col_ind = fuse_charges(charges[orig_partition:],
                              np.logical_not(flows[orig_partition:]))

  inv_row_map = -np.ones(
      orig_unique_row_qnums.unique_charges.shape[1], dtype=np.int16)
  for n in range(len(row_map)):
    inv_row_map[row_map[n]] = n

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

  if (tr_partition == 0):
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
    orig_row_posR, orig_col_posR = np.divmod(
        col_locs[col_ind.charge_labels == 0], orig_width)
    block_maps = [(all_cumul_degens[orig_row_posR] +
                   dense_to_sparse[orig_col_posR]).ravel()]
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(block_qnums, np.arange(block_qnums.shape[1], dtype=np.int16),
                 charges[0].charge_types)

  elif (tr_partition == len(charges)):
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
      orig_row_posL, orig_col_posL = np.divmod(
          row_locs[row_ind.charge_labels == n], orig_width)
      orig_row_posR, orig_col_posR = np.divmod(
          col_locs[col_ind.charge_labels == n], orig_width)
      block_maps[n] = (
          all_cumul_degens[np.add.outer(orig_row_posL, orig_row_posR)] +
          dense_to_sparse[np.add.outer(orig_col_posL, orig_col_posR)]).ravel()
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(block_qnums, np.arange(block_qnums.shape[1], dtype=np.int16),
                 charges[0].charge_types)

  return block_maps, obj, block_dims


class BlockSparseTensor:
  """
  Minimal class implementation of block sparsity.
  The class design follows Glen's proposal (Design 0).
  The class currently only supports a single U(1) symmetry
  and only numpy.ndarray.

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
    num_non_zero_elements = compute_num_nonzero(self.flat_charges,
                                                self.flat_flows)

    if num_non_zero_elements != len(data.flat):
      raise ValueError("number of tensor elements {} defined "
                       "by `charges` is different from"
                       " len(data)={}".format(num_non_zero_elements,
                                              len(data.flat)))

    self.data = np.asarray(data.flat)  #do not copy data

  def copy(self):
    return BlockSparseTensor(self.data.copy(), [i.copy() for i in self.indices])

  def todense(self) -> np.ndarray:
    """
    Map the sparse tensor to dense storage.
    
    """
    out = np.asarray(np.zeros(self.dense_shape, dtype=self.dtype).flat)
    charges = self.flat_charges
    out[np.nonzero(
        fuse_charges(charges, self.flat_flows) == charges[0].identity_charges)
        [0]] = self.data
    return np.reshape(out, self.dense_shape)

  @property
  def ndim(self):
    return len(self.indices)

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
    charges, flows = get_flat_meta_data(indices)
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
    charges, flows = get_flat_meta_data(indices)
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
    charges, flows = get_flat_meta_data(indices)
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
  def rank(self):
    return len(self.indices)

  @property
  def dense_shape(self) -> Tuple:
    """
    The dense shape of the tensor.
    Returns:
      Tuple: A tuple of `int`.
    """
    return tuple([i.dim for i in self.indices])

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

  @property
  def flat_charges(self):
    flat = []
    for i in self.indices:
      flat.extend(i.flat_charges)
    return flat

  @property
  def flat_flows(self):
    flat = []
    for i in self.indices:
      flat.extend(i.flat_flows)
    return flat

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
      return BlockSparseTensor(self.data, self.indices)
    flat_charges, flat_flows = get_flat_meta_data(self.indices)
    flat_order = get_flat_order(self.indices, order)
    print(flat_order)
    tr_partition = _find_best_partition(
        [len(flat_charges[n]) for n in flat_order])

    tr_sparse_blocks, tr_charges, tr_shapes = _find_transposed_diagonal_sparse_blocks(
        flat_charges, flat_flows, tr_partition, flat_order)

    sparse_blocks, charges, shapes = _find_diagonal_sparse_blocks(
        [flat_charges[n] for n in flat_order],
        [flat_flows[n] for n in flat_order], tr_partition)
    data = np.empty(len(self.data), dtype=self.dtype)
    for n in range(len(sparse_blocks)):
      sparse_block = sparse_blocks[n]
      ind = np.nonzero(tr_charges == charges[n])[0][0]
      permutation = tr_sparse_blocks[ind]
      data[sparse_block] = self.data[permutation]

    return BlockSparseTensor(data, [self.indices[o] for o in order])

  def reshape(self, shape: Union[Iterable[Index], Iterable[int]]) -> None:
    """
    Reshape `tensor` into `shape.
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
    new_shape = []
    for s in shape:
      if isinstance(s, Index):
        new_shape.append(s.dim)
      else:
        new_shape.append(s)

    # a few simple checks
    if np.prod(new_shape) != np.prod(self.dense_shape):
      raise ValueError("A tensor with {} elements cannot be "
                       "reshaped into a tensor with {} elements".format(
                           np.prod(self.shape), np.prod(new_shape)))

    flat_charges, flat_flows = get_flat_meta_data(self.indices)
    flat_dims = [f.dim for f in flat_charges]

    partitions = [0]
    for n in range(len(new_shape)):
      tmp = np.nonzero(np.cumprod(flat_dims) == new_shape[n])[0]
      if len(tmp) == 0:
        raise ValueError("The shape {} is incompatible with the "
                         "elementary shape {} of the tensor.".format(
                             new_shape, tuple([e.dim for e in flat_charges])))

      partitions.append(tmp[0] + 1)
      flat_dims = flat_dims[partitions[-1]:]
    partitions = np.cumsum(partitions)
    new_flat_charges = []
    new_flat_flows = []
    for n in range(1, len(partitions)):
      new_flat_charges.append(flat_charges[partitions[n - 1]:partitions[n]])
      new_flat_flows.append(flat_flows[partitions[n - 1]:partitions[n]])

    indices = [Index(c, f) for c, f in zip(new_flat_charges, new_flat_flows)]
    result = BlockSparseTensor(data=self.data, indices=indices)
    return result


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

  return tensor.reshape(shape)


def transpose(tensor: BlockSparseTensor,
              order: Union[List[int], np.ndarray]) -> "BlockSparseTensor":
  """
  Transpose `tensor` into the new order `order`. This routine currently shuffles
  data.
  Args: 
    tensor: The tensor to be transposed.
    order: The new order of indices.
  Returns:
    BlockSparseTensor: The transposed tensor.
  """
  return tensor.transpose()


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

  final_charges = tensor1.flat_charges + tensor2.flat_charges
  final_flows = tensor1.flat_flows + tensor2.flat_flows
  data = np.zeros(
      compute_num_nonzero(final_charges, final_flows), dtype=tensor1.dtype)
  if ((len(tensor1.data) > 0) and (len(tensor2.data) > 0)) and (len(data) > 0):
    # find the location of the zero block in the output
    final_block_maps, final_block_charges, final_block_dims = _find_diagonal_sparse_blocks(
        final_charges, final_flows, len(tensor1.flat_charges))
    index = np.nonzero(
        final_block_charges == final_block_charges.identity_charges)[0][0]
    data[final_block_maps[index].ravel()] = np.outer(tensor1.data,
                                                     tensor2.data).ravel()

  return BlockSparseTensor(data, tensor1.indices + tensor2.indices)


def tensordot(
    tensor1: BlockSparseTensor,
    tensor2: BlockSparseTensor,
    axes: Optional[Union[Sequence[Sequence[int]], int]] = 2,
    final_order: Optional[Union[List, np.ndarray]] = None) -> BlockSparseTensor:
  """
  Contract two `BlockSparseTensor`s along `axes`.
  Args:
    tensor1: First tensor.
    tensor2: Second tensor.
    axes: The axes to contract.
    final_order: An optional final order for the result
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
        "Some values in axes[1] = {} appear more than once!".format(axes2n))

  if len(axes1) == 0:
    res = outerproduct(tensor1, tensor2)
    if final_order is not None:
      return res.transpose(final_order)
    return res

  if (len(axes1) == tensor1.ndim) and (len(axes2) == tensor2.ndim):
    isort = np.argsort(axes1)
    data = np.dot(tensor1.data,
                  tensor2.transpose(np.asarray(axes2)[isort]).data)
    if len(tensor1.indices[0].flat_charges) > 0:
      identity_charges = tensor1.indices[0].flat_charges[0].identity_charges

    return BlockSparseTensor(
        data=data, indices=[Index(identity_charges, flow=False)])

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
    contr_flows_1.extend(tensor1.indices[a].flat_flows)
    contr_charges_1.extend(tensor1.indices[a].flat_charges)
  for a in axes2:
    contr_flows_2.extend(tensor2.indices[a].flat_flows)
    contr_charges_2.extend(tensor2.indices[a].flat_charges)

  if len(contr_charges_2) != len(contr_charges_1):
    raise ValueError(
        "axes1 and axes2 have incompatible elementary"
        " shapes {} and {}".format([e.dim for e in contr_charges_1],
                                   [e.dim for e in contr_charges_2]))
  if not np.all(
      np.asarray(contr_flows_1) == np.logical_not(np.asarray(contr_flows_2))):
    raise ValueError("axes1 and axes2 have incompatible elementary"
                     " flows {} and {}".format(contr_flows_1, contr_flows_2))

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

  flat_charges_1, flat_flows_1 = get_flat_meta_data(tensor1.indices)
  flat_charges_2, flat_flows_2 = get_flat_meta_data(tensor2.indices)

  flat_order_1 = get_flat_order(tensor1.indices, new_order1)
  flat_order_2 = get_flat_order(tensor2.indices, new_order2)

  left_charges = []
  right_charges = []
  left_flows = []
  right_flows = []
  free_indices = []
  for n in free_axes1:
    free_indices.append(tensor1.indices[n])
    left_charges.extend(tensor1.indices[n].flat_charges)
    left_flows.extend(tensor1.indices[n].flat_flows)
  for n in free_axes2:
    free_indices.append(tensor2.indices[n])
    right_charges.extend(tensor2.indices[n].flat_charges)
    right_flows.extend(tensor2.indices[n].flat_flows)

  tr_sparse_blocks_1, charges1, shapes_1 = _find_transposed_diagonal_sparse_blocks(
      flat_charges_1, flat_flows_1, len(left_charges), flat_order_1)

  tr_sparse_blocks_2, charges2, shapes_2 = _find_transposed_diagonal_sparse_blocks(
      flat_charges_2, flat_flows_2, len(contr_charges_2), flat_order_2)

  common_charges, label_to_common_1, label_to_common_2 = intersect(
      charges1.unique_charges,
      charges2.unique_charges,
      axis=1,
      return_indices=True)

  #initialize the data-vector of the output with zeros;
  if final_order is not None:
    #in this case we view the result of the diagonal multiplication
    #as a transposition of the final tensor
    final_indices = [free_indices[n] for n in final_order]
    _, reverse_order = np.unique(final_order, return_index=True)
    flat_reversed_order = get_flat_order(final_indices, reverse_order)
    flat_final_charges, flat_final_flows = get_flat_meta_data(final_indices)

    sparse_blocks_final, charges_final, shapes_final = _find_transposed_diagonal_sparse_blocks(
        flat_final_charges, flat_final_flows, len(left_charges),
        flat_reversed_order)

    num_nonzero_elements = np.sum([len(v) for v in sparse_blocks_final])
    data = np.zeros(
        num_nonzero_elements,
        dtype=np.result_type(tensor1.dtype, tensor2.dtype))
    label_to_common_final = intersect(
        charges_final.unique_charges,
        common_charges,
        axis=1,
        return_indices=True)[1]

    for n in range(common_charges.shape[1]):
      n1 = label_to_common_1[n]
      n2 = label_to_common_2[n]
      nf = label_to_common_final[n]

      data[sparse_blocks_final[nf].ravel()] = (
          tensor1.data[tr_sparse_blocks_1[n1].reshape(
              shapes_1[:, n1])] @ tensor2.data[tr_sparse_blocks_2[n2].reshape(
                  shapes_2[:, n2])]).ravel()

    return BlockSparseTensor(data=data, indices=final_indices)
  else:
    #Note: `cs` may contain charges that are not present in `common_charges`
    charges = left_charges + right_charges
    flows = left_flows + right_flows
    sparse_blocks, cs, shapes = _find_diagonal_sparse_blocks(
        charges, flows, len(left_charges))
    num_nonzero_elements = np.sum([len(v) for v in sparse_blocks])
    #Note that empty is not a viable choice here.
    data = np.zeros(
        num_nonzero_elements,
        dtype=np.result_type(tensor1.dtype, tensor2.dtype))

    label_to_common_final = intersect(
        cs.unique_charges, common_charges, axis=1, return_indices=True)[1]

    for n in range(common_charges.shape[1]):
      n1 = label_to_common_1[n]
      n2 = label_to_common_2[n]
      nf = label_to_common_final[n]

      data[sparse_blocks[nf].ravel()] = (
          tensor1.data[tr_sparse_blocks_1[n1].reshape(
              shapes_1[:, n1])] @ tensor2.data[tr_sparse_blocks_2[n2].reshape(
                  shapes_2[:, n2])]).ravel()
    return BlockSparseTensor(data=data, indices=free_indices)
