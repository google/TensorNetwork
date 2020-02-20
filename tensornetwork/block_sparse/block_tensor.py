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
from tensornetwork.block_sparse.charge import fuse_charges, fuse_degeneracies, BaseCharge, fuse_ndarray_charges, intersect, charge_equal, fuse_ndarrays
import scipy as sp
import copy
import time
# pylint: disable=line-too-long
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable, Sequence, Text
Tensor = Any


def flatten(list_of_list: List[List]) -> np.ndarray:
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


def get_flat_meta_data(indices: List[Index]) -> Tuple[List, List]:
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
    strides: An np.ndarray of (possibly permituted) strides.
  Returns:
    np.ndarray: Linear positions of tensor elements according to `strides`.
  """
  return fuse_ndarrays([
      np.arange(0, strides[n] * dims[n], strides[n], dtype=np.uint32)
      for n in range(len(dims))
  ])


def compute_sparse_lookup(charges: List[BaseCharge], flows: List[bool],
                          target_charges: BaseCharge
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
      `len(unique_charges)`. The position of values `n` in `lookup` are positions
       with charge values `unique_charges[n]`.
    unique_charges: The unique charges of fusion of `charges`
    label_to_unique: The  integer labels of the unique charges.
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


def compute_fused_charge_degeneracies(charges: List[BaseCharge],
                                      flows: List[bool]
                                     ) -> Tuple[BaseCharge, np.ndarray]:
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
  accumulated_charges, accumulated_degeneracies = (charges[0] *
                                                   flows[0]).unique(
                                                       return_counts=True)
  for n in range(1, len(charges)):
    leg_charges, leg_degeneracies = charges[n].unique(return_counts=True)
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
                                 flows: List[bool]) -> BaseCharge:
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
  single index, keeping only the quantum numbers that appear in `target_charges`.
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
