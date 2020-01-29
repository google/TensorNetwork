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
from tensornetwork.block_tensor.charge import fuse_degeneracies, fuse_charges, fuse_degeneracies, BaseCharge, BaseCharge, fuse_ndarray_charges, intersect
import numpy as np
import scipy as sp
import itertools
import time
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable, Sequence
Tensor = Any


def combine_index_strides(index_dims: np.ndarray,
                          strides: np.ndarray) -> np.ndarray:
  """
  Combine multiple indices of some dimensions and strides into a single index, 
  based on row-major order. Used when transposing SymTensors.
  Args:
    index_dims (np.ndarray): list of dim of each index.
    strides (np.ndarray): list of strides of each index.
  Returns:
    np.ndarray: strides of combined index.
  """
  num_ind = len(index_dims)
  comb_ind_locs = np.arange(
      0, strides[0] * index_dims[0], strides[0], dtype=np.uint32)
  for n in range(1, num_ind):
    comb_ind_locs = np.add.outer(
        comb_ind_locs,
        np.arange(0, strides[n] * index_dims[n], strides[n],
                  dtype=np.uint32)).ravel()

  return comb_ind_locs


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
  return


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


def compute_fused_charge_degeneracies(charges: List[BaseCharge],
                                      flows: List[bool]
                                     ) -> Tuple[BaseCharge, np.ndarray]:
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

    for n in range(len(accumulated_charges)):
      accumulated_degeneracies[n] = np.sum(fused_degeneracies[
          fused_charges.charge_labels == accumulated_charges.charge_labels[n]])

  return accumulated_charges, accumulated_degeneracies


def compute_unique_fused_charges(charges: List[BaseCharge],
                                 flows: List[Union[bool, int]]
                                ) -> Tuple[BaseCharge, np.ndarray]:
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
  if len(nz_inds) == 0:
    raise ValueError(
        "given leg-charges `charges` and flows `flows` are incompatible "
        "with a symmetric tensor")
  return np.squeeze(accumulated_degeneracies[nz_inds][0])


def _find_diagonal_sparse_blocks(charges: List[BaseCharge], flows: np.ndarray,
                                 partition: int
                                ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Find the location of all non-trivial symmetry blocks from the data vector of
  of SymTensor (when viewed as a matrix across some prescribed index 
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

    return block_maps, block_qnums, block_dims

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
        row_locs = np.concatenate([
            (row_ind.charge_labels == n) for n in range(num_blocks)
        ]).reshape(num_blocks, row_ind.dim)
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
      block_maps = [(cumulate_num_nz[row_locs[n, :]][:, None] +
                     np.arange(block_dims[1, n])[None, :]).ravel()
                    for n in range(num_blocks)]
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
  Find the location of all non-trivial symmetry blocks from the data vector of
  of SymTensor after transposition (and then viewed as a matrix across some 
  prescribed index bi-tr_partition). Produces and equivalent result to 
  retrieve_blocks acting on a transposed SymTensor, but is much faster.
  Args:
    charges (List[SymIndex]): list of SymIndex.
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

  else:
    # non-trivial transposition is required
    num_inds = len(charges)
    tensor_dims = np.array([charges[n].dim for n in range(num_inds)], dtype=int)
    strides = np.append(np.flip(np.cumprod(np.flip(tensor_dims[1:]))), 1)

    # define properties of new tensor resulting from transposition
    new_strides = strides[order]
    new_row_charges = [charges[n] for n in order[:tr_partition]]
    new_col_charges = [charges[n] for n in order[tr_partition:]]
    new_row_flows = flows[order[:tr_partition]]
    new_col_flows = flows[order[tr_partition:]]

    # compute qnums of row/cols in transposed tensor
    unique_row_qnums, new_row_degen = compute_fused_charge_degeneracies(
        new_row_charges, new_row_flows)

    # unique_row_qnums, new_row_degen = compute_qnum_degen(
    #     new_row_charges, new_row_flows)
    unique_col_qnums, new_col_degen = compute_fused_charge_degeneracies(
        new_col_charges, np.logical_not(new_col_flows))
    # unique_col_qnums, new_col_degen = compute_qnum_degen(
    #     new_col_charges, np.logical_not(new_col_flows))
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
    # compute qnums of row/cols in original tensor
    orig_partition = _find_best_partition(tensor_dims)
    orig_width = np.prod(tensor_dims[orig_partition:])

    orig_unique_row_qnums = compute_unique_fused_charges(
        charges[:orig_partition], flows[:orig_partition])
    orig_unique_col_qnums, orig_col_degen = compute_fused_charge_degeneracies(
        charges[orig_partition:], np.logical_not(flows[orig_partition:]))
    orig_block_qnums, row_map, col_map = intersect(
        orig_unique_row_qnums.unique_charges,
        orig_unique_col_qnums.unique_charges,
        axis=1,
        return_indices=True)
    orig_num_blocks = orig_block_qnums.shape[1]

    orig_row_ind = fuse_charges(charges[:orig_partition],
                                flows[:orig_partition])
    orig_col_ind = fuse_charges(charges[orig_partition:],
                                np.logical_not(flows[orig_partition:]))

    # compute row degens (i.e. number of non-zero elements per row)
    inv_row_map = -np.ones(
        orig_unique_row_qnums.unique_charges.shape[1], dtype=np.int16)
    for n in range(len(row_map)):
      inv_row_map[row_map[n]] = n

    all_degens = np.append(orig_col_degen[col_map],
                           0)[inv_row_map[orig_row_ind.charge_labels]]
    all_cumul_degens = np.cumsum(np.insert(all_degens[:-1], 0,
                                           0)).astype(np.uint32)
    # generate vector which translates from dense row position to sparse row position
    dense_to_sparse = np.zeros(orig_width, dtype=np.uint32)
    for n in range(orig_num_blocks):
      dense_to_sparse[orig_col_ind.charge_labels == col_map[n]] = np.arange(
          orig_col_degen[col_map[n]], dtype=np.uint32)

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
    out[np.nonzero(
        fuse_charges(charges, self.flows) == charges[0].identity_charges)
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
    flat_indices, flat_charges, flat_flows, _, flat_order, _ = flatten_meta_data(
        self.indices, order, 0)
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
    self.indices = [self.indices[o] for o in order]
    self.data = data
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
        new_shape.append(s.dimension)
      else:
        new_shape.append(s)
    # a few simple checks
    if np.prod(new_shape) != np.prod(self.dense_shape):
      raise ValueError("A tensor with {} elements cannot be "
                       "reshaped into a tensor with {} elements".format(
                           np.prod(self.shape), np.prod(self.dense_shape)))

    #keep a copy of the old indices for the case where reshaping fails
    #FIXME: this is pretty hacky!
    indices = [i.copy() for i in self.indices]
    flat_indices = []
    for i in indices:
      flat_indices.extend(i.get_elementary_indices())

    def raise_error():
      #if this error is raised then `shape` is incompatible
      #with the elementary indices. We then reset the shape
      #to what is was before the call to `reshape`.
      # self.indices = index_copy
      raise ValueError("The shape {} is incompatible with the "
                       "elementary shape {} of the tensor.".format(
                           new_shape,
                           tuple([e.dimension for e in flat_indices])))

    for n in range(len(new_shape)):
      if new_shape[n] > flat_indices[n].dimension:
        while new_shape[n] > flat_indices[n].dimension:
          #fuse flat_indices
          i1, i2 = flat_indices.pop(n), flat_indices.pop(n)
          #note: the resulting flow is set to one since the flow
          #is multiplied into the charges. As a result the tensor
          #will then be invariant in any case.
          flat_indices.insert(n, fuse_index_pair(i1, i2))
        if flat_indices[n].dimension > new_shape[n]:
          raise_error()
      elif new_shape[n] < flat_indices[n].dimension:
        raise_error()
    #at this point the first len(new_shape) flat_indices of the tensor
    #match the `new_shape`.
    while len(new_shape) < len(flat_indices):
      i2, i1 = flat_indices.pop(), flat_indices.pop()
      flat_indices.append(fuse_index_pair(i1, i2))

    result = BlockSparseTensor(data=self.data, indices=flat_indices)
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
  result = tensor.copy()
  result.transpose(order)
  return result


def _compute_transposed_sparse_blocks(indices: BlockSparseTensor,
                                      order: Union[List[int], np.ndarray],
                                      transposed_partition: Optional[int] = None
                                     ) -> Tuple[BaseCharge, Dict, int]:
  """
  Args:
    indices: A symmetric tensor.
    order: The new order of indices.
    permutation: An np.ndarray of int for reshuffling the data,
      typically the output of a prior call to `transpose`. Passing `permutation`
      can greatly speed up the transposition.
    return_permutation: If `True`, return the the permutation data.
  Returns:

  """
  if len(order) != len(indices):
    raise ValueError(
        "`len(order)={}` is different form `len(indices)={}`".format(
            len(order), len(indices)))
  flat_indices, flat_charges, flat_flows, flat_strides, flat_order, transposed_partition = flatten_meta_data(
      indices, order, transposed_partition)
  if transposed_partition is None:
    transposed_partition = _find_best_partition(
        [len(flat_charges[n]) for n in flat_order])

  return _find_transposed_diagonal_sparse_blocks(
      flat_charges,
      flat_flows,
      tr_partition=transposed_partition,
      order=flat_order)


def tensordot(tensor1: BlockSparseTensor,
              tensor2: BlockSparseTensor,
              axes: Sequence[Sequence[int]],
              final_order: Optional[Union[List, np.ndarray]] = None
             ) -> BlockSparseTensor:
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
      np.array([i.flow for i in elementary_1]) == np.array(
          [not i.flow for i in elementary_2])):
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

  #get the flattened indices for the output tensor
  left_indices = []
  right_indices = []
  for n in free_axes1:
    left_indices.extend(tensor1.indices[n].get_elementary_indices())
  for n in free_axes2:
    right_indices.extend(tensor2.indices[n].get_elementary_indices())
  indices = left_indices + right_indices

  _, flat_charges1, flat_flows1, flat_strides1, flat_order1, tr_partition1 = flatten_meta_data(
      tensor1.indices, new_order1, len(free_axes1))
  tr_sparse_blocks_1, charges1, shapes_1 = _find_transposed_diagonal_sparse_blocks(
      flat_charges1, flat_flows1, tr_partition1, flat_order1)
  _, flat_charges2, flat_flows2, flat_strides2, flat_order2, tr_partition2 = flatten_meta_data(
      tensor2.indices, new_order2, len(axes2))
  tr_sparse_blocks_2, charges2, shapes_2 = _find_transposed_diagonal_sparse_blocks(
      flat_charges2, flat_flows2, tr_partition2, flat_order2)
  #common_charges = charges1.intersect(charges2)
  common_charges, label_to_common_1, label_to_common_2 = intersect(
      charges1.unique_charges,
      charges2.unique_charges,
      axis=1,
      return_indices=True)

  #initialize the data-vector of the output with zeros;
  if final_order is not None:
    #in this case we view the result of the diagonal multiplication
    #as a transposition of the final tensor
    final_indices = [indices[n] for n in final_order]
    _, reverse_order = np.unique(final_order, return_index=True)

    flat_final_indices, flat_final_charges, flat_final_flows, flat_final_strides, flat_final_order, tr_partition = flatten_meta_data(
        final_indices, reverse_order, len(free_axes1))

    sparse_blocks_final, charges_final, shapes_final = _find_transposed_diagonal_sparse_blocks(
        flat_final_charges, flat_final_flows, tr_partition, flat_final_order)

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
    charges = [i.charges for i in indices]
    flows = [i.flow for i in indices]
    sparse_blocks, cs, shapes = _find_diagonal_sparse_blocks(
        charges, flows, len(left_indices))
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
    return BlockSparseTensor(data=data, indices=indices)


def flatten_meta_data(indices, order, partition):
  elementary_indices = {}
  flat_elementary_indices = []
  new_partition = 0
  for n in range(len(indices)):
    elementary_indices[n] = indices[n].get_elementary_indices()
    if n < partition:
      new_partition += len(elementary_indices[n])
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

  return flat_elementary_indices, flat_charges, flat_flows, flat_strides, flat_order, new_partition


#####################################################  DEPRECATED ROUTINES ############################


def _find_transposed_diagonal_sparse_blocks_old(
    charges: List[BaseCharge], flows: List[Union[bool, int]], order: np.ndarray,
    tr_partition: int) -> Tuple[BaseCharge, List[np.ndarray]]:
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
  _check_flows(flows)
  if len(flows) != len(charges):
    raise ValueError("`len(flows)` is different from `len(charges) ")
  if np.all(order == np.arange(len(order))):
    return _find_diagonal_sparse_blocks(charges, flows, tr_partition)

  strides = _get_strides([len(c) for c in charges])

  tr_row_charges = [charges[n] for n in order[:tr_partition]]
  tr_row_flows = [flows[n] for n in order[:tr_partition]]
  tr_row_strides = [strides[n] for n in order[:tr_partition]]

  tr_column_charges = [charges[n] for n in order[tr_partition:]]
  tr_column_flows = [flows[n] for n in order[tr_partition:]]
  tr_column_strides = [strides[n] for n in order[tr_partition:]]

  unique_tr_column_charges, tr_column_dims = compute_fused_charge_degeneracies(
      tr_column_charges, tr_column_flows)
  unique_tr_row_charges = compute_unique_fused_charges(tr_row_charges,
                                                       tr_row_flows)

  fused = unique_tr_row_charges + unique_tr_column_charges
  tr_li, tr_ri = np.divmod(
      np.nonzero(fused == unique_tr_column_charges.identity_charges)[0],
      len(unique_tr_column_charges))

  row_ind, row_locations = reduce_charges(
      charges=tr_row_charges,
      flows=tr_row_flows,
      target_charges=unique_tr_row_charges.charges[:, tr_li],
      return_locations=True,
      strides=tr_row_strides)

  col_ind, column_locations = reduce_charges(
      charges=tr_column_charges,
      flows=tr_column_flows,
      target_charges=unique_tr_column_charges.charges[:, tr_ri],
      return_locations=True,
      strides=tr_column_strides)

  partition = _find_best_partition([len(c) for c in charges])
  fused_row_charges = fuse_charges(charges[:partition], flows[:partition])
  fused_column_charges = fuse_charges(charges[partition:], flows[partition:])

  unique_fused_row, row_inverse = fused_row_charges.unique(return_inverse=True)
  unique_fused_column, column_inverse = fused_column_charges.unique(
      return_inverse=True)

  unique_column_charges, column_dims = compute_fused_charge_degeneracies(
      charges[partition:], flows[partition:])
  unique_row_charges = compute_unique_fused_charges(charges[:partition],
                                                    flows[:partition])
  fused = unique_row_charges + unique_column_charges
  li, ri = np.divmod(
      np.nonzero(fused == unique_column_charges.identity_charges)[0],
      len(unique_column_charges))

  common_charges, label_to_row, label_to_column = unique_row_charges.intersect(
      unique_column_charges * True, return_indices=True)
  num_blocks = len(label_to_row)
  tmp = -np.ones(len(unique_row_charges), dtype=np.int16)
  for n in range(len(label_to_row)):
    tmp[label_to_row[n]] = n

  degeneracy_vector = np.append(column_dims[label_to_column],
                                0)[tmp[row_inverse]]
  start_positions = np.cumsum(np.insert(degeneracy_vector[:-1], 0,
                                        0)).astype(np.uint32)

  column_dimension = np.prod([len(c) for c in charges[partition:]])

  column_lookup = compute_sparse_lookup(charges[partition:], flows[partition:],
                                        common_charges)

  blocks = []
  for n in range(num_blocks):
    rlocs = row_locations[row_ind.charge_labels == n]
    clocs = column_locations[col_ind.charge_labels == n]
    orig_row_posL, orig_col_posL = np.divmod(rlocs, np.uint32(column_dimension))
    orig_row_posR, orig_col_posR = np.divmod(clocs, np.uint32(column_dimension))
    inds = (start_positions[np.add.outer(orig_row_posL, orig_row_posR)] +
            column_lookup[np.add.outer(orig_col_posL, orig_col_posR)]).ravel()

    blocks.append([inds, (len(rlocs), len(clocs))])
  charges_out = unique_tr_row_charges[tr_li]
  return charges_out, blocks


def _find_diagonal_dense_blocks(
    row_charges: List[BaseCharge],
    column_charges: List[BaseCharge],
    row_flows: List[Union[bool, int]],
    column_flows: List[Union[bool, int]],
    row_strides: Optional[np.ndarray] = None,
    column_strides: Optional[np.ndarray] = None,
) -> Tuple[BaseCharge, List[np.ndarray]]:
  """

  Deprecated
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
      np.nonzero(fused == unique_column_charges.identity_charges)[0],
      len(unique_column_charges))
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


def _compute_sparse_lookups(row_charges: BaseCharge, row_flows, column_charges,
                            column_flows):
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
  column_lookup = np.empty(len(fused_column_charges), dtype=np.uint32)
  row_lookup = np.zeros(len(fused_row_charges), dtype=np.uint32)
  for n in range(len(common_charges)):
    column_lookup[col_ind_sort[col_start_positions[
        comm_col[n]]:col_start_positions[comm_col[n] + 1]]] = np.arange(
            col_charge_degeneracies[comm_col[n]])
    row_lookup[
        row_ind_sort[row_start_positions[comm_row[n]]:row_start_positions[
            comm_row[n] + 1]]] = col_charge_degeneracies[comm_col[n]]

  return np.append(0, np.cumsum(row_lookup[0:-1])), column_lookup


def _get_stride_arrays(dims):
  strides = np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))
  return [np.arange(dims[n]) * strides[n] for n in range(len(dims))]


def reduce_charges(charges: List[BaseCharge],
                   flows: Iterable[bool],
                   target_charges: np.ndarray,
                   return_locations: Optional[bool] = False,
                   strides: Optional[np.ndarray] = None
                  ) -> Tuple[BaseCharge, np.ndarray]:
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
      row_pos = combine_index_strides(tensor_dims[:partition],
                                      strides[:partition])
      col_pos = combine_index_strides(tensor_dims[partition:],
                                      strides[partition:])

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


def reduce_to_target_charges(charges: List[BaseCharge],
                             flows: List[Union[int, bool]],
                             target_charges: BaseCharge,
                             strides: Optional[np.ndarray] = None,
                             return_positions: Optional[bool] = False
                            ) -> np.ndarray:
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
  if len(charges) == 1:
    fused_charges = charges[0] * flows[0]
    unique, inverse = fused_charges.unique(return_inverse=True)
    common, label_to_unique, label_to_target = unique.intersect(
        target_charges, return_indices=True)
    inds = np.nonzero(np.isin(inverse, label_to_unique))[0]
    if strides is not None:
      permuted_inds = strides[0] * np.arange(len(charges[0]))
      if return_positions:
        return fused_charges[permuted_inds[inds]], inds
      return fused_charges[permuted_inds[inds]]

    if return_positions:
      return fused_charges[inds], inds
    return fused_charges[inds]

  partition = _find_best_partition([len(c) for c in charges])
  left_charges = fuse_charges(charges[:partition], flows[:partition])
  right_charges = fuse_charges(charges[partition:], flows[partition:])

  # unique_target_charges, inds = target_charges.unique(return_index=True)
  # target_charges = target_charges[np.sort(inds)]
  unique_left, left_inverse = left_charges.unique(return_inverse=True)
  unique_right, right_inverse = right_charges.unique(return_inverse=True)

  fused = unique_left + unique_right
  unique_fused, unique_fused_labels = fused.unique(return_inverse=True)

  relevant_charges, relevant_labels, _ = unique_fused.intersect(
      target_charges, return_indices=True)

  tmp = np.full(len(unique_fused), fill_value=-1, dtype=np.int16)
  tmp[relevant_labels] = np.arange(len(relevant_labels), dtype=np.int16)
  lookup_target = tmp[unique_fused_labels].reshape(
      [len(unique_left), len(unique_right)])

  if return_positions:
    if strides is not None:
      stride_arrays = [
          np.arange(len(charges[n])) * strides[n] for n in range(len(charges))
      ]
      permuted_left_inds = fuse_ndarrays(stride_arrays[0:partition])
      permuted_right_inds = fuse_ndarrays(stride_arrays[partition:])

      row_locations = [None] * len(unique_left)
      final_relevant_labels = [None] * len(unique_left)
      for n in range(len(unique_left)):
        labels = lookup_target[n, right_inverse]
        lookup = labels >= 0
        row_locations[n] = permuted_right_inds[lookup]
        final_relevant_labels[n] = labels[lookup]

      charge_labels = np.concatenate(
          [final_relevant_labels[n] for n in left_inverse])
      tmp_inds = [
          permuted_left_inds[n] + row_locations[left_inverse[n]]
          for n in range(len(left_charges))
      ]
      try:
        inds = np.concatenate(tmp_inds)
      except ValueError:
        inds = np.asarray(tmp_inds)

    else:
      row_locations = [None] * len(unique_left)
      final_relevant_labels = [None] * len(unique_left)
      for n in range(len(unique_left)):
        labels = lookup_target[n, right_inverse]
        lookup = labels >= 0
        row_locations[n] = np.nonzero(lookup)[0]
        final_relevant_labels[n] = labels[lookup]
      charge_labels = np.concatenate(
          [final_relevant_labels[n] for n in left_inverse])

      inds = np.concatenate([
          n * len(right_charges) + row_locations[left_inverse[n]]
          for n in range(len(left_charges))
      ])
      obj = charges[0].__new__(type(charges[0]))
      obj.__init__(relevant_charges.unique_charges, charge_labels,
                   charges[0].charge_types)
      return obj, inds

  else:
    final_relevant_labels = [None] * len(unique_left)
    for n in range(len(unique_left)):
      labels = lookup_target[n, right_inverse]
      lookup = labels >= 0
      final_relevant_labels[n] = labels[lookup]
    charge_labels = np.concatenate(
        [final_relevant_labels[n] for n in left_inverse])
    obj = charges[0].__new__(type(charges[0]))
    obj.__init__(relevant_charges.unique_charges, charge_labels,
                 charges[0].charge_types)
    return obj


def find_sparse_positions_new(charges: List[BaseCharge],
                              flows: List[Union[int, bool]],
                              target_charges: BaseCharge,
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
  if len(charges) == 1:
    fused_charges = charges[0] * flows[0]
    unique, inverse = fused_charges.unique(return_inverse=True)
    common, label_to_unique, label_to_target = unique.intersect(
        target_charges, return_indices=True)
    inds = np.nonzero(np.isin(inverse, label_to_unique))[0]
    if strides is not None:
      permuted_inds = strides[0] * np.arange(len(charges[0]))
      return fused_charges[permuted_inds[inds]], inds

    return fused_charges[inds], inds

  partition = _find_best_partition([len(c) for c in charges])
  left_charges = fuse_charges(charges[:partition], flows[:partition])
  right_charges = fuse_charges(charges[partition:], flows[partition:])

  # unique_target_charges, inds = target_charges.unique(return_index=True)
  # target_charges = target_charges[np.sort(inds)]
  unique_left, left_inverse = left_charges.unique(return_inverse=True)
  unique_right, right_inverse, right_degens = right_charges.unique(
      return_inverse=True, return_counts=True)

  fused = unique_left + unique_right

  unique_fused, labels_fused = fused.unique(return_inverse=True)

  relevant_charges, label_to_unique_fused, label_to_target = unique_fused.intersect(
      target_charges, return_indices=True)

  relevant_fused_positions = np.nonzero(
      np.isin(labels_fused, label_to_unique_fused))[0]
  relevant_left_labels, relevant_right_labels = np.divmod(
      relevant_fused_positions, len(unique_right))
  rel_l_labels = np.unique(relevant_left_labels)
  total_degen = {
      t: np.sum(right_degens[relevant_right_labels[relevant_left_labels == t]])
      for t in rel_l_labels
  }

  relevant_left_inverse = left_inverse[np.isin(left_inverse, rel_l_labels)]
  degeneracy_vector = np.empty(len(relevant_left_inverse), dtype=np.uint32)
  row_locations = [None] * len(unique_left)
  final_relevant_labels = [None] * len(unique_left)
  for n in range(len(relevant_left_labels)):
    degeneracy_vector[relevant_left_inverse ==
                      relevant_left_labels[n]] = total_degen[
                          relevant_left_labels[n]]
  start_positions = np.cumsum(degeneracy_vector) - degeneracy_vector
  tmp = np.full(len(unique_fused), fill_value=-1, dtype=np.int16)
  tmp[label_to_unique_fused] = np.arange(
      len(label_to_unique_fused), dtype=np.int16)
  lookup_target = tmp[labels_fused].reshape(
      [len(unique_left), len(unique_right)])

  final_relevant_labels = [None] * len(unique_left)
  for n in range(len(rel_l_labels)):
    labels = lookup_target[rel_l_labels[n], right_inverse]
    lookup = labels >= 0
    final_relevant_labels[rel_l_labels[n]] = labels[lookup]
  charge_labels = np.concatenate(
      [final_relevant_labels[n] for n in relevant_left_inverse])
  inds = np.concatenate([
      start_positions[n] +
      np.arange(total_degen[relevant_left_inverse[n]], dtype=np.uint32)
      for n in range(len(relevant_left_inverse))
  ])

  return relevant_charges[charge_labels], inds


def find_sparse_positions(charges: List[BaseCharge],
                          flows: List[Union[int, bool]],
                          target_charges: BaseCharge) -> Dict:
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
    charges: An np.ndarray of integer charges.
    flows: The flow direction of the left charges.
    target_charges: The target charges.
  Returns:
    dict: Mapping integers to np.ndarray of integers.
  """
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
  partition = _find_best_partition([len(c) for c in charges])
  left_charges = fuse_charges(charges[:partition], flows[:partition])
  right_charges = fuse_charges(charges[partition:], flows[partition:])

  # unique_target_charges, inds = target_charges.unique(return_index=True)
  # target_charges = target_charges[np.sort(inds)]
  unique_left, left_inverse = left_charges.unique(return_inverse=True)
  unique_right, right_inverse, right_dims = right_charges.unique(
      return_inverse=True, return_counts=True)

  fused_unique = unique_left + unique_right
  unique_inds = np.nonzero(fused_unique == target_charges)
  relevant_positions = unique_inds[0].astype(np.uint32)
  tmp_inds_left, tmp_inds_right = np.divmod(relevant_positions,
                                            len(unique_right))

  relevant_unique_left_inds = np.unique(tmp_inds_left)
  left_lookup = np.empty(np.max(relevant_unique_left_inds) + 1, dtype=np.uint32)
  left_lookup[relevant_unique_left_inds] = np.arange(
      len(relevant_unique_left_inds))
  relevant_unique_right_inds = np.unique(tmp_inds_right)
  right_lookup = np.empty(
      np.max(relevant_unique_right_inds) + 1, dtype=np.uint32)
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
  degeneracy_vector = np.empty(len(left_charge_labels[0]), dtype=np.uint32)
  for n in range(len(relevant_unique_left_inds)):
    degeneracy_vector[relevant_left_inverse[
        left_charge_labels[1] == n]] = np.sum(right_dims[tmp_inds_right[
            tmp_inds_left == relevant_unique_left_inds[n]]])

  start_positions = (np.cumsum(degeneracy_vector) - degeneracy_vector).astype(
      np.uint32)
  out = {}
  for n in range(len(target_charges)):
    block = []
    if len(unique_inds) > 1:
      lis, ris = np.divmod(unique_inds[0][unique_inds[1] == n],
                           len(unique_right))
    else:
      lis, ris = np.divmod(unique_inds[0], len(unique_right))

    for m in range(len(lis)):
      ri_tmp, arange, tmp_inds = right_block_information[lis[m]]
      block.append(
          np.add.outer(
              start_positions[relevant_left_inverse[left_charge_labels[1] ==
                                                    left_lookup[lis[m]]]],
              arange[tmp_inds == np.nonzero(
                  ri_tmp == ris[m])[0]]).ravel().astype(np.uint32))
    out[target_charges[n]] = np.concatenate(block)
  return out


# def find_dense_positions(charges: List[Union[BaseCharge, ChargeCollection]],
#                          flows: List[Union[int, bool]],
#                          target_charges: Union[BaseCharge, ChargeCollection],
#                          strides: Optional[np.ndarray] = None,
#                          store_dual: Optional[bool] = False) -> Dict:
#   """
#   Find the dense locations of elements (i.e. the index-values within the DENSE tensor)
#   in the vector of `fused_charges` resulting from fusing all elements of `charges`
#   that have a value of `target_charge`.
#   For example, given
#   ```
#   charges = [[-2,0,1,0,0],[-1,0,2,1]]
#   target_charge = 0
#   fused_charges = fuse_charges(charges,[1,1])
#   print(fused_charges) # [-3,-2,0,-1,-1,0,2,1,0,1,3,2,-1,0,2,1,-1,0,2,1]
#   ```
#   we want to find the index-positions of charges
#   that fuse to `target_charge=0`, i.e. where `fused_charges==0`,
#   within the dense array. As one additional wrinkle, `charges`
#   is a subset of the permuted charges of a tensor with rank R > len(charges),
#   and `stride_arrays` are their corresponding range of strides, i.e.

#   ```
#   R=5
#   D = [2,3,4,5,6]
#   tensor_flows = np.random.randint(-1,2,R)
#   tensor_charges = [np.random.randing(-5,5,D[n]) for n in range(R)]
#   order = np.arange(R)
#   np.random.shuffle(order)
#   tensor_strides = [360, 120,  30,   6,   1]

#   charges = [tensor_charges[order[n]] for n in range(3)]
#   flows = [tensor_flows[order[n]] for n in range(len(3))]
#   strides = [tensor_stride[order[n]] for n in range(3)]
#   _ = _find_transposed_dense_positions(charges, flows, 0, strides)

#   ```
#   `_find_transposed_dense_blocks` returns an np.ndarray containing the
#   index-positions of  these elements calculated using `stride_arrays`.
#   The result only makes sense in conjuction with the complementary
#   data computed from the complementary
#   elements in`tensor_charges`,
#   `tensor_strides` and `tensor_flows`.
#   This routine is mainly used in `_find_diagonal_dense_blocks`.

#   Args:
#     charges: A list of BaseCharge or ChargeCollection.
#     flows: The flow directions of the `charges`.
#     target_charge: The target charge.
#     strides: The strides for the `charges` subset.
#       if `None`, natural stride ordering is assumed.

#   Returns:
#     dict
#   """

#   _check_flows(flows)
#   out = {}
#   if store_dual:
#     store_charges = target_charges * (-1)
#   else:
#     store_charges = target_charges

#   if len(charges) == 1:
#     fused_charges = charges[0] * flows[0]
#     inds = np.nonzero(fused_charges == target_charges)
#     if len(target_charges) > 1:
#       for n in range(len(target_charges)):
#         i = inds[0][inds[1] == n]
#         if len(i) == 0:
#           continue
#         if strides is not None:
#           permuted_inds = strides[0] * np.arange(len(charges[0]))
#           out[store_charges.get_item(n)] = permuted_inds[i]
#         else:
#           out[store_charges.get_item(n)] = i
#       return out
#     else:
#       if strides is not None:
#         permuted_inds = strides[0] * np.arange(len(charges[0]))
#         out[store_charges.get_item(n)] = permuted_inds[inds[0]]
#       else:
#         out[store_charges.get_item(n)] = inds[0]
#       return out

#   partition = _find_best_partition([len(c) for c in charges])
#   left_charges = fuse_charges(charges[:partition], flows[:partition])
#   right_charges = fuse_charges(charges[partition:], flows[partition:])
#   if strides is not None:
#     stride_arrays = [
#         np.arange(len(charges[n])) * strides[n] for n in range(len(charges))
#     ]
#     permuted_left_inds = fuse_ndarrays(stride_arrays[0:partition])
#     permuted_right_inds = fuse_ndarrays(stride_arrays[partition:])

#   # unique_target_charges, inds = target_charges.unique(return_index=True)
#   # target_charges = target_charges[np.sort(inds)]
#   unique_left, left_inverse = left_charges.unique(return_inverse=True)
#   unique_right, right_inverse = right_charges.unique(return_inverse=True)

#   fused_unique = unique_left + unique_right
#   unique_inds = np.nonzero(fused_unique == target_charges)

#   relevant_positions = unique_inds[0]
#   tmp_inds_left, tmp_inds_right = np.divmod(relevant_positions,
#                                             len(unique_right))

#   relevant_unique_left_inds = np.unique(tmp_inds_left)
#   left_lookup = np.empty(np.max(relevant_unique_left_inds) + 1, dtype=np.uint32)
#   left_lookup[relevant_unique_left_inds] = np.arange(
#       len(relevant_unique_left_inds))
#   relevant_unique_right_inds = np.unique(tmp_inds_right)
#   right_lookup = np.empty(
#       np.max(relevant_unique_right_inds) + 1, dtype=np.uint32)
#   right_lookup[relevant_unique_right_inds] = np.arange(
#       len(relevant_unique_right_inds))

#   left_charge_labels = np.nonzero(
#       np.expand_dims(left_inverse, 1) == np.expand_dims(
#           relevant_unique_left_inds, 0))
#   right_charge_labels = np.nonzero(
#       np.expand_dims(right_inverse, 1) == np.expand_dims(
#           relevant_unique_right_inds, 0))

#   len_right = len(right_charges)

#   for n in range(len(target_charges)):
#     if len(unique_inds) > 1:
#       lis, ris = np.divmod(unique_inds[0][unique_inds[1] == n],
#                            len(unique_right))
#     else:
#       lis, ris = np.divmod(unique_inds[0], len(unique_right))
#     dense_positions = []
#     left_positions = []
#     lookup = []
#     for m in range(len(lis)):
#       li = lis[m]
#       ri = ris[m]
#       dense_left_positions = (left_charge_labels[0][
#           left_charge_labels[1] == left_lookup[li]]).astype(np.uint32)
#       dense_right_positions = (right_charge_labels[0][
#           right_charge_labels[1] == right_lookup[ri]]).astype(np.uint32)
#       if strides is None:
#         positions = np.expand_dims(dense_left_positions * len_right,
#                                    1) + np.expand_dims(dense_right_positions, 0)
#       else:
#         positions = np.expand_dims(
#             permuted_left_inds[dense_left_positions], 1) + np.expand_dims(
#                 permuted_right_inds[dense_right_positions], 0)

#       dense_positions.append(positions)
#       left_positions.append(dense_left_positions)
#       lookup.append(
#           np.stack([
#               np.arange(len(dense_left_positions), dtype=np.uint32),
#               np.full(len(dense_left_positions), fill_value=m, dtype=np.uint32)
#           ],
#                    axis=1))

#     if len(lookup) > 0:
#       ind_sort = np.argsort(np.concatenate(left_positions))
#       it = np.concatenate(lookup, axis=0)
#       table = it[ind_sort, :]
#       out[store_charges.get_item(n)] = np.concatenate([
#           dense_positions[table[n, 1]][table[n, 0], :].astype(np.uint32)
#           for n in range(table.shape[0])
#       ])
#     else:
#       out[store_charges.get_item(n)] = np.array([])

#   return out

# def _find_diagonal_sparse_blocks_old(charges: List[BaseCharge],
#                                  flows: List[Union[bool, int]],
#                                  partition: int) -> Tuple[BaseCharge, List]:
#   """
#   Given the meta data and underlying data of a symmetric matrix, compute
#   all diagonal blocks and return them in a dict.
#   `row_charges` and `column_charges` are lists of np.ndarray. The tensor
#   is viewed as a matrix with rows given by fusing `row_charges` and
#   columns given by fusing `column_charges`.

#   Args:
#     charges: A list of charges.
#     flows: A list of flows.
#     partition: The location of the partition of `charges` into rows and colums.
#   Returns:
#   return common_charges, blocks, start_positions, row_locations, column_degeneracies
#     List[Union[BaseCharge, ChargeCollection]]: A list of unique charges, one per block.
#     List[np.ndarray]: A list containing the blocks.
#   """
#   _check_flows(flows)
#   if len(flows) != len(charges):
#     raise ValueError("`len(flows)` is different from `len(charges)`")
#   row_charges = charges[:partition]
#   row_flows = flows[:partition]
#   column_charges = charges[partition:]
#   column_flows = flows[partition:]

#   #get the unique column-charges
#   #we only care about their degeneracies, not their order; that's much faster
#   #to compute since we don't have to fuse all charges explicitly
#   #`compute_fused_charge_degeneracies` multiplies flows into the column_charges
#   unique_column_charges, column_dims = compute_fused_charge_degeneracies(
#       column_charges, column_flows)
#   unique_row_charges = compute_unique_fused_charges(row_charges, row_flows)
#   #get the charges common to rows and columns (only those matter)
#   common_charges, label_to_row, label_to_column = unique_row_charges.intersect(
#       unique_column_charges * True, return_indices=True)

#   #convenience container for storing the degeneracies of each
#   #column charge
#   #column_degeneracies = dict(zip(unique_column_charges, column_dims))
#   #column_degeneracies = dict(zip(unique_column_charges * True, column_dims))
#   print(common_charges)
#   row_locations = find_sparse_positions(
#       charges=row_charges, flows=row_flows, target_charges=common_charges)

#   degeneracy_vector = np.empty(
#       np.sum([len(v) for v in row_locations.values()]), dtype=np.uint32)
#   #for each charge `c` in `common_charges` we generate a boolean mask
#   #for indexing the positions where `relevant_column_charges` has a value of `c`.
#   for c in common_charges:
#     degeneracy_vector[row_locations[c]] = column_degeneracies[c]

#   start_positions = (np.cumsum(degeneracy_vector) - degeneracy_vector).astype(
#       np.uint32)
#   blocks = []

#   for c in common_charges:
#     #numpy broadcasting is substantially faster than kron!
#     rlocs = row_locations[c]
#     rlocs.sort()  #sort in place (we need it again later)
#     cdegs = column_degeneracies[c]
#     inds = np.ravel(np.add.outer(start_positions[rlocs], np.arange(cdegs)))
#     blocks.append([inds, (len(rlocs), cdegs)])
#   return common_charges, blocks
