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
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index, fuse_charges, fuse_degeneracies
import numpy as np
import itertools
from typing import List, Union, Any, Tuple, Type, Optional
Tensor = Any


def check_flows(flows) -> None:
  if (set(flows) != {1}) and (set(flows) != {-1}) and (set(flows) != {-1, 1}):
    raise ValueError(
        "flows = {} contains values different from 1 and -1".format(flows))


def compute_num_nonzero(charges: List[np.ndarray],
                        flows: List[Union[bool, int]]) -> int:
  """
  Compute the number of non-zero elements, given the meta-data of 
  a symmetric tensor.
  Args:
    charges: List of np.ndarray, one for each leg of the 
      underlying tensor. Each np.ndarray `charges[leg]` 
      is of shape `(D[leg], Q)`.
      The bond dimension `D[leg]` can vary on each leg, the number of 
      symmetries `Q` has to be the same for each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
  Returns:
    int: The number of non-zero elements.
  """
  #TODO: this is not very efficient for large bond dimensions
  if len(charges) == 1:
    return len(np.nonzero(charges == 0)[0])
  #get unique charges and their degeneracies on each leg
  charge_degeneracies = [
      np.unique(charge, return_counts=True) for charge in charges
  ]
  accumulated_charges, accumulated_degeneracies = charge_degeneracies[0]
  #multiply the flow into the charges of first leg
  accumulated_charges *= flows[0]
  for n in range(1, len(charge_degeneracies)):
    #list of unique charges and list of their degeneracies
    #on the next unfused leg of the tensor
    leg_charge, leg_degeneracies = charge_degeneracies[n]

    #fuse the unique charges
    #Note: entries in `fused_charges` are not unique anymore.
    #flow1 = 1 because the flow of leg 0 has already been
    #mulitplied above
    fused_charges = fuse_charges(
        q1=accumulated_charges, flow1=1, q2=leg_charge, flow2=flows[n])
    #compute the degeneracies of `fused_charges` charges
    fused_degeneracies = fuse_degeneracies(accumulated_degeneracies,
                                           leg_degeneracies)
    #compute the new degeneracies resulting of fusing the vectors of unique charges
    #`accumulated_charges` and `leg_charge_2`
    accumulated_charges = np.unique(fused_charges)
    accumulated_degeneracies = []
    for n in range(len(accumulated_charges)):
      accumulated_degeneracies.append(
          np.sum(fused_degeneracies[fused_charges == accumulated_charges[n]]))

    accumulated_degeneracies = np.asarray(accumulated_degeneracies)
  if len(np.nonzero(accumulated_charges == 0)[0]) == 0:
    raise ValueError(
        "given leg-charges `charges` and flows `flows` are incompatible "
        "with a symmetric tensor")
  return np.sum(accumulated_degeneracies[accumulated_charges == 0])


def compute_nonzero_block_shapes(charges: List[np.ndarray],
                                 flows: List[Union[bool, int]]) -> dict:
  """
  Compute the blocks and their respective shapes of a symmetric tensor,
  given its meta-data.
  Args:
    charges: List of np.ndarray, one for each leg. 
      Each np.ndarray `charges[leg]` is of shape `(D[leg], Q)`.
      The bond dimension `D[leg]` can vary on each leg, the number of 
      symmetries `Q` has to be the same for each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
  Returns:
    dict: Dictionary mapping a tuple of charges to a shape tuple.
      Each element corresponds to a non-zero valued block of the tensor.
  """
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


def retrieve_non_zero_diagonal_blocks(data: np.ndarray,
                                      charges: List[np.ndarray],
                                      flows: List[Union[bool, int]]) -> dict:
  """
  Given the meta data and underlying data of a symmetric matrix, compute 
  all diagonal blocks and return them in a dict.
  Args: 
    data: An np.ndarray of the data. The number of elements in `data`
      has to match the number of non-zero elements defined by `charges` 
      and `flows`
    charges: List of np.ndarray, one for each leg. 
      Each np.ndarray `charges[leg]` is of shape `(D[leg], Q)`.
      The bond dimension `D[leg]` can vary on each leg, the number of 
      symmetries `Q` has to be the same for each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
  """

  if len(charges) != 2:
    raise ValueError("input has to be a two-dimensional symmetric matrix")
  check_flows(flows)
  if len(flows) != len(charges):
    raise ValueError("`len(flows)` is different from `len(charges)`")

  row_charges = flows[0] * charges[0]  # a list of charges on each row
  column_charges = flows[1] * charges[1]  # a list of charges on each column
  # for each matrix column find the number of non-zero elements in it
  # Note: the matrix is assumed to be symmetric, i.e. only elements where
  # ingoing and outgoing charge are identical are non-zero
  num_non_zero = [
      len(np.nonzero((row_charges + c) == 0)[0]) for c in column_charges
  ]
  #get the unique charges
  unique_row_charges, row_dims = np.unique(row_charges, return_counts=True)
  unique_column_charges, column_dims = np.unique(
      column_charges, return_counts=True)

  # get the degeneracies of each row and column charge
  row_degeneracies = dict(zip(unique_row_charges, row_dims))
  column_degeneracies = dict(zip(unique_column_charges, column_dims))
  blocks = {}
  for c in unique_row_charges:
    start = 0
    idxs = []
    for column in range(len(column_charges)):
      charge = column_charges[column]
      if (charge + c) != 0:
        start += num_non_zero[column]
      else:
        idxs.extend(start + np.arange(num_non_zero[column]))
    if idxs:
      blocks[c] = np.reshape(data[idxs],
                             (row_degeneracies[c], column_degeneracies[-c]))
  return blocks


class BlockSparseTensor:
  """
  Minimal class implementation of block sparsity.
  The class design follows Glen's proposal (Design 0).
  The class currently only supports a single U(1) symmetry
  and only nump.ndarray.
  Attributes:
    * self.data: A 1d np.ndarray storing the underlying 
      data of the tensor
    * self.charges: A list of `np.ndarray` of shape
      (D, Q), where D is the bond dimension, and Q the number
      of different symmetries (this is 1 for now).
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
  def shape(self) -> Tuple:
    return tuple([i.dimension for i in self.indices])

  @property
  def dtype(self) -> Type[np.number]:
    return self.data.dtype

  @property
  def flows(self):
    return [i.flow for i in self.indices]

  @property
  def charges(self):
    return [i.charges for i in self.indices]

  def reshape(self, shape):
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
      shape: The new shape.
    Returns:
      BlockSparseTensor: A new tensor reshaped into `shape`
    """

    # a few simple checks
    if np.prod(shape) != np.prod(self.shape):
      raise ValueError("A tensor with {} elements cannot be "
                       "reshaped into a tensor with {} elements".format(
                           np.prod(self.shape), np.prod(shape)))

    #keep a copy of the old indices for the case where reshaping fails
    #FIXME: this is pretty hacky!
    index_copy = [i.copy() for i in self.indices]

    def raise_error():
      #if this error is raised `shape` is incompatible
      #with the elementary indices. We have to reset them
      #to the original.
      self.indices = index_copy
      elementary_indices = []
      for i in self.indices:
        elementary_indices.extend(i.get_elementary_indices())
      print(elementary_indices)
      raise ValueError("The shape {} is incompatible with the "
                       "elementary shape {} of the tensor.".format(
                           shape,
                           tuple([e.dimension for e in elementary_indices])))

    for n in range(len(shape)):
      if shape[n] > self.shape[n]:
        while shape[n] > self.shape[n]:
          #fuse indices
          i1, i2 = self.indices.pop(n), self.indices.pop(n)
          #note: the resulting flow is set to one since the flow
          #is multiplied into the charges. As a result the tensor
          #will then be invariant in any case.
          self.indices.insert(n, fuse_index_pair(i1, i2))
        if self.shape[n] > shape[n]:
          raise_error()
      elif shape[n] < self.shape[n]:
        while shape[n] < self.shape[n]:
          #split index at n
          try:
            i1, i2 = split_index(self.indices.pop(n))
          except ValueError:
            raise_error()
          self.indices.insert(n, i1)
          self.indices.insert(n + 1, i2)
        if self.shape[n] < shape[n]:
          raise_error()

  def get_diagonal_blocks(self):
    if self.rank != 2:
      raise ValueError(
          "`get_diagonal_blocks` can only be called on a matrix, but found rank={}"
          .format(self.rank))
    return retrieve_non_zero_diagonal_blocks(
        data=self.data, charges=self.charges, flows=self.flows)


def reshape(tensor: BlockSparseTensor, shape: Tuple[int]):
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
    shape: The new shape.
  Returns:
    BlockSparseTensor: A new tensor reshaped into `shape`
  """
  result = BlockSparseTensor(
      data=tensor.data.copy(), indices=[i.copy() for i in tensor.indices])
  result.reshape(shape)
  return result
