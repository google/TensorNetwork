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
from tensornetwork.network_components import Node, contract, contract_between
# pylint: disable=line-too-long
from tensornetwork.backends import backend_factory
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index
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

  if len(charges) == 1:
    return len(charges)
  net_charges = flows[0] * charges[0]
  for i in range(1, len(flows)):
    net_charges = np.reshape(
        flows[i] * charges[i][:, None] + net_charges[None, :],
        len(charges[i]) * len(net_charges))

  return len(np.nonzero(net_charges == 0)[0])


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

  row_charges = charges[0]  # a list of charges on each row
  column_charges = charges[1]  # a list of charges on each column
  # for each matrix column find the number of non-zero elements in it
  # Note: the matrix is assumed to be symmetric, i.e. only elements where
  # ingoing and outgoing charge are identical are non-zero
  num_non_zero = [len(np.nonzero(row_charges == c)[0]) for c in column_charges]

  #get the unique charges
  #Note: row and column unique charges are the same due to symmetry
  unique_charges, row_dims = np.unique(row_charges, return_counts=True)
  _, column_dims = np.unique(column_charges, return_counts=True)

  # get the degenaricies of each row and column charge
  row_degeneracies = dict(zip(unique_charges, row_dims))
  column_degeneracies = dict(zip(unique_charges, column_dims))
  blocks = {}
  for c in unique_charges:
    start = 0
    idxs = []
    for column in range(len(column_charges)):
      charge = column_charges[column]
      if charge != c:
        start += num_non_zero[column]
      else:
        idxs.extend(start + np.arange(num_non_zero[column]))

    blocks[c] = np.reshape(data[idxs],
                           (row_degeneracies[c], column_degeneracies[c]))
  return blocks


class BlockSparseTensor:
  """
  Minimal class implementation of block sparsity.
  The class currently onluy supports a single U(1) symmetry.
  Currently only nump.ndarray is supported.
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
  # a few simple checks
  if np.prod(shape) != np.prod(tensor.shape):
    raise ValueError("A tensor with {} elements cannot be "
                     "reshaped into a tensor with {} elements".format(
                         np.prod(tensor.shape), np.prod(shape)))
  #copy indices
  result = BlockSparseTensor(
      data=tensor.data.copy(), indices=[i.copy() for i in tensor.indices])

  for n in range(len(shape)):
    if shape[n] > result.shape[n]:
      while shape[n] > result.shape[n]:
        #fuse indices
        i1, i2 = result.indices.pop(n), result.indices.pop(n)
        #note: the resulting flow is set to one since the flow
        #is multiplied into the charges. As a result the tensor
        #will then be invariant in any case.
        result.indices.insert(n, fuse_index_pair(i1, i2))
      if result.shape[n] > shape[n]:
        elementary_indices = []
        for i in tensor.indices:
          elementary_indices.extend(i.get_elementary_indices())
          raise ValueError("The shape {} is incompatible with the "
                           "elementary shape {} of the tensor.".format(
                               shape,
                               tuple(
                                   [e.dimension for e in elementary_indices])))
    elif shape[n] < result.shape[n]:
      while shape[n] < result.shape[n]:
        #split index at n
        try:
          i1, i2 = split_index(result.indices.pop(n))
        except ValueError:
          elementary_indices = []
          for i in tensor.indices:
            elementary_indices.extend(i.get_elementary_indices())
          raise ValueError("The shape {} is incompatible with the "
                           "elementary shape {} of the tensor.".format(
                               shape,
                               tuple(
                                   [e.dimension for e in elementary_indices])))
        result.indices.insert(n, i1)
        result.indices.insert(n + 1, i2)
      if result.shape[n] < shape[n]:
        raise ValueError(
            "shape {} is incompatible with the elementary result shape".format(
                shape))
  return result
