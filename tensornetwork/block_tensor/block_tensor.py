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

import numpy as np
import itertools
from typing import List, Union, Any, Tuple, Type, Optional
Tensor = Any


def check_flows(flows) -> None:
  if (set(flows) != {1}) and (set(flows) != {-1}) and (set(flows) != {-1, 1}):
    raise ValueError(
        "flows = {} contains values different from 1 and -1".format(flows))

  if set(flows) == {1}:
    raise ValueError("flows = {} has no outflowing index".format(flows))
  if set(flows) == {-1}:
    raise ValueError("flows = {} has no inflowing index".format(flows))


def fuse_quantum_numbers(q1: Union[List, np.ndarray],
                         q2: Union[List, np.ndarray]) -> np.ndarray:
  """
  Fuse quantumm numbers `q1` with `q2` by simple addition (valid
  for U(1) charges). `q1` and `q2` are typically two consecutive
  elements of `BlockSparseTensor.quantum_numbers`.
  Given `q1 = [0,1,2]` and `q2 = [10,100]`, this returns
  `[10, 11, 12, 100, 101, 102]`.
  When using column-major ordering of indices in `BlockSparseTensor`, 
  the position of q1 should be "to the left" of the position of q2.
  Args:
    q1: Iterable of integers
    q2: Iterable of integers
  Returns:
    np.ndarray: The result of fusing `q1` with `q2`.
  """
  return np.reshape(
      np.asarray(q2)[:, None] + np.asarray(q1)[None, :],
      len(q1) * len(q2))


def reshape(symmetric_tensor: BlockSparseTensor, shape: Tuple[int]):
  n = 0
  for s in shape:
    dim = 1
    while dim != s:
      dim *= symmetric_tensor.shape[n]
      n += 1
    if dim > s:
      raise ValueError(
          'desired shape = {} is incompatible with the symmetric tensor shape = {}'
          .format(shape, symmetric_tensor.shape))


def compute_num_nonzero(quantum_numbers: List[np.ndarray],
                        flows: List[Union[bool, int]]) -> int:
  """
  Compute the number of non-zero elements, given the meta-data of 
  a symmetric tensor.
  Args:
    quantum_numbers: List of np.ndarray, one for each leg. 
      Each np.ndarray `quantum_numbers[leg]` is of shape `(D[leg], Q)`.
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

  if len(quantum_numbers) == 1:
    return len(quantum_numbers)
  net_charges = flows[0] * quantum_numbers[0]
  for i in range(1, len(flows)):
    net_charges = np.reshape(
        flows[i] * quantum_numbers[i][:, None] + net_charges[None, :],
        len(quantum_numbers[i]) * len(net_charges))

  return len(np.nonzero(net_charges == 0)[0])


def compute_nonzero_block_shapes(quantum_numbers: List[np.ndarray],
                                 flows: List[Union[bool, int]]) -> dict:
  """
  Compute the blocks and their respective shapes of a symmetric tensor,
  given its meta-data.
  Args:
    quantum_numbers: List of np.ndarray, one for each leg. 
      Each np.ndarray `quantum_numbers[leg]` is of shape `(D[leg], Q)`.
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
  charges = []
  rank = len(quantum_numbers)
  #find the unique quantum numbers and their degeneracy on each leg
  for leg in range(rank):
    c, d = np.unique(quantum_numbers[leg], return_counts=True)
    charges.append(c)
    degeneracies.append(dict(zip(c, d)))

  #find all possible combination of leg charges c0, c1, ...
  #(with one charge per leg 0, 1, ...)
  #such that sum([flows[0] * c0, flows[1] * c1, ...]) = 0
  charge_combinations = list(
      itertools.product(
          *[charges[leg] * flows[leg] for leg in range(len(charges))]))
  net_charges = np.array([np.sum(c) for c in charge_combinations])
  zero_idxs = np.nonzero(net_charges == 0)[0]
  charge_shape_dict = {}
  for idx in zero_idxs:
    charges = charge_combinations[idx]
    shapes = [
        degeneracies[leg][flows[leg] * charges[leg]] for leg in range(rank)
    ]
    charge_shape_dict[charges] = shapes
  return charge_shape_dict


def retrieve_non_zero_diagonal_blocks(data: np.ndarray,
                                      quantum_numbers: List[np.ndarray],
                                      flows: List[Union[bool, int]]) -> dict:
  """
  Given the meta data and underlying data of a symmetric matrix, compute 
  all diagonal blocks and return them in a dict.
  Args: 
    data: An np.ndarray of the data. The number of elements in `data`
      has to match the number of non-zero elements defined by `quantum_numbers` 
      and `flows`
    quantum_numbers: List of np.ndarray, one for each leg. 
      Each np.ndarray `quantum_numbers[leg]` is of shape `(D[leg], Q)`.
      The bond dimension `D[leg]` can vary on each leg, the number of 
      symmetries `Q` has to be the same for each leg.
    flows: A list of integers, one for each leg,
      with values `1` or `-1`, denoting the flow direction
      of the charges on each leg. `1` is inflowing, `-1` is outflowing
      charge.
  """
  if len(quantum_numbers) != 2:
    raise ValueError("input has to be a two-dimensional symmetric matrix")
  check_flows(flows)
  if len(flows) != len(quantum_numbers):
    raise ValueError("`len(flows)` is different from `len(quantum_numbers)`")

  row_charges = quantum_numbers[0]  # a list of charges on each row
  column_charges = quantum_numbers[1]  # a list of charges on each column
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
    * self.quantum_numbers: A list of `np.ndarray` of shape
      (D, Q), where D is the bond dimension, and Q the number
      of different symmetries (this is 1 for now).
    * self.flows: A list of integers of length `k`.
        `self.flows` determines the flows direction of charges
        on each leg of the tensor. A value of `-1` denotes 
        outflowing charge, a value of `1` denotes inflowing
        charge.

  The tensor data is stored in self.data, a 1d np.ndarray.
  """

  def __init__(self, data: np.ndarray, quantum_numbers: List[np.ndarray],
               flows: List[Union[bool, int]]) -> None:
    """
    Args: 
      data: An np.ndarray of the data. The number of elements in `data`
        has to match the number of non-zero elements defined by `quantum_numbers` 
        and `flows`
      quantum_numbers: List of np.ndarray, one for each leg. 
        Each np.ndarray `quantum_numbers[leg]` is of shape `(D[leg], Q)`.
        The bond dimension `D[leg]` can vary on each leg, the number of 
        symmetries `Q` has to be the same for each leg.
      flows: A list of integers, one for each leg,
        with values `1` or `-1`, denoting the flow direction
        of the charges on each leg. `1` is inflowing, `-1` is outflowing
        charge.
    """
    block_dict = compute_nonzero_block_shapes(quantum_numbers, flows)
    num_non_zero_elements = np.sum([np.prod(s) for s in block_dict.values()])

    if num_non_zero_elements != len(data.flat):
      raise ValueError("number of tensor elements defined "
                       "by `quantum_numbers` is different from"
                       " len(data)={}".format(len(data.flat)))
    check_flows(flows)
    if len(flows) != len(quantum_numbers):
      raise ValueError(
          "len(flows) = {} is different from len(quantum_numbers) = {}".format(
              len(flows), len(quantum_numbers)))
    self.data = np.asarray(data.flat)  #do not copy data
    self.flows = flows
    self.quantum_numbers = quantum_numbers

  @classmethod
  def randn(cls,
            quantum_numbers: List[np.ndarray],
            flows: List[Union[bool, int]],
            dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a random symmetric tensor from random normal distribution.
    Args:
      quantum_numbers: List of np.ndarray, one for each leg. 
        Each np.ndarray `quantum_numbers[leg]` is of shape `(D[leg], Q)`.
        The bond dimension `D[leg]` can vary on each leg, the number of 
        symmetries `Q` has to be the same for each leg.
      flows: A list of integers, one for each leg,
        with values `1` or `-1`, denoting the flow direction
        of the charges on each leg. `1` is inflowing, `-1` is outflowing
        charge.
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    num_non_zero_elements = compute_num_nonzero(quantum_numbers, flows)
    backend = backend_factory.get_backend('numpy')
    data = backend.randn((num_non_zero_elements,), dtype=dtype)
    return cls(data=data, quantum_numbers=quantum_numbers, flows=flows)

  @property
  def shape(self) -> Tuple:
    return tuple([np.shape(q)[0] for q in self.quantum_numbers])

  @property
  def dtype(self) -> Type[np.number]:
    return self.data.dtype
