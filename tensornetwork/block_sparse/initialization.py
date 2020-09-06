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
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import _randn, _random
from tensornetwork.block_sparse.blocksparsetensor import BlockSparseTensor
from typing import Tuple, Type, Optional, Sequence


def ones(indices: Sequence[Index],
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


def zeros(indices: Sequence[Index],
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


def randn(indices: Sequence[Index],
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


def random(indices: Sequence[Index],
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


def empty_like(tensor: BlockSparseTensor) -> BlockSparseTensor:
  """
  Initialize a symmetric tensor with an uninitialized np.ndarray.
  The resulting tensor has the same shape and dtype as `tensor`.
  Args:
    tensor: A BlockSparseTensor.
  Returns:
    BlockSparseTensor
  """
  return BlockSparseTensor(
      np.empty(tensor.data.size, dtype=tensor.dtype),
      charges=tensor._charges,
      flows=tensor._flows,
      order=tensor._order,
      check_consistency=False)


def ones_like(tensor: BlockSparseTensor) -> BlockSparseTensor:
  """
  Initialize a symmetric tensor with ones.
  The resulting tensor has the same shape and dtype as `tensor`.
  Args:
    tensor: A BlockSparseTensor.
  Returns:
    BlockSparseTensor
  """
  return BlockSparseTensor(
      np.ones(tensor.data.size, dtype=tensor.dtype),
      charges=tensor._charges,
      flows=tensor._flows,
      order=tensor._order,
      check_consistency=False)


def zeros_like(tensor: BlockSparseTensor) -> BlockSparseTensor:
  """
  Initialize a symmetric tensor with zeros.
  The resulting tensor has the same shape and dtype as `tensor`.
  Args:
    tensor: A BlockSparseTensor.
  Returns:
    BlockSparseTensor
  """
  return BlockSparseTensor(
      np.zeros(tensor.data.size, dtype=tensor.dtype),
      charges=tensor._charges,
      flows=tensor._flows,
      order=tensor._order,
      check_consistency=False)


def randn_like(tensor: BlockSparseTensor) -> BlockSparseTensor:
  """
  Initialize a symmetric tensor with random gaussian numbers.
  The resulting tensor has the same shape and dtype as `tensor`.
  Args:
    tensor: A BlockSparseTensor.
  Returns:
    BlockSparseTensor
  """
  return BlockSparseTensor(
      _randn(tensor.data.size, dtype=tensor.dtype),
      charges=tensor._charges,
      flows=tensor._flows,
      order=tensor._order,
      check_consistency=False)


def random_like(
    tensor: BlockSparseTensor, boundaries: Tuple = (0, 1)) -> BlockSparseTensor:
  """
  Initialize a symmetric tensor with random uniform numbers.
  The resulting tensor has the same shape and dtype as `tensor`.
  Args:
    tensor: A BlockSparseTensor.
  Returns:
    BlockSparseTensor
  """
  return BlockSparseTensor(
      _random(tensor.data.size, dtype=tensor.dtype, boundaries=boundaries),
      charges=tensor._charges,
      flows=tensor._flows,
      order=tensor._order,
      check_consistency=False)
