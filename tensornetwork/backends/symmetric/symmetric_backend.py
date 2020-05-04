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
#pyling: disable=line-too-long
from typing import Optional, Any, Sequence, Tuple, Callable, List, Text, Type
from tensornetwork.backends import base_backend
from tensornetwork.backends.symmetric import decompositions
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import BlockSparseTensor
import tensornetwork.block_sparse as bs
import numpy
Tensor = Any

#TODO (mganahl): implement sparse solvers


#pylint: disable=abstract-method
class SymmetricBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(SymmetricBackend, self).__init__()
    self.bs = bs
    self.name = "symmetric"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return self.bs.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return self.bs.reshape(tensor, numpy.asarray(shape).astype(numpy.int32))

  def transpose(self, tensor, perm):
    return self.bs.transpose(tensor, perm)

  def svd_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(self.bs, tensor, split_axis,
                                            max_singular_values,
                                            max_truncation_error, relative)

  def qr_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.qr_decomposition(self.bs, tensor, split_axis)

  def rq_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(self.bs, tensor, split_axis)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return numpy.concatenate(values, axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.sparse_shape

  def shape_prod(self, values: Tensor) -> Tensor:
    return numpy.prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return self.bs.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    return self.bs.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, BlockSparseTensor):
      raise TypeError(
          "cannot convert tensor of type `{}` to `BlockSparseTensor`".format(
              type(tensor)))
    return tensor

  def trace(self, tensor: Tensor) -> Tensor:
    # Default np.trace uses first two axes.
    return self.bs.trace(tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.bs.tensordot(tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    raise NotImplementedError("`einsum` currently not implemented")

  def norm(self, tensor: Tensor) -> Tensor:
    return self.bs.norm(tensor)

  def eye(self,
          N: Index,
          dtype: Optional[numpy.dtype] = None,
          M: Optional[Index] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64

    return self.bs.eye(N, M, dtype=dtype)

  def ones(self,
           shape: List[Index],
           dtype: Optional[numpy.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64
    return self.bs.ones(shape, dtype=dtype)

  def zeros(self,
            shape: List[Index],
            dtype: Optional[numpy.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64
    return self.bs.zeros(shape, dtype=dtype)

  def randn(self,
            shape: List[Index],
            dtype: Optional[numpy.dtype] = None,
            seed: Optional[int] = None) -> Tensor:

    if seed:
      numpy.random.seed(seed)
    return self.bs.randn(shape, dtype)

  def random_uniform(self,
                     shape: List[Index],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[numpy.dtype] = None,
                     seed: Optional[int] = None) -> Tensor:

    if seed:
      numpy.random.seed(seed)
    dtype = dtype if dtype is not None else numpy.float64
    return self.bs.random(shape, boundaries, dtype)

  def conj(self, tensor: Tensor) -> Tensor:
    return self.bs.conj(tensor)

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return self.bs.eigh(matrix)

  def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 + tensor2

  def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 - tensor2

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 * tensor2

  def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 / tensor2

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError("input to symmetric backend method `inv` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    return self.bs.inv(matrix)

  def broadcast_right_multiplication(self, tensor1: Tensor, tensor2: Tensor):
    if tensor2.ndim != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor2`,"
                       " found `tensor2.shape = {}`".format(tensor2.shape))
    return self.tensordot(tensor1, self.diag(tensor2),
                          ([len(tensor1.shape) - 1], [0]))

  def broadcast_left_multiplication(self, tensor1: Tensor, tensor2: Tensor):
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tensor1.shape))
    return self.tensordot(self.diag(tensor1), tensor2, ([1], [0]))
