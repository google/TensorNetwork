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
from tensornetwork.block_sparse.block_tensor import BlockSparseTensor
# Note: this import has to stay here or some test will fail
# Some functions in block_tensor use isinstance(tensor,bt.ChargeArray)
# which, weirdly enough, gives the wrong result if block_tensor is imported
# within SymmetricBackend
#TODO: figure out what is going on here!
import tensornetwork.block_sparse.block_tensor as bt
import numpy
import scipy
Tensor = Any


class SymmetricBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(SymmetricBackend, self).__init__()
    self.bt = bt
    self.name = "symmetric"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return self.bt.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return self.bt.reshape(tensor, numpy.asarray(shape).astype(numpy.int32))

  def transpose(self, tensor, perm):
    return self.bt.transpose(tensor, perm)

  def svd_decomposition(self,
                        tensor: Tensor,
                        split_axis: int,
                        max_singular_values: Optional[int] = None,
                        max_truncation_error: Optional[float] = None
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(
        self.bt, tensor, split_axis, max_singular_values, max_truncation_error)

  def qr_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.qr_decomposition(self.bt, tensor, split_axis)

  def rq_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(self.bt, tensor, split_axis)

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
    return self.bt.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    return self.bt.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, BlockSparseTensor):
      raise TypeError(
          "cannot convert tensor of type `{}` to `BlockSparseTensor`".format(
              type(tensor)))
    return tensor

  def trace(self, tensor: Tensor) -> Tensor:
    # Default np.trace uses first two axes.
    return self.bt.trace(tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.bt.tensordot(tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    raise NotImplementedError("`einsum` currently not implemented")

  def norm(self, tensor: Tensor) -> Tensor:
    return self.bt.norm(tensor)

  def eye(self,
          N: Index,
          dtype: Optional[numpy.dtype] = None,
          M: Optional[Index] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64

    return self.bt.eye(N, M, dtype=dtype)

  def ones(self, shape: List[Index],
           dtype: Optional[numpy.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64
    return self.bt.ones(shape, dtype=dtype)

  def zeros(self, shape: List[Index],
            dtype: Optional[numpy.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64
    return self.bt.zeros(shape, dtype=dtype)

  def randn(self,
            shape: List[Index],
            dtype: Optional[numpy.dtype] = None,
            seed: Optional[int] = None) -> Tensor:

    if seed:
      numpy.random.seed(seed)
    return self.bt.randn(shape, dtype)

  def random_uniform(self,
                     shape: List[Index],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[numpy.dtype] = None,
                     seed: Optional[int] = None) -> Tensor:

    if seed:
      numpy.random.seed(seed)
    dtype = dtype if dtype is not None else numpy.float64
    return self.bt.rand(shape, boundaries, dtype)

  def conj(self, tensor: Tensor) -> Tensor:
    return self.bt.conj(tensor)

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return self.bt.eigh(matrix)

  def eigs(self,
           A: Callable,
           initial_state: Optional[Tensor] = None,
           num_krylov_vecs: Optional[int] = 200,
           numeig: Optional[int] = 6,
           tol: Optional[float] = 1E-8,
           which: Optional[Text] = 'LR',
           maxiter: Optional[int] = None,
           dtype: Optional[Type[numpy.number]] = None) -> Tuple[List, List]:
    raise NotImplementedError()

  def eigsh_lanczos(
      self,
      A: Callable,
      initial_state: Optional[Tensor] = None,
      num_krylov_vecs: Optional[int] = 200,
      numeig: Optional[int] = 1,
      tol: Optional[float] = 1E-8,
      delta: Optional[float] = 1E-8,
      ndiag: Optional[int] = 20,
      reorthogonalize: Optional[bool] = False) -> Tuple[List, List]:
    raise NotImplementedError()

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
    return self.bt.inv(matrix)
