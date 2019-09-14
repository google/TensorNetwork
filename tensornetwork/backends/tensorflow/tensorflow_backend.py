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
from typing import Optional, Any, Sequence, Tuple, Type
from tensornetwork.backends import base_backend
from tensornetwork.backends.tensorflow import decompositions
from tensornetwork.backends.tensorflow import tensordot2

# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
import numpy as np
Tensor = Any


class TensorFlowBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self, dtype: Optional[Type[np.number]] = None):
    super(TensorFlowBackend, self).__init__()
    try:
      import tensorflow as tf
    except ImportError:
      raise ImportError("Tensorflow not installed, please switch to a "
                        "different backend or install Tensorflow.")
    self.tf = tf
    self.name = "tensorflow"
    self.dtype = dtype

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return tensordot2.tensordot(self.tf, a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return self.tf.reshape(tensor, shape)

  def transpose(self, tensor, perm):
    return self.tf.transpose(tensor, perm)

  def svd_decomposition(self,
                        tensor: Tensor,
                        split_axis: int,
                        max_singular_values: Optional[int] = None,
                        max_truncation_error: Optional[float] = None
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(
        self.tf, tensor, split_axis, max_singular_values, max_truncation_error)

  def qr_decomposition(self, tensor: Tensor,
                       split_axis: int) -> Tuple[Tensor, Tensor]:
    return decompositions.qr_decomposition(self.tf, tensor, split_axis)

  def rq_decomposition(self, tensor: Tensor,
                       split_axis: int) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(self.tf, tensor, split_axis)

  def concat(self, values: Tensor, axis: int) -> Tensor:
    return self.tf.concat(values, axis)

  def shape(self, tensor: Tensor) -> Tensor:
    return self.tf.shape(tensor)

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tuple(tensor.shape.as_list())

  def prod(self, values: Tensor) -> Tensor:
    return self.tf.reduce_prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return self.tf.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    return self.tf.linalg.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    result = self.tf.convert_to_tensor(tensor)
    if self.dtype is not None and result.dtype is not self.dtype:
      raise TypeError(
          "Backend '{}' cannot convert tensor of dtype {} to dtype {}".format(
              self.name, result.dtype, self.dtype))
    return result

  def trace(self, tensor: Tensor) -> Tensor:
    return self.tf.linalg.trace(tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensordot2.tensordot(self.tf, tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return self.tf.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return self.tf.linalg.norm(tensor)

  def eye(self,
          N: int,
          dtype: Optional[Type[np.number]] = None,
          M: Optional[int] = None) -> Tensor:
    if not dtype:
      dtype = self.dtype if self.dtype is not None else self.tf.float64
    return self.tf.eye(num_rows=N, num_columns=M, dtype=dtype)

  def ones(self,
           shape: Tuple[int, ...],
           dtype: Optional[Type[np.number]] = None) -> Tensor:
    if not dtype:
      dtype = self.dtype if self.dtype is not None else self.tf.float64

    return self.tf.ones(shape=shape, dtype=dtype)

  def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None) -> Tensor:
    if not dtype:
      dtype = self.dtype if self.dtype is not None else self.tf.float64

    return self.tf.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None,
            seed: Optional[int] = None) -> Tensor:
    if seed:
      self.tf.random.set_random_seed(seed)

    if not dtype:
      dtype = self.dtype if self.dtype is not None else self.tf.float64

    if (dtype is self.tf.complex128) or (dtype is self.tf.complex64):
      return self.tf.complex(
          self.tf.random_normal(shape=shape, dtype=dtype.real_dtype),
          self.tf.random_normal(shape=shape, dtype=dtype.real_dtype))
    return self.tf.random_normal(shape=shape, dtype=dtype)

  def conj(self, tensor: Tensor) -> Tensor:
    return self.tf.conj(tensor)
