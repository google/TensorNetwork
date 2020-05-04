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
#pylint: disable=line-too-long
from typing import Optional, Any, Sequence, Tuple, Type, Callable, List, Text
from tensornetwork.backends import base_backend
from tensornetwork.backends.tensorflow import decompositions
from tensornetwork.backends.tensorflow import tensordot2

# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
import numpy as np
Tensor = Any

#pylint: disable=abstract-method


class TensorFlowBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    # pylint: disable=global-variable-undefined
    global tf
    super(TensorFlowBackend, self).__init__()
    try:
      #pylint: disable=import-outside-toplevel
      import tensorflow
    except ImportError:
      raise ImportError("Tensorflow not installed, please switch to a "
                        "different backend or install Tensorflow.")
    tf = tensorflow
    self.name = "tensorflow"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return tensordot2.tensordot(tf, a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return tf.reshape(tensor, shape)

  def transpose(self, tensor, perm):
    return tf.transpose(tensor, perm)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    return tf.slice(tensor, start_indices, slice_sizes)

  def svd_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(
        tf,
        tensor,
        split_axis,
        max_singular_values,
        max_truncation_error,
        relative=relative)

  def qr_decomposition(self, tensor: Tensor,
                       split_axis: int) -> Tuple[Tensor, Tensor]:
    return decompositions.qr_decomposition(tf, tensor, split_axis)

  def rq_decomposition(self, tensor: Tensor,
                       split_axis: int) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(tf, tensor, split_axis)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return tf.concat(values, axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return tf.shape(tensor)

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tuple(tensor.shape.as_list())

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return self.shape_tuple(tensor)

  def shape_prod(self, values: Tensor) -> Tensor:
    return tf.reduce_prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return tf.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    return tf.linalg.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    result = tf.convert_to_tensor(tensor)
    return result

  def trace(self, tensor: Tensor) -> Tensor:
    return tf.linalg.trace(tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensordot2.tensordot(tf, tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return tf.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return tf.linalg.norm(tensor)

  def eye(self,
          N: int,
          dtype: Optional[Type[np.number]] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else tf.float64
    return tf.eye(num_rows=N, num_columns=M, dtype=dtype)

  def ones(self,
           shape: Tuple[int, ...],
           dtype: Optional[Type[np.number]] = None) -> Tensor:
    dtype = dtype if dtype is not None else tf.float64
    return tf.ones(shape=shape, dtype=dtype)

  def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None) -> Tensor:
    dtype = dtype if dtype is not None else tf.float64
    return tf.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None,
            seed: Optional[int] = None) -> Tensor:
    if seed:
      tf.random.set_seed(seed)

    dtype = dtype if dtype is not None else tf.float64
    if (dtype is tf.complex128) or (dtype is tf.complex64):
      return tf.complex(
          tf.random.normal(shape=shape, dtype=dtype.real_dtype),
          tf.random.normal(shape=shape, dtype=dtype.real_dtype))
    return tf.random.normal(shape=shape, dtype=dtype)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[Type[np.number]] = None,
                     seed: Optional[int] = None) -> Tensor:
    if seed:
      tf.random.set_seed(seed)

    dtype = dtype if dtype is not None else tf.float64
    if (dtype is tf.complex128) or (dtype is tf.complex64):
      return tf.complex(
          tf.random.uniform(
              shape=shape,
              minval=boundaries[0],
              maxval=boundaries[1],
              dtype=dtype.real_dtype),
          tf.random.uniform(
              shape=shape,
              minval=boundaries[0],
              maxval=boundaries[1],
              dtype=dtype.real_dtype))
    tf.random.set_seed(10)
    a = tf.random.uniform(
        shape=shape, minval=boundaries[0], maxval=boundaries[1], dtype=dtype)
    return a

  def conj(self, tensor: Tensor) -> Tensor:
    return tf.math.conj(tensor)

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return tf.linalg.eigh(matrix)

  def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 + tensor2

  def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 - tensor2

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 * tensor2

  def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 / tensor2

  def index_update(self, tensor: Tensor, mask: Tensor,
                   assignee: Tensor) -> Tensor:
    #returns a copy (unfortunately)
    return tf.where(mask, assignee, tensor)

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError("input to tensorflow backend method `inv` has shape {}. "
                       "Only matrices are supported.".format(tf.shape(matrix)))
    return tf.linalg.inv(matrix)

  def broadcast_right_multiplication(self, tensor1: Tensor,
                                     tensor2: Tensor) -> Tensor:
    if len(tensor2.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor2`, "
                       "found `tensor2.shape = {}`".format(tf.shape(tensor2)))

    return tensor1 * tensor2

  def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tf.shape(tensor1)))

    t1_broadcast_shape = self.shape_concat(
        [self.shape_tensor(tensor1), [1] * (len(tensor2.shape) - 1)], axis=-1)
    return tensor2 * self.reshape(tensor1, t1_broadcast_shape)

  def sin(self, tensor: Tensor):
    return tf.math.sin(tensor)

  def cos(self, tensor: Tensor):
    return tf.math.cos(tensor)

  def exp(self, tensor: Tensor):
    return tf.math.exp(tensor)

  def log(self, tensor: Tensor):
    return tf.math.log(tensor)

  def expm(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) != 2:
      raise ValueError("input to tensorflow backend method `expm` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError("input to tensorflow backend method `expm` only supports"
                       "N*N matrix, {x}*{y} matrix is given".format(
                           x=matrix.shape[0], y=matrix.shape[1]))
    return tf.linalg.expm(matrix)
