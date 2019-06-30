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
from typing import Optional, Any, Sequence, Tuple
from tensornetwork.backends import base_backend
from tensornetwork.backends.tensorflow import decompositions

# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
Tensor = Any


class TensorFlowBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(TensorFlowBackend, self).__init__()
    try:
      import tensorflow
    except ImportError:
      raise AssertionError("tensorflow is not installed.")
    from tensornetwork.backends.tensorflow import tensordot2
    self.tensordot2 = tensordot2
    self.tf = tensorflow
    self.name = "tensorflow"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return self.tensordot2.tensordot(a, b, axes)

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
    return self.tf.convert_to_tensor(tensor)

  def trace(self, tensor: Tensor) -> Tensor:
    return self.tf.linalg.trace(tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.tensordot2.tensordot(tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return self.tf.einsum(expression, *tensors)
