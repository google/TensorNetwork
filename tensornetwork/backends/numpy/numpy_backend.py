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

Tensor = Any


class NumPyBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(NumPyBackend, self).__init__()
    try:
      import numpy
    except ImportError:
      raise AssertionError("numpy is not installed.")
    from tensornetwork.backends.numpy import decompositions
    self.np = numpy
    self.decompositions = decompositions
    self.name = "numpy"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return self.np.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return self.np.reshape(tensor, shape.astype(self.np.int32))

  def transpose(self, tensor, perm):
    return self.np.transpose(tensor, perm)

  def svd_decomposition(self,
                        tensor: Tensor,
                        split_axis: int,
                        max_singular_values: Optional[int] = None,
                        max_truncation_error: Optional[float] = None
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return self.decompositions.svd_decomposition(
        self.np, tensor, split_axis, max_singular_values, max_truncation_error)

  def concat(self, values: Tensor, axis: int) -> Tensor:
    return self.np.concatenate(values, axis)

  def shape(self, tensor: Tensor) -> Tensor:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def prod(self, values: Tensor) -> Tensor:
    return self.np.prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return self.np.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    if len(tensor.shape) != 1:
      raise TypeError("Only one dimensional tensors are allowed as input")
    return self.np.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    if (not isinstance(tensor, self.np.ndarray) 
        and not self.np.isscalar(tensor)):
      raise ValueError(
        "Expected a `np.array` or scalar. Got {}".format(type(tensor)))
    return self.np.asarray(tensor)

  def trace(self, tensor: Tensor) -> Tensor:
    # Default np.trace uses first two axes.
    return self.np.trace(tensor, axis1=-2, axis2=-1)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.np.tensordot(tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return self.np.einsum(expression, *tensors)
