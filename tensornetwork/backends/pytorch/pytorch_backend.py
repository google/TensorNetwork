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
from tensornetwork.backends.pytorch import decompositions
import numpy as np
# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
Tensor = Any


class PyTorchBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(PyTorchBackend, self).__init__()
    try:
      import torch
    except ImportError:
      raise AssertionError("pytorch is not installed.")
    self.torch = torch
    self.name = "pytorch"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return self.torch.tensordot(a, b, dims=axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return self.torch.reshape(tensor, tuple(np.array(shape).astype(int)))

  def transpose(self, tensor, perm):
    return tensor.permute(perm)

  def svd_decomposition(self,
                        tensor: Tensor,
                        split_axis: int,
                        max_singular_values: Optional[int] = None,
                        max_truncation_error: Optional[float] = None
                        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(self.torch, tensor, split_axis,
                                            max_singular_values,
                                            max_truncation_error)

  def concat(self, values: Tensor, axis: int) -> Tensor:
    return np.concatenate(values, axis)

  def shape(self, tensor: Tensor) -> Tensor:
    return self.torch.tensor(list(tensor.shape))

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tuple(tensor.shape)

  def prod(self, values: Tensor) -> int:
    return np.prod(np.array(values))

  def sqrt(self, tensor: Tensor) -> Tensor:
    return self.torch.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    return self.torch.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    return self.torch.as_tensor(tensor)

  def trace(self, tensor: Tensor) -> Tensor:
    return self.torch.einsum('...jj', tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.torch.tensordot(tensor1, tensor2, dims=0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return self.torch.einsum(expression, *tensors)
