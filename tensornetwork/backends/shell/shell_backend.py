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

import functools
import operator
from tensornetwork.backends import base_backend
from typing import Optional, Sequence, Tuple, List, Any


class ShellTensor:
  def __init__(self, shape: Tuple):
    self.shape = shape

  def reshape(self, new_shape: Tuple):
    self.shape = new_shape
    return self

Tensor = ShellTensor


class ShellBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(ShellBackend, self).__init__()
    self.name = "shell"

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Sequence[Sequence[int]]) -> Tensor:
    # Does not work when axis < 0
    gen_a = (x for i, x in enumerate(a.shape) if i not in axes[0])
    gen_b = (x for i, x in enumerate(b.shape) if i not in axes[1])
    return ShellTensor(tuple(self._concat_generators(gen_a, gen_b)))

  def _concat_generators(self, *gen):
    """Concatenates Python generators."""
    for g in gen:
      yield from g

  def reshape(self, tensor: Tensor, shape: Sequence[int]) -> Tensor:
    tensor = tensor.reshape(tuple(shape))
    return tensor

  def transpose(self, tensor: Tensor, perm: Sequence[int]) -> Tensor:
    shape = tuple(tensor.shape[i] for i in perm)
    tensor = tensor.reshape(tuple(shape))
    return tensor

  def svd_decomposition(self,
                        tensor: Tensor,
                        split_axis: int,
                        max_singular_values: Optional[int] = None,
                        max_truncation_error: Optional[float] = None
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if max_truncation_error is not None:
      raise NotImplementedError("SVD with truncation shape cannot be "
                                "calculated without explicit tensor values.")
    left_dims = tensor.shape[:split_axis]
    right_dims = tensor.shape[split_axis:]
    dim_s0 = min(functools.reduce(operator.mul, left_dims),
                 functools.reduce(operator.mul, right_dims))
    if max_singular_values is not None:
      dim_s = min(dim_s0, max_singular_values)
    else:
      dim_s = dim_s0

    u = ShellTensor(left_dims + (dim_s,))
    vh = ShellTensor((dim_s,) + right_dims)
    s = ShellTensor((dim_s,))
    s_rest = ShellTensor((dim_s0 - dim_s,))
    return u, s, vh, s_rest

  def concat(self, values: Sequence[Tensor], axis: int) -> Tensor:
    shape = values[0].shape
    if axis < 0:
      axis += len(shape)
    concat_size = sum(v.shape[axis] for v in values)
    new_shape = shape[:axis] + (concat_size,) + shape[axis + 1:]
    return ShellTensor(new_shape)

  def concat_shape(self, values) -> Sequence:
    tuple_values = (tuple(v) for v in values)
    return functools.reduce(operator.concat, tuple_values)

  def shape(self, tensor: Tensor) -> Tuple:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def prod(self, values: Tensor) -> int:
    # This is different from the BaseBackend prod!
    # prod calculates the product of tensor elements and cannot implemented
    # for shell tensors
    # This returns the product of sizes instead
    return self.shape_prod(values.shape)

  def shape_prod(self, shape: Sequence[int]) -> int:
    return functools.reduce(operator.mul, shape)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return tensor

  def diag(self, tensor: Tensor) -> Tensor:
    shape = tensor.shape
    new_tensor = ShellTensor((3 - len(shape)) * shape)
    return new_tensor

  def convert_to_tensor(self, tensor: Any) -> Tensor:
    shell_tensor = ShellTensor(tuple(tensor.shape))
    return shell_tensor

  def trace(self, tensor: Tensor) -> Tensor:
    return ShellTensor(tensor.shape[:-2])

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return ShellTensor(tensor1.shape + tensor2.shape)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    expr_list = expression.split(",")
    expr_list[-1], res = expr_list[-1].split("->")
    shape = tuple(self._find_char(expr_list, char, tensors) for char in res)
    return ShellTensor(shape)

  def _find_char(self, expr_list: List[str], char: str,
                 tensors: Sequence[Tensor]) -> int:
    """Finds character in einsum tensor expression.

    Args:
      expr_list: List with expression for input tensors in einsum.
      char: One character string (letter) that corresponds to a specific
        einsum component.

    Returns:
      size: Size of the axis that corresponds to this einsum expression
        character.
    """
    for i, expr in enumerate(expr_list):
      ind = expr.find(char)
      if ind != -1:
        return tensors[i].shape[ind]
    raise ValueError("Einsum output expression contains letters not given"
                     "in input.")
