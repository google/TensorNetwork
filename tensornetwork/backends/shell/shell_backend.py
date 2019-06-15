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
from typing import Optional, Sequence, Tuple, List
from tensornetwork.backends import base_backend

# Treat tensors as tuples that carry the real tensor's shape.
# Conversion from tensor to tuple will happen somewhere else?
Tensor = Tuple


class ShellBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(ShellBackend, self).__init__()
    self.name = "shell"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    # Does not work when axis < 0
    gen_a = (x for i, x in enumerate(a) if i not in axes[0])
    gen_b = (x for i, x in enumerate(b) if i not in axes[1])
    return tuple(self._concat_generators(gen_a, gen_b))

  def _concat_generators(self, *gen):
    """Concatenates Python generators."""
    for g in gen:
      yield from g

  def reshape(self, tensor: Tensor, shape: Tensor):
    return shape

  def transpose(self, tensor, perm):
    return tuple(tensor[i] for i in perm)

  def svd_decomposition(self,
                        tensor: Tensor,
                        split_axis: int,
                        max_singular_values: Optional[int] = None,
                        max_truncation_error: Optional[float] = None
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    raise NotImplementedError("SVD shape cannot be calculated without"
                              "explicit tensor values.")

  def concat(self, values: Sequence[Tensor], axis: int) -> Tensor:
    # Does not work when axis < 0
    concat_dim = len(values) * values[0][axis]
    return values[0][:axis] + (concat_dim,) + values[0][axis + 1:]

  def shape(self, tensor: Tensor) -> Tensor:
    return tensor

  def prod(self, values: Tensor) -> Tensor:
    return values

  def sqrt(self, tensor: Tensor) -> Tensor:
    return tensor

  def diag(self, tensor: Tensor) -> Tensor:
    return (3 - len(tensor)) * tensor

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    return tensor

  def trace(self, tensor: Tensor) -> Tensor:
    return tensor[:-2]

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 + tensor2

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    expr_list = expression.split(",")
    expr_list[-1], res = expr_list[-1].split("->")
    return tuple(self._find_char(expr_list, char, tensors) for char in res)

  def _find_char(self, expr_list, char, tensors):
    """Finds character in einsum tensor expression.

    Args:
      expr_list: List with expression for input tensors in einsum.
      char: One character string (letter) that corresponds to a specific
        einsum component.

    Returns:
      i: Index of the tensor that has `char` components.
      ind: Index of `char` in the i-th expression string.
    """
    for i, expr in enumerate(expr_list):
      ind = expr.find(char)
      if ind != -1:
        return tensors[i][ind]
    raise ValueError("Einsum output expression contains letters not given"
                     "in input.")
