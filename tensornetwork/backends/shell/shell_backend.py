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

import functools
import operator
from tensornetwork.backends import abstract_backend
#pylint: disable=line-too-long
from typing import Optional, Sequence, Tuple, List, Any, Union, Type, Callable, Text
import numpy as np


class ShellTensor:

  def __init__(self, shape: Tuple[int, ...], dtype=None):
    self.shape = shape
    self.dtype = dtype

  def reshape(self, new_shape: Tuple[int, ...]):
    self.shape = new_shape
    return self


Tensor = Any


class ShellBackend(abstract_backend.AbstractBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self) -> None:
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

  def svd_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if max_truncation_error is not None:
      raise NotImplementedError("SVD with truncation shape cannot be "
                                "calculated without explicit tensor values.")
    left_dims = tensor.shape[:split_axis]
    right_dims = tensor.shape[split_axis:]
    dim_s0 = min(
        functools.reduce(operator.mul, left_dims),
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

  def qr_decomposition(self, tensor: Tensor,
                       split_axis: int) -> Tuple[Tensor, Tensor]:

    left_dims = tensor.shape[:split_axis]
    right_dims = tensor.shape[split_axis:]
    center_dim = min(np.prod(left_dims), np.prod(right_dims))
    q = ShellTensor(left_dims + (center_dim,))
    r = ShellTensor((center_dim,) + right_dims)
    return q, r

  def rq_decomposition(self, tensor: Tensor,
                       split_axis: int) -> Tuple[Tensor, Tensor]:

    left_dims = tensor.shape[:split_axis]
    right_dims = tensor.shape[split_axis:]
    center_dim = min(np.prod(left_dims), np.prod(right_dims))
    q = ShellTensor(left_dims + (center_dim,))
    r = ShellTensor((center_dim,) + right_dims)
    return q, r

  def shape_concat(self, values: Sequence[Tensor], axis: int) -> Tensor:
    shape = values[0].shape
    if axis < 0:
      axis += len(shape)
    concat_size = sum(v.shape[axis] for v in values)
    new_shape = shape[:axis] + (concat_size,) + shape[axis + 1:]
    return ShellTensor(new_shape)

  def concat_shape(self, values) -> Sequence:
    tuple_values = (tuple(v) for v in values)
    return functools.reduce(operator.concat, tuple_values)

  def shape_tensor(self, tensor: Tensor) -> Tuple:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def sparse_shape(self, tensor):
    return self.shape_tuple(tensor)

  def shape_prod(self, values: Tensor) -> int:
    # This is different from the BaseBackend prod!
    # prod calculates the product of tensor elements and cannot implemented
    # for shell tensors
    # This returns the product of sizes instead
    return self.shape_product(values.shape)

  def shape_product(self, shape: Sequence[int]) -> int:
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

  #pylint: disable=unused-argument
  def einsum(self,
             expression: str,
             *tensors: Tensor,
             optimize: bool = True) -> Tensor:
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

  def norm(self, tensor: Tensor) -> Tensor:
    return ShellTensor(())

  def eye(self,
          N: int,
          dtype: Optional[Type[np.number]] = None,
          M: Optional[int] = None) -> Tensor:
    if not M:
      M = N
    return ShellTensor((N, M))

  def ones(self,
           shape: Tuple[int, ...],
           dtype: Optional[Type[np.number]] = None) -> Tensor:
    return ShellTensor(shape)

  def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None) -> Tensor:

    return ShellTensor(shape)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None,
            seed: Optional[int] = None) -> Tensor:
    return ShellTensor(shape)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[Type[np.number]] = None,
                     seed: Optional[int] = None) -> Tensor:
    return ShellTensor(shape)

  def conj(self, tensor: Tensor) -> Tensor:
    return tensor

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    shape = matrix.shape
    return ShellTensor((shape[0],)), ShellTensor(shape)

  def eigs(self,
           A: Callable,
           args: Optional[List[Tensor]] = None,
           initial_state: Optional[Tensor] = None,
           shape: Optional[Tuple[int, ...]] = None,
           dtype: Optional[Type[np.number]] = None,
           num_krylov_vecs: Optional[int] = 200,
           numeig: Optional[int] = 1,
           tol: Optional[float] = 1E-8,
           which: Optional[Text] = 'LR',
           maxiter: Optional[int] = None) -> Tuple[Tensor, List]:
    if args is None:
      args = []

    if num_krylov_vecs < numeig + 1:
      raise ValueError('`num_krylov_vecs` >= `numeig+1` required!')

    if initial_state is None:
      if (shape is None) or (dtype is None):
        raise ValueError("if no `initial_state` is passed, then `shape` and"
                         "`dtype` have to be provided")
      return [ShellTensor(tuple()) for _ in range(numeig)
             ], [ShellTensor(shape) for _ in range(numeig)]
    if not isinstance(initial_state, ShellTensor):
      raise TypeError("Expected a `ShellTensor`. Got {}".format(
          type(initial_state)))

    return [ShellTensor(tuple()) for _ in range(numeig)
           ], [ShellTensor(initial_state.shape) for _ in range(numeig)]

  def eigsh_lanczos(self,
                    A: Callable,
                    args: Optional[List[Tensor]] = None,
                    initial_state: Optional[Tensor] = None,
                    shape: Optional[Tuple] = None,
                    dtype: Optional[Type[np.number]] = None,
                    num_krylov_vecs: int = 20,
                    numeig: int = 1,
                    tol: float = 1E-8,
                    delta: float = 1E-8,
                    ndiag: int = 20,
                    reorthogonalize: bool = False) -> Tuple[Tensor, List]:
    if args is None:
      args = []
    if num_krylov_vecs < numeig:
      raise ValueError('`num_krylov_vecs` >= `numeig` required!')

    if numeig > 1 and not reorthogonalize:
      raise ValueError(
          "Got numeig = {} > 1 and `reorthogonalize = False`. "
          "Use `reorthogonalize=True` for `numeig > 1`".format(numeig))
    if initial_state is None:
      if (shape is None) or (dtype is None):
        raise ValueError("if no `initial_state` is passed, then `shape` and"
                         "`dtype` have to be provided")
      return [ShellTensor(tuple()) for _ in range(numeig)
             ], [ShellTensor(shape) for _ in range(numeig)]
    if not isinstance(initial_state, ShellTensor):
      raise TypeError("Expected a `ShellTensor`. Got {}".format(
          type(initial_state)))

    return [ShellTensor(tuple()) for _ in range(numeig)
           ], [ShellTensor(initial_state.shape) for _ in range(numeig)]

  def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    raise NotImplementedError("Shell tensor has not implemented addition( + )")

  def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    raise NotImplementedError(
        "Shell tensor has not implemented subtraction( - )")

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    a = np.ones(tensor1.shape)
    b = np.ones(tensor2.shape)
    return ShellTensor((a * b).shape)

  def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    raise NotImplementedError("Shell tensor has not implemented add( / )")

  def index_update(self, tensor: Tensor, mask: Tensor,
                   assignee: Tensor) -> Tensor:
    return ShellTensor(tensor.shape)

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError(
          "input to shell backend method `inv` has shape {}. Only matrices are supported."
          .format(matrix.shape))
    return ShellTensor(matrix.shape)

  def broadcast_right_multiplication(self, tensor1: Tensor,
                                     tensor2: Tensor) -> Tensor:
    if len(tensor2.shape) != 1:
      raise ValueError(
          "only order-1 tensors are allowed for `tensor2`, found `tensor2.shape = {}`"
          .format(tensor2.shape))

    shape2 = tuple(tensor2.shape)
    if len(shape2) < len(tensor1.shape):
      shape2 = tuple([1] * (len(tensor1.shape) - len(shape2))) + shape2
    shape = tuple([max([s1, s2]) for s1, s2 in zip(tensor1.shape, shape2)])
    return ShellTensor(shape)

  def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
    if len(tensor1.shape) != 1:
      raise ValueError(
          "only order-1 tensors are allowed for `tensor1`, found `tensor1.shape = {}`"
          .format(tensor1.shape))

    shape1 = tuple(tensor1.shape)
    if len(shape1) < len(tensor2.shape):
      shape1 = shape1 + tuple([1] * (len(tensor2.shape) - len(shape1)))
    shape = tuple([max([s1, s2]) for s1, s2 in zip(tensor2.shape, shape1)])
    return ShellTensor(shape)

  def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
    return fun

  def sum(self,
          tensor: Tensor,
          axis: Optional[Sequence[int]] = None,
          keepdims: bool = False) -> Tensor:
    if not keepdims:
      newshape = np.delete(tensor.shape, axis)
    else:
      newshape = np.array(tensor.shape)
      newshape[np.array(axis)] = 1
    return ShellTensor(newshape)

  def matmul(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    shape1 = np.array(tensor1.shape)[:-2]
    shape2 = np.array(tensor2.shape)[:-2]
    if not np.array_equal(shape1, shape2):
      raise ValueError("shape mismatch for matmul")
    new_shape = np.append(shape1, [tensor1.shape[-2], tensor2.shape[-1]])
    return ShellTensor(new_shape)
