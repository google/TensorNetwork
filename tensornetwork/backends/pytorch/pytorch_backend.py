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
from typing import Optional, Any, Sequence, Tuple, Callable, List, Text, Type
from tensornetwork.backends import base_backend
from tensornetwork.backends.pytorch import decompositions
import numpy as np

# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
Tensor = Any

#pylint: disable=abstract-method


class PyTorchBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(PyTorchBackend, self).__init__()
    # pylint: disable=global-variable-undefined
    global torchlib
    try:
      #pylint: disable=import-outside-toplevel
      import torch
    except ImportError:
      raise ImportError("PyTorch not installed, please switch to a different "
                        "backend or install PyTorch.")
    torchlib = torch
    self.name = "pytorch"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return torchlib.tensordot(a, b, dims=axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return torchlib.reshape(tensor, tuple(np.array(shape).astype(int)))

  def transpose(self, tensor, perm):
    return tensor.permute(perm)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    obj = tuple(
        slice(start, start + size)
        for start, size in zip(start_indices, slice_sizes))
    return tensor[obj]

  def svd_decomposition(self,
                        tensor: Tensor,
                        split_axis: int,
                        max_singular_values: Optional[int] = None,
                        max_truncation_error: Optional[float] = None,
                        relative: Optional[bool] = False
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(
        torchlib,
        tensor,
        split_axis,
        max_singular_values,
        max_truncation_error,
        relative=relative)

  def qr_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.qr_decomposition(torchlib, tensor, split_axis)

  def rq_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(torchlib, tensor, split_axis)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return np.concatenate(values, axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return torchlib.tensor(list(tensor.shape))

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tuple(tensor.shape)

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return self.shape_tuple(tensor)

  def shape_prod(self, values: Tensor) -> int:
    return np.prod(np.array(values))

  def sqrt(self, tensor: Tensor) -> Tensor:
    return torchlib.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    return torchlib.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    result = torchlib.as_tensor(tensor)
    return result

  def trace(self, tensor: Tensor) -> Tensor:
    return torchlib.einsum('...jj', tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return torchlib.tensordot(tensor1, tensor2, dims=0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return torchlib.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return torchlib.norm(tensor)

  def eye(self, N: int, dtype: Optional[Any] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else torchlib.float64
    if not M:
      M = N  #torch crashes if one passes M = None with dtype!=None
    return torchlib.eye(n=N, m=M, dtype=dtype)

  def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Tensor:
    dtype = dtype if dtype is not None else torchlib.float64
    return torchlib.ones(shape, dtype=dtype)

  def zeros(self, shape: Tuple[int, ...],
            dtype: Optional[Any] = None) -> Tensor:
    dtype = dtype if dtype is not None else torchlib.float64
    return torchlib.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Any] = None,
            seed: Optional[int] = None) -> Tensor:
    if seed:
      torchlib.manual_seed(seed)
    dtype = dtype if dtype is not None else torchlib.float64
    return torchlib.randn(shape, dtype=dtype)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[Any] = None,
                     seed: Optional[int] = None) -> Tensor:
    if seed:
      torchlib.manual_seed(seed)
    dtype = dtype if dtype is not None else torchlib.float64
    return torchlib.empty(shape, dtype=dtype).uniform_(*boundaries)

  def conj(self, tensor: Tensor) -> Tensor:
    return tensor  #pytorch does not support complex dtypes

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return matrix.symeig(eigenvectors=True)

  def eigsh_lanczos(
      self,
      A: Callable,
      args: List,
      initial_state: Optional[Tensor] = None,
      shape: Optional[Tuple] = None,
      dtype: Optional[Type[np.number]] = None,
      num_krylov_vecs: Optional[int] = 200,
      numeig: Optional[int] = 1,
      tol: Optional[float] = 1E-8,
      delta: Optional[float] = 1E-8,
      ndiag: Optional[int] = 20,
      reorthogonalize: Optional[bool] = False) -> Tuple[List, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a `LinearOperator` `A`.
    Args:
      A: A (sparse) implementation of a linear operator
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(*args, initial_state)`.
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `torch.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If both no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalus. Uses
        `torch.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
        as stopping criterion between two diagonalization steps of the
        tridiagonal operator.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped. It means that an (approximate) invariant subspace has
        been found.
      ndiag: The tridiagonal Operator is diagonalized every `ndiag`
        iterations to check convergence.
      reorthogonalize: If `True`, Krylov vectors are kept orthogonal by
        explicit orthogonalization (more costly than `reorthogonalize=False`)
    Returns:
      (eigvals, eigvecs)
       eigvals: A list of `numeig` lowest eigenvalues
       eigvecs: A list of `numeig` lowest eigenvectors
    """
    #TODO: make this work for tensorflow in graph mode
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
      initial_state = self.randn(shape, dtype)

    if not isinstance(initial_state, torchlib.Tensor):
      raise TypeError("Expected a `torch.Tensor`. Got {}".format(
          type(initial_state)))

    initial_state = self.convert_to_tensor(initial_state)
    vector_n = initial_state
    Z = self.norm(vector_n)
    vector_n /= Z
    norms_vector_n = []
    diag_elements = []
    krylov_vecs = []
    first = True
    eigvalsold = []
    for it in range(num_krylov_vecs):
      #normalize the current vector:
      norm_vector_n = torchlib.norm(vector_n)
      if abs(norm_vector_n) < delta:
        break
      norms_vector_n.append(norm_vector_n)
      vector_n = vector_n / norms_vector_n[-1]
      #store the Lanczos vector for later
      if reorthogonalize:
        for v in krylov_vecs:
          vector_n -= (v.view(-1).dot(vector_n.view(-1))) * torchlib.reshape(
              v, vector_n.shape)
      krylov_vecs.append(vector_n)
      A_vector_n = A(*args, vector_n)
      diag_elements.append(vector_n.view(-1).dot(A_vector_n.view(-1)))

      if ((it > 0) and (it % ndiag) == 0) and (len(diag_elements) >= numeig):
        #diagonalize the effective Hamiltonian
        A_tridiag = torchlib.diag(
            torchlib.tensor(diag_elements)) + torchlib.diag(
                torchlib.tensor(norms_vector_n[1:]), 1) + torchlib.diag(
                    torchlib.tensor(norms_vector_n[1:]), -1)
        eigvals, u = A_tridiag.symeig(eigenvectors=True)
        if not first:
          if torchlib.norm(eigvals[0:numeig] - eigvalsold[0:numeig]) < tol:
            break
        first = False
        eigvalsold = eigvals[0:numeig]
      if it > 0:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
        A_vector_n -= (krylov_vecs[-2] * norms_vector_n[-1])
      else:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
      vector_n = A_vector_n

    A_tridiag = torchlib.diag(torchlib.tensor(diag_elements)) + torchlib.diag(
        torchlib.tensor(norms_vector_n[1:]), 1) + torchlib.diag(
            torchlib.tensor(norms_vector_n[1:]), -1)
    eigvals, u = A_tridiag.symeig(eigenvectors=True)
    eigenvectors = []
    for n2 in range(min(numeig, len(eigvals))):
      state = self.zeros(initial_state.shape, initial_state.dtype)
      for n1, vec in enumerate(krylov_vecs):
        state += vec * u[n1, n2]
      eigenvectors.append(state / torchlib.norm(state))
    return eigvals[0:numeig], eigenvectors

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
    #make a copy
    t = torchlib.as_tensor(tensor).clone()
    t[mask] = assignee
    return t

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError(
          "input to pytorch backend method `inv` has shape {}. Only matrices are supported."
          .format(matrix.shape))
    return matrix.inverse()

  def broadcast_right_multiplication(self, tensor1: Tensor, tensor2: Tensor):
    if len(tensor2.shape) != 1:
      raise ValueError(
          "only order-1 tensors are allowed for `tensor2`, found `tensor2.shape = {}`"
          .format(tensor2.shape))

    return tensor1 * tensor2

  def broadcast_left_multiplication(self, tensor1: Tensor, tensor2: Tensor):
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tensor1.shape))

    t1_broadcast_shape = self.shape_concat(
        [self.shape_tensor(tensor1), [1] * (len(tensor2.shape) - 1)], axis=-1)
    return tensor2 * self.reshape(tensor1, t1_broadcast_shape)
