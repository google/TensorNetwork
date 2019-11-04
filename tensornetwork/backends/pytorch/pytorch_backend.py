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

from typing import Optional, Any, Sequence, Tuple, Callable, List
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
      #pylint: disable=import-outside-toplevel
      import torch
    except ImportError:
      raise ImportError("PyTorch not installed, please switch to a different "
                        "backend or install PyTorch.")
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

  def qr_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.qr_decomposition(self.torch, tensor, split_axis)

  def rq_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(self.torch, tensor, split_axis)

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
    result = self.torch.as_tensor(tensor)
    return result

  def trace(self, tensor: Tensor) -> Tensor:
    return self.torch.einsum('...jj', tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.torch.tensordot(tensor1, tensor2, dims=0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return self.torch.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return self.torch.norm(tensor)

  def eye(self, N: int, dtype: Optional[Any] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else self.torch.float64
    if not M:
      M = N  #torch crashes if one passes M = None with dtype!=None
    return self.torch.eye(n=N, m=M, dtype=dtype)

  def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Tensor:
    dtype = dtype if dtype is not None else self.torch.float64
    return self.torch.ones(shape, dtype=dtype)

  def zeros(self, shape: Tuple[int, ...],
            dtype: Optional[Any] = None) -> Tensor:
    dtype = dtype if dtype is not None else self.torch.float64
    return self.torch.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Any] = None,
            seed: Optional[int] = None) -> Tensor:
    if seed:
      self.torch.manual_seed(seed)
    dtype = dtype if dtype is not None else self.torch.float64
    return self.torch.randn(shape, dtype=dtype)

  def conj(self, tensor: Tensor) -> Tensor:
    return tensor  #pytorch does not support complex dtypes

  def eigsh_lanczos(self,
                    A: Callable,
                    initial_state: Optional[Tensor] = None,
                    ncv: Optional[int] = 200,
                    numeig: Optional[int] = 1,
                    tol: Optional[float] = 1E-8,
                    delta: Optional[float] = 1E-8,
                    ndiag: Optional[int] = 20,
                    reorthogonalize: Optional[bool] = False
                   ) -> Tuple[List, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a `LinearOperator` `A`.
    Args:
      A: A (sparse) implementation of a linear operator
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `torch.randn` method
      ncv: The number of iterations (number of krylov vectors).
      numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalus. Uses
        `torch.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
        as stopping criterion between two diagonalization steps of the
        tridiagonal operator.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector `x_n` has an L2 norm ||x_n|| < delta, the iteration 
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
    if ncv < numeig:
      raise ValueError('`ncv` >= `numeig` required!')
    if numeig > 1 and not reorthogonalize:
      raise ValueError(
          "Got numeig = {} > 1 and `reorthogonalize = False`. "
          "Use `reorthogonalize=True` for `numeig > 1`".format(numeig))

    if (initial_state is not None) and hasattr(A, 'shape'):
      if initial_state.shape != A.shape[1]:
        raise ValueError(
            "A.shape[1]={} and initial_state.shape={} are incompatible.".format(
                A.shape[1], initial_state.shape))
    if initial_state is None:
      if not hasattr(A, 'shape'):
        raise AttributeError("`A` has no  attribute `shape`. Cannot initialize "
                             "lanczos. Please provide a valid `initial_state`")

      if not hasattr(A, 'dtype'):
        raise AttributeError(
            "`A` has no  attribute `dtype`. Cannot initialize "
            "lanczos. Please provide a valid `initial_state` with "
            "a `dtype` attribute")
      initial_state = self.randn(A.shape[1], A.dtype)
    else:
      initial_state = self.convert_to_tensor(initial_state)
    vector_n = initial_state
    Z = self.norm(vector_n)
    vector_n /= Z
    norms_vector_n = []
    diag_elements = []
    krylov_vecs = []
    first = True
    eigvalsold = []
    for it in range(ncv):
      #normalize the current vector:
      norm_vector_n = self.torch.norm(vector_n)
      if abs(norm_vector_n) < delta:
        break
      norms_vector_n.append(norm_vector_n)
      vector_n = vector_n / norms_vector_n[-1]
      #store the Lanczos vector for later
      if reorthogonalize:
        for v in krylov_vecs:
          vector_n -= (v.view(-1).dot(vector_n.view(-1))) * v
      krylov_vecs.append(vector_n)
      A_vector_n = A(vector_n)
      diag_elements.append(vector_n.view(-1).dot(A_vector_n.view(-1)))

      if ((it > 0) and (it % ndiag) == 0) and (len(diag_elements) >= numeig):
        #diagonalize the effective Hamiltonian
        A_tridiag = self.torch.diag(
            self.torch.tensor(diag_elements)) + self.torch.diag(
                self.torch.tensor(norms_vector_n[1:]), 1) + self.torch.diag(
                    self.torch.tensor(norms_vector_n[1:]), -1)
        eigvals, u = A_tridiag.symeig()
        if not first:
          if self.torch.norm(eigvals[0:numeig] - eigvalsold[0:numeig]) < tol:
            break
        first = False
        eigvalsold = eigvals[0:numeig]
      if it > 0:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
        A_vector_n -= (krylov_vecs[-2] * norms_vector_n[-1])
      else:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
      vector_n = A_vector_n

    A_tridiag = self.torch.diag(
        self.torch.tensor(diag_elements)) + self.torch.diag(
            self.torch.tensor(norms_vector_n[1:]), 1) + self.torch.diag(
                self.torch.tensor(norms_vector_n[1:]), -1)
    eigvals, u = A_tridiag.symeig()
    eigenvectors = []
    for n2 in range(min(numeig, len(eigvals))):
      state = self.zeros(initial_state.shape, initial_state.dtype)
      for n1, vec in enumerate(krylov_vecs):
        state += vec * u[n1, n2]
      eigenvectors.append(state / self.torch.norm(state))
    return eigvals[0:numeig], eigenvectors

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 * tensor2
