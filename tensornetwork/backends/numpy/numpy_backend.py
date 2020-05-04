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
#pyling: disable=line-too-long
from typing import Optional, Any, Sequence, Tuple, Callable, List, Text, Type
from tensornetwork.backends import base_backend
from tensornetwork.backends.numpy import decompositions
import numpy as np
import scipy as sp
Tensor = Any


class NumPyBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(NumPyBackend, self).__init__()
    self.name = "numpy"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return np.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return np.reshape(tensor, np.asarray(shape).astype(np.int32))

  def transpose(self, tensor, perm):
    return np.transpose(tensor, perm)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    obj = tuple(
        slice(start, start + size)
        for start, size in zip(start_indices, slice_sizes))
    return tensor[obj]

  def svd_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(
        np,
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
    return decompositions.qr_decomposition(np, tensor, split_axis)

  def rq_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(np, tensor, split_axis)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return np.concatenate(values, axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return self.shape_tuple(tensor)

  def shape_prod(self, values: Tensor) -> Tensor:
    return np.prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return np.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    if len(tensor.shape) != 1:
      raise TypeError("Only one dimensional tensors are allowed as input")
    return np.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    if (not isinstance(tensor, np.ndarray) and not np.isscalar(tensor)):
      raise TypeError("Expected a `np.array` or scalar. Got {}".format(
          type(tensor)))
    result = np.asarray(tensor)
    return result

  def trace(self, tensor: Tensor) -> Tensor:
    # Default np.trace uses first two axes.
    return np.trace(tensor, axis1=-2, axis2=-1)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return np.tensordot(tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return np.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return np.linalg.norm(tensor)

  def eye(self,
          N,
          dtype: Optional[np.dtype] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else np.float64

    return np.eye(N, M=M, dtype=dtype)

  def ones(self,
           shape: Tuple[int, ...],
           dtype: Optional[np.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else np.float64
    return np.ones(shape, dtype=dtype)

  def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else np.float64
    return np.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None,
            seed: Optional[int] = None) -> Tensor:

    if seed:
      np.random.seed(seed)
    dtype = dtype if dtype is not None else np.float64
    if ((np.dtype(dtype) is np.dtype(np.complex128)) or
        (np.dtype(dtype) is np.dtype(np.complex64))):
      return np.random.randn(*shape).astype(
          dtype) + 1j * np.random.randn(*shape).astype(dtype)
    return np.random.randn(*shape).astype(dtype)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[np.dtype] = None,
                     seed: Optional[int] = None) -> Tensor:

    if seed:
      np.random.seed(seed)
    dtype = dtype if dtype is not None else np.float64
    if ((np.dtype(dtype) is np.dtype(np.complex128)) or
        (np.dtype(dtype) is np.dtype(np.complex64))):
      return np.random.uniform(
          boundaries[0],
          boundaries[1], shape).astype(dtype) + 1j * np.random.uniform(
              boundaries[0], boundaries[1], shape).astype(dtype)
    return np.random.uniform(boundaries[0], boundaries[1], shape).astype(dtype)

  def conj(self, tensor: Tensor) -> Tensor:
    return np.conj(tensor)

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return np.linalg.eigh(matrix)

  def eigs(self,
           A: Callable,
           initial_state: Optional[Tensor] = None,
           num_krylov_vecs: Optional[int] = 200,
           numeig: Optional[int] = 6,
           tol: Optional[float] = 1E-8,
           which: Optional[Text] = 'LR',
           maxiter: Optional[int] = None,
           dtype: Optional[Type[np.number]] = None) -> Tuple[List, List]:
    """
    Arnoldi method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`. `A` can be either a
    scipy.sparse.linalg.LinearOperator object or a regular callable.
    If no `initial_state` is provided then `A` has to have an attribute
    `shape` so that a suitable initial state can be randomly generated.
    This is a wrapper for scipy.sparse.linalg.eigs which only supports
    a subset of the arguments of scipy.sparse.linalg.eigs.

    Args:
      A: A (sparse) implementation of a linear operator
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `numpy.random.randn`
        method.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalus. Uses
      which : ['LM' | 'SM' | 'LR' | 'SR' | 'LI']
        Which `k` eigenvectors and eigenvalues to find:
            'LM' : largest magnitude
            'SM' : smallest magnitude
            'LR' : largest real part
            'SR' : smallest real part
            'LI' : largest imaginary part
      maxiter: The maximum number of iterations.
      dtype: An optional numpy-dtype. If provided, the
        return type will be cast to `dtype`.
    Returns:
       `np.ndarray`: An array of `numeig` lowest eigenvalues
       `np.ndarray`: An array of `numeig` lowest eigenvectors
    """
    if which == 'SI':
      raise ValueError('which = SI is currently not supported.')
    if which == 'LI':
      raise ValueError('which = LI is currently not supported.')

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

    if not isinstance(initial_state, np.ndarray):
      raise TypeError("Expected a `np.array`. Got {}".format(
          type(initial_state)))
    #initial_state is an np.ndarray of rank 1, so we can
    #savely deduce the shape from it
    lop = sp.sparse.linalg.LinearOperator(
        dtype=initial_state.dtype,
        shape=(initial_state.shape[0], initial_state.shape[0]),
        matvec=A)
    eta, U = sp.sparse.linalg.eigs(
        A=lop,
        k=numeig,
        which=which,
        v0=initial_state,
        ncv=num_krylov_vecs,
        tol=tol,
        maxiter=maxiter)
    if dtype:
      eta = eta.astype(dtype)
      U = U.astype(dtype)
    return list(eta), [U[:, n] for n in range(numeig)]

  def eigsh_lanczos(self,
                    A: Callable,
                    args: List,
                    initial_state: Optional[Tensor] = None,
                    shape: Optional[Tuple] = None,
                    dtype: Optional[Type[np.number]] = None,
                    num_krylov_vecs: int = 20,
                    numeig: int = 1,
                    tol: float = 1E-8,
                    delta: float = 1E-8,
                    ndiag: int = 20,
                    reorthogonalize: bool = False) -> Tuple[List, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`.
    Args:
      A: A (sparse) implementation of a linear operator.
         Call signature of `A` is `res = A(*args, vector)`, where `vector`
         can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(*args, initial_state)`.
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `backend.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If both no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalus. Uses
        `np.linalg.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
        as stopping criterion between two diagonalization steps of the
        tridiagonal operator.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped. It means that an (approximate) invariant subspace has
        been found.
      ndiag: The tridiagonal Operator is diagonalized every `ndiag` iterations
        to check convergence.
      reorthogonalize: If `True`, Krylov vectors are kept orthogonal by
        explicit orthogonalization (more costly than `reorthogonalize=False`)
    Returns:
      (eigvals, eigvecs)
       eigvals: A list of `numeig` lowest eigenvalues
       eigvecs: A list of `numeig` lowest eigenvectors
    """
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

    if not isinstance(initial_state, np.ndarray):
      raise TypeError("Expected a `np.ndarray`. Got {}".format(
          type(initial_state)))

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
      norm_vector_n = self.norm(vector_n)
      if abs(norm_vector_n) < delta:
        break
      norms_vector_n.append(norm_vector_n)
      vector_n = vector_n / norms_vector_n[-1]
      #store the Lanczos vector for later
      if reorthogonalize:
        for v in krylov_vecs:
          vector_n -= np.dot(np.ravel(np.conj(v)), np.ravel(vector_n)) * v
      krylov_vecs.append(vector_n)
      A_vector_n = A(*args, vector_n)
      diag_elements.append(
          np.dot(np.ravel(np.conj(vector_n)), np.ravel(A_vector_n)))

      if ((it > 0) and (it % ndiag) == 0) and (len(diag_elements) >= numeig):
        #diagonalize the effective Hamiltonian
        A_tridiag = np.diag(diag_elements) + np.diag(
            norms_vector_n[1:], 1) + np.diag(np.conj(norms_vector_n[1:]), -1)
        eigvals, u = np.linalg.eigh(A_tridiag)
        if not first:
          if np.linalg.norm(eigvals[0:numeig] - eigvalsold[0:numeig]) < tol:
            break
        first = False
        eigvalsold = eigvals[0:numeig]
      if it > 0:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
        A_vector_n -= (krylov_vecs[-2] * norms_vector_n[-1])
      else:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
      vector_n = A_vector_n

    A_tridiag = np.diag(diag_elements) + np.diag(
        norms_vector_n[1:], 1) + np.diag(np.conj(norms_vector_n[1:]), -1)
    eigvals, u = np.linalg.eigh(A_tridiag)
    eigenvectors = []
    if np.iscomplexobj(A_tridiag):
      eigvals = np.array(eigvals).astype(A_tridiag.dtype)

    for n2 in range(min(numeig, len(eigvals))):
      state = self.zeros(initial_state.shape, initial_state.dtype)
      for n1, vec in enumerate(krylov_vecs):
        state += vec * u[n1, n2]
      eigenvectors.append(state / np.linalg.norm(state))
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
    t = np.copy(tensor)
    t[mask] = assignee
    return t

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError("input to numpy backend method `inv` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    return np.linalg.inv(matrix)

  def broadcast_right_multiplication(self, tensor1: Tensor, tensor2: Tensor):
    if len(tensor2.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor2`,"
                       " found `tensor2.shape = {}`".format(tensor2.shape))
    return tensor1 * tensor2

  def broadcast_left_multiplication(self, tensor1: Tensor, tensor2: Tensor):
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tensor1.shape))

    t1_broadcast_shape = self.shape_concat(
        [self.shape_tensor(tensor1), [1] * (len(tensor2.shape) - 1)], axis=-1)
    return tensor2 * self.reshape(tensor1, t1_broadcast_shape)

  def sin(self, tensor: Tensor):
    return np.sin(tensor)

  def cos(self, tensor: Tensor):
    return np.cos(tensor)

  def exp(self, tensor: Tensor):
    return np.exp(tensor)

  def log(self, tensor: Tensor):
    return np.log(tensor)

  def expm(self, matrix: Tensor):
    if len(matrix.shape) != 2:
      raise ValueError("input to numpy backend method `expm` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError("input to numpy backend method `expm` only supports"
                       " N*N matrix, {x}*{y} matrix is given".format(
                           x=matrix.shape[0], y=matrix.shape[1]))
    # pylint: disable=no-member
    return sp.linalg.expm(matrix)
