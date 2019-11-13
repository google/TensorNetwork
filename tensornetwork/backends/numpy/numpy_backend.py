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
import numpy
import scipy
Tensor = Any


class NumPyBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(NumPyBackend, self).__init__()
    self.np = numpy
    self.name = "numpy"

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return self.np.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return self.np.reshape(tensor, self.np.asarray(shape).astype(self.np.int32))

  def transpose(self, tensor, perm):
    return self.np.transpose(tensor, perm)

  def svd_decomposition(self,
                        tensor: Tensor,
                        split_axis: int,
                        max_singular_values: Optional[int] = None,
                        max_truncation_error: Optional[float] = None
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(
        self.np, tensor, split_axis, max_singular_values, max_truncation_error)

  def qr_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.qr_decomposition(self.np, tensor, split_axis)

  def rq_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(self.np, tensor, split_axis)

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
    if (not isinstance(tensor, self.np.ndarray) and
        not self.np.isscalar(tensor)):
      raise TypeError("Expected a `np.array` or scalar. Got {}".format(
          type(tensor)))
    result = self.np.asarray(tensor)
    return result

  def trace(self, tensor: Tensor) -> Tensor:
    # Default np.trace uses first two axes.
    return self.np.trace(tensor, axis1=-2, axis2=-1)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.np.tensordot(tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return self.np.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return self.np.linalg.norm(tensor)

  def eye(self, N, dtype: Optional[numpy.dtype] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else self.np.float64

    return self.np.eye(N, M=M, dtype=dtype)

  def ones(self, shape: Tuple[int, ...],
           dtype: Optional[numpy.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else self.np.float64
    return self.np.ones(shape, dtype=dtype)

  def zeros(self, shape: Tuple[int, ...],
            dtype: Optional[numpy.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else self.np.float64
    return self.np.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[numpy.dtype] = None,
            seed: Optional[int] = None) -> Tensor:

    if seed:
      self.np.random.seed(seed)
    dtype = dtype if dtype is not None else self.np.float64
    if ((self.np.dtype(dtype) is self.np.dtype(self.np.complex128)) or
        (self.np.dtype(dtype) is self.np.dtype(self.np.complex64))):
      return self.np.random.randn(*shape).astype(
          dtype) + 1j * self.np.random.randn(*shape).astype(dtype)
    return self.np.random.randn(*shape).astype(dtype)

  def conj(self, tensor: Tensor) -> Tensor:
    return self.np.conj(tensor)

  def eigs(self,
           A: Callable,
           initial_state: Optional[Tensor] = None,
           num_krylov_vecs: Optional[int] = 200,
           numeig: Optional[int] = 6,
           tol: Optional[float] = 1E-8,
           which: Optional[Text] = 'LR',
           maxiter: Optional[int] = None,
           dtype: Optional[Type[numpy.number]] = None) -> Tuple[List, List]:
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

    if not isinstance(initial_state, self.np.ndarray):
      raise TypeError("Expected a `np.array`. Got {}".format(
          type(initial_state)))
    #initial_state is an np.ndarray of rank 1, so we can
    #savely deduce the shape from it
    lop = scipy.sparse.linalg.LinearOperator(
        dtype=initial_state.dtype,
        shape=(initial_state.shape[0], initial_state.shape[0]),
        matvec=A)
    eta, U = scipy.sparse.linalg.eigs(
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

  def eigsh_lanczos(
      self,
      A: Callable,
      initial_state: Optional[Tensor] = None,
      num_krylov_vecs: Optional[int] = 200,
      numeig: Optional[int] = 1,
      tol: Optional[float] = 1E-8,
      delta: Optional[float] = 1E-8,
      ndiag: Optional[int] = 20,
      reorthogonalize: Optional[bool] = False) -> Tuple[List, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`. If no `initial_state` is provided
    then `A` has to have an attribute `shape` so that a suitable initial
    state can be randomly generated.

    Args:
      A: A (sparse) implementation of a linear operator
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `numpy.random.randn` 
        method
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
    if not isinstance(initial_state, self.np.ndarray):
      raise TypeError("Expected a `np.array`. Got {}".format(
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
      norm_vector_n = self.np.linalg.norm(vector_n)
      if abs(norm_vector_n) < delta:
        break
      norms_vector_n.append(norm_vector_n)
      vector_n = vector_n / norms_vector_n[-1]
      #store the Lanczos vector for later
      if reorthogonalize:
        for v in krylov_vecs:
          vector_n -= self.np.dot(self.np.ravel(self.np.conj(v)), vector_n) * v
      krylov_vecs.append(vector_n)
      A_vector_n = A(vector_n)
      diag_elements.append(
          self.np.dot(
              self.np.ravel(self.np.conj(vector_n)), self.np.ravel(A_vector_n)))

      if ((it > 0) and (it % ndiag) == 0) and (len(diag_elements) >= numeig):
        #diagonalize the effective Hamiltonian
        A_tridiag = self.np.diag(diag_elements) + self.np.diag(
            norms_vector_n[1:], 1) + self.np.diag(
                self.np.conj(norms_vector_n[1:]), -1)
        eigvals, u = self.np.linalg.eigh(A_tridiag)
        if not first:
          if self.np.linalg.norm(eigvals[0:numeig] -
                                 eigvalsold[0:numeig]) < tol:
            break
        first = False
        eigvalsold = eigvals[0:numeig]
      if it > 0:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
        A_vector_n -= (krylov_vecs[-2] * norms_vector_n[-1])
      else:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
      vector_n = A_vector_n

    A_tridiag = self.np.diag(diag_elements) + self.np.diag(
        norms_vector_n[1:], 1) + self.np.diag(
            self.np.conj(norms_vector_n[1:]), -1)
    eigvals, u = self.np.linalg.eigh(A_tridiag)
    eigenvectors = []
    if self.np.iscomplexobj(A_tridiag):
      eigvals = self.np.array(eigvals).astype(A_tridiag.dtype)

    for n2 in range(min(numeig, len(eigvals))):
      state = self.zeros(initial_state.shape, initial_state.dtype)
      for n1, vec in enumerate(krylov_vecs):
        state += vec * u[n1, n2]
      eigenvectors.append(state / self.np.linalg.norm(state))
    return eigvals[0:numeig], eigenvectors

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 * tensor2

  def index_update(self, tensor: Tensor, mask: Tensor,
                   assignee: Tensor) -> Tensor:
    t = self.np.copy(tensor)
    t[mask] = assignee
    return t
