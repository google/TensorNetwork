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
"""Implementation of sparse Lanczos tridiagonalization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensornetwork.backends import backend_factory
import inspect
import warnings
from typing import Any, List, Optional, Text, Tuple, Type, Callable
import numpy as np
Tensor = Any


class LinearOperator:
  """
  A linear operator class. `LinearOperator` is used to provide a 
  common interface to `eigsh` and other sparse solvers across 
  all supported backends.
  """

  def __init__(self, matvec: Callable,
               shape: Tuple[Tuple[int, ...], Tuple[int, ...]],
               dtype: Type[np.number], backend: Text) -> None:
    """
    Initialize a `LinearOperator`
    Args:
      matvec: A function which implements a matrix-vector multiplication.
        Allowed signatures are: `matvec(x)`, `matvec(x, backend)`
          and `matvec(x, backend=default)` with `default` 'numpy', 'tensorflow'
          'pytorch', 'jax' or 'shell'.
         `result = `matvec` should return `result` of the same type as `x` (i.e.
          a `Tensor` object)
          Note that any default value for argument `backend` of `matvec` is 
          overridden
          by the value of the `backend` argument of `LinearOperator.__init__`
      shape: A tuple of outgoing and incoming shapes of the `LinearOperator`.
        For example, if `matvec(x)` takes as input a tensor `x` of shape 
        `x.shape = (2,3,4,5)` and returns `result` with 
         `result.shape = (4,5,2)`, then `shape` has to be 
         `shape = ((4,5,2), (2,3,4,5))`.
      dtype: The data-type of the `LinearOperator`.
      backend: The backend to be used with this `LinearOperator`. 
        `backend` can be 'numpy', 'tensorflow', 'pytorch', 'jax' or 'shell'
    Returns: 
      None
    Raises:
      ValueError if `matvec` signature is other than the above mentioned.
    """
    self.shape = shape
    self.backend = backend_factory.get_backend(backend, dtype)
    args = inspect.getargspec(matvec)
    if len(args[0]) > 2:
      raise ValueError(
          '`matvec` can have at most 2 arguments; found args = {}'.format(
              args[0]))
    if args[0][0] == 'backend':
      raise ValueError("Found `backend` as first argument of `matvec`. "
                       "It can only be the second argument!")

    if args[3] and (len(args[3]) != 1):
      N = len(args[0])
      params = [args[0][N - 1 - n] for n in range(len(args[3]))]
      defaults = [args[3][n] for n in reversed(range(len(args[3])))]
      raise ValueError(
          "The only allowed argument to `matvec` with defaults is `backend`. "
          "Found arguments {} with default values {}!".format(
              params[::-1], defaults[::-1]))

    if args[3] and (len(args[3]) == 1):
      if args[3][0] not in ('tensorflow', 'numpy', 'jax', 'pytorch', 'shell'):
        raise ValueError(
            "wrong default '{}' for argument `{}` of `matvec`. "
            "Only allowed values are 'numpy', 'tensorflow', 'pytorch', "
            " 'jax' and 'shell'".format(args[3][0], args[0][1]))
      if args[3][0] != self.backend.name:
        warnings.warn("default value of parameter `{0}` = '{1}' of `matvec` is "
                      "different from LinearOperator.backend.name='{2}'."
                      " Overriding the default to `{0}` = '{2}'".format(
                          args[0][1], args[3][0], self.backend.name))
    if len(args[0]) == 1:  #matvec takes only one argument
      self.matvec = matvec
    else:  #matvec takes two arguments (x, backend)

      def _matvec(x):
        return matvec(x, self.backend.name)

      self.matvec = _matvec

  @property
  def dtype(self):
    return self.backend.dtype

  def __call__(self, x):
    return self.matvec(x)


class ScalarProduct:

  def __init__(self, dot_product: Callable, dtype: Type[np.number],
               backend: Text) -> None:
    """
    Initialize a `ScalarProduct`
    Args:
      dot_product: A function which implements a dot-product between two vectors
        Allowed signatures: `dot_product(x, y)`, `dot_product(x, y, backend)`
          and `dot_product(x, y, backend=default)` with `default` 'numpy', 
          'tensorflow', 'pytorch', 'jax' or 'shell'.
         `result = `dot_product` should return `result` of the same type as `x` 
          (i.e. a `Tensor` object)
          Note that any default value for argument `backend` of `dot_product` 
          is overridden by the value of the `backend` argument of 
          `ScalarProduct.__init__`
      dtype: The data-type of the `ScalarProduct`.
      backend: The backend to be used with this `ScalarProduct`. 
        `backend` can be 'numpy', 'tensorflow', 'pytorch', 'jax' or 'shell'
    Returns: 
      None
    Raises:
      ValueError if `dot_product` signature is other than the above mentioned.
    """

    self.backend = backend_factory.get_backend(backend, dtype)
    args = inspect.getargspec(dot_product)
    if len(args[0]) > 3:
      raise ValueError(
          '`dot_product` can have at most 3 arguments; found args = {}'.format(
              args[0]))
    if (args[0][0] == 'backend') or (args[0][1] == 'backend'):
      raise ValueError(
          "Found `backend` as first or second argument of `dot_product`. "
          "It can only be the third argument!")
    if args[3] and (len(args[3]) != 1):
      N = len(args[0])
      params = [args[0][N - 1 - n] for n in range(len(args[3]))]
      defaults = [args[3][n] for n in reversed(range(len(args[3])))]
      raise ValueError(
          "The only allowed argument to `dot_product` with defaults is "
          "`backend`. Found arguments {} with default values {}!".format(
              params[::-1], defaults[::-1]))

    if args[3] and (len(args[3]) == 1):
      if args[3][0] not in ('tensorflow', 'numpy', 'jax', 'pytorch', 'shell'):
        raise ValueError(
            "wrong default '{}' for argument `{}` of `dot_product`. "
            "Only allowed values are 'numpy', 'tensorflow', 'pytorch', "
            "'jax' and 'shell'".format(args[3][0], args[0][1]))
      if args[3][0] != self.backend.name:
        warnings.warn(
            "default value of parameter `{0}` = '{1}' of `dot_product` is "
            "different from ScalarProduct.backend.name='{2}'."
            " Overriding the default to `{0}` = '{2}'".format(
                args[0][1], args[3][0], self.backend.name))

    if len(args[0]) == 2:  #dot_product takes only one argument
      self.dot_product = dot_product
    else:  #dot_product takes two arguments (x, backend)

      def _dot_product(x, y):
        return dot_product(x, y, self.backend.name)

      self.dot_product = _dot_product

  @property
  def dtype(self):
    return self.backend.dtype

  def __call__(self, x, y):
    return self.dot_product(x, y)


def eigsh_lanczos(A: LinearOperator,
                  dot_product: ScalarProduct,
                  initial_state: Optional[Tensor] = None,
                  ncv: Optional[int] = 200,
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
    dot_product: A (sparse) implementation of a scalar product
    initial_state: An initial vector for the Lanczos algorithm
    ncv: The number of iterations (number of krylov vectors).
    numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
      If `numeig > 1`, `reorthogonalize` has to be `True`.
    tol: The desired precision of the eigenvalus. Currently we use 
      `np.linalg.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
      as stopping criterion between two diagonalization steps of the
      tridiagonal operator.
    delta: Stopping criterion for Lanczos iteration.
      If two successive Krylov vectors `x_m` and `x_n`
      have an overlap abs(<x_m|x_n>) < delta, the iteration is stopped.
    ndiag: The tridiagonal Operator is diagonalized every `ndiag` iterations to
      check convergence.
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
  if A.backend.name != dot_product.backend.name:
    raise ValueError(
        "A.backend={} is different from dot_product.backend={}".format(
            A.backend.name, dot_product.backend.name))
  if A.dtype != dot_product.dtype:
    raise ValueError("A.dtype={} is different from dot_product.dtype={}".format(
        A.dtype, dot_product.dtype))
  if initial_state and (A.dtype is not initial_state.dtype):
    raise TypeError(
        "A.dtype={} is different from initial_state.dtype={}".format(
            A.dtype, initial_state.dtype))
  if initial_state and (initial_state.shape != A.shape[1]):
    raise ValueError(
        "A.shape[1]={} and initial_state.shape={} are incompatible.".format(
            A.shape[1], initial_state.shape))

  backend = A.backend
  if not initial_state:
    initial_state = backend.randn(A.shape[1])
  vector_n = initial_state
  Z = backend.norm(vector_n)
  vector_n /= Z

  converged = False
  it = 0
  norms_vector_n = []
  diag_elements = []
  krylov_vecs = []
  first = True
  eigvalsold = []
  while not converged:
    #normalize the current vector:
    norm_vector_n = backend.sqrt(dot_product(
        vector_n, vector_n))  #conj has to be implemented by the user
    if abs(norm_vector_n) < delta:
      converged = True
      break
    norms_vector_n.append(norm_vector_n)
    vector_n = vector_n / norms_vector_n[-1]
    #store the Lanczos vector for later
    if reorthogonalize:
      for v in krylov_vecs:
        vector_n -= dot_product(v, vector_n) * v
    krylov_vecs.append(vector_n)
    A_vector_n = A(vector_n)
    diag_elements.append(dot_product(vector_n, A_vector_n))

    if ((it > 0) and (it % ndiag) == 0) and (len(diag_elements) >= numeig):
      #diagonalize the effective Hamiltonian
      A_tridiag = np.diag(diag_elements) + np.diag(
          norms_vector_n[1:], 1) + np.diag(np.conj(norms_vector_n[1:]), -1)
      eigvals, u = np.linalg.eigh(A_tridiag)
      if first:
        if np.linalg.norm(eigvals[0:numeig] - eigvalsold[0:numeig]) < tol:
          converged = True
      first = False
      eigvalsold = eigvals[0:numeig]
    if it > 0:
      A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
      A_vector_n -= (krylov_vecs[-2] * norms_vector_n[-1])
    else:
      A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
    vector_n = A_vector_n
    it = it + 1
    if it >= ncv:
      break

  A_tridiag = np.diag(diag_elements) + np.diag(norms_vector_n[1:], 1) + np.diag(
      np.conj(norms_vector_n[1:]), -1)
  eigvals, u = np.linalg.eigh(A_tridiag)

  eigenvectors = []
  if np.iscomplexobj(A_tridiag):
    eigvals = np.array(eigvals).astype(A_tridiag.dtype)

  for n2 in range(min(numeig, len(eigvals))):
    state = backend.zeros(initial_state.shape)
    for n1, vec in enumerate(krylov_vecs):
      state += vec * u[n1, n2]
    eigenvectors.append(state / backend.sqrt(dot_product(state, state)))
  final_eigenvalues = [
      backend.convert_to_tensor(np.array(eigvals[n]))
      for n in range(min(numeig, len(eigvals)))
  ]
  final_eigenvectors = [
      backend.convert_to_tensor(eigenvectors[n])
      for n in range(min(numeig, len(eigvals)))
  ]

  return final_eigenvalues, final_eigenvectors
