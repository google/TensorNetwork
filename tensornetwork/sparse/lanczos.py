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
"""Implementation of TensorNetwork structure."""

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

  def __init__(self, matvec: Callable, shape: Tuple[Tuple[int]],
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
          Note that any default value for argument `backend` of `matvec` is overridden
          by the value of the `backend` argument of `LinearOperator.__init__`
      shape: A tuple of outgoing and incoming shapes of the `LinearOperator`.
        For example, if `matvec(x)` takes as input a tensor `x` of shape `x.shape = (2,3,4,5)`
        and returns `result` with `result.shape = (4,5,2)`, then `shape` has to be 
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
    if (args[0][0] == 'backend'):
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
            "Only allowed values are 'numpy', 'tensorflow', 'pytorch', 'jax' and 'shell'"
            .format(args[3][0], args[0][1]))
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

  def __init__(self, dotprod: Callable, dtype: Type[np.number],
               backend: Text) -> None:
    """
    Initialize a `ScalarProduct`
    Args:
      dotprod: A function which implements a dot-product between two vectors
        Allowed signatures are: `dotprod(x, y)`, `dotprod(x, y, backend)`
          and `dotprod(x, y, backend=default)` with `default` 'numpy', 'tensorflow'
          'pytorch', 'jax' or 'shell'.
         `result = `dotprod` should return `result` of the same type as `x` 
          (i.e. a `Tensor` object)
          Note that any default value for argument `backend` of `dotprod` is overridden
          by the value of the `backend` argument of `ScalarProduct.__init__`
      dtype: The data-type of the `ScalarProduct`.
      backend: The backend to be used with this `ScalarProduct`. 
        `backend` can be 'numpy', 'tensorflow', 'pytorch', 'jax' or 'shell'
    Returns: 
      None
    Raises:
      ValueError if `dotprod` signature is other than the above mentioned.
    """

    self.backend = backend_factory.get_backend(backend, dtype)
    args = inspect.getargspec(dotprod)
    if len(args[0]) > 3:
      raise ValueError(
          '`dotprod` can have at most 3 arguments; found args = {}'.format(
              args[0]))
    if (args[0][0] == 'backend') or (args[0][1] == 'backend'):
      raise ValueError(
          "Found `backend` as first or second argument of `dotprod`. "
          "It can only be the third argument!")
    if args[3] and (len(args[3]) != 1):
      N = len(args[0])
      params = [args[0][N - 1 - n] for n in range(len(args[3]))]
      defaults = [args[3][n] for n in reversed(range(len(args[3])))]
      raise ValueError(
          "The only allowed argument to `dotprod` with defaults is `backend`. "
          "Found arguments {} with default values {}!".format(
              params[::-1], defaults[::-1]))

    if args[3] and (len(args[3]) == 1):
      if args[3][0] not in ('tensorflow', 'numpy', 'jax', 'pytorch', 'shell'):
        raise ValueError(
            "wrong default '{}' for argument `{}` of `dotprod`. "
            "Only allowed values are 'numpy', 'tensorflow', 'pytorch', 'jax' and 'shell'"
            .format(args[3][0], args[0][1]))
      if args[3][0] != self.backend.name:
        warnings.warn(
            "default value of parameter `{0}` = '{1}' of `dotprod` is "
            "different from ScalarProduct.backend.name='{2}'."
            " Overriding the default to `{0}` = '{2}'".format(
                args[0][1], args[3][0], self.backend.name))

    if len(args[0]) == 2:  #dotprod takes only one argument
      self.dotprod = dotprod
    else:  #dotprod takes two arguments (x, backend)

      def _dotprod(x, y):
        return dotprod(x, y, self.backend.name)

      self.dotprod = _dotprod

  @property
  def dtype(self):
    return self.backend.dtype

  def __call__(self, x, y):
    return self.dotprod(x, y)


def eigsh_lanczos(A: LinearOperator,
                  vv: ScalarProduct,
                  v0: Optional[Tensor] = None,
                  ncv: Optional[int] = 200,
                  numeig: Optional[int] = 1,
                  tol: Optional[float] = 1E-8,
                  delta: Optional[float] = 1E-8,
                  ndiag: Optional[int] = 20,
                  reortho: Optional[bool] = False) -> Tuple[List]:
  """
  Lanczos method for finding the lowest eigenvector-eigenvalue pairs
  of a `LinearOperator` `A`.
  Args:
    A: A (sparse) implementation of a linear operator
    vv: A (sparse) implementation of a scalar product
    v0: An initial vector for the Lanczos algorithm
    ncv: The number of iterations (number of krylov vectors).
    numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
      If `numeig > 1`, `reortho` has to be `True`.
    tol: The desired precision of the eigenvalus. Currently we use 
      `np.linalg.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
      as stopping criterion between two diagonalization steps of the
      tridiagonal operator.
    delta: Stopping criterion for Lanczos iteration.
      If two successive Krylov vectors `x_m` and `x_n`
      have an overlap abs(<x_m|x_n>) < delta, the iteration is stopped.
    ndiag: The tridiagonal Operator is diagonalized every `ndiag` iterations to
      check convergence.
    reortho: If `True`, Krylov vectors are kept orthogonal by explicit orthogonalization
      (this is more costly than `reortho=False`)
  Returns:
    (eigvals, eigvecs)
     eigvals: A list of `numeig` lowest eigenvalues
     eigvecs: A list of `numeig` lowest eigenvectors
  """
  #TODO: make this work for tensorflow in graph mode
  if ncv < numeig:
    raise ValueError('`ncv` >= `numeig` required!')
  if numeig > 1 and not reortho:
    raise ValueError(
        'Got numeig = {} > 1 and `reortho = False`. Use `reortho=True` for `numeig > 1`'
        .format(numeig))
  if A.backend.name != vv.backend.name:
    raise ValueError("A.backend={} is different from vv.backend={}".format(
        A.backend.name, vv.backend.name))
  if A.dtype != vv.dtype:
    raise ValueError("A.dtype={} is different from vv.dtype={}".format(
        A.dtype, vv.dtype))
  if v0 and (A.dtype is not v0.dtype):
    raise TypeError("A.dtype={} is different from v0.dtype={}".format(
        A.dtype, v0.dtype))
  if v0 and (v0.shape != A.shape[1]):
    raise ValueError("A.shape[1]={} and v0.shape={} are incompatible.".format(
        A.shape[1], v0.shape))

  backend = A.backend
  if not v0:
    v0 = backend.randn(A.shape[1])
  xn = v0
  Z = backend.norm(xn)
  xn /= Z

  converged = False
  it = 0
  norms_xn = []
  epsn = []
  krylov_vecs = []
  first = True
  etaold = None
  while converged == False:
    #normalize the current vector:
    normxn = backend.sqrt(vv(xn, xn))  #conj has to be implemented by the user
    if abs(normxn) < delta:
      converged = True
      break
    norms_xn.append(normxn)
    xn = xn / norms_xn[-1]
    #store the Lanczos vector for later
    if reortho == True:
      for v in krylov_vecs:
        xn -= vv(v, xn) * v
    krylov_vecs.append(xn)
    Hxn = A(xn)
    epsn.append(vv(xn, Hxn))

    if ((it > 0) and (it % ndiag) == 0) and (len(epsn) >= numeig):
      #diagonalize the effective Hamiltonian
      Heff = np.diag(epsn) + np.diag(norms_xn[1:], 1) + np.diag(
          np.conj(norms_xn[1:]), -1)
      eta, u = np.linalg.eigh(Heff)
      if first == False:
        if np.linalg.norm(eta[0:numeig] - etaold[0:numeig]) < tol:
          converged = True
      first = False
      etaold = eta[0:numeig]
    if it > 0:
      Hxn -= (krylov_vecs[-1] * epsn[-1])
      Hxn -= (krylov_vecs[-2] * norms_xn[-1])
    else:
      Hxn -= (krylov_vecs[-1] * epsn[-1])
    xn = Hxn
    it = it + 1
    if it >= ncv:
      break

  Heff = np.diag(epsn) + np.diag(norms_xn[1:], 1) + np.diag(
      np.conj(norms_xn[1:]), -1)
  eta, u = np.linalg.eigh(Heff)
  states = []
  if np.iscomplexobj(Heff):  #only possible if backend
    eta = np.array(eta).astype(Heff.dtype)

  for n2 in range(min(numeig, len(eta))):
    state = backend.zeros(v0.shape)
    for n1 in range(len(krylov_vecs)):
      state += krylov_vecs[n1] * u[n1, n2]
    states.append(state / backend.sqrt(vv(state, state)))
  eigvals = [
      backend.convert_to_tensor(np.array(eta[n]))
      for n in range(min(numeig, len(eta)))
  ]
  eigvecs = [
      backend.convert_to_tensor(states[n])
      for n in range(min(numeig, len(eta)))
  ]

  return eigvals, eigvecs
