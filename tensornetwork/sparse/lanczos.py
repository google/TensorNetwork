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
from typing import Any, Sequence, List, Set, Optional, Union, Text, Tuple, Type, Dict, Callable
import numpy as np
Tensor = Any


class LinearOperator:
  """
  A linear operator class
  """

  def __init__(self, matvec: Callable, shape: Tuple[Tuple[int]],
               dtype: Type[np.number], backend: str) -> None:
    """
    Initialize a `LinearOperator`
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

  def __init__(self, vecvec: Callable, dtype: Type[np.number],
               backend: str) -> None:

    self.backend = backend_factory.get_backend(backend, dtype)
    args = inspect.getargspec(vecvec)
    if len(args[0]) > 3:
      raise ValueError(
          '`vecvec` can have at most 3 arguments; found args = {}'.format(
              args[0]))
    if (args[0][0] == 'backend') or (args[0][1] == 'backend'):
      raise ValueError(
          "Found `backend` as first or second argument of `vecvec`. "
          "It can only be the third argument!")
    if args[3] and (len(args[3]) != 1):
      N = len(args[0])
      params = [args[0][N - 1 - n] for n in range(len(args[3]))]
      defaults = [args[3][n] for n in reversed(range(len(args[3])))]
      raise ValueError(
          "The only allowed argument to `vecvec` with defaults is `backend`. "
          "Found arguments {} with default values {}!".format(
              params[::-1], defaults[::-1]))

    if args[3] and (len(args[3]) == 1):
      if args[3][0] not in ('tensorflow', 'numpy', 'jax', 'pytorch', 'shell'):
        raise ValueError(
            "wrong default '{}' for argument `{}` of `vecvec`. "
            "Only allowed values are 'numpy', 'tensorflow', 'pytorch', 'jax' and 'shell'"
            .format(args[3][0], args[0][1]))
      if args[3][0] != self.backend.name:
        warnings.warn("default value of parameter `{0}` = '{1}' of `vecvec` is "
                      "different from LinearOperator.backend.name='{2}'."
                      " Overriding the default to `{0}` = '{2}'".format(
                          args[0][1], args[3][0], self.backend.name))

    if len(args[0]) == 2:  #vecvec takes only one argument
      self.vecvec = vecvec
    else:  #vecvec takes two arguments (x, backend)

      def _vecvec(x, y):
        return vecvec(x, y, self.backend.name)

      self.vecvec = _vecvec

  @property
  def dtype(self):
    return self.backend.dtype

  def __call__(self, x, y):
    return self.vecvec(x, y)


def eigsh(A: LinearOperator,
          vv: ScalarProduct,
          v0: Optional[Tensor] = None,
          ncv: Optional[int] = 200,
          numeig: Optional[int] = 1,
          tol: Optional[float] = 1E-8,
          delta: Optional[float] = 1E-8,
          ndiag: Optional[int] = 20,
          reortho: Optional[bool] = False):

  if ncv < numeig:
    raise ValueError('`ncv` >= `numeig` required!')

  if A.backend.name != vv.backend.name:
    raise ValueError("A.backend={} is different from vv.backend={}".format(
        A.backend.name, vv.backend.name))

  backend = A.backend
  if v0 and (A.dtype is not v0.dtype):
    raise TypeError("A.dtype={} is different from v0.dtype={}".format(
        A.dtype, v0.dtype))

  if v0 and (v0.shape != A.shape[1]):
    raise ValueError("A.shape[1]={} and v0.shape={} are incompatible.".format(
        A.shape[1], v0.shape))

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
  while converged == False:
    #normalize the current vector:
    normxn = backend.sqrt(vv(xn, xn))  #conj has to be implemented by the user
    if normxn < delta:
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
  for n2 in range(min(numeig, len(eta))):
    state = backend.zeros(v0.shape, dtype=v0.dtype)
    for n1 in range(len(krylov_vecs)):
      state += krylov_vecs[n1] * u[n1, n2]
    states.append(state / backend.sqrt(vv(state, state)))
  eigvals = [
      backend.convert_to_tensor(np.array(eta[n]))
      for n in range(min(numeig, len(eta)))
  ]
  eig_states = [
      backend.convert_to_tensor(states[n])
      for n in range(min(numeig, len(eta)))
  ]

  return eigvals, eig_states
