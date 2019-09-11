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
from typing import Any, Sequence, List, Set, Optional, Union, Text, Tuple, Type, Dict, Callable
import numpy as np
Tensor = Any


class LinearOperator:

  def __init__(self, matvec: Callable, shape: Tuple[Tuple[int]], backend: str,
               dtype: Type[np.number]):

    self.matvec = matvec
    self.shape = shape
    self.backend = backend_factory.get_backend(backend, dtype)

  @property
  def dtype(self):
    return self.backend.dtype

  def __call__(self, x):
    return self.matvec(x, self.backend.name)


def eigsh(A: LinearOperator,
          vv: Callable,
          v0: Optional[Tensor] = None,
          ncv: Optional[int] = 200,
          numeig: Optional[int] = 1,
          tol: Optional[float] = 1E-8,
          delta: Optional[float] = 1E-8,
          ndiag: Optional[int] = 20,
          reortho: Optional[bool] = False):

  if ncv < numeig:
    raise ValueError('`ncv` >= `numeig` required!')
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
    normxn = backend.sqrt(vv(
        xn, xn, backend.name))  #conj has to be implemented by the user
    if normxn < delta:
      converged = True
      break
    norms_xn.append(normxn)
    xn = xn / norms_xn[-1]
    #store the Lanczos vector for later
    if reortho == True:
      for v in krylov_vecs:
        xn -= vv(v, xn, backend.name) * v
    krylov_vecs.append(xn)
    Hxn = A(xn)
    epsn.append(vv(xn, Hxn, backend.name))

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
    states.append(state / backend.sqrt(vv(state, state, backend.name)))
  print([eta[n].dtype for n in range(min(numeig, len(eta)))])
  print(states[0].dtype)
  eigvals = [
      backend.convert_to_tensor(np.array(eta[n]))
      for n in range(min(numeig, len(eta)))
  ]
  eig_states = [
      backend.convert_to_tensor(states[n])
      for n in range(min(numeig, len(eta)))
  ]

  return eigvals, eig_states
