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

  def __init__(self, matvec: Callable,
               shape: Tuple[Tuple[int]], backend: str, dtype: Type[np.number]):
    self.matvec = matvec
    self.shape = shape
    self.backend = backend_factory.get_backend(backend, dtype)

  @property
  def dtype(self):
    return self.backend.dtype


def eigsh(A: LinearOperator,
          vv: Callable,
          v0: Optional[Tensor] = None,
          ncv: Optional[int] = 200,
          tol: Optional[float] = 1E-8,
          delta: Optional[float] = 1E-8
          ndiag: Optional[int] = 20,
          reortho: Optional[bool] = False):

  if v0 and (A.dtype is not v0.dtype):
    raise TypeError("A.dtype={} is different from v0.dtype={}".format(
        A.dtype, v0.dtype))
  if v0:
    xn = v0
  else:
    xn = self.backend.randn(self.shape[1])

  Z = self.backend.norm(xn)
  xn /= Z

  converged = False
  it = 0
  norms_xn = []
  epsn = []
  krylov_vecs = []
  first = True
  while converged == False:
    #normalize the current vector:
    normxn = self.backend.sqrt(vv(xn, xn))
    if normxn < delta:
      converged = True
      break
    norms_xn.append(normxn)
    xn = xn / norms_xn[-1]
    #store the Lanczos vector for later
    if reortho == True:
      for v in krylov_vecs:
        xn -= self.scalar_product(v, xn) * v
    krylov_vecs.append(xn)
    Hxn = A.matvec(xn)
    epsn.append(self.scalar_product(xn, Hxn))

    if ((it > 0) and (it % self.Ndiag) == 0) & (len(epsn) >= self.numeig):
      #diagonalize the effective Hamiltonian
      Heff = np.diag(epsn) + np.diag(norms_xn[1:], 1) + np.diag(np.conj(norms_xn[1:]), -1)
      eta, u = np.linalg.eigh(Heff)
      if first == False:
        if np.linalg.norm(eta[0:self.numeig] -
                          etaold[0:self.numeig]) < self.deltaEta:
          converged = True
      first = False
      etaold = eta[0:self.numeig]
    if it > 0:
      Hxn -= (krylov_vecs[-1] * epsn[-1])
      Hxn -= (krylov_vecs[-2] * norms_xn[-1])
    else:
      Hxn -= (krylov_vecs[-1] * epsn[-1])
    xn = Hxn
    it = it + 1
    if it >= self.ncv:
      break

  if walltime_log:
    walltime_log(
        lan=[(time.time() - t1) / len(epsn)] * len(epsn),
        QR=[],
        add_layer=[],
        num_lan=[len(epsn)])

  self.Heff = np.diag(epsn) + np.diag(norms_xn[1:], 1) + np.diag(np.conj(norms_xn[1:]), -1)
  eta, u = np.linalg.eigh(self.Heff)
  states = []
  for n2 in range(min(self.numeig, len(eta))):
    if zeros is None:
      state = initialstate.zeros(initialstate.shape, dtype=initialstate.dtype)
    else:
      state = copy.deepcopy(zeros)
    for n1 in range(len(krylov_vecs)):
      state += krylov_vecs[n1] * u[n1, n2]
    states.append(state / np.sqrt(self.scalar_product(state, state)))

  return eta[0:min(self.numeig, len(eta))], states[0:min(self.numeig, len(eta)
                                                        )], it  #,epsn,kn
