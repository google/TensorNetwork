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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensornetwork.matrixproductstates.base_mps import BaseMPS
from tensornetwork.matrixproductstates.mpo import BaseMPO
from tensornetwork.ncon_interface import ncon
from sys import stdout
from typing import Any, Text, Union
Tensor = Any


class BaseDMRG:
  """
  A base class for DMRG (and possibly other) simulations.
  Finite DMRG and infinite DMRG are subclassed from `BaseDMRG`.
  """

  def __init__(self, mps: BaseMPS, mpo: BaseMPO, left_boundary: Tensor,
               right_boundary: Tensor, name: Text):
    """
    Base class for DMRG simulations.
    Args:
      mps: The initial mps. Should be either FiniteMPS or InfiniteMPS 
        (latter is not yet supported).
      mpo: A `FiniteMPO` or `InfiniteMPO` object.
      lb:  The left boundary environment. `lb` has to have shape 
        (mpo[0].shape[0],mps[0].shape[0],mps[0].shape[0])
      rb: The right environment. `rb` has to have shape 
        (mpo[-1].shape[1],mps[-1].shape[1],mps[-1].shape[1])
    Raises:
      TypeError: If mps and mpo have different backends.
      ValueError: If len(mps) != len(mpo).
     """
    if mps.backend is not mpo.backend:
      raise TypeError('mps and mpo use different backends.')

    if not mps.dtype == mpo.dtype:
      raise TypeError('mps.dtype = {} is different from mpo.dtype = {}'.format(
          mps.dtype, mpo.dtype))

    if len(mps) != len(mpo):
      raise ValueError('len(mps) = {} is different from len(mpo) = {}'.format(
          len(mps), len(mpo)))

    self.mps = mps
    self.mpo = mpo
    self.left_envs = {0: self.backend.convert_to_tensor(left_boundary)}
    self.right_envs = {
        len(mps) - 1: self.backend.convert_to_tensor(right_boundary)
    }

    def _add_left_layer(L, mps_tensor, mpo_tensor):
      return ncon([L, mps_tensor, mpo_tensor,
                   self.backend.conj(mps_tensor)],
                  [[2, 1, 5], [1, 3, -2], [2, -1, 4, 3], [5, 4, -3]],
                  backend=self.backend.name)

    self.add_left_layer = self.backend.jit(_add_left_layer)

    def _add_right_layer(R, mps_tensor, mpo_tensor):
      return ncon([R, mps_tensor, mpo_tensor,
                   self.backend.conj(mps_tensor)],
                  [[2, 1, 5], [-2, 3, 1], [-1, 2, 4, 3], [-3, 4, 5]],
                  backend=self.backend.name)

    self.add_left_layer = self.backend.jit(_add_left_layer)
    self.add_right_layer = self.backend.jit(_add_right_layer)

    def _single_site_matvec(L, mpotensor, R, mpstensor):
      return ncon([L, mpstensor, mpotensor, R],
                  [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
                  backend=self.backend.name)

    self.single_site_matvec = _single_site_matvec  #jitting happens inside eigsh_lanczos
    self.name = name

  @property
  def backend(self):
    return self.mps.backend

  @property
  def dtype(self):
    """
    return the dtype of BaseMPS
    """
    if not self.mps.dtype == self.mpo.dtype:
      raise TypeError('mps.dtype = {} is different from mpo.dtype = {}'.format(
          self.mps.dtype, self.mpo.dtype))
    return self.mps.dtype

  def position(self, site: int):
    """
    Shifts the center position of mps to site `n`, and updates left and 
    right environments accordingly. Left blocks at site > n are set 
    to `None`, and right blocks at site < n are `None`. 
    Args:
      site: The bond to which the position should be shifted

    Returns: self
    """
    if site > len(self.mps):
      raise IndexError("site > len(mps)")
    if site < 0:
      raise IndexError("site < 0")
    if site == self.mps.center_position:
      return

    elif site > self.mps.center_position:
      pos = self.mps.center_position
      self.mps.position(site)
      for m in range(pos, site):
        self.left_envs[m + 1] = self.add_left_layer(self.left_envs[m],
                                                    self.mps.tensors[m],
                                                    self.mpo.tensors[m])

    elif site < self.mps.center_position:
      pos = self.mps.center_position
      self.mps.position(site)
      for m in reversed(range(site, pos)):
        self.right_envs[m] = self.add_right_layer(self.right_envs[m + 1],
                                                  self.mps.tensors[m + 1],
                                                  self.mpo.tensors[m + 1])

    for m in range(site + 1, len(self.mps) + 1):
      try:
        del self.left_envs[m]
      except KeyError:
        pass
    for m in range(-1, site):
      try:
        del self.right_envs[m]
      except KeyError:
        pass

    return self

  def compute_left_envs(self):
    """
    Compute all left environment blocks up to self.mps.center_position.
    """
    lb = self.left_envs[0]
    self.left_envs = {0: lb}

    for n in range(self.mps.center_position):
      self.left_envs[n + 1] = self.add_left_layer(self.left_envs[n],
                                                  self.mps.tensors[n],
                                                  self.mpo.tensors[n])

  def compute_right_envs(self):
    """
    Compute all right environment blocks up to self.mps.center_position.
    """
    rb = self.right_envs[len(self.mps) - 1]
    self.right_envs = {len(self.mps) - 1: rb}
    for n in reversed(range(self.mps.center_position + 1, len(self.mps))):
      self.right_envs[n - 1] = self.add_right_layer(self.right_envs[n],
                                                    self.mps.tensors[n],
                                                    self.mpo.tensors[n])

  def _optimize_1s_local(self,
                         sweep_dir,
                         num_krylov_vecs=10,
                         tol=1E-5,
                         delta=1E-6,
                         ndiag=10,
                         verbose=0):
    site = self.mps.center_position
    initial = self.mps.tensors[self.mps.center_position]
    #note: some backends will jit functions
    self.left_envs[site]
    self.right_envs[site]
    energies, states = self.backend.eigsh_lanczos(
        A=self.single_site_matvec,
        args=[
            self.left_envs[site], self.mpo.tensors[site], self.right_envs[site]
        ],
        initial_state=self.mps.tensors[site],
        num_krylov_vecs=num_krylov_vecs,
        numeig=1,
        tol=tol,
        delta=delta,
        ndiag=ndiag,
        reorthogonalize=False)
    local_ground_state = states[0]
    energy = energies[0]
    local_ground_state /= self.backend.norm(local_ground_state)

    if sweep_dir in ('r', 'right'):
      Q, R = self.mps.qr_decomposition(local_ground_state)
      self.mps.tensors[site] = Q
      if site < len(self.mps.tensors) - 1:
        self.mps.center_position += 1
        self.mps.tensors[site + 1] = self.mps.lcontract(
            R, self.mps.tensors[site + 1])
        self.left_envs[site + 1] = self.add_left_layer(self.left_envs[site], Q,
                                                       self.mpo.tensors[site])

    elif sweep_dir in ('l', 'left'):
      R, Q = self.mps.rq_decomposition(local_ground_state)
      self.mps.tensors[site] = Q
      if site > 0:
        self.mps.center_position -= 1
        self.mps.tensors[site - 1] = self.mps.rcontract(
            self.mps.tensors[site - 1], R)
        self.right_envs[site - 1] = self.add_right_layer(
            self.right_envs[site], Q, self.mpo.tensors[site])

    return energy

  def run_one_site(self,
                   num_sweeps=4,
                   precision=1E-6,
                   num_krylov_vecs=10,
                   verbose=0,
                   delta=1E-6,
                   tol=1E-6,
                   ndiag=10):
    """
    Run a single-site DMRG optimization of the MPS.
    Args:
      num_sweeps: Number of DMRG sweeps. A sweep optimizes all sites
        starting at the left side, moving to the right side, and back
        to the left side.
      precision: The desired precision of the energy. If `precision` is
        reached, optimization is terminated.
      num_krylov_vecs: Krylov space dimension used in the iterative eigsh_lanczos
        method.
      verbose: Verbosity flag. Us`verbose=0` to suppress any output. Larger values
        prpoduce increasingly more output.
      delta: Convergence parameter of `eigsh_lanczos` to determine if an invariant
        subspace has been found.
      tol: Tolerance parameter of `eigsh_lanczos`. If eigenvalues in `eigsh_lanczos`
        have converged within `tol`, `eighs_lanczos` is terminted.
      ndiag: Inverse frequency at which eigenvalues of the tridiagonal Hamiltonian
        produced by `eigsh_lanczos` are tested for convergence. `ndiag=10` tests
        at every tenth step.
    Returns:
      float: The energy upon termination of `run_one_site`.
    """
    converged = False
    final_energy = 1E100
    iteration = 0
    initial_site = 0
    self.mps.position(0)  #move center position to the left end
    self.compute_right_envs()

    def print_msg(site):
      if verbose > 0:
        stdout.write("\rSS-DMRG it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f" %
                     (iteration, num_sweeps, site, len(
                         self.mps), np.real(energy), np.imag(energy)))
        stdout.flush()
      if verbose > 1:
        print("")

    while not converged:
      if initial_site == 0:
        self.position(0)
        #the part outside the loop covers the len(self)==1 case
        energy = self._optimize_1s_local(
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag,
            verbose=verbose)
        initial_site += 1
        print_msg(site=0)

      for site in range(initial_site, len(self.mps) - 1):
        #_optimize_1site_local shifts the center site internally
        energy = self._optimize_1s_local(
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag,
            verbose=verbose)
        print_msg(site=site)

      #prepare for right sweep: move center all the way to the right
      self.position(len(self.mps) - 1)
      for site in reversed(range(len(self.mps) - 1)):
        #_optimize_1site_local shifts the center site internally
        energy = self._optimize_1s_local(
            sweep_dir='left',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag,
            verbose=verbose)
        print_msg(site=site)

      if np.abs(final_energy - energy) < precision:
        converged = True
      final_energy = energy
      iteration += 1
      if iteration > num_sweeps:
        if verbose > 0:
          print()
          print(
              'dmrg did not converge to desired precision {0} after {1} iterations'
              .format(precision, num_sweeps))
        break
    return final_energy


class FiniteDMRG(BaseDMRG):
  """
    DMRGUnitCellEngine
    simulation container for density matrix renormalization group optimization

    """

  def __init__(self, mps, mpo, name='FiniteDMRG'):
    lshape = (mpo.tensors[0].shape[0], mps.tensors[0].shape[0],
              mps.tensors[0].shape[0])
    rshape = (mpo.tensors[-1].shape[1], mps.tensors[-1].shape[2],
              mps.tensors[-1].shape[2])
    lb = mps.backend.ones(lshape, dtype=mps.dtype)
    rb = mps.backend.ones(rshape, dtype=mps.dtype)
    super().__init__(
        mps=mps, mpo=mpo, left_boundary=lb, right_boundary=rb, name=name)
