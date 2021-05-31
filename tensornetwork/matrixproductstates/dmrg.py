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
import numpy as np
from tensornetwork.matrixproductstates.base_mps import BaseMPS
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
from tensornetwork.matrixproductstates.mpo import BaseMPO, FiniteMPO
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
      name: An optional name for the simulation.
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
    if mps.center_position is None:
      raise ValueError(
          "Found mps in non-canonical form. Please canonicalize mps.")
    self.mps = mps
    self.mpo = mpo
    self.left_envs = {0: self.backend.convert_to_tensor(left_boundary)}
    self.right_envs = {
        len(mps) - 1: self.backend.convert_to_tensor(right_boundary)
    }
    if self.left_envs[0].dtype != self.dtype:
      raise TypeError(
          'left_boundary.dtype = {} is different from BaseDMRG.dtype = {}'
          .format(self.left_envs[0].dtype.dtype, self.dtype))
    if self.right_envs[len(mps) - 1].dtype != self.dtype:
      raise TypeError(
          'right_boundary.dtype = {} is different from BaseDMRG.dtype = {}'
          .format(self.right_envs[0].dtype, self.dtype))

    self.name = name

  @property
  def backend(self):
    return self.mps.backend

  @property
  def dtype(self):
    """
    Return the dtype of BaseMPS.
    """
    if not self.mps.dtype == self.mpo.dtype:
      raise TypeError('mps.dtype = {} is different from mpo.dtype = {}'.format(
          self.mps.dtype, self.mpo.dtype))
    return self.mps.dtype

  def single_site_matvec(self, mpstensor, L, mpotensor, R):
    return ncon([L, mpstensor, mpotensor, R],
                [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
                backend=self.backend.name)

  def two_site_matvec(self, mps_bond_tensor, L, left_mpotensor,
                      right_mpotensor, R):
    return ncon([L, mps_bond_tensor, left_mpotensor, right_mpotensor, R],
                [[3, 1, -1], [1, 2, 5, 6], [3, 4, -2, 2], [4, 7, -3, 5],
                 [7, 6, -4]],
                backend=self.backend.name)

  def add_left_layer(self, L, mps_tensor, mpo_tensor):
    return ncon([L, mps_tensor, mpo_tensor,
                 self.backend.conj(mps_tensor)],
                [[2, 1, 5], [1, 3, -2], [2, -1, 4, 3], [5, 4, -3]],
                backend=self.backend.name)

  def add_right_layer(self, R, mps_tensor, mpo_tensor):
    return ncon([R, mps_tensor, mpo_tensor,
                 self.backend.conj(mps_tensor)],
                [[2, 1, 5], [-2, 3, 1], [-1, 2, 4, 3], [-3, 4, 5]],
                backend=self.backend.name)

  def position(self, site: int):
    """
    Shifts the center position `site`, and updates left and
    right environments accordingly. Left blocks at sites > `site` are set
    to `None`, and right blocks at sites < `site` are `None`.
    Args:
      site: The site to which the position of the center-site should be shifted.
    Returns: BaseDMRG
    """
    if site >= len(self.mps):
      raise IndexError("site > length of mps")
    if site < 0:
      raise IndexError("site < 0")
    if site == self.mps.center_position:
      return self

    if site > self.mps.center_position:
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

  def compute_left_envs(self) -> None:
    """
    Compute all left environment blocks of sites up to
    (including) self.mps.center_position.
    """
    lb = self.left_envs[0]
    self.left_envs = {0: lb}

    for n in range(self.mps.center_position):
      self.left_envs[n + 1] = self.add_left_layer(self.left_envs[n],
                                                  self.mps.tensors[n],
                                                  self.mpo.tensors[n])

  def compute_right_envs(self) -> None:
    """
    Compute all right environment blocks of sites up to
    (including) self.mps.center_position.
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
                         ndiag=10) -> np.number:
    """
    Single-site optimization at the current position of the center site.
    The method shifts the center position of the mps by one site
    to the left or to the right, depending on the value of `sweep_dir`.
    Args:
      sweep_dir: Sweep direction; 'left' or 'l' for a sweep from right to left,
        'right' or 'r' for a sweep from left to right.
      num_krylov_vecs: Dimension of the Krylov space used in `eighs_lanczos`.
      tol: The desired precision of the eigenvalues in `eigsh_lanczos'.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped.
      ndiag: Inverse frequencey of tridiagonalizations in `eighs_lanczos`.
    Returns:
      float/complex: The local energy after optimization.
    """
    site = self.mps.center_position
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
      Q, R = self.mps.qr(local_ground_state)
      self.mps.tensors[site] = Q
      if site < len(self.mps.tensors) - 1:
        self.mps.center_position += 1
        self.mps.tensors[site + 1] = ncon([R, self.mps.tensors[site + 1]],
                                          [[-1, 1], [1, -2, -3]],
                                          backend=self.backend.name)
        self.left_envs[site + 1] = self.add_left_layer(self.left_envs[site], Q,
                                                       self.mpo.tensors[site])

    elif sweep_dir in ('l', 'left'):
      R, Q = self.mps.rq(local_ground_state)
      self.mps.tensors[site] = Q
      if site > 0:
        self.mps.center_position -= 1
        self.mps.tensors[site - 1] = ncon([self.mps.tensors[site - 1], R],
                                          [[-1, -2, 1], [1, -3]],
                                          backend=self.backend.name)
        self.right_envs[site - 1] = self.add_right_layer(
            self.right_envs[site], Q, self.mpo.tensors[site])

    return energy

  def _optimize_2s_local(self,
                         max_bond_dim,
                         sweep_dir,
                         num_krylov_vecs=10,
                         tol=1E-5,
                         delta=1E-6,
                         ndiag=10) -> np.number:
    """
    Two-site optimization at the current position of the center site.
    The method shifts the center position of the mps by one site
    to the left or to the right, depending on the value of `sweep_dir`.
    Args:
      max_bond_dim: Maximum MPS bond dimension. During DMRG optimization,
        an MPS exceeding this dimension is truncated via SVD.
      sweep_dir: Sweep direction; 'left' or 'l' for a sweep from right to left,
        'right' or 'r' for a sweep from left to right.
      num_krylov_vecs: Dimension of the Krylov space used in `eighs_lanczos`.
      tol: The desired precision of the eigenvalues in `eigsh_lanczos'.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped.
      ndiag: Inverse frequencey of tridiagonalizations in `eighs_lanczos`.
    Returns:
      float/complex: The local energy after optimization.
    """
    site = self.mps.center_position
    #note: some backends will jit functions
    if sweep_dir in ('r', 'right'):
      bond_mps = ncon([self.mps.tensors[site], self.mps.tensors[site + 1]],
                      [[-1, -2, 1], [1, -3, -4]],
                      backend=self.backend.name)
      energies, states = self.backend.eigsh_lanczos(
          A=self.two_site_matvec,
          args=[
              self.left_envs[site], self.mpo.tensors[site],
              self.mpo.tensors[site + 1], self.right_envs[site + 1]
          ],
          initial_state=bond_mps,
          num_krylov_vecs=num_krylov_vecs,
          numeig=1,
          tol=tol,
          delta=delta,
          ndiag=ndiag,
          reorthogonalize=False)
      local_ground_state = states[0]
      energy = energies[0]
      local_ground_state /= self.backend.norm(local_ground_state)

      u, s, vh, _ = self.mps.svd(local_ground_state, 2, max_bond_dim, None)
      s = self.backend.diagflat(s)
      self.mps.tensors[site] = u
      if site < len(self.mps.tensors) - 1:
        self.mps.center_position += 1
        self.mps.tensors[site + 1] = ncon([s, vh], [[-1, 1], [1, -2, -3]],
                                          backend=self.backend.name)
        self.left_envs[site + 1] = self.add_left_layer(self.left_envs[site], u,
                                                       self.mpo.tensors[site])

    elif sweep_dir in ('l', 'left'):
      bond_mps = ncon([self.mps.tensors[site - 1], self.mps.tensors[site]],
                      [[-1, -2, 1], [1, -3, -4]],
                      backend=self.backend.name)
      energies, states = self.backend.eigsh_lanczos(
          A=self.two_site_matvec,
          args=[
              self.left_envs[site - 1], self.mpo.tensors[site - 1],
              self.mpo.tensors[site], self.right_envs[site]
          ],
          initial_state=bond_mps,
          num_krylov_vecs=num_krylov_vecs,
          numeig=1,
          tol=tol,
          delta=delta,
          ndiag=ndiag,
          reorthogonalize=False)
      local_ground_state = states[0]
      energy = energies[0]
      local_ground_state /= self.backend.norm(local_ground_state)

      u, s, vh, _ = self.mps.svd(local_ground_state, 2, max_bond_dim, None)
      s = self.backend.diagflat(s)
      self.mps.tensors[site] = vh
      if site > 0:
        self.mps.center_position -= 1
        self.mps.tensors[site - 1] = ncon([u, s],
                                          [[-1, -2, 1], [1, -3]],
                                          backend=self.backend.name)
        self.right_envs[site - 1] = \
          self.add_right_layer(self.right_envs[site], vh,
                               self.mpo.tensors[site])

    return energy

  def run_one_site(self,
                   num_sweeps=4,
                   precision=1E-6,
                   num_krylov_vecs=10,
                   verbose=0,
                   delta=1E-6,
                   tol=1E-6,
                   ndiag=10) -> np.number:
    """
    Run a single-site DMRG optimization of the MPS.
    Args:
      num_sweeps: Number of DMRG sweeps. A sweep optimizes all sites
        starting at the left side, moving to the right side, and back
        to the left side.
      precision: The desired precision of the energy. If `precision` is
        reached, optimization is terminated.
      num_krylov_vecs: Krylov space dimension used in the iterative
        eigsh_lanczos method.
      verbose: Verbosity flag. Us`verbose=0` to suppress any output.
        Larger values produce increasingly more output.
      delta: Convergence parameter of `eigsh_lanczos` to determine if
        an invariant subspace has been found.
      tol: Tolerance parameter of `eigsh_lanczos`. If eigenvalues in
        `eigsh_lanczos` have converged within `tol`, `eighs_lanczos`
        is terminted.
      ndiag: Inverse frequency at which eigenvalues of the
        tridiagonal Hamiltonian produced by `eigsh_lanczos` are tested
        for convergence. `ndiag=10` tests at every tenth step.
    Returns:
      float: The energy upon termination of `run_one_site`.
    """
    if num_sweeps == 0:
      return self.compute_energy()

    converged = False
    final_energy = 1E100
    iteration = 1
    initial_site = 0

    self.mps.position(0)  #move center position to the left end
    self.compute_right_envs()

    def print_msg(site):
      if verbose < 2:
        stdout.write(f"\rSS-DMRG sweep={iteration}/{num_sweeps}, "
                     f"site={site}/{len(self.mps)}: optimized E={energy}  ")
        stdout.flush()

      if verbose >= 2:
        print(f"SS-DMRG sweep={iteration}/{num_sweeps}, "
              f"site={site}/{len(self.mps)}: optimized E={energy}  ")

    while not converged:
      if initial_site == 0:
        self.position(0)
        #the part outside the loop covers the len(self)==1 case
        energy = self._optimize_1s_local(
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        initial_site += 1
        print_msg(site=0)
      while self.mps.center_position < len(self.mps) - 1:
        #_optimize_1site_local shifts the center site internally
        energy = self._optimize_1s_local(
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        print_msg(site=self.mps.center_position - 1)
      #prepare for left sweep: move center all the way to the right
      self.position(len(self.mps) - 1)
      while self.mps.center_position > 0:
        #_optimize_1site_local shifts the center site internally
        energy = self._optimize_1s_local(
            sweep_dir='left',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        print_msg(site=self.mps.center_position + 1)

      if np.abs(final_energy - energy) < precision:
        converged = True
      final_energy = energy
      iteration += 1
      if iteration > num_sweeps:
        if verbose > 0:
          print()
          print("dmrg did not converge to desired precision {0} "
                "after {1} iterations".format(precision, num_sweeps))
        break
    return final_energy

  def run_two_site(self,
                   max_bond_dim,
                   num_sweeps=4,
                   precision=1E-6,
                   num_krylov_vecs=10,
                   verbose=0,
                   delta=1E-6,
                   tol=1E-6,
                   ndiag=10) -> np.number:
    """
    Run a two-site DMRG optimization of the MPS.
    Args:
      max_bond_dim: Maximum MPS bond dimension. During DMRG optimization,
        an MPS exceeding this dimension is truncated via SVD.
      num_sweeps: Number of DMRG sweeps. A sweep optimizes all sites
        starting at the left side, moving to the right side, and back
        to the left side.
      precision: The desired precision of the energy. If `precision` is
        reached, optimization is terminated.
      num_krylov_vecs: Krylov space dimension used in the iterative
        eigsh_lanczos method.
      verbose: Verbosity flag. Us`verbose=0` to suppress any output.
        Larger values produce increasingly more output.
      delta: Convergence parameter of `eigsh_lanczos` to determine if
        an invariant subspace has been found.
      tol: Tolerance parameter of `eigsh_lanczos`. If eigenvalues in
        `eigsh_lanczos` have converged within `tol`, `eighs_lanczos`
        is terminted.
      ndiag: Inverse frequency at which eigenvalues of the
        tridiagonal Hamiltonian produced by `eigsh_lanczos` are tested
        for convergence. `ndiag=10` tests at every tenth step.
    Returns:
      float: The energy upon termination of `run_two_site`.
    """
    if num_sweeps == 0:
      return self.compute_energy()

    converged = False
    final_energy = 1E100
    iteration = 1
    initial_site = 0

    self.mps.position(0)  #move center position to the left end
    self.compute_right_envs()

    # TODO (pedersor): print max truncation errors
    def print_msg(left_site, right_site):
      if verbose == 0:
        stdout.write(f"\rTS-DMRG sweep={iteration}/{num_sweeps}, "
                     f"sites=({left_site},{right_site})/{len(self.mps)}: "
                     f"optimized E={energy}    ")
        stdout.flush()
      if verbose == 1:
        D = self.mps.bond_dimensions[right_site]
        stdout.write(f"\rTS-DMRG sweep={iteration}/{num_sweeps}, "
                     f"sites=({left_site},{right_site})/{len(self.mps)}: "
                     f"optimized E={energy}, D = {D}     ")
        stdout.flush()

      if verbose >= 2:
        D = self.mps.bond_dimensions[left_site]
        print(f"TS-DMRG sweep={iteration}/{num_sweeps}, "
              f"sites=({left_site},{right_site})/{len(self.mps)}: "
              f"optimized E={energy}, D = {D}     ")

    while not converged:
      if initial_site == 0:
        self.position(0)
        #the part outside the loop covers the len(self)==1 case
        energy = self._optimize_2s_local(
            max_bond_dim=max_bond_dim,
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        initial_site += 1
        print_msg(left_site=0, right_site=1)
      while self.mps.center_position < len(self.mps) - 1:
        #_optimize_2site_local shifts the center site internally
        energy = self._optimize_2s_local(
            max_bond_dim=max_bond_dim,
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        print_msg(self.mps.center_position - 1, self.mps.center_position)
      #prepare for left sweep: move center all the way to the right
      self.position(len(self.mps) - 1)
      while self.mps.center_position > 0:
        #_optimize_2site_local shifts the center site internally
        energy = self._optimize_2s_local(
            max_bond_dim=max_bond_dim,
            sweep_dir='left',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        print_msg(self.mps.center_position, self.mps.center_position + 1)

      if np.abs(final_energy - energy) < precision:
        converged = True
      final_energy = energy
      iteration += 1
      if iteration > num_sweeps:
        if verbose > 0:
          print()
          print("dmrg did not converge to desired precision {0} "
                "after {1} iterations".format(precision, num_sweeps))
        break
    return final_energy

  def compute_energy(self):
    self.mps.position(0)  #move center position to the left end
    self.compute_right_envs()
    return ncon([
        self.add_right_layer(self.right_envs[0], self.mps.tensors[0],
                             self.mpo.tensors[0]),
        self.left_envs[0],
    ], [[1, 2, 3], [1, 2, 3]],
                backend=self.backend.name).item()


class FiniteDMRG(BaseDMRG):
  """
  Class for simulating finite DMRG.
  """

  def __init__(self,
               mps: FiniteMPS,
               mpo: FiniteMPO,
               name: Text = 'FiniteDMRG') -> None:
    """
    Initialize a finite DRMG simulation.
    Args:
      mps: A FiniteMPS object.
      mpo: A FiniteMPO object.
      name: An optional name for the simulation.
    """
    backend = mps.backend
    conmpo0 = backend.conj(mpo.tensors[0])
    conmps0 = backend.conj(mps.tensors[0])
    mps0 = mps.tensors[0]

    conmpoN = backend.conj(mpo.tensors[-1])
    conmpsN = backend.conj(mps.tensors[-1])
    mpsN = mps.tensors[-1]

    lshape = (backend.sparse_shape(conmpo0)[0],
              backend.sparse_shape(conmps0)[0], backend.sparse_shape(mps0)[0])
    rshape = (backend.sparse_shape(conmpoN)[1],
              backend.sparse_shape(conmpsN)[2], backend.sparse_shape(mpsN)[2])
    lb = backend.ones(lshape, dtype=mps.dtype)
    rb = backend.ones(rshape, dtype=mps.dtype)
    super().__init__(
        mps=mps, mpo=mpo, left_boundary=lb, right_boundary=rb, name=name)
