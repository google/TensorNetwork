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
from tensornetwork.network_components import BaseNode, Node
# pylint: disable=line-too-long
from tensornetwork.network_operations import split_node_qr, split_node_rq, split_node_full_svd, norm, conj
from tensornetwork.matrixproductstates.base_mps import BaseMPS
from tensornetwork.matrixproductstates.mpo import BaseMPO
from tensornetwork import ncon
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
    Base class for simulation objects; upon initialization
    Args:
      mps: The initial mps. Should be either FiniteMPS or InfiniteMPS 
        (latter is not yet supported)
      mpo: Hamiltonian in MPO format.
      lb:  The left boundary environment. `lb` has to have shape 
        (mps[0].shape[0],mps[0].shape[0],mpo[0].shape[0])
      rb: The right environment. `rb` has to have shape 
        (mps[-1].shape[1],mps[-1].shape[1],mpo[-1].shape[1])
    Raises:
      TypeError: If mps and mpo have different backends.
      ValueError: If len(mps) != len(mpo).
     """
    if not mps.backend.name == mpo.backend.name:
      raise TypeError(
          'mps.backend.name={} is different from mpo.backend.name={}.'.format(
              mps.backend.name, mpo.backend.name))

    if not mps.dtype == mpo.dtype:
      raise TypeError('mps.dtype={} is different from mpo.dtype={}'.format(
          mps.dtype, mpo.dtype))

    if not mps.dtype == mpo.dtype:
      raise TypeError('mps.dtype={} is different from mpo.dtype={}'.format(
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

    self.single_site_matvec = self.backend.make_passable_to_jit(
        self.backend.jit(_single_site_matvec))

  def __len__(self):
    """
    return the length of the mps/mpo
    """
    return len(self.mps)

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

  def position(self, n: int):
    """
    Shifts the center position of mps to bond n, and updates left and 
    right environments accordingly. Left blocks at site > n are set 
    to `None`, and right blocks at site < n are `None`. 
    Args:
      n: The bond to which the position should be shifted

    returns: self
    """
    if n > len(self.mps):
      raise IndexError("n > len(mps)")
    if n < 0:
      raise IndexError("n < 0")
    if n == self.mps.center_position:
      return

    elif n > self.mps.center_position:
      pos = self.mps.center_position
      self.mps.position(n)
      for m in range(pos, n):
        self.left_envs[m + 1] = self.add_left_layer(self.left_envs[m],
                                                    self.mps.tensors[m],
                                                    self.mpo.tensors[m])

    elif n < self.mps.center_position:
      pos = self.mps.center_position
      self.mps.position(n)
      for m in reversed(range(n, pos)):
        self.right_envs[m] = self.add_right_layer(self.right_envs[m + 1],
                                                  self.mps.tensors[m + 1],
                                                  self.mpo.tensors[m + 1])

    for m in range(n + 1, len(self.mps) + 1):
      try:
        del self.left_envs[m]
      except KeyError:
        pass
    for m in range(-1, n):
      try:
        del self.right_envs[m]
      except KeyError:
        pass

    return self

  def compute_left_envs(self):
    """
    Compute all left environment blocks.
    """
    lb = self.left_envs[0]
    self.left_envs = {0: lb}

    for n in range(self.mps.center_position):
      self.left_envs[n + 1] = self.add_left_layer(self.left_envs[n],
                                                  self.mps.tensors[n],
                                                  self.mpo.tensors[n])

  def compute_right_envs(self):
    """
    Compute all right environment blocks.
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

  def run_one_site_timing(self,
                          start,
                          stop,
                          num_sweeps=4,
                          precision=1E-6,
                          num_krylov_vecs=10,
                          verbose=0,
                          delta=1E-6,
                          tol=1E-6,
                          ndiag=10):
    converged = False
    final_energy = 1E100
    iteration = 0
    initial_site = start
    self.mps.position(start)  #move center position to the left end
    self.compute_left_envs()
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
      for site in range(start, stop - 1):
        #_optimize_1site_local shifts the center site internally
        energy = self._optimize_1s_local(
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag,
            verbose=verbose)
        print_msg(site=self.mps.center_position)

      #prepare for right sweep: move center all the way to the right
      self.position(stop)
      for site in reversed(range(start + 1, stop)):
        #_optimize_1site_local shifts the center site internally
        energy = self._optimize_1s_local(
            sweep_dir='left',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag,
            verbose=verbose)
        print_msg(site=self.mps.center_position)

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

  # def _optimize_2s_local(self,
  #                        thresh=1E-10,
  #                        D=None,
  #                        ncv=40,
  #                        Ndiag=10,
  #                        landelta=1E-5,
  #                        landeltaEta=1E-5,
  #                        verbose=0):
  #   raise NotImplementedError()
  #   mpol = self.mpo.tensors[self.mpo.pos - 1]
  #   mpor = self.mpo.tensors[self.mpo.pos]
  #   Ml, Mc, dl, dlp = mpol.shape
  #   Mc, Mr, dr, drp = mpor.shape
  #   mpo = tf.reshape(
  #       misc_mps.ncon([mpol, mpor], [[-1, 1, -3, -5], [1, -2, -4, -6]]),
  #       [Ml, Mr, dl * dr, dlp * drp])
  #   initial = misc_mps.ncon([
  #       self.mps.tensors[self.mps.center_position - 1], self.mps.mat,
  #       self.mps.tensors[self.mps.center_position]
  #   ], [[-1, -2, 1], [1, 2], [2, -3, -4]])
  #   Dl, dl, dr, Dr = initial.shape
  #   tf.reshape(initial, [Dl, dl * dr, Dr])
  #   if self.walltime_log:
  #     t1 = time.time()

  #   nit, vecs, alpha, beta = LZ.do_lanczos(
  #       L=self.left_envs[self.mps.center_position - 1],
  #       mpo=mpo,
  #       R=self.right_envs[self.mps.center_position],
  #       initial_state=initial,
  #       ncv=ncv,
  #       delta=landelta)
  #   if self.walltime_log:
  #     self.walltime_log(
  #         lan=[(time.time() - t1) / float(nit)] * int(nit),
  #         QR=[],
  #         add_layer=[],
  #         num_lan=[int(nit)])

  #   temp = tf.reshape(
  #       tf.reshape(opt, [
  #           self.mps.bond_dimensions[self.mps.center_position - 1], dlp, drp,
  #           self.mps.bond_dimensions[self.mps.center_position + 1]
  #       ]), [])
  #   opt.split(mps_merge_data).transpose(0, 2, 3, 1).merge([[0, 1], [2, 3]])

  #   U, S, V = temp.svd(truncation_threshold=thresh, D=D)
  #   Dnew = S.shape[0]
  #   if verbose > 0:
  #     stdout.write(
  #         "\rTS-DMRG it=%i/%i, sites=(%i,%i)/%i: optimized E=%.16f+%.16f at D=%i"
  #         % (self._it, self.Nsweeps,
  #            self.mps.center_position - 1, self.mps.center_position,
  #            len(self.mps), tf.real(e), tf.imag(e), Dnew))
  #     stdout.flush()
  #   if verbose > 1:
  #     print("")

  #   Z = np.sqrt(misc_mps.ncon([S, S], [[1], [1]]))
  #   self.mps.mat = S.diag() / Z

  #   self.mps.tensors[self.mps.center_position - 1] = U.split(
  #       [merge_data[0], [U.shape[1]]]).transpose(0, 2, 1)
  #   self.mps.tensors[self.mps.center_position] = V.split(
  #       [[V.shape[0]], merge_data[1]]).transpose(0, 2, 1)
  #   self.left_envs[self.mps.center_position] = self.add_layer(
  #       B=self.left_envs[self.mps.center_position - 1],
  #       mps_tensor=self.mps.tensors[self.mps.center_position - 1],
  #       mpo_tensor=self.mpo.tensors[self.mps.center_position - 1],
  #       conj_mps_tensor=self.mps.tensors[self.mps.center_position - 1],
  #       direction=1)

  #   self.right_envs[self.mps.center_position - 1] = self.add_layer(
  #       B=self.right_envs[self.mps.center_position],
  #       mps_tensor=self.mps.tensors[self.mps.center_position],
  #       mpo_tensor=self.mpo.tensors[self.mps.center_position],
  #       conj_mps_tensor=self.mps.tensors[self.mps.center_position],
  #       direction=-1)
  #   return e

  # def _optimize_1s_local(self,
  #                        site,
  #                        sweep_dir,
  #                        ncv=40,
  #                        Ndiag=10,
  #                        landelta=1E-5,
  #                        landeltaEta=1E-5,
  #                        verbose=0):

  #   if sweep_dir in (-1, 'r', 'right'):
  #     if self.mps.center_position != site:
  #       raise ValueError(
  #           '_optimize_1s_local for sweep_dir={2}: site={0} != mps.center_position={1}'
  #           .format(site, self.mps.center_position, sweep_dir))
  #   if sweep_dir in (1, 'l', 'left'):
  #     if self.mps.center_position != (site + 1):
  #       raise ValueError(
  #           '_optimize_1s_local for sweep_dir={2}: site={0}, mps.center_position={1}'
  #           .format(site, self.mps.center_position, sweep_dir))

  #   if sweep_dir in (-1, 'r', 'right'):
  #     #NOTE (martin) don't use get_tensor here
  #     initial = misc_mps.ncon([self.mps.mat, self.mps.tensors[site]],
  #                             [[-1, 1], [1, -2, -3]])
  #   elif sweep_dir in (1, 'l', 'left'):
  #     #NOTE (martin) don't use get_tensor here
  #     initial = misc_mps.ncon([self.mps.tensors[site], self.mps.mat],
  #                             [[-1, -2, 1], [1, -3]])
  #   if self.walltime_log:
  #     t1 = time.time()
  #   nit, vecs, alpha, beta = LZ.do_lanczos(
  #       L=self.left_envs[site],
  #       mpo=self.mpo.tensors[site],
  #       R=self.right_envs[site],
  #       initial_state=initial,
  #       ncv=np.min([
  #           ncv,
  #           int(initial.shape[0]) * int(initial.shape[1]) * int(
  #               initial.shape[2])
  #       ]),
  #       delta=landelta)

  #   if self.walltime_log:
  #     self.walltime_log(
  #         lan=[(time.time() - t1) / float(nit)] * int(nit),
  #         QR=[],
  #         add_layer=[],
  #         num_lan=[int(nit)])

  #   e, opt = LZ.tridiag(vecs, alpha, beta)
  #   Dnew = opt.shape[2]
  #   # if verbose == (-1):
  #   #     print(f"SS-DMRG  site={site}: optimized E={e}")

  #   if verbose > 0:
  #     stdout.write(
  #         "\rSS-DMRG it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i" %
  #         (self._it, self.Nsweeps, site, len(self.mps), np.real(e), np.imag(e),
  #          Dnew))
  #     stdout.flush()

  #   if verbose > 1:
  #     print("")

  #   if self.walltime_log:
  #     t1 = time.time()
  #   if sweep_dir in (-1, 'r', 'right'):
  #     A, mat, Z = misc_mps.prepare_tensor_QR(opt, direction='l')
  #     A /= Z
  #   elif sweep_dir in (1, 'l', 'left'):
  #     mat, B, Z = misc_mps.prepare_tensor_QR(opt, direction='r')
  #     B /= Z
  #   if self.walltime_log:
  #     self.walltime_log(lan=[], QR=[time.time() - t1], add_layer=[], num_lan=[])

  #   self.mps.mat = mat
  #   if sweep_dir in (-1, 'r', 'right'):
  #     self.mps._tensors[site] = A
  #     self.mps.center_position += 1
  #     self.left_envs[site + 1] = self.add_layer(
  #         B=self.left_envs[site],
  #         mps_tensor=self.mps.tensors[site],
  #         mpo_tensor=self.mpo.tensors[site],
  #         conj_mps_tensor=self.mps.tensors[site],
  #         direction=1,
  #         walltime_log=self.walltime_log)

  #   elif sweep_dir in (1, 'l', 'left'):
  #     self.mps._tensors[site] = B
  #     self.mps.center_position = site
  #     self.right_envs[site - 1] = self.add_layer(
  #         B=self.right_envs[site],
  #         mps_tensor=self.mps.tensors[site],
  #         mpo_tensor=self.mpo.tensors[site],
  #         conj_mps_tensor=self.mps.tensors[site],
  #         direction=-1,
  #         walltime_log=self.walltime_log)
  #   return e


# class FiniteDMRGEngine(DMRGUnitCellEngine):

#   def __init__(self, mps, mpo, name='FiniteDMRG'):
#     # if not isinstance(mps, FiniteMPSCentralGauge):
#     #     raise TypeError(
#     #         'in FiniteDMRGEngine.__init__(...): mps of type FiniteMPSCentralGauge expected, got {0}'
#     #         .format(type(mps)))

#     lb = tf.ones([
#         mps.bond_dimensions[0], mps.bond_dimensions[0], mpo.bond_dimensions[0]
#     ],
#                  dtype=mps.dtype)
#     rb = tf.ones([
#         mps.bond_dimensions[-1], mps.bond_dimensions[-1],
#         mpo.bond_dimensions[-1]
#     ],
#                  dtype=mps.dtype)
#     super().__init__(mps=mps, mpo=mpo, lb=lb, rb=rb, name=name)
