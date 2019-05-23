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
"""implementations of finite and infinite Density Matrix Renormalization Group algorithms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../')
import time
import ncon as ncon
import numpy as np
import tensorflow as tf
import Lanczos as LZ
from sys import stdout
import misc_mps
import functools as fct
from matrixproductstates import InfiniteMPSCentralGauge, FiniteMPSCentralGauge


class MPSSimulationBase:

  def __init__(self, mps, mpo, lb, rb, name):
    """
        Base class for simulation objects; upon initialization, creates all 
        left and right envvironment blocks
        mps:      MPS object
                  the initial mps
        mpo:      MPO object
                  Hamiltonian in MPO format
        name: str
                  the name of the simulation
        lb:       np.ndarray of shape (D,D,M), or None
                  the left environment; 
                  lb has to have shape (mps[0].shape[0],mps[0].shape[0],mpo[0].shape[0])
                  if None, obc are assumed, and lb=ones((mps[0].shape[0],mps[0].shape[0],mpo[0].shape[0]))
        rb:       np.ndcarray of shape (D,D,M), or None
                  the right environment
                  rb has to have shape (mps[-1].shape[1],mps[-1].shape[1],mpo[-1].shape[1])
                  if None, obc are assumed, and rb=ones((mps[-1].shape[1],mps[-1].shape[1],mpo[-1].shape[1]))
        """
    self.mps = mps
    self.mpo = mpo
    if not self.mps.dtype == self.mpo.dtype:
      raise TypeError('the types of mps and mpo are not compatible')
    if len(mps) != len(mpo):
      raise ValueError('len(mps)!=len(mpo)')
    self.mps.position(0)
    self.lb = lb
    self.rb = rb
    self.left_envs = {0: self.lb}
    self.right_envs = {len(mps) - 1: self.rb}

  def __len__(self):
    """
        return the length of the simulation
        """
    return len(self.mps)

  @property
  def dtype(self):
    """
        return the data-type of the MPSSimulationBase

        type is obtained from applying np.result_type 
        to the mps and mpo objects
        """
    assert (self.mps.dtype == self.mpo.dtype)
    return self.mps.dtype

  @staticmethod
  def add_layer(B,
                mps_tensor,
                mpo_tensor,
                conj_mps_tensor,
                direction,
                walltime_log=None):
    """
        adds an mps-mpo-mps layer to a left or right block "E"; used in dmrg to calculate the left and right
        environments
        Parameters:
        ---------------------------
        B:               Tensor object  
                         a tensor of shape (D1,D1',M1) (for direction>0) or (D2,D2',M2) (for direction>0)
        mps_tensor:      Tensor object of shape =(Dl,Dr,d)
        mpo_tensor:      Tensor object of shape = (Ml,Mr,d,d')
        conj_mps_tensor: Tensor object of shape =(Dl',Dr',d')
                         the mps tensor on the conjugated side
                         this tensor will be complex conjugated inside the routine; usually, the user will like to pass 
                         the unconjugated tensor
        direction:       int or str
                         direction in (1,'l','left'): add a layer to the right of ```B```
                  direction in (-1,'r','right'): add a layer to the left of ```B```

        Return:
        -----------------
        Tensor of shape (Dr,Dr',Mr) for direction in (1,'l','left')
        Tensor of shape (Dl,Dl',Ml) for direction in (-1,'r','right')
        """
    if walltime_log:
      t1 = time.time()
    out = misc_mps.add_layer(
        B, mps_tensor, mpo_tensor, conj_mps_tensor, direction=direction)

    if walltime_log:
      walltime_log(lan=[], QR=[], add_layer=[time.time() - t1], num_lan=[])
    return out

  def position(self, n):
    """
        shifts the center position of mps to bond n, and updates left and right environments
        accordingly; Left blocks at site > n are None, and right blocks at site < n are None
        Note that the index convention for R blocks is reversed, i.e. self.right_envs[0] is self.rb, 
        self.right_envs[1] is the second right most R-block, a.s.o
        Parameters:
        ------------------------------------
        n: int
           the bond to which the position should be shifted

        returns: self
        """
    if n > len(self.mps):
      raise IndexError("MPSSimulationBase.position(n): n>len(mps)")
    if n < 0:
      raise IndexError("MPSSimulationBase.position(n): n<0")
    if n == self.mps.pos:
      return

    elif n > self.mps.pos:
      pos = self.mps.pos
      if self.walltime_log:
        t1 = time.time()
      self.mps.position(n)
      if self.walltime_log:
        self.walltime_log(
            lan=[], QR=[time.time() - t1], add_layer=[], num_lan=[])

      for m in range(pos, n):
        self.left_envs[m + 1] = self.add_layer(
            self.left_envs[m],
            self.mps[m],
            self.mpo[m],
            self.mps[m],
            direction=1,
            walltime_log=self.walltime_log)

    elif n < self.mps.pos:
      pos = self.mps.pos
      if self.walltime_log:
        t1 = time.time()
      self.mps.position(n)
      if self.walltime_log:
        self.walltime_log(
            lan=[], QR=[time.time() - t1], add_layer=[], num_lan=[])

      for m in reversed(range(n, pos)):
        self.right_envs[m - 1] = self.add_layer(
            self.right_envs[m],
            self.mps[m],
            self.mpo[m],
            self.mps[m],
            direction=-1,
            walltime_log=self.walltime_log)

    for m in range(n + 1, len(self.mps) + 1):
      try:
        del self.left_envs[m]
      except KeyError:
        pass
    for m in range(-1, n - 1):
      try:
        del self.right_envs[m]
      except KeyError:
        pass

    return self

  def update(self):
    """
        shift center site of the MPSSimulationBase to 0 and recalculate all left and right blocks
        """
    self.mps.position(0)
    self.compute_left_envs()
    self.compute_right_envs()
    return self


class DMRGUnitCellEngine(MPSSimulationBase):
  """
    DMRGUnitCellEngine
    simulation container for density matrix renormalization group optimization

    """

  def __init__(self, mps, mpo, lb, rb, name='DMRG'):
    """
        initialize an MPS object
        mps:      MPS object
                  the initial mps
        mpo:      MPO object
                  Hamiltonian in MPO format
        name:     str
                  the name of the simulation
        lb,rb:    None or np.ndarray
                  left and right environment boundary conditions
                  if None, obc are assumed
                  user can provide lb and rb to fix the boundary condition of the mps
                  shapes of lb, rb, mps[0] and mps[-1] have to be consistent
        """
    self.walltime_log = None
    super().__init__(mps=mps, mpo=mpo, name=name, lb=lb, rb=rb)
    self.compute_right_envs()

  def compute_left_envs(self):
    """
        compute all left environment blocks
        up to self.mps.position; all blocks for site > self.mps.position are set to None
        """
    self.left_envs = {}
    self.left_envs[0] = self.lb
    for n in range(self.mps.pos):
      self.left_envs[n + 1] = self.add_layer(
          B=self.left_envs[n],
          mps_tensor=self.mps[n],
          mpo_tensor=self.mpo[n],
          conj_mps_tensor=self.mps[n],
          direction=1,
          walltime_log=self.walltime_log)

  def compute_right_envs(self):
    """
        compute all right environment blocks
        up to self.mps.position; all blocks for site < self.mps.position are set to None
        """
    self.right_envs = {}
    self.right_envs[len(self.mps) - 1] = self.rb
    for n in reversed(range(self.mps.pos, len(self.mps))):
      self.right_envs[n - 1] = self.add_layer(
          B=self.right_envs[n],
          mps_tensor=self.mps[n],
          mpo_tensor=self.mpo[n],
          conj_mps_tensor=self.mps[n],
          direction=-1,
          walltime_log=self.walltime_log)

  def _optimize_2s_local(self,
                         thresh=1E-10,
                         D=None,
                         ncv=40,
                         Ndiag=10,
                         landelta=1E-5,
                         landeltaEta=1E-5,
                         verbose=0):
    raise NotImplementedError()
    mpol = self.mpo[self.mpo.pos - 1]
    mpor = self.mpo[self.mpo.pos]
    Ml, Mc, dl, dlp = mpol.shape
    Mc, Mr, dr, drp = mpor.shape
    mpo = tf.reshape(
        ncon.ncon([mpol, mpor], [[-1, 1, -3, -5], [1, -2, -4, -6]]),
        [Ml, Mr, dl * dr, dlp * drp])
    initial = ncon.ncon(
        [self.mps[self.mps.pos - 1], self.mps.mat, self.mps[self.mps.pos]],
        [[-1, -2, 1], [1, 2], [2, -3, -4]])
    Dl, dl, dr, Dr = initial.shape
    tf.reshape(initial, [Dl, dl * dr, Dr])
    if self.walltime_log:
      t1 = time.time()

    nit, vecs, alpha, beta = LZ.do_lanczos(
        L=self.left_envs[self.mps.pos - 1],
        mpo=mpo,
        R=self.right_envs[self.mps.pos],
        initial_state=initial,
        ncv=ncv,
        delta=landelta)
    if self.walltime_log:
      self.walltime_log(
          lan=[(time.time() - t1) / float(nit)] * int(nit),
          QR=[],
          add_layer=[],
          num_lan=[int(nit)])

    temp = tf.reshape(
        tf.reshape(opt, [
            self.mps.D[self.mps.pos - 1], dlp, drp, self.mps.D[self.mps.pos + 1]
        ]), [])
    opt.split(mps_merge_data).transpose(0, 2, 3, 1).merge([[0, 1], [2, 3]])

    U, S, V = temp.svd(truncation_threshold=thresh, D=D)
    Dnew = S.shape[0]
    if verbose > 0:
      stdout.write(
          "\rTS-DMRG it=%i/%i, sites=(%i,%i)/%i: optimized E=%.16f+%.16f at D=%i"
          % (self._it, self.Nsweeps, self.mps.pos - 1, self.mps.pos,
             len(self.mps), tf.real(e), tf.imag(e), Dnew))
      stdout.flush()
    if verbose > 1:
      print("")

    Z = np.sqrt(ncon.ncon([S, S], [[1], [1]]))
    self.mps.mat = S.diag() / Z

    self.mps[self.mps.pos - 1] = U.split([merge_data[0],
                                          [U.shape[1]]]).transpose(0, 2, 1)
    self.mps[self.mps.pos] = V.split([[V.shape[0]], merge_data[1]]).transpose(
        0, 2, 1)
    self.left_envs[self.mps.pos] = self.add_layer(
        B=self.left_envs[self.mps.pos - 1],
        mps_tensor=self.mps[self.mps.pos - 1],
        mpo_tensor=self.mpo[self.mps.pos - 1],
        conj_mps_tensor=self.mps[self.mps.pos - 1],
        direction=1)

    self.right_envs[self.mps.pos - 1] = self.add_layer(
        B=self.right_envs[self.mps.pos],
        mps_tensor=self.mps[self.mps.pos],
        mpo_tensor=self.mpo[self.mps.pos],
        conj_mps_tensor=self.mps[self.mps.pos],
        direction=-1)
    return e

  def _optimize_1s_local(self,
                         site,
                         sweep_dir,
                         ncv=40,
                         Ndiag=10,
                         landelta=1E-5,
                         landeltaEta=1E-5,
                         verbose=0):

    if sweep_dir in (-1, 'r', 'right'):
      if self.mps.pos != site:
        raise ValueError(
            '_optimize_1s_local for sweep_dir={2}: site={0} != mps.pos={1}'.
            format(site, self.mps.pos, sweep_dir))
    if sweep_dir in (1, 'l', 'left'):
      if self.mps.pos != (site + 1):
        raise ValueError(
            '_optimize_1s_local for sweep_dir={2}: site={0}, mps.pos={1}'.
            format(site, self.mps.pos, sweep_dir))

    if sweep_dir in (-1, 'r', 'right'):
      #NOTE (martin) don't use get_tensor here
      initial = ncon.ncon([self.mps.mat, self.mps[site]],
                          [[-1, 1], [1, -2, -3]])
    elif sweep_dir in (1, 'l', 'left'):
      #NOTE (martin) don't use get_tensor here
      initial = ncon.ncon([self.mps[site], self.mps.mat],
                          [[-1, -2, 1], [1, -3]])

    if self.walltime_log:
      t1 = time.time()
    nit, vecs, alpha, beta = LZ.do_lanczos(
        L=self.left_envs[site],
        mpo=self.mpo[site],
        R=self.right_envs[site],
        initial_state=initial,
        ncv=np.min([
            ncv,
            int(initial.shape[0]) * int(initial.shape[1]) * int(
                initial.shape[2])
        ]),
        delta=landelta)

    if self.walltime_log:
      self.walltime_log(
          lan=[(time.time() - t1) / float(nit)] * int(nit),
          QR=[],
          add_layer=[],
          num_lan=[int(nit)])

    e, opt = LZ.tridiag(vecs, alpha, beta)
    Dnew = opt.shape[2]
    # if verbose == (-1):
    #     print(f"SS-DMRG  site={site}: optimized E={e}")

    if verbose > 0:
      stdout.write(
          "\rSS-DMRG it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i" %
          (self._it, self.Nsweeps, site, len(self.mps), np.real(e), np.imag(e),
           Dnew))
      stdout.flush()

    if verbose > 1:
      print("")

    if self.walltime_log:
      t1 = time.time()
    if sweep_dir in (-1, 'r', 'right'):
      A, mat, Z = misc_mps.prepare_tensor_QR(opt, direction='l')
      A /= Z
    elif sweep_dir in (1, 'l', 'left'):
      mat, B, Z = misc_mps.prepare_tensor_QR(opt, direction='r')
      B /= Z
    if self.walltime_log:
      self.walltime_log(lan=[], QR=[time.time() - t1], add_layer=[], num_lan=[])

    self.mps.mat = mat
    if sweep_dir in (-1, 'r', 'right'):
      self.mps._tensors[site] = A
      self.mps.pos += 1
      self.left_envs[site + 1] = self.add_layer(
          B=self.left_envs[site],
          mps_tensor=self.mps[site],
          mpo_tensor=self.mpo[site],
          conj_mps_tensor=self.mps[site],
          direction=1,
          walltime_log=self.walltime_log)

    elif sweep_dir in (1, 'l', 'left'):
      self.mps._tensors[site] = B
      self.mps.pos = site
      self.right_envs[site - 1] = self.add_layer(
          B=self.right_envs[site],
          mps_tensor=self.mps[site],
          mpo_tensor=self.mpo[site],
          conj_mps_tensor=self.mps[site],
          direction=-1,
          walltime_log=self.walltime_log)
    return e

  def run_one_site(self,
                   Nsweeps=4,
                   precision=1E-6,
                   ncv=40,
                   verbose=0,
                   delta=1E-10,
                   deltaEta=1E-10,
                   walltime_log=None):
    """
        do a one-site DMRG optimzation for an open system
        Paramerters:
        Nsweeps:         int
                         number of left-right  sweeps
        precision:       float    
                         desired precision of the ground state energy
        ncv:             int
                         number of krylov vectors

        verbose:         int
                         verbosity flag
        delta:    float
                  orthogonality threshold; once the next vector of the iteration is orthogonal to the previous ones 
                  within ```delta``` precision, iteration is terminated
        deltaEta: float
                  desired precision of the energies; once eigenvalues of tridiad Hamiltonian are converged within ```deltaEta```
                  iteration is terminated
        walltime_log:  callable or None
                       if not None, walltime_log is passed to do_lanczos, add_layer and prepare_tensor_QR to 
                       log runtimes


        """

    self.walltime_log = walltime_log
    converged = False
    energy = 1E100
    self._it = 1
    self.Nsweeps = Nsweeps
    while not converged:
      self.position(0)
      #the part outside the loop covers the len(self)==1 case
      e = self._optimize_1s_local(
          site=0,
          sweep_dir='right',
          ncv=ncv,
          landelta=delta,
          landeltaEta=deltaEta,
          verbose=verbose)

      for n in range(1, len(self.mps) - 1):
        #_optimize_1site_local shifts the center site internally
        e = self._optimize_1s_local(
            site=n,
            sweep_dir='right',
            ncv=ncv,
            landelta=delta,
            landeltaEta=deltaEta,
            verbose=verbose)
      #prepare for right weep: move center all the way to the right
      self.position(len(self.mps))
      for n in range(len(self.mps) - 1, 0, -1):
        #_optimize_1site_local shifts the center site internally
        e = self._optimize_1s_local(
            site=n,
            sweep_dir='left',
            ncv=ncv,
            landelta=delta,
            landeltaEta=deltaEta,
            verbose=verbose)

      if np.abs(e - energy) < precision:
        converged = True
      energy = e
      self._it += 1
      if self._it > Nsweeps:
        if verbose > 0:
          print()
          print(
              'dmrg did not converge to desired precision {0} after {1} iterations'
              .format(precision, Nsweeps))
        break
    return e


class FiniteDMRGEngine(DMRGUnitCellEngine):

  def __init__(self, mps, mpo, name='FiniteDMRG'):
    # if not isinstance(mps, FiniteMPSCentralGauge):
    #     raise TypeError(
    #         'in FiniteDMRGEngine.__init__(...): mps of type FiniteMPSCentralGauge expected, got {0}'
    #         .format(type(mps)))

    lb = tf.ones([mps.D[0], mps.D[0], mpo.D[0]], dtype=mps.dtype)
    rb = tf.ones([mps.D[-1], mps.D[-1], mpo.D[-1]], dtype=mps.dtype)
    super().__init__(mps=mps, mpo=mpo, lb=lb, rb=rb, name=name)


class InfiniteDMRGEngine(DMRGUnitCellEngine):

  def __init__(self,
               mps,
               mpo,
               name='InfiniteDMRG',
               precision=1E-12,
               precision_canonize=1E-12,
               nmax=1000,
               nmax_canonize=1000,
               ncv=40,
               numeig=1,
               pinv=1E-20,
               power_method=False):

    # if not isinstance(mps, InfiniteMPSCentralGauge):
    #     raise TypeError(
    #         'in InfiniteDMRGEngine.__init__(...): mps of type InfiniteMPSCentralGauge expected, got {0}'
    #         .format(type(mps)))

    mps.restore_form(
        precision=precision_canonize,
        ncv=ncv,
        nmax=nmax_canonize,
        numeig=numeig,
        power_method=power_method,
        pinv=pinv)  #this leaves state in left-orthogonal form

    lb, hl = misc_mps.compute_steady_state_Hamiltonian_GMRES(
        'l',
        mps,
        mpo,
        left_dominant=tf.diag(tf.ones(mps.D[-1], dtype=mps.dtype)),
        right_dominant=ncon.ncon([mps.mat, tf.conj(mps.mat)],
                                 [[-1, 1], [-2, 1]]),
        precision=precision,
        nmax=nmax)

    rmps = mps.get_right_orthogonal_imps(
        precision=precision_canonize,
        ncv=ncv,
        nmax=nmax_canonize,
        numeig=numeig,
        pinv=pinv,
        restore_form=False)

    rb, hr = misc_mps.compute_steady_state_Hamiltonian_GMRES(
        'r',
        rmps,
        mpo,
        right_dominant=tf.diag(tf.ones(mps.D[0], dtype=mps.dtype)),
        left_dominant=ncon.ncon([mps.mat, tf.conj(mps.mat)],
                                [[1, -1], [1, -2]]),
        precision=precision,
        nmax=nmax)

    left_dominant = ncon.ncon([mps.mat, tf.conj(mps.mat)], [[1, -1], [1, -2]])
    out = mps.unitcell_transfer_op('l', left_dominant)

    super().__init__(mps=mps, mpo=mpo, lb=lb, rb=rb, name=name)

  def shift_unitcell(self, sites):
    """
        
        """
    self.position(sites)
    new_lb = self.left_envs[sites]
    new_rb = self.right_envs[sites - 1]
    centermatrix = self.mps.mat
    self.mps.position(len(self.mps))  #move centermatrix to the right
    new_center_matrix = ncon.ncon([self.mps.mat, self.mps.connector],
                                  [[-1, 1], [1, -2]])

    self.mps.pos = sites
    self.mps.mat = centermatrix
    self.mps.position(0)
    new_center_matrix = ncon.ncon([new_center_matrix, self.mps.mat],
                                  [[-1, 1], [1, -2]])
    tensors = [self.mps[n] for n in range(sites, len(self.mps))
              ] + [self.mps[n] for n in range(sites)]
    self.mps._tensors = tensors
    self.mpo._tensors = [self.mpo[n] for n in range(sites, len(self.mps))
                        ] + [self.mpo[n] for n in range(sites)]
    self.mps.connector = tf.linalg.inv(centermatrix)
    self.mps.mat = new_center_matrix
    self.mps.pos = len(self.mps) - sites
    self.lb = new_lb
    self.rb = new_rb
    self.update()

  def run_one_site(self,
                   Nsweeps=1,
                   precision=1E-6,
                   ncv=40,
                   verbose=0,
                   delta=1E-10,
                   deltaEta=1E-10):
    self._idmrg_it = 0
    converged = False
    eold = 0.0
    while not converged:
      e = super().run_one_site(
          Nsweeps=1,
          precision=precision,
          ncv=ncv,
          verbose=verbose - 1,
          delta=delta,
          deltaEta=deltaEta)

      self.shift_unitcell(sites=len(self.mps) // 2)
      if verbose > 0:
        stdout.write(
            "\rSS-IDMRG  it=%i/%i, energy per unit-cell E/N=%.16f+%.16f" %
            (self._idmrg_it, Nsweeps, np.real((e - eold) / len(self.mps)),
             np.imag((e - eold) / len(self.mps))))
        stdout.flush()
        if verbose > 1:
          print('')
      eold = e
      self._idmrg_it += 1
      if self._idmrg_it > Nsweeps:
        converged = True
        break
