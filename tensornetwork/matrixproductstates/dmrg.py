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
from tensornetwork.matrixproductstates.mps import BaseMPS
from tensornetwork.matrixproductstates.mpo import BaseMPO
from typing import Any, Text, Union
Tensor = Any


class BaseDMRG:
  """
  A base class for DMRG (and possibly other) simulations.
  Finite DMRG and infinite DMRG are subclassed from `BaseDMRG`.
  """

  def __init__(self, mps: BaseMPS, mpo: BaseMPO,
               left_boundary: Union[BaseNode, Tensor],
               right_boundary: Union[BaseNode, Tensor], name: Text):
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

    self.mps = mps
    self.mpo = mpo
    if not self.mps.dtype == self.mpo.dtype:
      raise TypeError('mps.dtype={} is different from mpo.dtype={}'.format(
          self.mps.dtype, self.mpo.dtype))
    if len(mps) != len(mpo):
      raise ValueError('len(mps) = {} is different from len(mpo) = {}'.format(
          len(mps), len(mpo)))
    self.left_envs = {0: Node(left_boundary, backend=mps.backend.name)}
    self.right_envs = {
        len(mps) - 1: Node(right_boundary, backend=mps.backend.name)
    }

  def __len__(self):
    """
    return the length of the mps/mpo
    """
    return len(self.mps)

  @property
  def dtype(self):
    """
    return the dtype of BaseMPS
    """
    if not self.mps.dtype == self.mpo.dtype:
      raise TypeError('mps.dtype={} is different from mpo.dtype={}'.format(
          self.mps.dtype, self.mpo.dtype))
    return self.mps.dtype

  @staticmethod
  def add_layer(B: BaseNode, mps_node: BaseNode, mpo_node: BaseNode,
                conj_mps_node: BaseNode, direction: Union[str, int]):
    """
    Adds an mps-mpo-mps layer to a left or DMRG right block `B`.
    Args:
      B: A tensor of shape (Dl,Dl',Ml) (for direction in {1,'l',left'}) or (Dr,Dr',Mr)
        (for direction in {-1,'r','right'})
      mps_node: Node of shape =(Dl,Dr,d).
      mpo_node: Node of shape = (Ml,Mr,d,d').
      conj_mps_node: Node of shape =(Dl',Dr',d').
        The mps tensor on the conjugated side.
        The node will be complex conjugated inside the routine, so do NOT pass  
        a conjugated Node.
      direction: If direction is 1,'l' or 'left': add a layer to the right of `B`
                 If direction is -1,'r' or 'right': add a layer to the left of `B`
    Returns:
      Tensor of shape (Dr,Dr',Mr) for direction in (1,'l','left')
      Tensor of shape (Dl,Dl',Ml) for direction in (-1,'r','right')
    """
    mps_node[1] ^ mpo_node[3]
    conj_mps_node[1] ^ mpo_node[2]

    if direction in (1, 'l', 'left'):
      B[0] ^ mps_node[0]
      B[1] ^ conj_mps_node[0]
      B[2] ^ mpo_node[0]
      order = [mps_node[2], conj_mps_node[2], mpo_node[1]]

    elif direction in (-1, 'r', 'right'):
      B[0] ^ mps_node[2]
      B[1] ^ conj_mps_node[2]
      B[2] ^ mpo_node[1]
      order = [mps_node[0], conj_mps_node[0], mpo_node[0]]

    result = B @ mps_node @ mpo_node @ conj_mps_node
    result.reorder_edges(order)
    return result

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
    if n == self.mps.pos:
      return

    elif n > self.mps.pos:
      pos = self.mps.pos
      self.mps.position(n)
      for m in range(pos, n):
        self.left_envs[m + 1] = self.add_layer(
            self.left_envs[m],
            self.mps[m],
            self.mpo[m],
            self.mps[m],
            direction=1)

    elif n < self.mps.pos:
      pos = self.mps.pos
      self.mps.position(n)
      for m in reversed(range(n, pos)):
        self.right_envs[m - 1] = self.add_layer(
            self.right_envs[m],
            self.mps[m],
            self.mpo[m],
            self.mps[m],
            direction=-1)

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

  def compute_left_envs(self):
    """
    Compute all left environment blocks.
    """
    lb = self.left_envs[0]
    self.left_envs = {0: lb}

    for n in range(self.mps.pos):
      self.left_envs[n + 1] = self.add_layer(
          B=self.left_envs[n],
          mps_tensor=self.mps[n],
          mpo_tensor=self.mpo[n],
          conj_mps_tensor=self.mps[n],
          direction=1)

  def compute_right_envs(self):
    """
    Compute all right environment blocks.
    """
    rb = self.right_envs[len(self.mps) - 1]
    self.right_envs = {len(self.mps) - 1: rb}
    for n in reversed(range(self.mps.pos, len(self.mps))):
      self.right_envs[n - 1] = self.add_layer(
          B=self.right_envs[n],
          mps_tensor=self.mps[n],
          mpo_tensor=self.mpo[n],
          conj_mps_tensor=self.mps[n],
          direction=-1)

  def update(self):
    """
        shift center site of the MPSSimulationBase to 0 and recalculate all left and right blocks
        """
    self.mps.position(0)
    self.compute_left_envs()
    self.compute_right_envs()
    return self

  def _optimize_1s_local(self,
                         site,
                         sweep_dir,
                         ncv=40,
                         Ndiag=10,
                         landelta=1E-5,
                         landeltaEta=1E-5,
                         verbose=0):

    initial = self.mps.nodes[self.center_position]
    def mv(tensor):
      mps=Node(tensor, backend=initial.backend.name)
      L = self.left_envs[site]
      R = self.right_envs[site]
      mpo=self.mpo.nodes[site]
      
    initial.backend.eigsh_lanczos(
      A: Callable,
      initial_state: Optional[Tensor] = None,
      ncv: Optional[int] = 200,
      numeig: Optional[int] = 1,
      tol: Optional[float] = 1E-8,
      delta: Optional[float] = 1E-8,
      ndiag: Optional[int] = 20,
      reorthogonalize: Optional[bool] = False)
    
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


class FiniteDMRG(BaseDMRG):
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
    self.mps.position(0)
    super().__init__(mps=mps, mpo=mpo, name=name, lb=lb, rb=rb)
    self.compute_right_envs()

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
        misc_mps.ncon([mpol, mpor], [[-1, 1, -3, -5], [1, -2, -4, -6]]),
        [Ml, Mr, dl * dr, dlp * drp])
    initial = misc_mps.ncon(
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
            self.mps.bond_dimensions[self.mps.pos - 1], dlp, drp,
            self.mps.bond_dimensions[self.mps.pos + 1]
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

    Z = np.sqrt(misc_mps.ncon([S, S], [[1], [1]]))
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
            '_optimize_1s_local for sweep_dir={2}: site={0} != mps.pos={1}'
            .format(site, self.mps.pos, sweep_dir))
    if sweep_dir in (1, 'l', 'left'):
      if self.mps.pos != (site + 1):
        raise ValueError(
            '_optimize_1s_local for sweep_dir={2}: site={0}, mps.pos={1}'
            .format(site, self.mps.pos, sweep_dir))

    if sweep_dir in (-1, 'r', 'right'):
      #NOTE (martin) don't use get_tensor here
      initial = misc_mps.ncon([self.mps.mat, self.mps[site]],
                              [[-1, 1], [1, -2, -3]])
    elif sweep_dir in (1, 'l', 'left'):
      #NOTE (martin) don't use get_tensor here
      initial = misc_mps.ncon([self.mps[site], self.mps.mat],
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

    lb = tf.ones([
        mps.bond_dimensions[0], mps.bond_dimensions[0], mpo.bond_dimensions[0]
    ],
                 dtype=mps.dtype)
    rb = tf.ones([
        mps.bond_dimensions[-1], mps.bond_dimensions[-1],
        mpo.bond_dimensions[-1]
    ],
                 dtype=mps.dtype)
    super().__init__(mps=mps, mpo=mpo, lb=lb, rb=rb, name=name)
