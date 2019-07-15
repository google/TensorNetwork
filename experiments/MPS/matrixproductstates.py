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
"""implementation of different Matrix Product States types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensornetwork as tn
from experiments.MPS import misc_mps
from experiments.MPS import tensornetwork_tools as tnt
from sys import stdout

def is_mps_tensor(tensor):
  """
    test if `tensor` is of rank 3
    Args: 
        tensor (tf.Tensor)
    Returns:
        tf.Tensor of dtype bool: True if `tensor` is of rank 3, `False` otherwise
    """
  return tf.equal(tf.rank(tensor), 3)


def _float_res(dtype):
  """Get the "resolution" of a floating-point dtype.
    FIXME: This probably doesn't work for bfloat16!
    """
  return np.finfo(dtype.as_numpy_dtype).resolution


def orthonormalization(A, which):
  """
    The deviation from left or right orthonormalization of an MPS tensor
    Args:
        A (tf.Tensor):  mps tensor
        which (str):    can take values in ('l','left','r','right)
                      which  orthonormalization to be checked
    Returns:
        tf.Tensor:     the deviation from left or right orthogonality of `A`
    """
  if which in ('l', 'left', 1):
    eye = tf.eye(tf.cast(tf.shape(A)[2], tf.int32), dtype=A.dtype)
    M = misc_mps.ncon([A, tf.conj(A)], [[1, 2, -1], [1, 2, -2]])
    return tf.norm(M - eye)
  elif which in ('r', 'right', -1):
    eye = tf.eye(tf.cast(tf.shape(A)[0], tf.int32), dtype=A.dtype)
    M = misc_mps.ncon([A, tf.conj(A)], [[-1, 2, 1], [-2, 2, 1]])
    return tf.norm(M - eye)
  else:
    raise ValueError("{} is not a valid direction.".format(which))


def mps_from_dense(psi, max_bond_dim=None, auto_trunc_max_err=0.0):
  """Construct an MPS from a dense vector psi.
    The shape of psi must be the sequence of site Hilbert space dimensions.
    The number of sites is `rank(psi)`.
    Args:
        psi: The dense state as a tensor.
    Returns:
        Gs: Gamma tensors, length rank(psi).
        Ls: Lambda matrices (Schmidt coefficients), length rank(psi).
        nrm: The norm of the original state.
    """
  psi_dims = tf.unstack(tf.shape(psi))  # shape as a list of scalar Tensors
  num_sites = len(psi_dims)

  nrm = tf.norm(psi)
  psi = tf.divide(psi, nrm)

  psiR = tf.reshape(psi, (1, *psi_dims, 1))  # adds boundary bond dimensions
  Gs = []
  Ls = [tf.ones((1, 1), dtype=psi.dtype)]  # left boundary Lambda
  s_prev = tf.ones((1,), dtype=psi.dtype)

  sites_remaining = num_sites + 2  # add the virtual boundary "sites"
  while sites_remaining > 3:
    # prep for SVD by folding the previous singular values back into psiR
    Lm1 = tf.diag(s_prev)
    psiR = misc_mps.ncon([Lm1, psiR],
                         [(-1, 1), (1, *range(-2, -sites_remaining - 1, -1))])

    U, s, psiR, s_rest = tnt.svd_tensor(
        psiR, [0, 1],
        list(range(2, sites_remaining)),
        nsv_max=max_bond_dim,
        auto_trunc_max_err=auto_trunc_max_err)

    # turn U into a Gamma using the previous inverse singular values
    Lm1_i = tf.diag(tf.reciprocal(s_prev))
    G = misc_mps.ncon([Lm1_i, U], [(-1, 1), (1, -2, -3)])

    Gs.append(G)
    Ls.append(tf.diag(s))
    s_prev = s
    sites_remaining -= 1

  Gs.append(psiR)

  return Gs, Ls, nrm


class AbstractMPSUnitCell:
  """Defines a common interface to Matrix Product State (MPS) objects.
    This includes some methods, relying only on this interface, for extracting
    some basic information from an MPS (correlation functions, entanglement
    entropy, etc.).

    Any implemented methods in this class should be efficient in the sense that
    they are local: They never act across the entire system and depend on 
    `get_env_...()` methods to get the appropriate environment tensors.
    """

  def __init__(self, name=None):
    self.name = name

  @classmethod
  def from_tensors(cls, tensors, name=None):
    """Construct an MPS from a list of MPS tensors.
        The tensors need not be in a canonical form."""
    raise NotImplementedError()

  @classmethod
  def from_mps(cls, mps, name=None):
    """Construct an MPS from an existing MPS object."""
    return cls.from_tensors([A for A in mps.tensors_itr], name=name)

  @classmethod
  def random(cls,
             d,
             D,
             name=None,
             dtype=tf.float32,
             initializer_function=tf.random_uniform,
             *args,
             **kwargs):
    """
        Creates a random MPS. Tensors are initialized using 
        initializer_function.

        Args:
            d: list of int
                the physical Hilbert space dimension on each site
            D: list of int of len(d) + 1
                the bond dimensions of the MPS
            name: str or None
                name of the MPS
            dtype: tensorflow dtype object
                the datatype of the MPS
            initializer_functions: callable
                initialization function;
            *args,**kwargs: further arguments passed to initializer_function

        Returns:
            mps: An initialized MPS object.
        """
    if not len(d) == (len(D) - 1):
      raise ValueError('.random(): len(d)! = len(D)-1!')

    kwargs['minval'] = kwargs.get('minval', -0.1)
    kwargs['maxval'] = kwargs.get('maxval', 0.1)

    tensors = misc_mps.initialize_mps_tensors(initializer_function, D, d, dtype,
                                              *args, **kwargs)
    return cls.from_tensors(tensors, name=name)

  @property
  def dtype(self):
    raise NotImplementedError()

  def get_tensor(self, n):
    """
    MPS tensor for site n, compatible with get_env_...().
    An MPS tensor always has 3 dimensions, although some may have size 1.
    Args:
        n (int): site
    Returns:
        tf.Tensor
    """
    raise NotImplementedError()

  def set_tensor(self, n, tensor):
    """
    Sets MPS tensor for site n (need not be implemented)
    Args:
        n (int): site
        tensor (tf.Tensor):  mps-tensor
    """
    raise NotImplementedError()

  @property
  def num_sites(self):
    """
    Number of sites.
    For an Infinite MPS, this is the number of sites in the unit cell.
    """
    raise NotImplementedError()

  def __len__(self):
    return self.num_sites

  @property
  def tensors_itr(self):
    """Iterates over the MPS tensors."""
    return (self.get_tensor(n) for n in range(self.num_sites))

  @property
  def D(self):
    """Returns a vector of bond dimensions, length `num_sites+1`."""
    raise NotImplementedError()

  @property
  def d(self):
    """Returns the vector of physical dimensions, length `num_sites`."""
    raise NotImplementedError()

  def get_env_left(self, n):
    """Left environment of site n, compatible with __get_item__()"""
    raise NotImplementedError()

  def get_env_right(self, n):
    """Right environment of site n, compatible with __get_item__()"""
    raise NotImplementedError()

  def get_envs_right(self, sites):
    """Right environments for a selection of sites.
        Returns the environments as a dictionary, indexed by site number.

        Args:
            sites: list of site numbers for which the environments should be 
                   calculated.
        Returns:
            rs: dictionary mapping site numbers (int) to right environments (tf.Tensor)
        """
    # NOTE: This default implementation is not necessarily optimal for all
    #       descendant classes.
    if not np.all(np.array(sites) >= 0):
      raise ValueError('get_envs_right: sites have to be >= 0')
    n2 = max(sites)
    n1 = min(sites)
    rs = {n2: self.get_env_right(n2)}
    r = rs[n2]
    for n in reversed(range(n1, n2)):
      r = self.transfer_op(n + 1, 'r', r)
      if n in sites:
        rs[n] = r
    return rs

  def get_envs_left(self, sites):
    """Left environments for a selection of sites.
        Returns the environments as a dictionary, indexed by site number.

        Args:
            sites: list of site numbers for which the environments should be 
                   calculated.
        Returns:
            ls:   dictionary mapping site numbers (int) to left environments (tf.Tensor)
        """
    # NOTE: This default implementation is not necessarily optimal for all
    #       descendant classes.
    if not np.all(np.array(sites) >= 0):
      raise ValueError('get_envs_left: sites have to be >= 0')
    n2 = max(sites)
    n1 = min(sites)
    ls = {n1: self.get_env_left(n1)}
    l = ls[n1]
    for n in range(n1 + 1, n2 + 1):
      l = self.transfer_op(n - 1, 'l', l)
      if n in sites:
        ls[n] = l
    return ls

  def transfer_op(self, site, direction, x):
    """Applies the 1-site transfer operator, acting left or right, to x.
        x is assumed to have the form of a left or right environment, for
        direction `left` or `right`, respectively.
        Args:
            site (int):    the site at which the transfer operator should be applied
            x (tf.Tensor of shape (D, D)): the environment to be transfered
        """
    A = self.get_tensor(site)
    return misc_mps.transfer_op([A], [A], direction=direction, x=x)

  def norm(self):
    raise NotImplementedError()

  def normalize(self):
    """Normalizes the state and returns the previous norm."""
    raise NotImplementedError()

  def _check_env(self, tol=None):
    """Test environments returned by get_env_...().
        They must be compatible with the transfer operator.
        NOTE: This only works in eager mode.
        """
    if tol is None:
      tol = _float_res(self.dtype)
    l = self.get_env_left(0)
    for n in range(self.num_sites - 1):
      l = self.transfer_op(n, 'l', l)
      l_check = self.get_env_left(n + 1)
      diff = tf.norm(l - l_check).numpy()
      if diff > tol:
        print("l:", n, diff)

    r = self.get_env_right(self.num_sites - 1)
    for n in reversed(range(1, self.num_sites)):
      r = self.transfer_op(n, 'r', r)
      r_check = self.get_env_right(n - 1)
      diff = tf.norm(r - r_check).numpy()
      if diff > tol:
        print("r:", n, diff)

  def _check_envs(self, tol=None):
    """Test environments returned by get_envs_...().
        They must be compatible with the transfer operator.
        NOTE: This only works in eager mode.
        """
    if tol is None:
      tol = _float_res(self.dtype)
    l = self.get_env_left(0)
    ls_check = self.get_envs_left(list(range(self.num_sites)))
    for n in range(self.num_sites - 1):
      l = self.transfer_op(n, 'l', l)
      l_check = ls_check[n + 1]
      diff = tf.norm(l - l_check).numpy()
      if diff > tol:
        print("l:", n, diff)

    r = self.get_env_right(self.num_sites - 1)
    rs_check = self.get_envs_right(list(range(self.num_sites)))
    for n in reversed(range(1, self.num_sites)):
      r = self.transfer_op(n, 'r', r)
      r_check = rs_check[n - 1]
      diff = tf.norm(r - r_check).numpy()
      if diff > tol:
        print("r:", n, diff)

  def expval_1site(self, op, n):
    """Expectation value of a single-site operator on site n.
        NOTE: Assumes the state is normalized!
        Args: 
            opt (tf.Tensor):   the operator for which the expectation value should be calculated
            n (int):           the site at which the expectation value should be calculated
        Returns:
            tf.Tensor:         the expectation value
        """
    return self.expvals_1site([op], [n])[0]

  def expvals_1site(self, ops, sites):
    """
        Expectation value of list of single-site operators at `sites`.
        NOTE: Assumes the state is normalized!

        Args:
            ops (list of tf.Tensor):  local operators to be measure
            sites (list of int):      sites where the operators live
                                      `sites` can be in any order and have any number of sites appear arbitrarily often
        Returns:
             list of tf.Tensor:       a list of measurements, in the same order as `sites` were passed
        """
    sites = [s % self.num_sites for s in sites]
    if not len(ops) == len(sites):
      raise ValueError('measure_1site_ops: len(ops) has to be len(sites)!')
    right_envs = self.get_envs_right(sites)
    left_envs = self.get_envs_left(sites)
    res = []
    for n in range(len(sites)):
      op = ops[n]
      r = right_envs[sites[n]]
      l = left_envs[sites[n]]
      A = self.get_tensor(sites[n])
      expval = misc_mps.ncon([l, A, op, r, tf.conj(A)], [(2, 4), (2, 1, 3),
                                                         (5, 1), (3, 6),
                                                         (4, 5, 6)])
      res.append(expval)

    return tf.convert_to_tensor(res)

  def schmidt_spec_cut(self, n):
    """
        Schmidt spectrum for the cut between sites n and n+1.
        Args:
            n (int):   the position of the cut
        Returns:
            tf.Tensor of shape (self.D[n]):  the Schmidt-values of the cut across `n`
        """

    l = self.get_env_left(n + 1)
    r = self.get_env_right(n)
    lr = tf.transpose(l) @ r
    schmidt_sq = tf.svd(lr, compute_uv=False)
    # In case the state is not normalized, ensure schmidt_sq sums to 1
    schmidt_sq = tf.divide(schmidt_sq, tf.reduce_sum(schmidt_sq))
    return tf.sqrt(schmidt_sq)

  def correlator_1site(self, op1, op2, site1, sites2):
    """
        Compute expectation values <op1,op2> for all pairs (site1, n2) with n2 from `sites2`
        if site1 == n2, op2 will be applied first
        Args:
            op1, op2 (tf.Tensor):  local operators to be measured
            site1 (int):           the sites of op1
            sites2 (list of int):  the sites of op2
        Returns:
            c (tf.Tensor):   the measurements of the same order as `sites`
                             i.e.  c[n] = <"op1(site1)" "op2(sites2[n])"> (abusing notation, op1 and op2 are not callable)
        """

    N = self.num_sites
    if site1 < 0:
      raise ValueError(
          "Site site1 out of range: {} not between 0 and {}.".format(site1, N))
    sites2 = np.array(sites2)

    c = []

    left_sites = sorted(sites2[sites2 < site1])
    rs = self.get_envs_right([site1])
    if len(left_sites) > 0:
      left_sites_mod = list(set([n % N for n in left_sites]))

      ls = self.get_envs_left(left_sites_mod)

      A = self.get_tensor(site1)
      r = misc_mps.ncon([A, tf.conj(A), op1, rs[site1]], [(-1, 2, 1), (-2, 3, 4),
                                                    (3, 2), (1, 4)])

      n1 = np.min(left_sites)
      for n in range(site1 - 1, n1 - 1, -1):
        if n in left_sites:
          l = ls[n % N]
          A = self.get_tensor(n % N)
          res = misc_mps.ncon([l, A, op2, tf.conj(A), r],
                        [[1, 4], [1, 2, 5], [3, 2], [4, 3, 6], [5, 6]])
          c.append(res)
        if n > n1:
          r = self.transfer_op(n % N, 'right', r)

      c = list(reversed(c))

    ls = self.get_envs_left([site1])

    if site1 in sites2:
      A = self.get_tensor(site1)
      op = misc_mps.ncon([op2, op1], [[-1, 1], [1, -2]])
      res = misc_mps.ncon([ls[site1], A, op, tf.conj(A), rs[site1]],
                    [[1, 4], [1, 2, 5], [3, 2], [4, 3, 6], [5, 6]])
      c.append(res)

    right_sites = sites2[sites2 > site1]
    if len(right_sites) > 0:
      right_sites_mod = list(set([n % N for n in right_sites]))

      rs = self.get_envs_right(right_sites_mod)

      A = self.get_tensor(site1)
      l = misc_mps.ncon([ls[site1], A, op1, tf.conj(A)], [(1, 2), (1, 3, -1), (4, 3),
                                                    (2, 4, -2)])

      n2 = np.max(right_sites)
      for n in range(site1 + 1, n2 + 1):
        if n in right_sites:
          r = rs[n % N]
          A = self.get_tensor(n % N)
          res = misc_mps.ncon([l, A, op2, tf.conj(A), r],
                        [[1, 4], [1, 2, 5], [3, 2], [4, 3, 6], [5, 6]])
          c.append(res)

        if n < n2:
          l = self.transfer_op(n % N, 'left', l)

    return tf.convert_to_tensor(c)

  def apply_2site(self, op, n):
    """Applies a nearest-neighbor operator to sites n and n+1 (in place)."""
    raise NotImplementedError()


class AbstractFiniteMPS(AbstractMPSUnitCell):

  def __init__(self, name=None):
    super().__init__(name=name)
    if tf.executing_eagerly():
      D = self.D
      if not tf.equal(D[0], 1) or not tf.equal(D[self.num_sites], 1):
        raise ValueError("Boundary bond dimensions must both equal 1.")

  @classmethod
  def from_dense(cls, psi, max_bond_dim=None, auto_trunc_max_err=0.0,
                 name=None):
    """Construct an MPS from a dense vector psi.
        The shape of psi must be the sequence of site Hilbert space dimensions.
        Args:
          psi: The dense state as a tensor (treated as a vector).
        """
    Gams, Lams, norm = mps_from_dense(
        psi, max_bond_dim=max_bond_dim, auto_trunc_max_err=auto_trunc_max_err)
    mps = FiniteMPS_Schmidt(Gams, Lams, norm=norm, name=name)
    return cls.from_mps(mps)

  @classmethod
  def from_product(cls, site_states, name=None):
    """Construct an MPS from a product of dense site states.
        Each site state must be a vector. The length of the vector is the
        Hilbert space dimension of the site.
        Args:
          site_states: A list of vectors, each representing a wavefunction for
            a particular site.
        """
    As = [tf.reshape(psi, (1, tf.size(psi), 1)) for psi in site_states]
    m = FiniteMPS_Generic(As)
    return cls.from_mps(m, name=name)

  @classmethod
  def random(cls,
             d,
             D,
             name=None,
             dtype=tf.float32,
             initializer_function=tf.random_uniform,
             *args,
             **kwargs):
    """
        Creates a random finite MPS. Tensors are initialized using 
        initializer_function.

        Args:
            d: list of int
                the physical Hilbert space dimension on each site
            D: list of int of len(d) - 1
                the bond dimensions of the MPS
            name: str or None
                name of the MPS
            dtype: tensorflow dtype object
                the datatype of the MPS
            initializer_functions: callable
                initialization function;
            *args,**kwargs: further arguments passed to initializer_function

        Returns:
            mps: An initialized MPS object.
        """
    if not len(d) == (len(D) + 1):
      raise ValueError('.random(): len(d)! = len(D)+1!')
    return super().random(
        d, [1] + D + [1],
        name=name,
        dtype=dtype,
        initializer_function=initializer_function,
        *args,
        **kwargs)

  def to_dense(self):
    """Convert to a dense vector with shape `self.d`.
        """
    N = self.num_sites

    if N == 0:
      return tf.zeros((0,), dtype=self.dtype)

    D = self.D

    psi = self.get_tensor(0)
    physdim = tf.shape(psi)[1]
    for j in range(1, N):
      A = self.get_tensor(j)
      physdim *= tf.shape(A)[1]
      psi = misc_mps.ncon([psi, A], [('l', 'p1', 'ri'), ('ri', 'p2', 'r')],
                          out_order=['l', 'p1', 'p2', 'r'],
                          con_order=['ri'])
      psi = tf.reshape(psi, (D[0], physdim, D[j + 1]))

    psi = misc_mps.ncon([psi], [(1, -1, 1)])
    psi = tf.reshape(psi, self.d)

    return psi

  def overlap(self, other):
    """Computes the overlap between two finite MPS."""
    if self.num_sites != other.num_sites:
      raise ValueError("MPS have different numbers of sites")

    if tf.executing_eagerly():
      d = self.d
      d_other = other.d
      if not np.all(tf.equal(d[m], d_other[m]) for m in range(self.num_sites)):
        raise ValueError("MPS have different physical dimensions")

    l = tf.ones(shape=(1, 1), dtype=self.dtype)
    for n in range(self.num_sites):
      l = misc_mps.transfer_op([self.get_tensor(n)], [other.get_tensor(n)],
                               direction='l',
                               x=l)
    return l


class AbstractInfiniteMPS(AbstractMPSUnitCell):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_unitcell_transfer_op(self, direction):
    """
        Returns a function that implements the transfer operator for the
        entire unit cell.
        """
    if direction not in ('l', 'left', 1, 'r', 'right', -1):
      raise ValueError("Invalid direction: {}".format(direction))
    As = [A for A in self.tensors_itr]

    def t_op(x):
      return misc_mps.transfer_op(As, As, direction, x)

    return t_op

  def unitcell_transfer_op(self, direction, x):
    """
        Compute action of the unit-cell transfer operator on ```x```
        Args:
            direction (int or str):    if direction in (1,'l','left'): left-multiply x
                                       if direction in (-1,'r','right'): right-multiply x
            x (tf.Tensor):             tensor of shape (mps.D[0],mps.D[0]) or  (mps.D[-1],mps.D[-1])
                                       the left/right environment that should be transfered across the mps
        Returns:
            tf.Tensor of same shape as `x`
        """
    if tf.executing_eagerly():
      if direction in ('l', 'left', 1):
        if not tf.equal(tf.shape(x)[0], self.D[0]):
          raise ValueError('shape of x[0] does not match the shape of mps.D[0]')
        if not tf.equal(tf.shape(x)[1], self.D[0]):
          raise ValueError('shape of x[1] does not match the shape of mps.D[0]')

      if direction in ('r', 'right', -1):
        if not tf.equal(tf.shape(x)[0], self.D[-1]):
          raise ValueError(
              'shape of x[0] does not match the shape of mps.D[-1]')
        if not tf.equal(tf.shape(x)[1], self.D[-1]):
          raise ValueError(
              'shape of x[1] does not match the shape of mps.D[-1]')

    t_op = self.get_unitcell_transfer_op(direction)
    return t_op(x)

  def TMeigs_power_method(self,
                          direction,
                          init=None,
                          precision=1E-12,
                          nmax=100000):
    tensors = [self.get_tensor(n) for n in range(len(self))]
    return misc_mps.TMeigs_power_method(
        tensors=tensors,
        direction=direction,
        init=init,
        precision=precision,
        nmax=nmax)

  def TMeigs(self,
             direction,
             init=None,
             precision=1E-12,
             ncv=50,
             nmax=1000,
             numeig=1,
             which='LR'):
    """
        calculate the left or right dominant eigenvector of the MPS-unit-cell transfer operator using sparse 
        method.
        Notes: - Currently  only works in eager mode.
               - the implementation uses scipy's sparse module (eigs). tf.Tensor are mapped to numpy arrays and back
                 to tf.Tensor for each call to matrix-vector product. This is not optimal and will be fixed at some alter stage
        Args:
            direction (int or str):     if direction in (1,'l','left')   return the left dominant EV
                                        if direction in (-1,'r','right') return the right dominant EV
            init (tf.Tensor):           initial guess for the eigenvector
            precision (float):          desired precision of the dominant eigenvalue
            ncv(int):                   number of Krylov vectors
            nmax (int):                 max number of iterations
            numeig (int):               hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                                        to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
            which (str):                hyperparameter, passed to scipy.sparse.linalg.eigs; which eigen-vector to target
                                        can be ('LM','LA,'SA','LR'), refer to scipy.sparse.linalg.eigs documentation for details
        Returns:
            eta (tf.Tensor):        the eigenvalue
            x (tf.Tensor):          the dominant eigenvector (in matrix form)
        """
    #FIXME: add graph-mode execution
    tensors = [self.get_tensor(n) for n in range(len(self))]
    return misc_mps.TMeigs(
        tensors=tensors,
        direction=direction,
        init=init,
        precision=precision,
        ncv=ncv,
        nmax=nmax,
        numeig=numeig,
        which=which)


class MPSUnitCell_Generic(AbstractMPSUnitCell):

  def __init__(self, tensors, name=None):
    self.tensors = tensors
    super().__init__(name=name)

  @classmethod
  def random(cls,
             d=list,
             D=list,
             name=None,
             dtype=tf.float32,
             initializer_function=tf.random_uniform,
             *args,
             **kwargs):
    """
        Creates a random finite MPS. Tensors are initialized using 
        initializer_function.

        Args:
            d: list of int
                the physical Hilbert space dimension on each site
            D: list of int of len(d) - 1
                the bond dimensions of the MPS
            name: str or None
                name of the MPS
            dtype: tensorflow dtype object
                the datatype of the MPS
            initializer_functions: callable
                initialization function;
            *args,**kwargs: further arguments passed to initializer_function

        Returns:
            mps: An initialized MPS object.
        """
    if not len(d) == (len(D) - 1):
      raise ValueError('MPSUnitCell_Generic.random: len(d)! = len(D)-1!')

    kwargs['minval'] = kwargs.get('minval', -0.1)
    kwargs['maxval'] = kwargs.get('maxval', 0.1)
    return cls(
        tensors=misc_mps.initialize_mps_tensors(initializer_function, D, d,
                                                dtype, *args, **kwargs),
        name=name)

  @classmethod
  def from_tensors(cls, tensors, name=None):
    return cls(tensors, name=name)

  @property
  def num_sites(self):
    return len(self.tensors)

  def get_tensor(self, n):
    return self.tensors[n]

  def set_tensor(self, n, tensor):
    self.tensors[n] = tensor

  @property
  def D(self):
    return [tf.shape(self.tensors[0])[0]] + \
        [tf.shape(t)[2] for t in self.tensors]

  @property
  def d(self):
    return [tf.shape(t)[1] for t in self.tensors]

  @property
  def dtype(self):
    dtype = self.tensors[0].dtype
    assert (np.all([dtype == t.dtype for t in self.tensors]))
    return dtype


class FiniteMPS_Generic(MPSUnitCell_Generic, AbstractFiniteMPS):

  def __init__(self, tensors, name=None):
    super().__init__(tensors, name=name)

  @classmethod
  def random(cls,
             d=list,
             D=list,
             name=None,
             dtype=tf.float32,
             initializer_function=tf.random_uniform,
             *args,
             **kwargs):
    """
        Creates a random finite MPS. Tensors are initialized using 
        initializer_function.

        Args:
            d: list of int
                the physical Hilbert space dimension on each site
            D: list of int of len(d) - 1
                the bond dimensions of the MPS
            name: str or None
                name of the MPS
            dtype: tensorflow dtype object
                the datatype of the MPS
            initializer_functions: callable
                initialization function;
            *args,**kwargs: further arguments passed to initializer_function

        Returns:
            mps: An initialized MPS object.
        """

    if not len(d) == (len(D) + 1):
      raise ValueError('FiniteMPS_Generic.random: len(d)! = len(D)+1!')
    D = [1] + D + [1]

    kwargs['minval'] = kwargs.get('minval', -0.1)
    kwargs['maxval'] = kwargs.get('maxval', 0.1)
    return cls(
        tensors=misc_mps.initialize_mps_tensors(initializer_function, D, d,
                                                dtype, *args, **kwargs),
        name=name)

  def norm(self):
    """
        return the norm of the centermatrix
        """
    r = self.get_env_right(-1)
    return tf.sqrt(r[0, 0])

  def normalize(self):
    """
        normalize the centermatrix
        """
    nrm = self.norm()
    # FIXME: Keep norms of all tensors similar.
    self.tensors[0] = tf.divide(self.tensors[0], nrm)
    return nrm

  def get_env_left(self, n):
    """
        get left environment of site `n`
        """
    l = tf.ones(shape=(1, 1), dtype=self.dtype)
    for n in range(n):
      l = self.transfer_op(n, 'l', l)
    return l

  def get_env_right(self, n):
    """
        get right environment of site `n`
        """
    r = tf.ones(shape=(1, 1), dtype=self.dtype)
    for n in reversed(range(n + 1, self.num_sites)):
      r = self.transfer_op(n, 'r', r)
    return r


class InfiniteMPS_Generic(MPSUnitCell_Generic, AbstractInfiniteMPS):

  def __init__(self, tensors, name=None):
    super().__init__(tensors, name=name)


class MPSUnitCell_Schmidt(AbstractMPSUnitCell):

  def __init__(self, Gams, Lams, name=None):
    self.Gams = Gams
    if len(Lams) != len(Gams):
      raise ValueError("len(Lams) != len(Gams)")
    self.Lams = Lams
    super().__init__(name)
    self.check_dims()

  def get_tensor(self, n):
    # Return tensors in left canonical form
    Gam = self.Gams[n]
    Lam = self.Lams[n]
    A = misc_mps.ncon([Lam, Gam], [(-1, 1), (1, -2, -3)])
    return A

  def get_env_left(self, n):
    # Return the env. for left canonical form
    l = tf.eye(tf.cast(self.D[n], tf.int32), dtype=self.dtype)
    return l

  def get_env_right(self, n):
    # Return the env. for left canonical form
    if n == self.num_sites - 1:
      Lam = self.Lams[0]
    else:
      Lam = self.Lams[n + 1]
    return Lam @ Lam

  def get_envs_left(self, sites):
    # Efficient: Each of these is a local operation.
    return {n: self.get_env_left(n) for n in sites}

  def get_envs_right(self, sites):
    # Efficient: Each of these is a local operation.
    return {n: self.get_env_right(n) for n in sites}

  def check_dims(self):
    """Check that tensor dimensions are compatible.
        Note: This will only work in eager mode.
        """
    Gams = self.Gams
    Lams = self.Lams
    D0s = [G.shape[0] for G in Gams]
    D1s = [G.shape[2] for G in Gams]
    for n in range(self.num_sites):
      if D0s[n] != D1s[n - 1]:
        raise ValueError("Gammas have inconsistent bond dimensions.")
      if D0s[n] != Lams[n].shape[1]:
        raise ValueError(
            "Lambdas and Gammas have inconsistent bond dimensions.")
      if Lams[n].shape[0] != Lams[n].shape[1]:
        raise ValueError("Lambda {} is not square!".format(n))

  def check_form(self):
    """Check that the tensors fulfill the properties of the canonical form.
        Note: This will only work in eager mode.
        """
    N = self.num_sites
    Gams = self.Gams
    Lams = self.Lams
    res = _float_res(self.dtype)

    trLams = [tf.trace(Lam).numpy() for Lam in Lams]
    for (n, trLam) in enumerate(trLams):
      if not np.abs(1 - trLam) > res:
        print("trace(Lams[{}]) == {}".format(n, trLam))

    for n in range(N):
      AL = misc_mps.ncon([Lams[n], Gams[n]], [(-1, 1), (1, -2, -3)])
      orthcheck = orthonormalization(AL, 'l').numpy()
      if orthcheck > res:
        print("left ortho failed at site {}: {}".format(n, orthcheck))

      AR = misc_mps.ncon([Gams[n], Lams[(n + 1) % N]], [(-1, -2, 1), (1, -3)])
      orthcheck = orthonormalization(AR, 'r').numpy()
      if orthcheck > res:
        print("right ortho failed at site {}: {}".format(n, orthcheck))

  @property
  def num_sites(self):
    return len(self.Gams)

  def set_tensor(self, n, tensor):
    raise NotImplementedError()

  @property
  def dtype(self):
    dtype = self.Gams[0].dtype
    assert (np.all([dtype == t.dtype for t in self.Gams]))
    assert (np.all([dtype == t.dtype for t in self.Lams]))
    return dtype

  @property
  def D(self):
    # Override for efficiency
    Gams = self.Gams
    return [tf.shape(Gams[0])[0]] + \
        [tf.shape(Gams[n])[2] for n in range(self.num_sites)]

  @property
  def d(self):
    # Override for efficiency
    Gams = self.Gams
    return [tf.shape(Gams[n])[1] for n in range(self.num_sites)]

  def schmidt_spec_cut(self, n):
    """Schmidt spectrum for the cut between sites n and n+1."""
    return tf.diag_part(self.Lams[(n + 1) % self.num_sites])


class FiniteMPS_Schmidt(MPSUnitCell_Schmidt, AbstractFiniteMPS):

  def __init__(self, Gams, Lams, norm=1.0, name=None):
    self._norm = norm
    super().__init__(Gams, Lams, name=name)

  def get_tensor(self, n):
    # Override to account for the norm factor
    A = super().get_tensor(n)
    if n == 0:
      A = tf.multiply(A, self._norm)
    return A

  def get_env_left(self, n):
    # Override to account for the norm factor
    l = super().get_env_left(n)
    if n > 0:
      l = tf.multiply(l, self._norm**2)
    return l

  def norm(self):
    return self._norm

  def normalize(self):
    nrm = self._norm
    self._norm = 1.0
    return nrm

  @classmethod
  def from_dense(cls, psi, name=None):
    Gams, Lams, norm = mps_from_dense(psi)
    return cls(Gams, Lams, norm=norm, name=name)

  def apply_2site(self, op, n, max_bond_dim=None, auto_trunc_max_err=0.0):
    """Applies a 2-site gate on sites n and n+1.
        NOTE: Only for a unitary gate, and in the absence of truncation.
              will the canonical form be preserved. In other cases, 
              .check_form() can be used to view the damage!
        """
    G = self.Gams
    L = self.Lams

    res = misc_mps.apply_2site_schmidt_canonical(
        op,
        L[n],
        G[n],
        L[n + 1],
        G[n + 1],
        L[(n + 2) % self.num_sites],
        max_bond_dim=max_bond_dim,
        auto_trunc_max_err=auto_trunc_max_err)
    G[n], L[n + 1], G[n + 1], normfac, trunc_err = res
    self._norm *= normfac

    if tf.executing_eagerly():
      # Check if we broke the canonical form!
      A = self.get_tensor(n + 1)
      form_check = orthonormalization(A, 'l')
      res = _float_res(self.dtype)
      # FIXME: This will not work outside of eager mode
      if form_check > 10 * res:
        # FIXME: restore_form() here based on threshold? Alternatively,
        #        return the violation and let the caller decide.
        print("Warning: Now violating canonical form by at least {}".format(
            form_check))
    # FIXME: How to deal with form-breaking in non-eager?

    return trunc_err

  def restore_form(self):
    raise NotImplementedError()
    # TODO: Simplest implementation - hook up to CentralGauge...
    #       But it's in another file...


class InfiniteMPS_Schmidt(MPSUnitCell_Schmidt, AbstractInfiniteMPS):

  def __init__(self, Gams, Lams, name=None):
    super().__init__(Gams, Lams, name=name)

  def restore_form(self):
    # TODO: This needs to get the transfer matrix dom. eigenvectors and get
    #       into, say, left canonical form, before using inverted Schmidt
    #       coefficients to arrive at the Schmidt form.
    raise NotImplementedError()


class MPSUnitCellCentralGauge(AbstractMPSUnitCell):

  @classmethod
  def from_tensors(cls, tensors, name=None):
    """
        initializes an MPSUnitCellCentralGauge from a list of tensors.
        mps.pos is set to len(tensors)
        connector, centralmatrix and right_mat are all set to identity matrices
        Args:
            tensors (list of tf.Tensor): list of mps tensors
        Returns:
            MPSUnitCellCentralGauge
        """
    dtype = tensors[0].dtype
    D_end = tensors[-1].shape[2]
    centermatrix = tf.eye(int(D_end), dtype=dtype)
    connector = tf.eye(int(D_end), dtype=dtype)
    right_mat = tf.eye(int(D_end), dtype=dtype)
    position = len(tensors)
    mps = cls(
        tensors=tensors,
        centermatrix=centermatrix,
        connector=connector,
        right_mat=right_mat,
        position=position,
        name=name)
    mps.position(0)  # obtain canonical form
    return mps

  def __init__(self,
               tensors,
               centermatrix,
               connector,
               right_mat,
               position,
               name=None):
    """
        initializes an MPSUnitCellCentralGauge .
        Args:
            tensors (list of tf.Tensor): list of mps tensors
            centermatrix (tf.Tensor):    initial center matrix 
            connector (tf.Tensor):       connector matrix
            right_mat (tf.Tensor):       right matrix used for efficient calculation of observables
            position (int):              initial position of the center-matrix
            name (str):                  a name for the MPS
        Returns:
            MPSUnitCellCentralGauge
        """

    self._tensors = tensors
    self.pos = position
    self.mat = centermatrix
    self.connector = connector
    self._right_mat = right_mat
    super().__init__(name=name)

  @property
  def D(self):
    """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N  ==  num_sites`."""
    ts = self._tensors
    return (
        [tf.shape(ts[0])[0]] + [tf.shape(ts[n])[2] for n in range(len(self))])

  @property
  def d(self):
    """Returns a vector of all physical dimensions."""
    return [tf.shape(t)[1] for t in self._tensors]

  @property
  def num_sites(self):
    return len(self._tensors)

  def __getitem__(self, n):
    """
        Martin's API for getting center gauge site tensors.
        NOTE: Does not return the same as .get_tensor(n).
        """
    return self._tensors[n]

  def __setitem__(self, n, tensor):
    """
        Martin's API for setting center gauge site tensors.
        NOTE: Does not do the same as .set_tensor(n).
        """
    self._tensors[n] = tensor

  def __iter__(self):
    """
        Martin's API for getting center gauge site tensors.
        NOTE: Does not return the same as .tensors_itr
        """
    return iter(self._tensors)

  def get_tensor(self, n):
    """
        `get_tensor(n)` returns an mps tensors, possibly contracted with the center matrix and or the connector
        By convention, the center matrix is contracted if n == self.pos
        The connector is always absorbed at the right end of the mps.
        """
    N = self.num_sites

    if n < 0:
      raise ValueError("n = {} is less than zero.".format(n))
    elif n >= N:
      raise ValueError("n = {} is >=  num_sites.".format(n))

    if self.pos < N:
      # absorb the centermatrix if it is to the left of the tensor at site n
      if n == self.pos:
        out = misc_mps.ncon([self.centermatrix, self._tensors[n]],
                            [[-1, 1], [1, -2, -3]])
      else:
        out = self._tensors[n]
    elif self.pos == N:
      # exception: if the centermatrix is at the end of the cell (next
      #            to the connector), absorb it into the rightmost tensor.
      if n == (N - 1):
        out = misc_mps.ncon([self._tensors[n], self.centermatrix],
                            [[-1, -2, 1], [1, -3]])
      else:
        out = self._tensors[n]
    else:
      raise ValueError("Unexpected value of .pos = {}.".format(self.pos))

    if n == (N - 1):
      # Absorb the connector into the rightmost tensor.
      # This makes the complete unit cell left canonical.
      out = misc_mps.ncon([out, self.connector], [[-1, -2, 1], [1, -3]])

    return out

  def set_tensor(self, n, tensor):
    raise NotImplementedError()

  @property
  def dtype(self):
    assert (np.all([self._tensors[0].dtype == t.dtype for t in self._tensors]))
    return self._tensors[0].dtype

  # TODO: Provide optimal get_envs_left and get_envs_right.
  def get_env_left(self, site):
    """
        compute the left environment of site `site`
        """
    if site >= len(self) or site < 0:
      raise IndexError(
          'index {0} out of bounds for MPSUnitCellCentralGauge of length {1}'.
          format(site, len(self)))

    if site <= self.pos:
      return tf.eye(int(self.D[site]), dtype=self.dtype)
    else:
      l = tf.eye(int(self.D[self.pos]), dtype=self.dtype)
      for n in range(self.pos, site):
        l = self.transfer_op(n, direction='l', x=l)
      return l

  def get_env_right(self, site):
    """
        compute the right environment of site `site`
        """

    site = site % len(self)
    if site >= len(self) or site < 0:
      raise IndexError(
          'index {0} out of bounds for MPSUnitCellCentralGauge of length {1}'.
          format(site, len(self)))

    if site == len(self) - 1:
      return misc_mps.ncon(
          [self._right_mat, tf.conj(self._right_mat)], [[-1, 1], [-2, 1]])

    elif site >= self.pos and site < len(self) - 1:
      return tf.eye(int(self.D[site + 1]), dtype=self.dtype)
    else:
      r = misc_mps.ncon(
          [self.centermatrix, tf.conj(self.centermatrix)], [[-1, 1], [-2, 1]])
      for n in range(self.pos - 1, site, -1):
        r = self.transfer_op(n, 'r', r)
      return r

  def norm(self):
    """
        return the norm of the center matrix
        """
    return tf.sqrt(
        misc_mps.ncon(
            [self.centermatrix, tf.conj(self.centermatrix)], [[1, 2], [1, 2]]))

  def normalize(self):
    """
    normalizes the center matrix
    """
    Z = tf.sqrt(misc_mps.ncon([self.mat, tf.conj(self.mat)], [[1, 2], [1, 2]]))
    self.mat /= Z
    return Z

  def restore_form(self):
    raise NotImplementedError('')

  @property
  def centermatrix(self):
    return self.mat

  def diagonalize_center_matrix(self):
    """
        diagonalizes the center matrix and pushes U and V onto the left and right MPS tensors
        """

    if self.pos == 0:
      return
    elif self.pos == len(self):
      return
    else:
      S, U, V = tf.linalg.svd(self.centermatrix)
      S = tf.cast(S, self.dtype)
      self._tensors[self.pos - 1] = misc_mps.ncon(
          [self._tensors[self.pos - 1], U], [[-1, -2, 1], [1, -3]])
      self._tensors[self.pos] = misc_mps.ncon(
          [tf.conj(V), self._tensors[self.pos]], [[1, -1], [1, -2, -3]])
      self.mat = tf.diag(S)

  def position(self, bond, D=None, thresh=1E-32,normalize=False):
    """
        position(bond,schmidt_thresh = 1E-16):
        shifts the center site of the MPS to "bond".
        bond n is the bond to the *left* of site n.
        Args:
            bond (int):  the bond onto which to put the center matrix
        Returns:
            None
        """
    if bond > self.pos:
      self._tensors[self.pos] = misc_mps.ncon(
          [self.mat, self._tensors[self.pos]], [[-1, 1], [1, -2, -3]])

      for n in range(self.pos, bond):
        if (D is not None) or (thresh > 1E-16):
          tensor, s, v, _ = misc_mps.prepare_tensor_SVD(
            self._tensors[n], direction=1, D=D, thresh=thresh, normalize=normalize)
          mat = misc_mps.ncon([s,v],[[-1,1],[1,-2]])
        else:
          tensor, mat, _ = misc_mps.prepare_tensor_QR(
            self._tensors[n], direction=1)
        self.mat = mat
        self._tensors[n] = tensor
        if (n + 1) < bond:
          self._tensors[n + 1] = misc_mps.ncon([self.mat, self._tensors[n + 1]],
                                               [[-1, 1], [1, -2, -3]])

    if bond < self.pos:
      self._tensors[self.pos - 1] = misc_mps.ncon(
          [self._tensors[self.pos - 1], self.centermatrix],
          [[-1, -2, 1], [1, -3]])
      for n in range(self.pos - 1, bond - 1, -1):
        if (D is not None) or (thresh > 1E-16):        
          u, s, tensor, _ = misc_mps.prepare_tensor_SVD(
            self._tensors[n], direction=-1, D=D, thresh=thresh, normalize=normalize)
          mat = misc_mps.ncon([u,s],[[-1,1],[1,-2]])

        else:        
          mat, tensor, _ = misc_mps.prepare_tensor_QR(
            self._tensors[n], direction=-1)
        self.mat = mat
        self._tensors[n] = tensor
        if n > bond:
          self._tensors[n - 1] = misc_mps.ncon([self._tensors[n - 1], self.mat],
                                               [[-1, -2, 1], [1, -3]])
    self.pos = bond

  @staticmethod
  def ortho_deviation(tensor, which):
    """
        returns the deviation from left or right orthonormalization of the MPS tensors
        Args:
            which (str):  can take values in ('l','left','r','right')
                          determines which orthogonality should be tested
        Returns:
            tf.Tensor:   the deviation of orthogonality over all tensor entries
        """
    return orthonormalization(tensor, which)

  @staticmethod
  def check_ortho(tensor, which='l', thresh=1E-8):
    """
        checks if orthogonality condition on tensor is obeyed up to ```thresh```
        NOTE: Only works in eager mode.
        """
    return MPSUnitCellCentralGauge.ortho_deviation(tensor,
                                                   which).numpy().real < thresh

  def check_form(self, thresh=1E-8):
    """
        check if the MPS is in canonical form, i.e. if all tensors to the left of self.pos are left isometric, and 
        all tensors to the right of self.pos are right isometric.

        NOTE: Only works in eager mode.

        Args:
            thresh (float):  threshold for allowed deviation from orthogonality
        Returns:
            bool
        """
    pos = self.pos
    self.position(self.num_sites)
    a = np.all([
        self.check_ortho(self.get_tensor(site), 'l', thresh)
        for site in range(self.num_sites)
    ])
    self.position(0)
    b = np.all([
        self.check_ortho(self.get_tensor(site), 'r', thresh)
        for site in range(self.num_sites)
    ])
    self.position(pos)
    return np.all(a + b)


class FiniteMPSCentralGauge(MPSUnitCellCentralGauge, AbstractFiniteMPS):
  """
  A simple MPS class for finite systems;
  """
  @classmethod  
  def from_dense(cls, psi, name=None):
    
    ds = psi.shape
    Dl = 1
    tensors=[]
    for n in range(len(ds)-1):
        mat = np.reshape(psi,(ds[n]*Dl,np.prod(ds[n+1:])))
        Q,R = np.linalg.qr(mat)
        Dr = Q.shape[1]
        tensors.append(tf.convert_to_tensor(np.reshape(Q,(Dl,ds[n],Dr))))
        psi = np.reshape(R,((Dr,) + ds[n+1:]))
        Dl = Dr
    tensors.append(tf.convert_to_tensor(np.reshape(R,(Dl,ds[-1],1))))
    dtype = tensors[0].dtype
    D_end = tensors[-1].shape[2]
    centermatrix = tf.eye(int(D_end), dtype=dtype)
    position = len(tensors)
    mps = cls(
        tensors=tensors,
        centermatrix=centermatrix,
        position=position,
        name=name)
    return mps
   
  @classmethod
  def from_tensors(cls, tensors, name=None):
    """
        initializes an FiniteMPSCentralGauge from a list of tensors.
        mps.pos is set to len(tensors)
        connector, centralmatrix and right_mat are all set to identity matrices
        Args:
            tensors (list of tf.Tensor): list of mps tensors
        Returns:
            MPSUnitCellCentralGauge
        """
    dtype = tensors[0].dtype
    D_end = tensors[-1].shape[2]
    centermatrix = tf.eye(int(D_end), dtype=dtype)
    position = len(tensors)
    mps = cls(
        tensors=tensors,
        centermatrix=centermatrix,
        position=position,
        name=name)
    mps.position(0)  # obtain canonical form
    return mps

  def __init__(self, tensors, centermatrix, position, name=None):
    """
        initializes a FiniteMPSCentralGauge .
        Args:
            tensors (list of tf.Tensor): list of mps tensors
            centermatrix (tf.Tensor):    initial center matrix 
            position (int):              initial position of the center-matrix
            name (str):                  a name for the MPS
        Returns:
            FiniteMPSCentralGauge
        """

    if not (np.all([tensors[0].dtype == t.dtype for t in tensors])):
      raise TypeError(
          'FiniteMPSCentralGauge.__init__: tensors need to have same types')
    if not tensors[0].dtype == centermatrix.dtype:
      raise TypeError(
          'FiniteMPSCentralGauge.__init__: tensors need to have same type as centermatrix'
      )
    super().__init__(
        tensors=tensors,
        centermatrix=centermatrix,
        position=position,
        connector=tf.ones(shape=[1, 1], dtype=centermatrix.dtype),
        right_mat=tf.ones(shape=[1, 1], dtype=centermatrix.dtype),
        name=name)
    
  def get_amplitude(self,sigmas):
      t = self.get_tensor(0)[:,sigmas[0],:]
      for n in range(1,len(self)):
        t = misc_mps.ncon([t,self.get_tensor(n)[:,sigmas[n],:]],[[-1,1],[1,-2]])
      return t

  def canonize(self, name=None):
    """
        bring mps into canonical form, i.e. brings it into Gamma,Lambda form; 
        Args:
            name (str, optional):   a name for the canonized MPS
        Returns:
            FiniteMPS_Schmidt:      canonical form of the MPS
        """
    Lambdas, Gammas = [], []

    self.position(len(self))
    self.position(0)
    Lambdas.append(self.centermatrix)
    for n in range(len(self)):
      self.position(n + 1)
      self.diagonalize_center_matrix()
      Gammas.append(
          misc_mps.ncon(
              [tf.diag(1.0 / tf.diag_part(Lambdas[-1])), self._tensors[n]],
              [[-1, 1], [1, -2, -3]]))
      if n < len(self) - 1:
        Lambdas.append(self.centermatrix)

    return FiniteMPS_Schmidt(Gams=Gammas, Lams=Lambdas, name=name)

  def apply_2site(self, op, n, max_bond_dim=None, auto_trunc_max_err=0.0):
    """Applies an arbitrary two-site gate to the state at sites n and n+1.
        Optionally truncate the bond between sites n and n+1 after application.
        Note: If the tensors were initially in central gauge before this, they
              will still be in central gauge afterwards (up to the norm of the
              center matrix).
        """
    # move center matrix to bond between sites n and n+1
    self.position(n + 1)

    # normalize first for consistent truncation
    norm = self.normalize()

    A1 = self.get_tensor(n)
    A2 = self.get_tensor(n + 1)
    res = misc_mps.apply_2site_generic(
        op,
        A1,
        A2,
        max_bond_dim=max_bond_dim,
        auto_trunc_max_err=auto_trunc_max_err)
    A1, cmat, A2, trunc_err = res
    self._tensors[n] = A1
    self.mat = tf.divide(cmat, norm)  # account for previous normalization
    self._tensors[n + 1] = A2

    return trunc_err
  
  def generate_samples(self, num_samples):
      """
      generate samples from the MPS probability amplitude
      Args:
          num_samples(int): number of samples
      Returns:
          tf.Tensor of shape (num_samples, len(self):  the samples
      """
      dtype = self.dtype
      
      self.position(len(self))
      self.position(0)
      ds = self.d
      Ds = self.D
      right_envs = self.get_envs_right(range(len(self)))
      it = 0
      sigmas = []
      p_joint_1 = tf.ones(shape=[num_samples, 1], dtype=dtype)
      lenv = tf.stack([tf.eye(Ds[0], dtype=dtype) for _ in range(num_samples)], axis=0) #shape (num_samples, 1, 1)
      Z1 = tf.ones(shape=[num_samples,1], dtype=dtype)#shape (num_samples, 1)
      for site in range(len(self)):
        stdout.write( "\rgenerating samples at site %i/%i" % (site,len(self)))
        Z0 = tf.expand_dims(tf.linalg.norm(tf.reshape(lenv,(num_samples, Ds[site] * Ds[site])), axis=1),1) #shape (num_samples, 1)
        lenv /= tf.expand_dims(Z0,2)
        p_joint_0 = tf.linalg.diag_part(tn.ncon([lenv,self.get_tensor(site), tf.conj(self.get_tensor(site)), right_envs[site]],
                                                [[-1, 1, 2], [1, -2, 3],[2, -3, 4], [3, 4]])) #shape (Nt, d)
        #print(p_joint_0.shape, Z0.shape, Z1.shape, p_joint_1.shape)

        p_cond = Z0 / Z1 * tf.abs(p_joint_0/p_joint_1)

        p_cond /= np.expand_dims(tf.math.reduce_sum(p_cond,axis=1),1)

        #print(tf.math.reduce_sum(p_cond,1))
        sigmas.append(tf.squeeze(tf.random.categorical(tf.math.log(p_cond),1)))
        p_joint_1 = tf.expand_dims(tf.math.reduce_sum(p_cond * tf.one_hot(sigmas[-1], ds[site], dtype=dtype), axis=1),1)

        one_hots = tf.one_hot(sigmas[-1],ds[site], dtype=dtype)
        tmp = tn.ncon([self.get_tensor(site), one_hots],[[-2, 1, -3], [-1, 1]])          #tmp has shape (Nt, Dl, Dr)
        tmp2 = tf.transpose(tf.matmul(tf.transpose(lenv,(0, 2, 1)), tmp), (0, 2, 1)) #has shape (Nt, Dr, Dl')
        lenv = tf.matmul(tmp2, tf.conj(tmp)) #has shape (Nt, Dr, Dr')
        Z1 = Z0
      return tf.stack(sigmas, axis=1)
    


class InfiniteMPSCentralGauge(MPSUnitCellCentralGauge, AbstractInfiniteMPS):
  """
    A simple MPS class for infinite systems;
    """

  @classmethod
  def random(cls,
             d=list,
             D=list,
             name=None,
             dtype=tf.float32,
             precision=1E-10,
             power_method=True,
             numeig=1,
             initializer_function=tf.random_uniform,
             *args,
             **kwargs):

    out = super().random(
        d=d,
        D=D,
        name=name,
        dtype=dtype,
        initializer_function=initializer_function,
        *args,
        **kwargs)

    out.restore_form(
        numeig=numeig, precision=precision, power_method=power_method)
    return out

  def roll(self, shift_by):
    """Rol the sites in the unit cell.
        Moves sites to the left by `shift_by`, i.e. site `n` becomes site 
        `n - shift_by`.
        """
    self.position(shift_by)
    centermatrix = tf.Variable(self.mat)  # copy the center matrix
    self.position(len(self))  # move center matrix to the right
    new_center_matrix = misc_mps.ncon([self.mat, self.connector],
                                      [[-1, 1], [1, -2]])

    self.pos = shift_by
    self.mat = tf.Variable(centermatrix)
    self.position(0)
    new_center_matrix = misc_mps.ncon([new_center_matrix, self.mat],
                                      [[-1, 1], [1, -2]])
    tensors = ([self._tensors[n] for n in range(shift_by, len(self._tensors))] +
               [self._tensors[n] for n in range(shift_by)])
    self._tensors = tensors
    self.connector = tf.linalg.inv(centermatrix)
    self._right_mat = centermatrix
    self.mat = new_center_matrix
    self.pos = len(self) - shift_by

  def __init__(self,
               tensors,
               centermatrix,
               connector,
               right_mat,
               position,
               name=None,
               gauge=None):
    """
        initialize an infinite MPS in central gauge. The state is initialized in left orthogonal form 

        """
    if not (np.all([tensors[0].dtype == t.dtype for t in tensors])):
      raise TypeError(
          'FiniteMPSCentralGauge.__init__: tensors have to have same types')
    if not tensors[0].dtype == centermatrix.dtype:
      raise TypeError(
          'FiniteMPSCentralGauge.__init__: tensors have to have same type as centermatrix'
      )
    if not connector.dtype == centermatrix.dtype:
      raise TypeError(
          'FiniteMPSCentralGauge.__init__: connector has to have same type as centermatrix'
      )
    if not right_mat.dtype == centermatrix.dtype:
      raise TypeError(
          'FiniteMPSCentralGauge.__init__: right_mat has to have same type as centermatrix'
      )

    if not (tensors[0].shape[0] == tensors[-1].shape[2]):
      raise ValueError(
          'InfiniteMPSCentralGauge: boundary bond-dimensions are different: D[0] = {0}, D[1] = {1}'
          .format(tensors[0].shape[0], tensors[-1].shape[2]))
    super().__init__(
        tensors=tensors,
        centermatrix=centermatrix,
        connector=connector,
        right_mat=right_mat,
        position=position,
        name=name)
    #guarantee a properly initialized state
    #random() cannot be called with gauge = 's', because all **kwargs are passed to
    #tf.random_uniform which has no keyword gauge (it will raise an exception)
    if gauge in ('s', 'symmetric', 0):
      self.restore_form()

  def get_tensor(self, site):
    """
        get_tensor returns an mps tensors, possibly contracted with the center matrix
        by convention, the center matrix is contracted if n == self.pos
        the centermatrix is always absorbed from the left into the mps tensor, unless site == N-1, 
        in which case it is absorbed from the right

        the connector matrix is always absorbed into the right-most mps tensors, unless
        self.pos == 0, in which case it absorbed into the left-most mps tensor

        """
    return super().get_tensor(site % len(self))

  def set_tensor(self, n, tensor):
    raise NotImplementedError()

  def __setitem__(self, site, tensor):
    """
        Martin's API for setting center gauge site tensors.
        NOTE: Does not do the same as .set_tensor(n).
        """
    n = site % len(self)
    self._tensors[n] = tensor

  def restore_form(self,
                   init=None,
                   precision=1E-12,
                   power_method=True,
                   ncv=50,
                   nmax=1000,
                   numeig=1,
                   pinv=1E-30):
    """
        bring the MPS into Schmidt canonical form; normalizes the state
        Args:
            init (tf.tensor):     initial guess for the eigenvector
            precision (float):    desired precision of the dominant eigenvalue
            power_method (bool):  use power-method instead of sparse solver
            ncv (int):            number of Krylov vectors
            nmax (int):           max number of iterations
            numeig (int):         hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                                  to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
            pinv (float):         pseudoinverse cutoff
        Returns:
            None
        """

    self.position(0)
    tensors = [self.get_tensor(n) for n in range(len(self))]
    if not power_method:
      self._tensors, self.mat, self.connector, self._right_mat = misc_mps.restore_helper(
          tensors=tensors,
          init=init,
          precision=precision,
          ncv=ncv,
          nmax=nmax,
          numeig=numeig,
          pinv=pinv)
    elif power_method:
      self._tensors, self.mat, self.connector, self._right_mat = misc_mps.restore_helper_power_method(
          tensors=tensors, init=init, precision=precision, nmax=nmax, pinv=pinv)

    self.pos = len(self)

  def apply_2site(self, op, n, max_bond_dim=None, auto_trunc_max_err=0.0):
    """Applies an arbitrary two-site gate to the state at sites n and n+1.
        Optionally truncate the bond between sites n and n+1 after application.
        """
    N = self.num_sites

    if n + 1 >= N:
      # Sadly, we have to roll sites to avoid the connector matrix.
      self.roll(1)
      n -= 1
      cycled = True
    else:
      cycled = False

    # move center matrix to bond between sites n and n+1
    self.position((n + 1) % N)

    r_init = self.get_env_right((n - 1) % N)

    A1 = self.get_tensor(n)
    A2 = self.get_tensor(n + 1)
    res = misc_mps.apply_2site_generic(
        op,
        A1,
        A2,
        max_bond_dim=max_bond_dim,
        auto_trunc_max_err=auto_trunc_max_err)
    A1, cmat, A2, trunc_err = res
    self._tensors[n] = A1  # in left canonical form
    self.mat = cmat
    self._tensors[n + 1] = A2  # in right canonical form

    # FIXME: Is it enough to just check the right env?
    r_new = self.get_env_right(n - 1)
    diff = tf.norm(r_init - r_new)
    res = _float_res(self.dtype)
    if diff > 10 * res:
      print("Automatically restoring form due to effects on transfer "
            "matrix of order {}".format(diff))
      self.restore_form()

    if cycled:
      # Move sites back to where they were!
      self.roll(N - 1)

    return trunc_err

  def get_left_orthogonal_imps(self,
                               init=None,
                               precision=1E-12,
                               power_method=True,
                               ncv=50,
                               nmax=1000,
                               numeig=1,
                               pinv=1E-30,
                               restore_form=True,
                               name=None):
    """
        return the left-orthogonal form of the mps by shifting center-matrix to 
        the right end of the MPS and contracting `connector` and `centermatrix` into 
        the mps
        Args:
            init (tf.tensor):     initial guess for the eigenvector
            precision (float):    desired precision of the dominant eigenvalue
            power_method (bool):  use power-method instead of sparse solver
            ncv (int):            number of Krylov vectors
            nmax (int):           max number of iterations
            numeig (int):         hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                                  to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
            pinv (float):         pseudoinverse cutoff
            restore_form (bool):  if `True`, restore form prior to shifting center-matrix
        Returns:
            InfiniteMPSCentralGauge in left-orthogonal form
        """

    if restore_form:
      self.restore_form(
          init=init,
          precision=precision,
          power_method=power_method,
          ncv=ncv,
          nmax=nmax,
          numeig=numeig,
          pinv=pinv)
    dtype = self.dtype
    D_end = self.D[-1]
    centermatrix = tf.eye(int(D_end), dtype=dtype)
    connector = tf.eye(int(D_end), dtype=dtype)
    right_mat = tf.eye(int(D_end), dtype=dtype)
    position = len(self)
    return InfiniteMPSCentralGauge(
        tensors=[self.get_tensor(n) for n in range(len(self))],
        centermatrix=centermatrix,
        connector=connector,
        right_mat=right_mat,
        position=position,
        name=name)

  def get_right_orthogonal_imps(self,
                                init=None,
                                precision=1E-12,
                                power_method=True,
                                ncv=50,
                                nmax=1000,
                                numeig=1,
                                pinv=1E-30,
                                restore_form=True,
                                name=None):
    """
        return the right-orthogonal form of the mps by shifting center-matrix to 
        the left end of the MPS and contracting `connector` and `centermatrix` into 
        the mps
        Args:
            init (tf.tensor):     initial guess for the eigenvector
            precision (float):    desired precision of the dominant eigenvalue
            power_method (bool):  use power-method instead of sparse solver
            ncv (int):            number of Krylov vectors
            nmax (int):           max number of iterations
            numeig (int):         hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                                  to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
            pinv (float):         pseudoinverse cutoff
            restore_form (bool):  if `True`, restore form prior to shifting center-matrix
        Returns:
            InfiniteMPSCentralGauge in right-orthogonal form
    """
    if restore_form:
      self.restore_form(
          init=init,
          precision=precision,
          power_method=power_method,
          ncv=ncv,
          nmax=nmax,
          numeig=numeig,
          pinv=pinv)
    self.position(0)

    A = misc_mps.ncon([self.connector, self.mat, self._tensors[0]],
                      [[-1, 1], [1, 2], [2, -2, -3]])
    tensors = [A] + [self._tensors[n] for n in range(1, len(self))]
    dtype = self.dtype
    D_end = self.D[-1]
    centermatrix = tf.eye(int(D_end), dtype=dtype)
    connector = tf.eye(int(D_end), dtype=dtype)
    right_mat = tf.eye(int(D_end), dtype=dtype)
    position = 0

    return InfiniteMPSCentralGauge(
        tensors=tensors,
        centermatrix=centermatrix,
        connector=connector,
        right_mat=right_mat,
        position=position,
        name=name)
