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
from tensornetwork.network import TensorNetwork
import numpy as np
from typing import Any, List, Optional, Text, Type, Union, Dict
Tensor = Any


class FiniteMPS(TensorNetwork):
  """
  An MPS class for finite systems.
  `FiniteMPS` keeps track of the nodes of the network by storing them in a list
  `FiniteMPS._nodes`. Any external changes to this list will potentially corrupt 
  the mps. `FiniteMPS` has a central site. The position of this central site is
  stored in `FiniteMPS.center_position`. This center position can be  
  shifted using the `FiniteMPS.position` method. 
  If the state is initialized with `center_positon=0`, 
  then `FiniteMPS.position(len(FiniteMPS)-1)` shifts the `center_position`
  to `len(FiniteMPS) - 1`. If the shift is a "right-shift" (i.e. 
  `center_position` is moved from left to right), then all sites that are 
  visited in between are left in left-orthogonal form. If the shift is a 
  "left-shift" (i.e. `center_position` is shifted from right to left), 
  then all sites that are visited in between are left in right-orthogonal form. 
  For random initial tensors `tensors` and `center_position=0`, 
  doing one sweep from left to right and a successive sweep from right to left 
  brings the state into central canonical form. In this state, 
  all sites to the left of `center_position` are left orthogonal, 
  and all sites to the right of `center_position` are right orthogonal, 
  and the state is normalized. Due to efficiency reasons, the state upon 
  initialization is usually NOT brought into the central canonical form.
  """

  def __init__(self,
               tensors: List[Tensor],
               center_position: int,
               backend: Optional[Text] = None,
               dtype: Optional[Type[np.number]] = None) -> None:
    """
    Initialize a FiniteMPS.
    Args:
      tensors: A list of `Tensor` objects.
      center_position: The initial position of the center site.
      backend: The name of the backend that should be used to perform 
        contractions. See documentation of TensorNetwork.__init__ for 
        a list of supported backends.
      dtype: An optional `dtype` for the FiniteMPS. See documentation of 
        TensorNetwork.__init__ for more details.
    Returns:
      None

    """
    if center_position < 0 or center_position >= len(tensors):
      raise ValueError(
          'center_position = {} not between 0 <= center_position < {}'.format(
              center_position, len(tensors)))
    super().__init__(backend=backend, dtype=dtype)
    self._nodes = [
        self.add_node(tensors[n], name='node{}'.format(n))
        for n in range(len(tensors))
    ]  #redundant?!
    for site in range(len(self._nodes) - 1):
      self.connect(self._nodes[site][2], self._nodes[site + 1][0])
    self.center_position = center_position

  def save(self, path: str):
    raise NotImplementedError()

  @property
  def nodes(self):
    return self._nodes

  @property
  def bond_dimensions(self) -> List:
    """
    Return a list of bond dimensions of FiniteMPS
    """
    return [self._nodes[0].shape[0]] + [node.shape[2] for node in self._nodes]

  @property
  def physical_dimensions(self) -> List:
    """
    Return a list of physical Hilbert-space dimensions of FiniteMPS
    """

    return [node.shape[1] for node in self._nodes]

  def position(self, site: int, normalize: Optional[bool] = True) -> np.number:
    """
    Shift FiniteMPS.center_position to `site`.
    Args:
      site: The site to which FiniteMPS.center_position should be shifted
      normalize: If `True`, normalize matrices when shifting.
    Returns:
      Tensor: The norm of the tensor at FiniteMPS.center_position
    """
    #`site` has to be between 0 and len(mps) - 1
    if site >= len(self._nodes) or site < 0:
      raise ValueError('site = {} not between values'
                       ' 0 < site < N = {}'.format(site, len(self)))
    #nothing to do
    if site == self.center_position:
      return self.backend.norm(self._nodes[self.center_position].tensor)

    #shift center_position to the right using QR decomposition
    if site > self.center_position:
      n = self.center_position
      for n in range(self.center_position, site):
        Q, R = self.split_node_qr(
            self._nodes[n],
            left_edges=[self._nodes[n][0], self._nodes[n][1]],
            right_edges=[self._nodes[n][2]],
            left_name=self._nodes[n].name)

        self._nodes[n] = Q  #Q is a left-isometric tensor of rank 3
        self._nodes[n + 1] = self.contract(R[1], name=self._nodes[n + 1].name)
        Z = self.backend.norm(self._nodes[n + 1].tensor)

        # for an mps with > O(10) sites one needs to normalize to avoid
        # over or underflow errors; this takes care of the normalization
        if normalize:
          self._nodes[n + 1].tensor /= Z

      self.center_position = site

    #shift center_position to the left using RQ decomposition
    elif site < self.center_position:
      for n in reversed(range(site + 1, self.center_position + 1)):
        R, Q = self.split_node_rq(
            self._nodes[n],
            left_edges=[self._nodes[n][0]],
            right_edges=[self._nodes[n][1], self._nodes[n][2]],
            right_name=self._nodes[n].name)

        # for an mps with > O(10) sites one needs to normalize to avoid
        # over or underflow errors; this takes care of the normalization
        self._nodes[n] = Q  #Q is a right-isometric tensor of rank 3
        self._nodes[n - 1] = self.contract(R[0], name=self._nodes[n - 1].name)
        Z = self.backend.norm(self._nodes[n - 1].tensor)
        if normalize:
          self._nodes[n - 1].tensor /= Z

      self.center_position = site
    #return the norm of the last R tensor (useful for checks)
    return Z

  def check_orthonormality(self, which: Text, site: int) -> Tensor:
    """
    Check orthonormality of tensor at site `site`.
    Args:
      which: if 'l' or 'left': check left orthogonality
             if 'r' or'right': check right orthogonality
      site:  The site of the tensor.
    Returns:
      scalar Tensor: The L2 norm of the deviation from identity.
    Raises:
      ValueError: If which is different from 'l','left', 'r' or 'right'.
    """
    if which not in ('l', 'left', 'r', 'right'):
      raise ValueError(
          "Wrong value `which`={}. "
          "`which` as to be 'l','left', 'r' or 'right.".format(which))
    net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
    n1 = net.add_node(self._nodes[site].tensor)
    n2 = net.add_node(self.backend.conj(self._nodes[site].tensor))
    if which in ('l', 'left'):
      n1[0] ^ n2[0]
      n1[1] ^ n2[1]
    elif which in ('r', 'right'):
      n1[2] ^ n2[2]
      n1[1] ^ n2[1]
    result = net.contract_between(n1, n2)
    return self.backend.norm(
        abs(result.tensor -
            self.backend.eye(N=result.shape[0], M=result.shape[1])))

  def left_envs(self, sites: List[int]) -> Dict:
    """
    Compute left reduced density matrices for site `sites`.
    This returns a dict `left_envs` mapping sites (int) to Tensors.
    `left_envs[site]` is the left-reduced density matrix to the left of
    site `site`.
    Args:
      sites (list of int): A list of sites of the MPS.
    Returns:
      dict maping int to Tensor: The left-reduced density matrices 
        at each  site in `sites`.

    """
    n2 = max(sites)
    sites = np.array(sites)  #enable logical indexing

    #check if all elements of `sites` are within allowed range
    if not np.all(sites <= len(self)):
      raise ValueError('all elements of `sites` have to be <= N = {}'.format(
          len(self)))
    if not np.all(sites >= 0):
      raise ValueError('all elements of `sites` have to be positive')

    # left-reduced density matrices to the left of `center_position`
    # (including center_position) are all identities
    left_sites = sites[sites <= self.center_position]
    left_envs = {}
    for site in left_sites:
      left_envs[site] = self.backend.eye(N=self._nodes[site].shape[0])

    # left reduced density matrices at sites > center_position
    # have to be calculated from a network contraction
    if n2 > self.center_position:
      net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
      nodes = {}
      conj_nodes = {}
      for site in range(self.center_position, n2):
        nodes[site] = net.add_node(self._nodes[site].tensor)
        conj_nodes[site] = net.add_node(
            self.backend.conj(self._nodes[site].tensor))
      nodes[self.center_position][0] ^ conj_nodes[self.center_position][0]
      nodes[self.center_position][1] ^ conj_nodes[self.center_position][1]

      for site in range(self.center_position + 1, n2):
        nodes[site][0] ^ nodes[site - 1][2]
        conj_nodes[site][0] ^ conj_nodes[site - 1][2]
        nodes[site][1] ^ conj_nodes[site][1]

      edges = {site: node[2] for site, node in nodes.items()}
      conj_edges = {site: node[2] for site, node in conj_nodes.items()}

      left_env = net.contract_between(nodes[self.center_position],
                                      conj_nodes[self.center_position])
      left_env.reorder_edges(
          [edges[self.center_position], conj_edges[self.center_position]])
      if self.center_position + 1 in sites:
        left_envs[self.center_position + 1] = left_env.tensor
      for site in range(self.center_position + 1, n2):
        left_env = net.contract_between(left_env, nodes[site])
        left_env = net.contract_between(left_env, conj_nodes[site])
        if site + 1 in sites:
          left_env.reorder_edges([edges[site], conj_edges[site]])
          left_envs[site + 1] = left_env.tensor
    return left_envs

  def right_envs(self, sites: List[int]) -> Dict:
    """
    Compute right reduced density matrices for site `sites.
    This returns a dict `right_envs` mapping sites (int) to Tensors.
    `right_envs[site]` is the right-reduced density matrix to the right of
    site `site`.
    Args:
      sites (list of int): A list of sites of the MPS.
    Returns:
      dict maping int to Tensors: The right-reduced density matrices 
        at each  site in `sites`.

    """

    n1 = min(sites)
    sites = np.array(sites)
    #check if all elements of `sites` are within allowed range
    if not np.all(np.array(sites) < len(self)):
      raise ValueError('all elements of `sites` have to be < N = {}'.format(
          len(self)))
    if not np.all(np.array(sites) >= -1):
      raise ValueError('all elements of `sites` have to be >= -1')

    # right-reduced density matrices to the right of `center_position`
    # (including center_position) are all identities
    right_sites = sites[sites >= self.center_position]
    right_envs = {}
    for site in right_sites:
      right_envs[site] = self.backend.eye(N=self._nodes[site].shape[2])

    # right reduced density matrices at sites < center_position
    # have to be calculated from a network contraction
    if n1 < self.center_position:
      net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
      nodes = {}
      conj_nodes = {}
      for site in reversed(range(n1 + 1, self.center_position + 1)):
        nodes[site] = net.add_node(self._nodes[site].tensor)
        conj_nodes[site] = net.add_node(
            self.backend.conj(self._nodes[site].tensor))
      nodes[self.center_position][2] ^ conj_nodes[self.center_position][2]
      nodes[self.center_position][1] ^ conj_nodes[self.center_position][1]

      for site in reversed(range(n1 + 1, self.center_position)):
        nodes[site][2] ^ nodes[site + 1][0]
        conj_nodes[site][2] ^ conj_nodes[site + 1][0]
        nodes[site][1] ^ conj_nodes[site][1]

      edges = {site: node[0] for site, node in nodes.items()}
      conj_edges = {site: node[0] for site, node in conj_nodes.items()}

      right_env = net.contract_between(nodes[self.center_position],
                                       conj_nodes[self.center_position])
      if self.center_position - 1 in sites:
        right_env.reorder_edges(
            [edges[self.center_position], conj_edges[self.center_position]])
        right_envs[self.center_position - 1] = right_env.tensor
      for site in reversed(range(n1 + 1, self.center_position)):
        right_env = net.contract_between(right_env, nodes[site])
        right_env = net.contract_between(right_env, conj_nodes[site])
        if site - 1 in sites:
          right_env.reorder_edges([edges[site], conj_edges[site]])
          right_envs[site - 1] = right_env.tensor

    return right_envs

  def __len__(self):
    return len(self._nodes)

  def apply_transfer_operator(self, site: int, direction: Union[Text, int],
                              matrix: Tensor) -> Tensor:
    """
    Compute the action of the MPS transfer-operator at site `site`.
    Args:
      site (int): a site of the MPS
      direction (str or int): if 1, 'l' or 'left': compute the left-action 
                                of the MPS transfer-operator at `site` on the
                                input `matrix`
                              if -1, 'r' or 'right': compute the right-action 
                                of the MPS transfer-operator at `site` on the
                                input `matrix`
      matrix (Tensor): A rank-2 tensor or matrix.
    Returns:
      Tensor: the result of applying the MPS transfer-operator to `matrix`
    """
    net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
    mat = net.add_node(matrix)
    node = net.add_node(self._nodes[site].tensor)
    conj_node = net.add_node(self.backend.conj(self._nodes[site].tensor))
    node[1] ^ conj_node[1]
    if direction in (1, 'l', 'left'):
      mat[0] ^ node[0]
      mat[1] ^ conj_node[0]
      edge_order = [node[2], conj_node[2]]
    elif direction in (-1, 'r', 'right'):
      mat[0] ^ node[2]
      mat[1] ^ conj_node[2]
      edge_order = [node[0], conj_node[0]]
    result = net.contract_between(mat, node)
    result = net.contract_between(result, conj_node)

    return result.reorder_edges(edge_order).tensor

  def apply_two_site_gate(self,
                          gate: Tensor,
                          site1: int,
                          site2: int,
                          max_singular_values: Optional[int] = None,
                          max_truncation_err: Optional[float] = None) -> Tensor:
    """
    Apply a two-site gate to an MPS. This routine will in general 
    destroy any canonical form of the state. If a canonical form is needed, 
    the user can restore it using MPS.position
    Args:
      gate (Tensor): a two-body gate
      site1, site2 (int, int): the sites where the gate should be applied
      max_singular_values (int): The maximum number of singular values to keep.
      max_truncation_err (float): The maximum allowed truncation error.
    """
    if len(gate.shape) != 4:
      raise ValueError('rank of gate is {} but has to be 4'.format(
          len(gate.shape)))
    if site1 < 0 or site1 >= len(self) - 1:
      raise ValueError(
          'site1 = {} is not between 0 <= site < N - 1 = {}'.format(
              site1, len(self)))
    if site2 < 1 or site2 >= len(self):
      raise ValueError('site2 = {} is not between 1 <= site < N = {}'.format(
          site2, len(self)))
    if site2 <= site1:
      raise ValueError('site2 = {} has to be larger than site2 = {}'.format(
          site2, site1))
    if site2 != site1 + 1:
      raise ValueError(
          'site2 ={} != site1={}. Only nearest neighbor gates are currently '
          'supported'.format(site2, site1))

    if (max_singular_values or
        max_truncation_err) and self.center_position not in (site1, site2):
      raise ValueError(
          'center_position = {}, but gate is applied at sites {}, {}. '
          'Truncation should only be done if the gate '
          'is applied at the center position of the MPS'.format(
              self.center_position, site1, site2))

    gate_node = self.add_node(gate)
    gate_node[2] ^ self._nodes[site1][1]
    gate_node[3] ^ self._nodes[site2][1]
    left_edges = [self._nodes[site1][0], gate_node[0]]
    right_edges = [gate_node[1], self._nodes[site2][2]]
    result = self.contract_between(self._nodes[site1], self._nodes[site2])
    result = self.contract_between(result, gate_node)
    U, S, V, tw = self.split_node_full_svd(
        result,
        left_edges=left_edges,
        right_edges=right_edges,
        max_singular_values=max_singular_values,
        max_truncation_err=max_truncation_err,
        left_name=self._nodes[site1].name,
        right_name=self._nodes[site2].name)
    V.reorder_edges([S[1]] + right_edges)
    left_edges = left_edges + [S[1]]
    self._nodes[site1] = self.contract_between(
        U, S, name=U.name).reorder_edges(left_edges)
    self._nodes[site2] = V
    return tw

  def apply_one_site_gate(self, gate: Tensor, site: int) -> None:
    """
    Apply a one-site gate to an MPS. This routine will in general 
    destroy any canonical form of the state. If a canonical form is needed, 
    the user can restore it using MPS.position
    Args:
      gate (Tensor): a one-body gate
      site (int): the site where the gate should be applied
      
    """
    if len(gate.shape) != 2:
      raise ValueError('rank of gate is {} but has to be 2'.format(
          len(gate.shape)))
    if site < 0 or site >= len(self):
      raise ValueError('site = {} is not between 0 <= site < N={}'.format(
          site, len(self)))
    gate_node = self.add_node(gate)
    gate_node[1] ^ self._nodes[site][1]
    edge_order = [self._nodes[site][0], gate_node[0], self._nodes[site][2]]
    self._nodes[site] = self.contract_between(
        gate_node, self._nodes[site],
        name=self._nodes[site].name).reorder_edges(edge_order)

  def measure_local_operator(self, ops: List[Tensor], sites: List[int]) -> List:
    """
    Measure the expectation value of local operators `ops` site `sites`.
    Args:
      ops: list Tensors of rank 2; the local operators to be measured
      sites: sites where `ops` acts.
    Returns:
      List: measurements <op[n]> for n in `sites`
    Raises:
      ValueError if `len(ops) != len(sites)`
    """
    if not len(ops) == len(sites):
      raise ValueError('measure_1site_ops: len(ops) has to be len(sites)!')
    right_envs = self.right_envs(sites)
    left_envs = self.left_envs(sites)
    res = []
    for n, site in enumerate(sites):
      net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
      O = net.add_node(ops[n])
      R = net.add_node(right_envs[site])
      L = net.add_node(left_envs[site])
      A = net.add_node(self._nodes[site].tensor)
      conj_A = net.add_node(self.backend.conj(self._nodes[site].tensor))
      O[1] ^ A[1]
      O[0] ^ conj_A[1]
      R[0] ^ A[2]
      R[1] ^ conj_A[2]
      L[0] ^ A[0]
      L[1] ^ conj_A[0]
      result = L @ A @ O @ conj_A @ R
      res.append(result.tensor)
    return res

  def measure_two_body_correlator(self, op1: Tensor, op2: Tensor, site1: int,
                                  sites2: List[int]) -> List:
    """
    Commpute the correlator <op1,op2> between `site1` and all sites in `s` in 
    `sites2`. if `site1 == s`, op2 will be applied first
    Args:
      op1, op2: Tensors of rank 2; the local operators to be measured
      site1: the site where `op1`  acts
      sites2: sites where `op2` acts.
    Returns:
      List: correlator <op1, op2>
    Raises:
      ValueError if `site1` is out of range
    """
    N = len(self)
    if site1 < 0:
      raise ValueError(
          "Site site1 out of range: {} not between 0 <= site < N = {}.".format(
              site1, N))
    sites2 = np.array(sites2)  #enable logical indexing

    # we break the computation into two parts:
    # first we get all correlators <op2(site2) op1(site1)> with site2 < site1
    # then all correlators <op1(site1) op2(site2)> with site1 >= site1

    # get all sites smaller than site1
    left_sites = sorted(sites2[sites2 < site1])
    # get all sites larger than site1
    right_sites = sorted(sites2[sites2 > site1])

    # compute all neccessary right reduced
    # density matrices in one go. This is
    # more efficient than calling right_envs
    # for each site individually
    if right_sites:
      right_sites_mod = list({n % N for n in right_sites})
      rs = self.right_envs([site1] + right_sites_mod)

    c = []
    if left_sites:

      left_sites_mod = list({n % N for n in left_sites})

      ls = self.left_envs(left_sites_mod + [site1])
      net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)

      A = net.add_node(self._nodes[site1].tensor)
      O1 = net.add_node(op1)
      conj_A = net.add_node(self.backend.conj(self._nodes[site1].tensor))
      R = net.add_node(rs[site1])
      R[0] ^ A[2]
      R[1] ^ conj_A[2]
      A[1] ^ O1[1]
      conj_A[1] ^ O1[0]
      R = ((R @ A) @ O1) @ conj_A
      n1 = np.min(left_sites)
      r = R.tensor

      #          -- A--------
      #             |        |
      # compute   op1(site1) |
      #             |        |
      #          -- A*-------
      # and evolve it to the left by contracting tensors at site2 < site1
      # if site2 is in `sites2`, calculate the observable
      #
      #  ---A--........-- A--------
      # |   |             |        |
      # |  op2(site2)    op1(site1)|
      # |   |             |        |
      #  ---A--........-- A*-------

      for n in range(site1 - 1, n1 - 1, -1):
        if n in left_sites:
          net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
          A = net.add_node(self._nodes[n % N].tensor)
          conj_A = net.add_node(self.backend.conj(self._nodes[n % N].tensor))
          O2 = net.add_node(op2)
          L = net.add_node(ls[n % N])
          R = net.add_node(r)
          L[0] ^ A[0]
          L[1] ^ conj_A[0]
          O2[0] ^ conj_A[1]
          O2[1] ^ A[1]
          R[0] ^ A[2]
          R[1] ^ conj_A[2]

          res = (((L @ A) @ O2) @ conj_A) @ R
          c.append(res.tensor)
        if n > n1:
          r = self.apply_transfer_operator(n % N, 'right', r)

      c = list(reversed(c))

    # compute <op2(site1)op1(site1)>
    if site1 in sites2:
      net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
      O1 = net.add_node(op1)
      O2 = net.add_node(op2)
      L = net.add_node(ls[site1])
      R = net.add_node(rs[site1])
      A = net.add_node(self._nodes[site1].tensor)
      conj_A = net.add_node(self.backend.conj(self._nodes[site1].tensor))

      O1[1] ^ O2[0]
      L[0] ^ A[0]
      L[1] ^ conj_A[0]
      R[0] ^ A[2]
      R[1] ^ conj_A[2]
      A[1] ^ O2[1]
      conj_A[1] ^ O1[0]
      O = O1 @ O2
      res = (((L @ A) @ O) @ conj_A) @ R
      c.append(res.tensor)

    # compute <op1(site1) op2(site2)> for site1 < site2
    right_sites = sorted(sites2[sites2 > site1])
    if right_sites:
      net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
      A = net.add_node(self._nodes[site1].tensor)
      conj_A = net.add_node(self.backend.conj(self._nodes[site1].tensor))
      L = net.add_node(ls[site1])
      O1 = net.add_node(op1)
      L[0] ^ A[0]
      L[1] ^ conj_A[0]
      A[1] ^ O1[1]
      conj_A[1] ^ O1[0]
      L = L @ A @ O1 @ conj_A
      l = L.tensor
      del net

      n2 = np.max(right_sites)
      #          -- A--
      #         |   |
      # compute | op1(site1)
      #         |   |
      #          -- A*--
      # and evolve it to the right by contracting tensors at site2 > site1
      # if site2 is in `sites2`, calculate the observable
      #
      #  ---A--........-- A--------
      # |   |             |        |
      # |  op1(site1)    op2(site2)|
      # |   |             |        |
      #  ---A--........-- A*-------

      for n in range(site1 + 1, n2 + 1):
        if n in right_sites:
          net = TensorNetwork(backend=self.backend.name, dtype=self.dtype)
          L = net.add_node(l)
          R = net.add_node(rs[n % N])
          A = net.add_node(self._nodes[n % N].tensor)
          conj_A = net.add_node(self.backend.conj(self._nodes[n % N].tensor))
          O2 = net.add_node(op2)
          A[0] ^ L[0]
          conj_A[0] ^ L[1]
          O2[0] ^ conj_A[1]
          O2[1] ^ A[1]
          R[0] ^ A[2]
          R[1] ^ conj_A[2]
          res = L @ A @ O2 @ conj_A @ R
          c.append(res.tensor)

        if n < n2:
          l = self.apply_transfer_operator(n % N, 'left', l)
    return np.array(c)
