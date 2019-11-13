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
import functools
# pylint: disable=line-too-long
from tensornetwork.network_components import Node, contract, contract_between, BaseNode
from tensornetwork.backends import backend_factory
# pylint: disable=line-too-long
from tensornetwork.network_operations import split_node_qr, split_node_rq, split_node_full_svd, norm, conj
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
Tensor = Any


class BaseMPS:
  """
  The base class for MPS. All MPS should be derived from BaseMPS
  `BaseMPS` is an infinite matrix product state with a 
  finite unitcell.

  Important attributes:

    * `BaseMPS.nodes`: stores the tensors in a list of `Node` objects
    * `BaseMPS.center_position`: the location of the orthogonality site
    * `BaseMPS.connector_matrix`: a rank-2 `Tensor` stored in a `Node`.
      `BaseMPS.connector_matrix` connects unit cells back to themselves.
       To stack different unit cells, the `BaseMPS.connector_matrix` is
       absorbed into the rightmost (by convention) mps tensor prior
       to stacking.

  To obtain a sequence of `Node` objects `[node_1,...,node_N]`
  which can be arbitrarily stacked, i.e.
  `stacked_nodes=[node_1,...,node_N, node_1, ..., node_N,...]`
  use the `BaseMPS.get_node` function. This function automatically
  absorbs `BaseNode.connector_matrix` into the correct `Node` object
  to ensure that `Node`s (i.e. the mps tensors) can be consistently
  stacked without gauge jumps.

  The orthogonality center can be be shifted using the 
  `BaseMPS.position` method, which uses uses QR and RQ methods to shift 
  `center_position`.

  
  """

  def __init__(self,
               tensors: List[Union[BaseNode, Tensor]],
               center_position: Optional[int] = 0,
               connector_matrix: Optional[Union[BaseNode, Tensor]] = None,
               backend: Optional[Text] = None) -> None:
    """
    Initialize a BaseMPS.
    Args:
      tensors: A list of `Tensor` or `BaseNode` objects.
      center_position: The initial position of the center site.
      connector_matrix: A `Tensor` or `BaseNode` of rank 2 connecting
        different unitcells. A value `None` is equivalent to an identity
        `connector_matrix`.
      backend: The name of the backend that should be used to perform 
        contractions. Available backends are currently 'numpy', 'tensorflow',
        'pytorch', 'jax'
    """
    if center_position < 0 or center_position >= len(tensors):
      raise ValueError(
          'center_position = {} not between 0 <= center_position < {}'.format(
              center_position, len(tensors)))

    # we're no longer connecting MPS nodes because it's barely needed
    # the dtype is deduced from the tensor object.
    self.nodes = [
        Node(tensors[n], backend=backend, name='node{}'.format(n))
        for n in range(len(tensors))
    ]

    self.connector_matrix = Node(
        connector_matrix,
        backend=backend) if connector_matrix is not None else connector_matrix
    self.center_position = center_position

  def __len__(self):
    return len(self.nodes)

  def position(self, site: int, normalize: Optional[bool] = True) -> np.number:
    """
    Shift `FiniteMPS.center_position` to `site`.

    Args:
      site: The site to which FiniteMPS.center_position should be shifted
      normalize: If `True`, normalize matrices when shifting.
    Returns:
      `Tensor`: The norm of the tensor at `FiniteMPS.center_position`
    """
    #`site` has to be between 0 and len(mps) - 1
    if site >= len(self.nodes) or site < 0:
      raise ValueError('site = {} not between values'
                       ' 0 < site < N = {}'.format(site, len(self)))
    #nothing to do
    if site == self.center_position:
      Z = self.backend.norm(self.nodes[self.center_position].tensor)
      if normalize:
        self.nodes[self.center_position].tensor /= Z
      return Z

    #shift center_position to the right using QR decomposition
    if site > self.center_position:
      n = self.center_position
      for n in range(self.center_position, site):
        Q, R = split_node_qr(
            self.nodes[n],
            left_edges=[self.nodes[n][0], self.nodes[n][1]],
            right_edges=[self.nodes[n][2]],
            left_name=self.nodes[n].name)
        Q[2] | R[0]  #break the edge between Q and R
        order = [R[0], self.nodes[n + 1][1], self.nodes[n + 1][2]]
        R[1] ^ self.nodes[n + 1][0]  #connect R to the right node
        self.nodes[n] = Q  #Q is a left-isometric tensor of rank 3
        self.nodes[n + 1] = contract(R[1], name=self.nodes[n + 1].name)
        self.nodes[n + 1].reorder_edges(order)
        Z = norm(self.nodes[n + 1])

        # for an mps with > O(10) sites one needs to normalize to avoid
        # over or underflow errors; this takes care of the normalization
        if normalize:
          self.nodes[n + 1].tensor /= Z

      self.center_position = site

    #shift center_position to the left using RQ decomposition
    elif site < self.center_position:
      for n in reversed(range(site + 1, self.center_position + 1)):

        R, Q = split_node_rq(
            self.nodes[n],
            left_edges=[self.nodes[n][0]],
            right_edges=[self.nodes[n][1], self.nodes[n][2]],
            right_name=self.nodes[n].name)
        #print(self.nodes[n].shape, R.shape, Q.shape)
        R[1] | Q[0]  #break the edge between R and Q
        R[0] ^ self.nodes[n - 1][2]  #connect R to the left node
        order = [self.nodes[n - 1][0], self.nodes[n - 1][1], R[1]]

        # for an mps with > O(10) sites one needs to normalize to avoid
        # over or underflow errors; this takes care of the normalization
        self.nodes[n] = Q  #Q is a right-isometric tensor of rank 3
        self.nodes[n - 1] = contract(R[0], name=self.nodes[n - 1].name)
        self.nodes[n - 1].reorder_edges(order)
        Z = norm(self.nodes[n - 1])
        if normalize:
          self.nodes[n - 1].tensor /= Z

      self.center_position = site
    #return the norm of the last R tensor (useful for checks)
    return Z

  @property
  def backend(self):
    if not all([
        self.nodes[0].backend.name == node.backend.name for node in self.nodes
    ]):
      raise ValueError('not all backends in FiniteMPS.nodes are the same')
    return self.nodes[0].backend

  @property
  def dtype(self):
    if not all([self.nodes[0].dtype == node.dtype for node in self.nodes]):
      raise ValueError('not all dtype in FiniteMPS.nodes are the same')

    return self.nodes[0].dtype

  def save(self, path: str):
    raise NotImplementedError()

  @property
  def bond_dimensions(self) -> List:
    """
    A list of bond dimensions of `BaseMPS`
    """
    return [self.nodes[0].shape[0]] + [node.shape[2] for node in self.nodes]

  @property
  def physical_dimensions(self) -> List:
    """
    A list of physical Hilbert-space dimensions of `BaseMPS`
    """

    return [node.shape[1] for node in self.nodes]

  def right_envs(self, sites: Sequence[int]) -> Dict:
    raise NotImplementedError()

  def left_envs(self, sites: Sequence[int]) -> Dict:
    raise NotImplementedError()

  def apply_transfer_operator(self, site: int, direction: Union[Text, int],
                              matrix: Union[BaseNode, Tensor]) -> BaseNode:
    """
    Compute the action of the MPS transfer-operator at site `site`.

    Args:
      site: a site of the MPS
      direction: 
        * if `1, 'l'` or `'left'`: compute the left-action 
          of the MPS transfer-operator at `site` on the input `matrix`.
        * if `-1, 'r'` or `'right'`: compute the right-action 
          of the MPS transfer-operator at `site` on the input `matrix`
      matrix: A rank-2 tensor or matrix.
    Returns:
      `Node`: the result of applying the MPS transfer-operator to `matrix`
    """
    mat = Node(matrix, backend=self.backend.name)
    node = self.get_node(site)
    conj_node = conj(node)
    node[1] ^ conj_node[1]
    if direction in (1, 'l', 'left'):
      mat[0] ^ node[0]
      mat[1] ^ conj_node[0]
      edge_order = [node[2], conj_node[2]]
    elif direction in (-1, 'r', 'right'):
      mat[0] ^ node[2]
      mat[1] ^ conj_node[2]
      edge_order = [node[0], conj_node[0]]
    result = mat @ node @ conj_node
    return result.reorder_edges(edge_order)

  def measure_local_operator(self, ops: List[Union[BaseNode, Tensor]],
                             sites: Sequence[int]) -> List:
    """
    Measure the expectation value of local operators `ops` site `sites`.

    Args:
      ops: A list Tensors of rank 2; the local operators to be measured.
      sites: Sites where `ops` act.

    Returns:
      List: measurements :math:`\\langle` `ops[n]`:math:`\\rangle` for n in `sites`
    Raises:
      ValueError if `len(ops) != len(sites)`
    """
    if not len(ops) == len(sites):
      raise ValueError('measure_1site_ops: len(ops) has to be len(sites)!')
    right_envs = self.right_envs(sites)
    left_envs = self.left_envs(sites)
    res = []
    for n, site in enumerate(sites):
      O = Node(ops[n], backend=self.backend.name)
      R = right_envs[site]
      L = left_envs[site]
      A = Node(self.nodes[site], backend=self.backend.name)
      conj_A = conj(A)
      O[1] ^ A[1]
      O[0] ^ conj_A[1]
      R[0] ^ A[2]
      R[1] ^ conj_A[2]
      L[0] ^ A[0]
      L[1] ^ conj_A[0]
      result = L @ A @ O @ conj_A @ R
      res.append(result.tensor)
    return res

  def measure_two_body_correlator(self, op1: Union[BaseNode, Tensor],
                                  op2: Union[BaseNode, Tensor], site1: int,
                                  sites2: Sequence[int]) -> List:
    """
    Compute the correlator 
    :math:`\\langle` `op1[site1], op2[s]`:math:`\\rangle`
    between `site1` and all sites `s` in `sites2`. if `s==site1`, 
    `op2[s]` will be applied first

    Args:
      op1: Tensor of rank 2; the local operator at `site1`
      op2: List of tensors of rank 2; the local operators 
        at `sites2`.
      site1: The site where `op1`  acts
      sites2: Sites where operators `op2` act.
    Returns:
      List: Correlator :math:`\\langle` `op1[site1], op2[s]`:math:`\\rangle`
        for `s` :math:`\\in` `sites2`.
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
      A = Node(self.nodes[site1], backend=self.backend.name)
      O1 = Node(op1, backend=self.backend.name)
      conj_A = conj(A)
      R = rs[site1]
      R[0] ^ A[2]
      R[1] ^ conj_A[2]
      A[1] ^ O1[1]
      conj_A[1] ^ O1[0]
      R = ((R @ A) @ O1) @ conj_A
      n1 = np.min(left_sites)
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
          A = Node(self.nodes[n % N], backend=self.backend.name)
          conj_A = conj(A)
          O2 = Node(op2, backend=self.backend.name)
          L = ls[n % N]
          L[0] ^ A[0]
          L[1] ^ conj_A[0]
          O2[0] ^ conj_A[1]
          O2[1] ^ A[1]
          R[0] ^ A[2]
          R[1] ^ conj_A[2]

          res = (((L @ A) @ O2) @ conj_A) @ R
          c.append(res.tensor)
        if n > n1:
          R = self.apply_transfer_operator(n % N, 'right', R)

      c = list(reversed(c))

    # compute <op2(site1)op1(site1)>
    if site1 in sites2:
      O1 = Node(op1, backend=self.backend.name)
      O2 = Node(op2, backend=self.backend.name)
      L = ls[site1]
      R = rs[site1]
      A = Node(self.nodes[site1], backend=self.backend.name)
      conj_A = conj(A)

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
      A = Node(self.nodes[site1], backend=self.backend.name)
      conj_A = conj(A)
      L = ls[site1]
      O1 = Node(op1, backend=self.backend.name)
      L[0] ^ A[0]
      L[1] ^ conj_A[0]
      A[1] ^ O1[1]
      conj_A[1] ^ O1[0]
      L = L @ A @ O1 @ conj_A
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
          R = rs[n % N]
          A = Node(self.nodes[n % N], backend=self.backend.name)
          conj_A = conj(A)
          O2 = Node(op2, backend=self.backend.name)
          A[0] ^ L[0]
          conj_A[0] ^ L[1]
          O2[0] ^ conj_A[1]
          O2[1] ^ A[1]
          R[0] ^ A[2]
          R[1] ^ conj_A[2]
          res = L @ A @ O2 @ conj_A @ R
          c.append(res.tensor)

        if n < n2:
          L = self.apply_transfer_operator(n % N, 'left', L)
    return c

  def apply_two_site_gate(self,
                          gate: Union[BaseNode, Tensor],
                          site1: int,
                          site2: int,
                          max_singular_values: Optional[int] = None,
                          max_truncation_err: Optional[float] = None) -> Tensor:
    """
    Apply a two-site gate to an MPS. This routine will in general 
    destroy any canonical form of the state. If a canonical form is needed, 
    the user can restore it using `FiniteMPS.position`.

    Args:
      gate (Tensor): a two-body gate
      site1, site2 (int, int): the sites where the gate should be applied
      max_singular_values (int): The maximum number of singular values to keep.
      max_truncation_err (float): The maximum allowed truncation error.
    Returns:
      scalar `Tensor`: the truncated weight of the truncation.
    
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

    gate_node = Node(gate, backend=self.backend.name)

    self.nodes[site1][2] ^ self.nodes[site2][0]
    gate_node[2] ^ self.nodes[site1][1]
    gate_node[3] ^ self.nodes[site2][1]
    left_edges = [self.nodes[site1][0], gate_node[0]]
    right_edges = [gate_node[1], self.nodes[site2][2]]
    result = self.nodes[site1] @ self.nodes[site2] @ gate_node
    U, S, V, tw = split_node_full_svd(
        result,
        left_edges=left_edges,
        right_edges=right_edges,
        max_singular_values=max_singular_values,
        max_truncation_err=max_truncation_err,
        left_name=self.nodes[site1].name,
        right_name=self.nodes[site2].name)
    V.reorder_edges([S[1]] + right_edges)
    left_edges = left_edges + [S[1]]
    self.nodes[site1] = contract_between(
        U, S, name=U.name).reorder_edges(left_edges)
    self.nodes[site2] = V
    self.nodes[site1][2] | self.nodes[site2][0]
    return tw

  def apply_one_site_gate(self, gate: Union[BaseNode, Tensor],
                          site: int) -> None:
    """
    Apply a one-site gate to an MPS. This routine will in general 
    destroy any canonical form of the state. If a canonical form is needed, 
    the user can restore it using `FiniteMPS.position`

    Args:
      gate: a one-body gate
      site: the site where the gate should be applied
      
    """
    if len(gate.shape) != 2:
      raise ValueError('rank of gate is {} but has to be 2'.format(
          len(gate.shape)))
    if site < 0 or site >= len(self):
      raise ValueError('site = {} is not between 0 <= site < N={}'.format(
          site, len(self)))
    gate_node = Node(gate, backend=self.backend.name)
    gate_node[1] ^ self.nodes[site][1]
    edge_order = [self.nodes[site][0], gate_node[0], self.nodes[site][2]]
    self.nodes[site] = contract_between(
        gate_node, self.nodes[site],
        name=self.nodes[site].name).reorder_edges(edge_order)

  def check_orthonormality(self, which: Text, site: int) -> Tensor:
    """
    Check orthonormality of tensor at site `site`.

    Args:
      which: * if `'l'` or `'left'`: check left orthogonality
             * if `'r`' or `'right'`: check right orthogonality
      site:  The site of the tensor.
    Returns:
      scalar `Tensor`: The L2 norm of the deviation from identity.
    Raises:
      ValueError: If which is different from 'l','left', 'r' or 'right'.
    """
    if which not in ('l', 'left', 'r', 'right'):
      raise ValueError(
          "Wrong value `which`={}. "
          "`which` as to be 'l','left', 'r' or 'right.".format(which))
    n1 = self.nodes[site]
    n2 = conj(n1)
    if which in ('l', 'left'):
      n1[0] ^ n2[0]
      n1[1] ^ n2[1]
    elif which in ('r', 'right'):
      n1[2] ^ n2[2]
      n1[1] ^ n2[1]
    result = n1 @ n2
    return self.backend.norm(
        abs(result.tensor - self.backend.eye(
            N=result.shape[0], M=result.shape[1], dtype=self.dtype)))

  def get_node(self, site: int) -> BaseNode:
    """
    Returns the `Node` object at `site`.
    If `site==len(self) - 1` `BaseMPS.connector_matrix`
    is absorbed fromt the right-hand side into the returned 
    `Node` object.

    Args:
      site: The site for which to return the `Node`.
    Returns:
      `Node`: The node at `site`.
    """
    if site >= len(self):
      raise IndexError(
          'index `site` = {} is out of range for len(mps)= {}'.format(
              site, len(self)))
    if site < 0:
      raise ValueError(
          'index `site` has to be larger than 0 (found `site`={}).'.format(
              site))
    if (site == len(self) - 1) and (self.connector_matrix is not None):
      self.nodes[site][2] ^ self.connector_matrix[0]
      order = [
          self.nodes[site][0], self.nodes[site][1], self.connector_matrix[1]
      ]
      return contract_between(
          self.nodes[site],
          self.connector_matrix,
          name=self.nodes[site].name,
          output_edge_order=order)
    return self.nodes[site]

  def canonicalize(self, normalize: Optional[bool] = True) -> np.number:
    raise NotImplementedError()


class InfiniteMPS(BaseMPS):
  """
  An MPS class for infinite systems. 

  MPS tensors are stored as a list of `Node` objects in the `InfiniteMPS.nodes`
  attribute.
  `InfiniteMPS` has a central site, also called orthogonality center. 
  The position of this central site is stored in `InfiniteMPS.center_position`, 
  and it can be be shifted using the `InfiniteMPS.position` method. 
  `InfiniteMPS.position` uses QR and RQ methods to shift `center_position`.
  
  `InfiniteMPS` can be initialized either from a `list` of tensors, or
  by calling the classmethod `InfiniteMPS.random`.
  """

  def __init__(self,
               tensors: List[Union[BaseNode, Tensor]],
               center_position: Optional[int] = 0,
               connector_matrix: Optional[Union[BaseNode, Tensor]] = None,
               backend: Optional[Text] = None) -> None:
    """
    Initialize a FiniteMPS.
    Args:
      tensors: A list of `Tensor` or `BaseNode` objects.
      center_position: The initial position of the center site.
      connector_matrix: A `Tensor` or `BaseNode` of rank 2 connecting
        different unitcells. A value `None` is equivalent to an identity
        `connector_matrix`.
      backend: The name of the backend that should be used to perform 
        contractions. Available backends are currently 'numpy', 'tensorflow',
        'pytorch', 'jax'
    """

    super().__init__(
        tensors=tensors,
        center_position=center_position,
        connector_matrix=connector_matrix,
        backend=backend)

  @classmethod
  def random(cls,
             d: List[int],
             D: List[int],
             dtype: Type[np.number],
             backend: Optional[Text] = None):
    """
    Initialize a random `FiniteMPS`. The resulting state
    is normalized. Its center-position is at 0.

    Args:
      d: A list of physical dimensions.
      D: A list of bond dimensions.
      dtype: A numpy dtype.
      backend: An optional backend.
    Returns:
      `FiniteMPS`
    """
    #use numpy backend for tensor initialization
    be = backend_factory.get_backend('numpy')
    if len(D) != len(d) + 1:
      raise ValueError('len(D) = {} is different from len(d) + 1= {}'.format(
          len(D),
          len(d) + 1))
    if D[-1] != D[0]:
      raise ValueError('D[0]={} != D[-1]={}.'.format(D[0], D[-1]))

    tensors = [
        be.randn((D[n], d[n], D[n + 1]), dtype=dtype) for n in range(len(d))
    ]
    return cls(tensors=tensors, center_position=0, backend=backend)

  def unit_cell_transfer_operator(self, direction: Union[Text, int],
                                  matrix: Union[BaseNode, Tensor]) -> BaseNode:
    sites = range(len(self))
    if direction in (-1, 'r', 'right'):
      sites = reversed(sites)

    for site in sites:
      matrix = self.apply_transfer_operator(site, direction, matrix)
    return matrix

  def transfer_matrix_eigs(
      self,
      direction: Union[Text, int],
      initial_state: Optional[Union[BaseNode, Tensor]] = None,
      precision: Optional[float] = 1E-10,
      num_krylov_vecs: Optional[int] = 30,
      maxiter: Optional[int] = None):
    """
    Compute the dominant eigenvector of the MPS transfer matrix.
    
    Ars:
      direction: 
        * If `'1','l''left'`: return the left dominant eigenvalue
          and eigenvector
        * If `'-1','r''right'`: return the right dominant eigenvalue
          and eigenvector
      initial_state: An optional initial state.
      num_krylov_vecs: Number of Krylov vectors to be used in `eigs`.
      precision: The desired precision of the eigen values.
      maxiter: The maximum number of iterations.
    Returns:
      `float` or `complex`: The dominant eigenvalue.
      Node: The dominant eigenvector.
    """
    D = self.bond_dimensions[0]

    def mv(vector):
      result = self.unit_cell_transfer_operator(
          direction, self.backend.reshape(vector, (D, D)))
      return self.backend.reshape(result.tensor, (D * D,))

    if not initial_state:
      initial_state = self.backend.randn((self.bond_dimensions[0]**2,),
                                         dtype=self.dtype)
    else:
      if isinstance(initial_state, BaseNode):
        initial_state = initial_state.tensor
      initial_state = self.backend.reshape(initial_state,
                                           (self.bond_dimensions[0]**2,))

    #note: for real dtype eta and dens are real.
    #but scipy.linalg.eigs returns complex dtypes in any case
    #since we know that for an MPS transfer matrix the largest
    #eigenvalue and corresponding eigenvector are real
    # we cast them.
    eta, dens = self.backend.eigs(
        A=mv,
        initial_state=initial_state,
        num_krylov_vecs=num_krylov_vecs,
        numeig=1,
        tol=precision,
        which='LR',
        maxiter=maxiter,
        dtype=self.dtype)
    result = self.backend.reshape(
        dens[0], (self.bond_dimensions[0], self.bond_dimensions[0]))
    return eta[0], Node(result, backend=self.backend.name)

  def right_envs(self, sites: Sequence[int]) -> Dict:
    raise NotImplementedError()

  def left_envs(self, sites: Sequence[int]) -> Dict:
    raise NotImplementedError()

  def save(self, path: str):
    raise NotImplementedError()

  def canonicalize(self,
                   initial_state: Optional[Union[BaseNode, Tensor]] = None,
                   precision: Optional[float] = 1E-10,
                   truncation_threshold: Optional[float] = 1E-15,
                   D: Optional[int] = None,
                   num_krylov_vecs: Optional[int] = 50,
                   maxiter: Optional[int] = 1000,
                   pseudo_inverse_cutoff: Optional[float] = None):
    """
    Canonicalize an InfiniteMPS (i.e. bring it into Schmidt-canonical form).

    Parameters:
    ------------------------------
    init:          Tensor
                   initial guess for the eigenvector
    precision:     float
                   desired precision of the dominant eigenvalue
    num_krylov_vecs:           int
                   number of Krylov vectors
    nmax:          int
                   max number of iterations
    numeig:        int
                   hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                   to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
    pinv:          float
                   pseudoinverse cutoff
    truncation_threshold: float 
                          truncation threshold for the MPS, if < 1E-15, no truncation is done
    D:             int or None 
                   if int is given, bond dimension will be reduced to `D`; `D=None` has no effect
    warn_thresh:   float 
                   threshold value; if TMeigs returns an eigenvalue with imaginary value larger than 
                   ```warn_thresh```, a warning is issued 

    Returns:
    ----------------------------------
    None
    """

    #bring center-position to 0
    self.position(0)
    #dtype of eta is the same as InfiniteMPS.dtype
    #this is assured in the backend.
    eta, l = self.transfer_matrix_eigs(
        direction='left',
        initial_state=initial_state,
        precision=precision,
        num_krylov_vecs=num_krylov_vecs,
        maxiter=maxiter)
    sqrteta = self.backend.sqrt(eta)
    self.nodes[0].tensor /= sqrteta

    # if np.abs(np.imag(eta)) / np.abs(np.real(eta)) > warn_thresh:
    #   print(
    #       'in mpsfunctions.py.regaugeIMPS: warning: found eigenvalue eta with large imaginary part: ',
    #       eta)

    #TODO: would be nice to do the algebra directly on the nodes here
    l.tensor /= self.backend.trace(l.tensor)
    l.tensor = (l.tensor +
                self.backend.transpose(self.backend.conj(l.tensor),
                                       (1, 0))) / 2.0
    #eigvals_left and u_left are both `Tensor` objects
    eigvals_left, u_left = self.backend.eigh(l.tensor)
    eigvals_left /= self.backend.norm(eigvals_left)
    if pseudo_inverse_cutoff:
      mask = eigvals_left <= pseudo_inverse_cutoff

    inveigvals_left = 1.0 / eigvals_left
    if pseudo_inverse_cutoff:
      inveigvals_left = self.backend.index_update(inveigvals_left, mask, 0.0)

    # u_left = Node(u_left, backend=self.backend.name)
    # sqrt_eigvals_left = Node(
    #     self.backend.sqrt(self.backend.diag(eigvals_left)),
    #     backend=self.backend.name)

    # inv_sqrt_eigvals_left = Node(
    #     self.backend.sqrt(self.backend.diag(inv_eigvals_left)),
    #     backend=self.backend.name)

    # sqrt_eigvals_left[0] ^ u_left[0]
    # order = [sqrt_eigvals_left[0], u_left[0]]
    # y = sqrt_eigvals_left[0] @ u_left[1]
    # y.reorder_edges(order)

    y = Node(
        ncon(
            [u_left, self.backend.diag(self.backend.sqrt(eigvals_left))],
            [[-2, 1], [1, -1]],
            backend=self.backend.name),
        backend=self.backend.name)
    invy = Node(
        ncon([
            self.backend.diag(self.backend.sqrt(inveigvals_left)),
            self.backend.conj(u_left)
        ], [[-2, 1], [-1, 1]],
             backend=self.backend.name),
        backend=self.backend.name)

    eta, r = self.transfer_matrix_eigs(
        direction='right',
        initial_state=initial_state,
        precision=precision,
        num_krylov_vecs=num_krylov_vecs,
        maxiter=maxiter)

    r.tensor /= self.backend.trace(r.tensor)
    r.tensor = (r.tensor +
                self.backend.transpose(self.backend.conj(r.tensor),
                                       (1, 0))) / 2.0
    #eigvals_left and u_left are both `Tensor` objects
    eigvals_right, u_right = self.backend.eigh(r.tensor)
    eigvals_right /= self.backend.norm(eigvals_right)
    if pseudo_inverse_cutoff:
      mask = eigvals_right <= pseudo_inverse_cutoff

    inveigvals_right = 1.0 / eigvals_right
    if pseudo_inverse_cutoff:
      inveigvals_right = self.backend.index_update(inveigvals_right, mask, 0.0)

    # r = r / r.tr()
    # r = (r + r.conj().transpose()) / 2.0
    # eigvals_right, u_right = r.eigh()
    # eigvals_right[eigvals_right <= pinv] = 0.0

    # eigvals_right /= np.sqrt(
    #     ncon.ncon([eigvals_right, eigvals_right.conj()], [[1], [1]]))
    # inveigvals_right = eigvals_right.zeros(eigvals_right.shape[0])
    # inveigvals_right[
    #     eigvals_right > pinv] = 1.0 / eigvals_right[eigvals_right > pinv]

    x = Node(
        ncon([u_right,
              self.backend.diag(self.backend.sqrt(eigvals_right))],
             [[-1, 1], [1, -2]],
             backend=self.backend.name),
        backend=self.backend.name)

    invx = Node(
        ncon([
            self.backend.diag(self.backend.sqrt(inveigvals_right)),
            self.backend.conj(u_right)
        ], [[-1, 1], [-2, 1]],
             backend=self.backend.name),
        backend=self.backend.name)

    tmp = Node(
        ncon([y, x], [[-1, 1], [1, -2]], backend=self.backend.name),
        backend=self.backend.name)
    U, lam, V, _ = split_node_full_svd(
        tmp, [tmp[0]], [tmp[1]],
        max_singular_values=D,
        max_truncation_err=truncation_threshold)
    # U, lam, V, _ = .svd(
    #     truncation_threshold=truncation_threshold, D=D)

    #lam[1] ^ V[0]
    #V[1] ^ invx[0]
    #invx[1] ^ self.nodes[0][0]

    #absorb lam*V*invx into the left-most mps tensor
    self.nodes[0] = ncon([lam, V, invx, self.nodes[0]],
                         [[-1, 1], [1, 2], [2, 3], [3, -2, -3]])

    #absorb connector * invy * U * lam into the right-most tensor
    #Note that lam is absorbed here, which means that the state
    #is in the parallel decomposition
    #Note that we absorb connector_matrix here
    self.nodes[-1] = ncon([self.get_node(len(self) - 1), invy, U, lam],
                          [[-1, -2, 1], [1, 2], [2, 3], [3, -3]])
    #now do a sweep of QR decompositions to bring the mps tensors into
    #left canonical form (except the last one)
    self.position(len(self) - 2)
    # Z = norm(self.nodes[-1])
    # Z = ncon([self.nodes[-1], conj(self.nodes[-1])],
    #          [[1, 2, 3], [1, 2, 3]]) / np.sum(self.D[-1])
    #self._tensors[-1] /= np.sqrt(Z)

    #TODO: lam is a diagonal matrix, but we're not making use of it the moment
    lam_norm = self.backend.norm(lam.tensor)
    lam.tensor /= lam_norm
    self.center_position = len(self) - 1
    self._connector = (1.0 / lam).diag()
    self._right_mat = lam.diag()
    self._norm = self.dtype.type(1)


class FiniteMPS(BaseMPS):
  """
  An MPS class for finite systems. 

  MPS tensors are stored as a list of `Node` objects in the `FiniteMPS.nodes`
  attribute.
  `FiniteMPS` has a central site, also called orthogonality center. 
  The position of this central site is stored in `FiniteMPS.center_position`, 
  and it can be be shifted using the `FiniteMPS.position` method. 
  `FiniteMPS.position` uses QR and RQ methods to shift `center_position`.
  
  `FiniteMPS` can be initialized either from a `list` of tensors, or
  by calling the classmethod `FiniteMPS.random`.
  
  By default, `FiniteMPS` is initialized in *canonical* form, i.e.
  the state is normalized, and all tensors to the left of 
  `center_position` are left orthogonal, and all tensors 
  to the right of `center_position` are right orthogonal. The tensor
  at `FiniteMPS.center_position` is neither left nor right orthogonal.

  Note that canonicalization can be computationally relatively 
  costly and scales :math:`\\propto ND^3`.
  """

  def __init__(self,
               tensors: List[Union[BaseNode, Tensor]],
               center_position: Optional[int] = 0,
               canonicalize: Optional[bool] = True,
               backend: Optional[Text] = None) -> None:
    """
    Initialize a FiniteMPS.
    Args:
      tensors: A list of `Tensor` or `BaseNode` objects.
      center_position: The initial position of the center site.
      canonicalize: If `True` the mps is canonicalized at initialization.
      backend: The name of the backend that should be used to perform 
        contractions. Available backends are currently 'numpy', 'tensorflow',
        'pytorch', 'jax'
    """

    super().__init__(
        tensors=tensors,
        center_position=center_position,
        connector_matrix=None,
        backend=backend)
    if canonicalize:
      if center_position == 0:
        self.center_position = len(self) - 1
        self.position(center_position)
      elif center_position == len(self) - 1:
        self.center_position = 0
        self.position(center_position)
      else:
        self.center_position = 0
        self.position(len(self) - 1)
        self.position(center_position)

  @classmethod
  def random(cls,
             d: List[int],
             D: List[int],
             dtype: Type[np.number],
             backend: Optional[Text] = None):
    """
    Initialize a random `FiniteMPS`. The resulting state
    is normalized. Its center-position is at 0.

    Args:
      d: A list of physical dimensions.
      D: A list of bond dimensions.
      dtype: A numpy dtype.
      backend: An optional backend.
    Returns:
      `FiniteMPS`
    """
    #use numpy backend for tensor initialization
    be = backend_factory.get_backend('numpy')
    if len(D) != len(d) - 1:
      raise ValueError('len(D) = {} is different from len(d) - 1 = {}'.format(
          len(D),
          len(d) - 1))
    D = [1] + D + [1]
    tensors = [
        be.randn((D[n], d[n], D[n + 1]), dtype=dtype) for n in range(len(d))
    ]
    return cls(tensors=tensors, center_position=0, backend=backend)

  def canonicalize(self, normalize: Optional[bool] = True) -> np.number:
    """
    Bring the MPS into canonical form according to `FiniteMPS.center_position`.

    Assuming nothing about the content of the current tensors, brings the
    tensors into canonical form with a center site at
    `FiniteMPS.center_position`.

    Args:
      normalize: If `True`, normalize matrices when shifting.
    Returns:
      `Tensor`: The norm of the MPS.
    """
    pos = self.center_position
    self.position(0, normalize=False)
    self.position(len(self.nodes) - 1, normalize=False)
    return self.position(pos, normalize=normalize)

  def check_canonical(self) -> Tensor:
    """
    Check whether the MPS is in the expected canonical form.
    Returns:
      The L2 norm of the vector of local deviations.
    """
    deviations = []
    for site in range(len(self.nodes)):
      if site < self.center_position:
        deviation = self.check_orthonormality('l', site)
      elif site > self.center_position:
        deviation = self.check_orthonormality('r', site)
      else:
        continue
      deviations.append(deviation**2)
    return self.backend.sqrt(sum(deviations))

  def left_envs(self, sites: Sequence[int]) -> Dict:
    """
    Compute left reduced density matrices for site `sites`.
    This returns a dict `left_envs` mapping sites (int) to Tensors.
    `left_envs[site]` is the left-reduced density matrix to the left of
    site `site`.

    Args:
      sites (list of int): A list of sites of the MPS.
    Returns:
      `dict` mapping `int` to `Tensor`: The left-reduced density matrices 
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
      left_envs[site] = Node(
          self.backend.eye(N=self.nodes[site].shape[0], dtype=self.dtype),
          backend=self.backend.name)

    # left reduced density matrices at sites > center_position
    # have to be calculated from a network contraction
    if n2 > self.center_position:
      nodes = {}
      conj_nodes = {}
      for site in range(self.center_position, n2):
        nodes[site] = Node(self.nodes[site], backend=self.backend.name)
        conj_nodes[site] = conj(self.nodes[site])

      nodes[self.center_position][0] ^ conj_nodes[self.center_position][0]
      nodes[self.center_position][1] ^ conj_nodes[self.center_position][1]

      for site in range(self.center_position + 1, n2):
        nodes[site][0] ^ nodes[site - 1][2]
        conj_nodes[site][0] ^ conj_nodes[site - 1][2]
        nodes[site][1] ^ conj_nodes[site][1]

      edges = {site: node[2] for site, node in nodes.items()}
      conj_edges = {site: node[2] for site, node in conj_nodes.items()}

      left_env = contract_between(nodes[self.center_position],
                                  conj_nodes[self.center_position])
      left_env.reorder_edges(
          [edges[self.center_position], conj_edges[self.center_position]])
      if self.center_position + 1 in sites:
        left_envs[self.center_position + 1] = left_env
      for site in range(self.center_position + 1, n2):
        left_env = contract_between(left_env, nodes[site])
        left_env = contract_between(left_env, conj_nodes[site])
        if site + 1 in sites:
          left_env.reorder_edges([edges[site], conj_edges[site]])
          left_envs[site + 1] = left_env
    return left_envs

  def right_envs(self, sites: Sequence[int]) -> Dict:
    """
    Compute right reduced density matrices for site `sites.
    This returns a dict `right_envs` mapping sites (int) to Tensors.
    `right_envs[site]` is the right-reduced density matrix to the right of
    site `site`.

    Args:
      sites (list of int): A list of sites of the MPS.
    Returns:
      `dict` mapping `int` to `Tensor`: The right-reduced density matrices 
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
      right_envs[site] = Node(
          self.backend.eye(N=self.nodes[site].shape[2], dtype=self.dtype),
          backend=self.backend.name)

    # right reduced density matrices at sites < center_position
    # have to be calculated from a network contraction
    if n1 < self.center_position:
      nodes = {}
      conj_nodes = {}
      for site in reversed(range(n1 + 1, self.center_position + 1)):
        nodes[site] = Node(self.nodes[site], backend=self.backend.name)
        conj_nodes[site] = conj(self.nodes[site])

      nodes[self.center_position][2] ^ conj_nodes[self.center_position][2]
      nodes[self.center_position][1] ^ conj_nodes[self.center_position][1]

      for site in reversed(range(n1 + 1, self.center_position)):
        nodes[site][2] ^ nodes[site + 1][0]
        conj_nodes[site][2] ^ conj_nodes[site + 1][0]
        nodes[site][1] ^ conj_nodes[site][1]

      edges = {site: node[0] for site, node in nodes.items()}
      conj_edges = {site: node[0] for site, node in conj_nodes.items()}

      right_env = contract_between(nodes[self.center_position],
                                   conj_nodes[self.center_position])
      if self.center_position - 1 in sites:
        right_env.reorder_edges(
            [edges[self.center_position], conj_edges[self.center_position]])
        right_envs[self.center_position - 1] = right_env
      for site in reversed(range(n1 + 1, self.center_position)):
        right_env = contract_between(right_env, nodes[site])
        right_env = contract_between(right_env, conj_nodes[site])
        if site - 1 in sites:
          right_env.reorder_edges([edges[site], conj_edges[site]])
          right_envs[site - 1] = right_env

    return right_envs

  def save(self, path: str):
    raise NotImplementedError()
