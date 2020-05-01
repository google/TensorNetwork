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
# pylint: disable=line-too-long
from tensornetwork.network_components import Node, contract, contract_between, BaseNode
# pylint: disable=line-too-long
from tensornetwork.network_operations import split_node_qr, split_node_rq, split_node_full_svd, norm, conj, switch_backend
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
from tensornetwork.backends.base_backend import BaseBackend
Tensor = Any


class BaseMPS:
  """The base class for MPS. All MPS should be derived from BaseMPS `BaseMPS`
  is an infinite matrix product state with a finite unitcell.

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
               backend: Optional[Union[Text, BaseBackend]] = None) -> None:
    """Initialize a BaseMPS.

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
    """Shift `FiniteMPS.center_position` to `site`.

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
    else:
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
    """A list of bond dimensions of `BaseMPS`"""
    return [self.nodes[0].shape[0]] + [node.shape[2] for node in self.nodes]

  @property
  def physical_dimensions(self) -> List:
    """A list of physical Hilbert-space dimensions of `BaseMPS`"""

    return [node.shape[1] for node in self.nodes]

  def switch_backend(self, new_backend: Text) -> None:
    """Change the backend of all the nodes in the MPS.

    Args:
      new_backend (str): The new backend.
    """
    switch_backend(self.nodes, new_backend)

  def right_envs(self, sites: Sequence[int]) -> Dict:
    raise NotImplementedError()

  def left_envs(self, sites: Sequence[int]) -> Dict:
    raise NotImplementedError()

  def apply_transfer_operator(self, site: int, direction: Union[Text, int],
                              matrix: Union[BaseNode, Tensor]) -> BaseNode:
    """Compute the action of the MPS transfer-operator at site `site`.

    Args:
      site: a site of the MPS
      direction:
        * if `1, 'l'` or `'left'`: compute the left-action
          of the MPS transfer-operator at `site` on the input `matrix`.
        * if `-1, 'r'` or `'right'`: compute the right-action
          of the MPS transfer-operator at `site` on the input `matrix`
      matrix: A rank-2 tensor or matrix.
    Returns:
      `Node`: The result of applying the MPS transfer-operator to `matrix`
    """
    mat = Node(matrix, backend=self.backend)
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
    else:
      raise ValueError("direction must be one of {1, 'l', 'left', "
                       "'-1', 'r', 'right'}")
    result = mat @ node @ conj_node
    return result.reorder_edges(edge_order)

  def measure_local_operator(self, ops: List[Union[BaseNode, Tensor]],
                             sites: Sequence[int]) -> List:
    """Measure the expectation value of local operators `ops` site `sites`.

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
      O = Node(ops[n], backend=self.backend)
      R = right_envs[site]
      L = left_envs[site]
      A = Node(self.nodes[site], backend=self.backend)
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
    between `site1` and all sites `s` in `sites2`. If `s == site1`, 
    `op2[s]` will be applied first.

    Args:
      op1: Tensor of rank 2; the local operator at `site1`.
      op2: Tensor of rank 2; the local operator at `sites2`.
      site1: The site where `op1`  acts
      sites2: Sites where operator `op2` acts.
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
    # then all correlators <op1(site1) op2(site2)> with site2 >= site1

    # get all sites smaller than site1
    left_sites = np.sort(sites2[sites2 < site1])
    # get all sites larger than site1
    right_sites = np.sort(sites2[sites2 > site1])

    # compute all neccessary right reduced
    # density matrices in one go. This is
    # more efficient than calling right_envs
    # for each site individually
    rs = self.right_envs(
        np.append(site1, np.mod(right_sites, N)).astype(np.int64))
    ls = self.left_envs(
        np.append(np.mod(left_sites, N), site1).astype(np.int64))

    c = []
    if len(left_sites) > 0:

      A = Node(self.nodes[site1], backend=self.backend)
      O1 = Node(op1, backend=self.backend)
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
          A = Node(self.nodes[n % N], backend=self.backend)
          conj_A = conj(A)
          O2 = Node(op2, backend=self.backend)
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
      O1 = Node(op1, backend=self.backend)
      O2 = Node(op2, backend=self.backend)
      L = ls[site1]
      R = rs[site1]
      A = Node(self.nodes[site1], backend=self.backend)
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
    if len(right_sites) > 0:
      A = Node(self.nodes[site1], backend=self.backend)
      conj_A = conj(A)
      L = ls[site1]
      O1 = Node(op1, backend=self.backend)
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
          A = Node(self.nodes[n % N], backend=self.backend)
          conj_A = conj(A)
          O2 = Node(op2, backend=self.backend)
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
    """Apply a two-site gate to an MPS. This routine will in general destroy
    any canonical form of the state. If a canonical form is needed, the user
    can restore it using `FiniteMPS.position`.

    Args:
      gate: A two-body gate.
      site1: The first site where the gate acts.
      site2: The second site where the gate acts.
      max_singular_values: The maximum number of singular values to keep.
      max_truncation_err: The maximum allowed truncation error.

    Returns:
      `Tensor`: A scalar tensor containing the truncated weight of the
        truncation.
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
          'Found site2 ={}, site1={}. Only nearest neighbor gates are currently '
          'supported'.format(site2, site1))

    if (max_singular_values or
        max_truncation_err) and self.center_position not in (site1, site2):
      raise ValueError(
          'center_position = {}, but gate is applied at sites {}, {}. '
          'Truncation should only be done if the gate '
          'is applied at the center position of the MPS'.format(
              self.center_position, site1, site2))

    gate_node = Node(gate, backend=self.backend)

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
    """Apply a one-site gate to an MPS. This routine will in general destroy
    any canonical form of the state. If a canonical form is needed, the user
    can restore it using `FiniteMPS.position`

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
    gate_node = Node(gate, backend=self.backend)
    gate_node[1] ^ self.nodes[site][1]
    edge_order = [self.nodes[site][0], gate_node[0], self.nodes[site][2]]
    self.nodes[site] = contract_between(
        gate_node, self.nodes[site],
        name=self.nodes[site].name).reorder_edges(edge_order)

  def check_orthonormality(self, which: Text, site: int) -> Tensor:
    """Check orthonormality of tensor at site `site`.

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
    n1 = self.get_node(site)  #we need to absorb the connector_matrix
    n2 = conj(n1)
    if which in ('l', 'left'):
      n1[0] ^ n2[0]
      n1[1] ^ n2[1]
    else:
      n1[2] ^ n2[2]
      n1[1] ^ n2[1]
    result = n1 @ n2
    return self.backend.norm(
        abs(result.tensor - self.backend.eye(
            N=result.shape[0], M=result.shape[1], dtype=self.dtype)))

  def check_canonical(self) -> Tensor:
    """Check whether the MPS is in the expected canonical form.

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

  def get_node(self, site: int) -> BaseNode:
    """Returns the `Node` object at `site`.

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

  def canonicalize(self, *args, **kwargs) -> np.number:
    raise NotImplementedError()
