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
import tensornetwork
import numpy as np
from typing import Any, List, Optional, Text, Type, Union
Tensor = Any


class FiniteMPS(tensornetwork.TensorNetwork):
  """
  An MPS class for finite systems.
  FiniteMPS keeps track of the nodes of the network by storing them in a list
  `FiniteMPS.nodes`. Any external changes to this list will corrupt the 
  mps. FiniteMPS has a central site. The position of this central site is
  stored in `FiniteMPS.center_position`. This center position can be  
  shifted using the `FiniteMPS.position` method. 
  If the state is initialized with `center_positon=0`, 
  then `FiniteMPS.position(len(FiniteMPS)-1)` shifts the center_position 
  to `len(FiniteMPS) - 1`. If the shift is a "left-shift", all sites that 
  are visited in between are left in left-orthogonal form. If the shift is a
  "right-shift", all sites that are visited in between are left in right-orthogonal
  form. For a random initial tensors `tensors`, doing one sweep from left to right 
  and a successive sweep from right to left brings the state into central canonical 
  form. In this state, all sites to the left of center_position are left orthogonal,
  and all sites to the right of center_position are right orthogonal.
  Due to efficiency reasons, the state upon initialization is usually NOT brought 
  into the central canonical form.
  """

  def __init__(self,
               tensors: List[Tensor],
               center_position: int,
               name: Optional[Text] = None,
               backend: Optional[Text] = None,
               dtype: Optional[Type[np.number]] = None) -> None:
    """
    Initialize a FiniteMPS.
    Args:
      tensors: A list of `Tensor` objects.
      center_position: The initial position of the center site.
      name: A name for the object.
      backend: The name of the backend that should be used to perform contractions.
        See documentation of TensorNetwork.__init__ for a list of supported
        backends.
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
    self.nodes = [
        self.add_node(tensors[n], name='node{}'.format(n))
        for n in range(len(tensors))
    ]  #redundant?!
    for site in range(len(self.nodes) - 1):
      self.connect(self.nodes[site][2], self.nodes[site + 1][0])
    self.center_position = center_position

  @property
  def D(self):
    """
    Return a list of bond dimensions of FiniteMPS
    """
    return [self.nodes[0].shape[0]] + [node.shape[2] for node in self.nodes]

  @property
  def d(self):
    """
    Return a list of physical Hilbert-space dimensions of FiniteMPS
    """

    return [node.shape[1] for node in self.nodes]

  def position(self, site: int, normalize: Optional[bool] = True) -> None:
    """
    Shift FiniteMPS.center_position to `site`.
    Args:
      site: The site to which FiniteMPS.center_position should be shifted
      normalize: If `True`, normalize matrices when shifting.
    Returns:
      Tensor: The norm of the tensor at FiniteMPS.center_position
    """

    if site >= len(self.nodes) or site < 0:
      raise ValueError('site = {} not between values'
                       ' 0 < site < N = {}'.format(site, len(self)))
    if site == self.center_position:
      return self.backend.norm(self.nodes[self.center_position].tensor)
    elif site > self.center_position:
      n = self.center_position
      for n in range(self.center_position, site):
        Q, R = self.split_node_qr(
            self.nodes[n],
            left_edges=[self.nodes[n][0], self.nodes[n][1]],
            right_edges=[self.nodes[n][2]],
            left_name=self.nodes[n].name)

        self.nodes[n] = Q
        self.nodes[n + 1] = self.contract(R[1], name=self.nodes[n + 1].name)
        Z = self.backend.norm(self.nodes[n + 1].tensor)
        if normalize:
          self.nodes[n + 1].tensor /= Z

      self.center_position = site

    elif site < self.center_position:
      for n in reversed(range(site + 1, self.center_position + 1)):
        R, Q = self.split_node_rq(
            self.nodes[n],
            left_edges=[self.nodes[n][0]],
            right_edges=[self.nodes[n][1], self.nodes[n][2]],
            right_name=self.nodes[n].name)
        Z = self.backend.norm(R.tensor)
        if normalize:
          R.tensor /= Z
        self.nodes[n] = Q
        self.nodes[n - 1] = self.contract(R[0], name=self.nodes[n - 1].name)
        Z = self.backend.norm(self.nodes[n - 1].tensor)
        if normalize:
          self.nodes[n - 1].tensor /= Z

      self.center_position = site
    return Z

  def check_orthonormality(self, which: Text, site: int) -> Tensor:
    """
    Check orthonormality of tensor at site `site`.
    Args:
      which: if in ('l','left'): check left orthogonality
             if in ('r','right'): check right orthogonality
      site:  the site of the tensor
    """
    net = tensornetwork.TensorNetwork(
        backend=self.backend.name, dtype=self.dtype)
    n1 = net.add_node(self.nodes[site].tensor)
    n2 = net.add_node(self.backend.conj(self.nodes[site].tensor))
    if which in ('l', 'left'):
      n1[0] ^ n2[0]
      n1[1] ^ n2[1]
    elif which in ('r', 'right'):
      n1[2] ^ n2[2]
      n1[1] ^ n2[1]
    result = net.contract_between(n1, n2).tensor
    return self.backend.norm(
        result - self.backend.eye(N=result.shape[0], M=result.shape[1]))

  def left_envs(self, sites: List[int]):
    n1 = min(sites)
    n2 = max(sites)
    sites = np.array(sites)
    if not np.all(sites <= len(self)):
      raise ValueError('all elements of `sites` have to be <= N = {}'.format(
          len(self)))
    if not np.all(sites >= 0):
      raise ValueError('all elements of `sites` have to be positive')

    left_sites = sites[sites <= self.center_position]

    left_envs = {}
    for site in left_sites:
      left_envs[site] = self.backend.eye(N=self.nodes[site].shape[0])

    if n2 > self.center_position:
      net = tensornetwork.TensorNetwork(
          backend=self.backend.name, dtype=self.dtype)
      nodes = {}
      conj_nodes = {}
      for site in range(self.center_position, n2):
        nodes[site] = net.add_node(self.nodes[site].tensor)
        conj_nodes[site] = net.add_node(
            self.backend.conj(self.nodes[site].tensor))
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
      if (self.center_position + 1) in sites:
        left_envs[self.center_position + 1] = left_env.tensor
      for site in range(self.center_position + 1, n2):
        left_env = net.contract_between(left_env, nodes[site])
        left_env = net.contract_between(left_env, conj_nodes[site])
        if site + 1 in sites:
          left_env.reorder_edges([edges[site], conj_edges[site]])
          left_envs[site + 1] = left_env.tensor
    return left_envs

  def right_envs(self, sites: List[int]):
    n1 = min(sites)
    n2 = max(sites)
    sites = np.array(sites)
    if not np.all(np.array(sites) < len(self)):
      raise ValueError('all elements of `sites` have to be < N = {}'.format(
          len(self)))
    if not np.all(np.array(sites) >= -1):
      raise ValueError('all elements of `sites` have to be >= -1')

    right_sites = sites[sites >= self.center_position]
    right_envs = {}
    for site in right_sites:
      right_envs[site] = self.backend.eye(N=self.nodes[site].shape[2])

    if n1 < self.center_position:
      net = tensornetwork.TensorNetwork(
          backend=self.backend.name, dtype=self.dtype)
      nodes = {}
      conj_nodes = {}
      for site in reversed(range(n1 + 1, self.center_position + 1)):
        nodes[site] = net.add_node(self.nodes[site].tensor)
        conj_nodes[site] = net.add_node(
            self.backend.conj(self.nodes[site].tensor))
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
      if (self.center_position - 1) in sites:
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
    return len(self.nodes)

  def transfer_operator(self, site: int, direction: Union[Text, int],
                        tensor: Tensor) -> Tensor:

    net = tensornetwork.TensorNetwork(
        backend=self.backend.name, dtype=self.dtype)
    mat = net.add_node(tensor)
    node = net.add_node(self.nodes[site].tensor)
    conj_node = net.add_node(self.backend.conj(self.nodes[site].tensor))
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
              site, len(self)))
    if site2 < 1 or site2 >= len(self):
      raise ValueError('site2 = {} is not between 1 <= site < N = {}'.format(
          site, len(self)))
    if site2 <= site1:
      raise ValueError('site2 = {} has to be larger than site2 = {}'.format(
          site2, site1))
    if site2 != site1 + 1:
      raise ValueError(
          'site2 ={} != site1={}. Only nearest neighbor gates are currenlty supported'
          .format(site2, site1))

    if (max_singular_values or
        max_truncation_err) and self.center_position not in (site1, site2):
      raise ValueError(
          'center_position = {}, but gate is applied at sites {}, {}. '
          'Truncation should only be done if the gate '
          'is applied at the center position of the MPS'.format(
              self.center_position, site1, site2))

    gate_node = self.add_node(gate)
    e1 = gate_node[2] ^ self.nodes[site1][1]
    e2 = gate_node[3] ^ self.nodes[site2][1]
    left_edges = [self.nodes[site1][0], gate_node[0]]
    right_edges = [gate_node[1], self.nodes[site2][2]]
    result = self.contract_between(self.nodes[site1], self.nodes[site2])
    result = self.contract_between(result, gate_node)
    U, S, V, tw = self.split_node_full_svd(
        result,
        left_edges=left_edges,
        right_edges=right_edges,
        max_singular_values=max_singular_values,
        max_truncation_err=max_truncation_err,
        left_name=[self.nodes[site1].name],
        right_name=[self.nodes[site2].name])
    V.reorder_edges([S[1]] + right_edges)
    left_edges = left_edges + [S[1]]
    self.nodes[site1] = self.contract_between(
        U, S, name=U.name).reorder_edges(left_edges)
    self.nodes[site2] = V
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
    e = gate_node[1] ^ self.nodes[site][1]
    edge_order = [self.nodes[site][0], gate_node[0], self.nodes[site][2]]
    self.nodes[site] = self.contract_between(
        gate_node, self.nodes[site],
        name=self.nodes[site].name).reorder_edges(edge_order)
