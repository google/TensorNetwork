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
from typing import Any, Sequence, List, Set, Optional, Union, Text, Tuple, Type, Dict
Tensor = Any


class FiniteMPS(tensornetwork.TensorNetwork):
  """
    An MPS class for finite systems.
    """

  def __init__(self,
               tensors: List[Tensor],
               center_position: int,
               name: Optional[Text] = None,
               backend: Optional[Text] = None,
               dtype: Optional[Type[np.number]] = None):

    super().__init__(backend=backend, dtype=dtype)
    self.nodes = [
        self.add_node(tensors[n], name='node{}'.format(n))
        for n in range(len(tensors))
    ]  #redundant?!
    for site in range(len(self.nodes) - 1):
      self.connect(self.nodes[site][2], self.nodes[site + 1][0])
    self.center_position = center_position

  def position(self, site: int, normalize: Optional[bool] = True) -> None:
    if site >= len(self.nodes) or site < 0:
      raise ValueError('site = {} not between values'
                       ' 0 < site < N = {}'.format(site, len(self)))
    if site == self.center_position:
      return
    elif site > self.center_position:
      n = self.center_position
      for n in range(self.center_position, site):
        Q, R = self.split_node_qr(
            self.nodes[n],
            left_edges=[self.nodes[n][0], self.nodes[n][1]],
            right_edges=[self.nodes[n][2]],
            left_name=self.nodes[n].name)
        Z = self.backend.norm(R.tensor)
        if normalize:
          R.tensor /= Z
        self.nodes[n] = Q
        self.nodes[n + 1] = self.contract(R[1], name=self.nodes[n + 1].name)
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

      self.center_position = site
    return Z

  def check_orthonormality(self, which: Text, site: int) -> Tensor:
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
    if not np.all(np.array(sites) < len(self)):
      raise ValueError('all elements of `sites` have to be < N={}'.format(
          len(self)))
    if not np.all(np.array(sites) >= 0):
      raise ValueError('all elements of `sites` have to be positive')

    left_envs = {}
    for site in range(n1, min(self.center_position + 1, n2 + 1)):
      if site in sites:
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

      left_env = net.contract_between(nodes[self.center_position],
                                      conj_nodes[self.center_position])
      if (self.center_position + 1) in sites:
        left_envs[self.center_position + 1] = left_env.tensor
      for site in range(self.center_position + 1, n2):
        left_env = net.contract_between(left_env, nodes[site])
        left_env = net.contract_between(left_env, conj_nodes[site])
        if site + 1 in sites:
          left_envs[site + 1] = left_env.tensor
    return left_envs

  def __len__(self):
    return len(self.nodes)
