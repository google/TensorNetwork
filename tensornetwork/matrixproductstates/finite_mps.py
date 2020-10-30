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
import functools

from tensornetwork.network_components import Node, contract_between
from tensornetwork.backends import backend_factory
from tensornetwork.linalg.node_linalg import conj
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
from tensornetwork.matrixproductstates.base_mps import BaseMPS
from tensornetwork.backends.abstract_backend import AbstractBackend
Tensor = Any


class FiniteMPS(BaseMPS):
  """An MPS class for finite systems.


  MPS tensors are stored as a list.
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
               tensors: List[Tensor],
               center_position: Optional[int] = None,
               canonicalize: Optional[bool] = True,
               backend: Optional[Union[AbstractBackend, Text]] = None) -> None:
    """Initialize a `FiniteMPS`. If `canonicalize` is `True` the state
       is brought into canonical form, with `BaseMPS.center_position`
       at `center_position`. if `center_position` is `None` and
       `canonicalize = True`, `BaseMPS.center_position` is set to 0.

    Args:
      tensors: A list of `Tensor` objects.
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
    if (center_position is not None) and (not canonicalize):
      raise ValueError("can only set center_position of canonical mps")
    if canonicalize:
      if center_position is None:
        center_position = 0
      if center_position == len(self) - 1:
        self.center_position = len(self) - 1
        self.position(0)
        self.position(len(self) - 1)
      else:
        self.center_position = 0
        self.position(len(self) - 1)
        self.position(center_position)

  @classmethod
  def random(
      cls,
      d: List[int],
      D: List[int],
      dtype: Type[np.number],
      canonicalize: bool = True,
      backend: Optional[Union[AbstractBackend, Text]] = None) -> "FiniteMPS":
    """Initialize a random `FiniteMPS`. The resulting state is normalized. Its
    center-position is at 0.

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
    return cls(
        tensors=tensors,
        center_position=None,
        canonicalize=canonicalize,
        backend=backend)

  # pylint: disable=arguments-differ
  def canonicalize(self, normalize: bool = True) -> np.number:
    """Bring the MPS into canonical form according to
    `center_position`. If `center_position` is `None`, the
    MPS is canonicalized with `center_position = 0`.

    Args:
      normalize: If `True`, normalize matrices when shifting the
        orthogonality center.
    Returns:
      `Tensor`: The norm of the MPS.
    """
    N = len(self.tensors)
    if self.center_position is not None:
      pos = self.center_position
      if pos >= N // 2:
        self.center_position = 0
        self.position(N - 1, normalize=normalize)
      else:
        self.center_position = len(self.tensors) - 1
        self.position(0, normalize=normalize)
      return self.position(pos, normalize=normalize)

    self.center_position = len(self.tensors) - 1
    return self.position(0, normalize=normalize)

  def check_canonical(self) -> Tensor:
    """Check whether the MPS is in the expected canonical form.

    Returns:
      The L2 norm of the vector of local deviations.
    """
    if self.center_position is None:
      raise ValueError(
          "FiniteMPS.center_positions is `None`. Cannot check canonical form.")
    deviations = []
    for site in range(len(self.tensors)):
      if site < self.center_position:
        deviation = self.check_orthonormality('l', site)
      elif site > self.center_position:
        deviation = self.check_orthonormality('r', site)
      else:
        continue
      deviations.append(deviation**2)
    return self.backend.sqrt(sum(deviations[1:], deviations[0]))

  def left_envs(self, sites: Sequence[int]) -> Dict:
    """Compute left reduced density matrices for site `sites`. This returns a
    dict `left_envs` mapping sites (int) to Tensors. `left_envs[site]` is the
    left-reduced density matrix to the left of site `site`.

    Args:
      sites (list of int): A list of sites of the MPS.
    Returns:
      `dict` mapping `int` to `Tensor`: The left-reduced density matrices
        at each  site in `sites`.
    """
    sites = np.array(sites)  #enable logical indexing
    if len(sites) == 0:
      return {}
    if self.center_position is not None:
      center_position = self.center_position
    else:
      center_position = 0

    n2 = np.max(sites)

    #check if all elements of `sites` are within allowed range
    if not np.all(sites <= len(self)):
      raise ValueError('all elements of `sites` have to be <= N = {}'.format(
          len(self)))
    if not np.all(sites >= 0):
      raise ValueError('all elements of `sites` have to be positive')

    # left-reduced density matrices to the left of `center_position`
    # (including center_position) are all identities
    left_sites = sites[sites <= center_position]
    left_envs = {}
    for site in left_sites:
      left_envs[site] = Node(
          self.backend.eye(
              N=self.backend.sparse_shape(
                  self.backend.conj(self.tensors[site]))[0],
              dtype=self.dtype),
          backend=self.backend)

    # left reduced density matrices at sites > center_position
    # have to be calculated from a network contraction
    if n2 > center_position:
      nodes = {}
      conj_nodes = {}
      for site in range(center_position, n2):
        nodes[site] = Node(self.tensors[site], backend=self.backend)
        conj_nodes[site] = conj(nodes[site])

      nodes[center_position][0] ^ conj_nodes[center_position][0]
      nodes[center_position][1] ^ conj_nodes[center_position][1]

      for site in range(center_position + 1, n2):
        nodes[site][0] ^ nodes[site - 1][2]
        conj_nodes[site][0] ^ conj_nodes[site - 1][2]
        nodes[site][1] ^ conj_nodes[site][1]

      edges = {site: node[2] for site, node in nodes.items()}
      conj_edges = {site: node[2] for site, node in conj_nodes.items()}

      left_env = contract_between(nodes[center_position],
                                  conj_nodes[center_position])
      left_env.reorder_edges(
          [edges[center_position], conj_edges[center_position]])
      if center_position + 1 in sites:
        left_envs[center_position + 1] = left_env
      for site in range(center_position + 1, n2):
        left_env = contract_between(left_env, nodes[site])
        left_env = contract_between(left_env, conj_nodes[site])
        if site + 1 in sites:
          left_env.reorder_edges([edges[site], conj_edges[site]])
          left_envs[site + 1] = left_env
    return {k: v.tensor for k, v in left_envs.items()}

  def right_envs(self, sites: Sequence[int]) -> Dict:
    """Compute right reduced density matrices for site `sites. This returns a
    dict `right_envs` mapping sites (int) to Tensors. `right_envs[site]` is the
    right-reduced density matrix to the right of site `site`.

    Args:
      sites (list of int): A list of sites of the MPS.
    Returns:
      `dict` mapping `int` to `Tensor`: The right-reduced density matrices
        at each  site in `sites`.
    """
    sites = np.array(sites)
    if len(sites) == 0:
      return {}

    if self.center_position is not None:
      center_position = self.center_position
    else:
      center_position = len(self.tensors) - 1

    n1 = np.min(sites)
    #check if all elements of `sites` are within allowed range
    if not np.all(sites < len(self)):
      raise ValueError('all elements of `sites` have to be < N = {}'.format(
          len(self)))
    if not np.all(sites >= -1):
      raise ValueError('all elements of `sites` have to be >= -1')

    # right-reduced density matrices to the right of `center_position`
    # (including center_position) are all identities
    right_sites = sites[sites >= center_position]
    right_envs = {}
    for site in right_sites:
      right_envs[site] = Node(
          self.backend.eye(
              N=self.backend.sparse_shape(
                  self.backend.conj(self.tensors[site]))[2],
              dtype=self.dtype),
          backend=self.backend)

    # right reduced density matrices at sites < center_position
    # have to be calculated from a network contraction
    if n1 < center_position:
      nodes = {}
      conj_nodes = {}
      for site in reversed(range(n1 + 1, center_position + 1)):
        nodes[site] = Node(self.tensors[site], backend=self.backend)
        conj_nodes[site] = conj(nodes[site])

      nodes[center_position][2] ^ conj_nodes[center_position][2]
      nodes[center_position][1] ^ conj_nodes[center_position][1]

      for site in reversed(range(n1 + 1, center_position)):
        nodes[site][2] ^ nodes[site + 1][0]
        conj_nodes[site][2] ^ conj_nodes[site + 1][0]
        nodes[site][1] ^ conj_nodes[site][1]

      edges = {site: node[0] for site, node in nodes.items()}
      conj_edges = {site: node[0] for site, node in conj_nodes.items()}

      right_env = contract_between(nodes[center_position],
                                   conj_nodes[center_position])
      if center_position - 1 in sites:
        right_env.reorder_edges(
            [edges[center_position], conj_edges[center_position]])
        right_envs[center_position - 1] = right_env
      for site in reversed(range(n1 + 1, center_position)):
        right_env = contract_between(right_env, nodes[site])
        right_env = contract_between(right_env, conj_nodes[site])
        if site - 1 in sites:
          right_env.reorder_edges([edges[site], conj_edges[site]])
          right_envs[site - 1] = right_env
    return {k: v.tensor for k, v in right_envs.items()}

  def save(self, path: str):
    raise NotImplementedError()
