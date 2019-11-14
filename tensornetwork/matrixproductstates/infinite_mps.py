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
from tensornetwork.matrixproductstates.base_mps import BaseMPS
Tensor = Any


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
    return eta[0], Node(result, backend=self.backend)

  def right_envs(self, sites: Sequence[int]) -> Dict:
    raise NotImplementedError()

  def left_envs(self, sites: Sequence[int]) -> Dict:
    raise NotImplementedError()

  def save(self, path: str):
    raise NotImplementedError()
