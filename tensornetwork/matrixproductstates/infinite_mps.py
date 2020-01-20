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
from tensornetwork.ncon_interface import ncon
Tensor = Any


class InfiniteMPS(BaseMPS):
  """An MPS class for infinite systems.

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
    """Initialize a InfiniteMPS.

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
    """Initialize a random `InfiniteMPS`. The resulting state is normalized.
    Its center-position is at 0.

    Args:
      d: A list of physical dimensions.
      D: A list of bond dimensions.
      dtype: A numpy dtype.
      backend: An optional backend.
    Returns:
      `InfiniteMPS`
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

  def transfer_matrix_eigs(self,
                           direction: Union[Text, int],
                           initial_state: Optional[Union[BaseNode,
                                                         Tensor]] = None,
                           precision: Optional[float] = 1E-10,
                           num_krylov_vecs: Optional[int] = 30,
                           maxiter: Optional[int] = None):
    """Compute the dominant eigenvector of the MPS transfer matrix.

    Ars:
      direction:
        * If `'1','l' or 'left'`: return the left dominant eigenvalue
          and eigenvector
        * If `'-1','r' or 'right'`: return the right dominant eigenvalue
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

  # pylint: disable=arguments-differ
  def canonicalize(self,
                   left_initial_state: Optional[Union[BaseNode, Tensor]] = None,
                   right_initial_state: Optional[Union[BaseNode,
                                                       Tensor]] = None,
                   precision: Optional[float] = 1E-10,
                   truncation_threshold: Optional[float] = 1E-15,
                   D: Optional[int] = None,
                   num_krylov_vecs: Optional[int] = 50,
                   maxiter: Optional[int] = 1000,
                   pseudo_inverse_cutoff: Optional[float] = None):
    """Canonicalize an InfiniteMPS (i.e. bring it into Schmidt-canonical form).

    Args:
      left_initial_state: An initial guess for the left eigenvector of
        the unit-cell mps transfer matrix
      right_initial_state: An initial guess for the right eigenvector of
        the unit-cell transfer matrix
      precision: The desired precision of the dominant eigenvalues (passed
        to InfiniteMPS.transfer_matrix_eigs)
      truncation_threshold: Truncation threshold for Schmidt-values at the
        boundaries of the mps.
      D: The maximum number of Schmidt values to be kept at the boundaries
        of the mps.
      num_krylov_vecs: Number of Krylov vectors to diagonalize transfer_matrix
      maxiter: Maximum number of iterations in `eigs`
      pseudo_inverse_cutoff: A cutoff for taking the Moore-Penrose pseudo-inverse
        of a matrix. Given the SVD of a matrix :math:`M=U S V`, the inverse is
        is computed as :math:`V^* S^{-1}_+ U^*`, where :math:`S^{-1}_+` equals
        `S^{-1}` for all values in `S` which are larger than `pseudo_inverse_cutoff`,
         and is 0 for all others.
    Returns:
      None
    """
    # bring center_position to 0
    self.position(0)
    # dtype of eta is the same as InfiniteMPS.dtype
    # this is assured in the backend.
    eta, l = self.transfer_matrix_eigs(
        direction='left',
        initial_state=left_initial_state,
        precision=precision,
        num_krylov_vecs=num_krylov_vecs,
        maxiter=maxiter)
    sqrteta = self.backend.sqrt(eta)
    self.nodes[0].tensor /= sqrteta

    # TODO: would be nice to do the algebra directly on the nodes here
    l.tensor /= self.backend.trace(l.tensor)
    l.tensor = (l.tensor +
                self.backend.transpose(self.backend.conj(l.tensor),
                                       (1, 0))) / 2.0

    # eigvals_left and u_left are both `Tensor` objects
    eigvals_left, u_left = self.backend.eigh(l.tensor)
    eigvals_left /= self.backend.norm(eigvals_left)
    if pseudo_inverse_cutoff:
      mask = eigvals_left <= pseudo_inverse_cutoff

    inveigvals_left = 1.0 / eigvals_left
    if pseudo_inverse_cutoff:
      inveigvals_left = self.backend.index_update(inveigvals_left, mask, 0.0)

    sqrtl = Node(
        ncon(
            [u_left, self.backend.diag(self.backend.sqrt(eigvals_left))],
            [[-2, 1], [1, -1]],
            backend=self.backend.name),
        backend=self.backend)
    inv_sqrtl = Node(
        ncon([
            self.backend.diag(self.backend.sqrt(inveigvals_left)),
            self.backend.conj(u_left)
        ], [[-2, 1], [-1, 1]],
             backend=self.backend.name),
        backend=self.backend)

    eta, r = self.transfer_matrix_eigs(
        direction='right',
        initial_state=right_initial_state,
        precision=precision,
        num_krylov_vecs=num_krylov_vecs,
        maxiter=maxiter)

    r.tensor /= self.backend.trace(r.tensor)
    r.tensor = (r.tensor +
                self.backend.transpose(self.backend.conj(r.tensor),
                                       (1, 0))) / 2.0
    # eigvals_left and u_left are both `Tensor` objects
    eigvals_right, u_right = self.backend.eigh(r.tensor)
    eigvals_right /= self.backend.norm(eigvals_right)
    if pseudo_inverse_cutoff:
      mask = eigvals_right <= pseudo_inverse_cutoff

    inveigvals_right = 1.0 / eigvals_right
    if pseudo_inverse_cutoff:
      inveigvals_right = self.backend.index_update(inveigvals_right, mask, 0.0)

    sqrtr = Node(
        ncon([u_right,
              self.backend.diag(self.backend.sqrt(eigvals_right))],
             [[-1, 1], [1, -2]],
             backend=self.backend.name),
        backend=self.backend)

    inv_sqrtr = Node(
        ncon([
            self.backend.diag(self.backend.sqrt(inveigvals_right)),
            self.backend.conj(u_right)
        ], [[-1, 1], [-2, 1]],
             backend=self.backend.name),
        backend=self.backend)

    tmp = Node(
        ncon([sqrtl, sqrtr], [[-1, 1], [1, -2]], backend=self.backend.name),
        backend=self.backend)
    U, lam, V, _ = split_node_full_svd(
        tmp, [tmp[0]], [tmp[1]],
        max_singular_values=D,
        max_truncation_err=truncation_threshold)
    # absorb lam*V*invx into the left-most mps tensor
    self.nodes[0] = ncon([lam, V, inv_sqrtr, self.nodes[0]],
                         [[-1, 1], [1, 2], [2, 3], [3, -2, -3]])

    # absorb connector * inv_sqrtl * U * lam into the right-most tensor
    # Note that lam is absorbed here, which means that the state
    # is in the parallel decomposition
    # Note that we absorb connector_matrix here
    self.nodes[-1] = ncon([self.get_node(len(self) - 1), inv_sqrtl, U, lam],
                          [[-1, -2, 1], [1, 2], [2, 3], [3, -3]])
    # now do a sweep of QR decompositions to bring the mps tensors into
    # left canonical form (except the last one)
    self.position(len(self) - 1)
    # TODO: lam is a diagonal matrix, but we're not making
    # use of it the moment
    lam_norm = self.backend.norm(lam.tensor)
    lam.tensor /= lam_norm
    self.center_position = len(self) - 1
    self.connector_matrix = Node(
        self.backend.inv(lam.tensor), backend=self.backend)

    return lam_norm
