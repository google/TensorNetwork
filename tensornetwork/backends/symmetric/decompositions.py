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
"""Tensor Decomposition Implementations."""

from typing import Optional, Any, Tuple
from tensornetwork.block_tensor.block_tensor import _find_diagonal_sparse_blocks, BlockSparseTensor
from tensornetwork.block_tensor.index import Index
import numpy as np
import warnings
Tensor = Any


def svd_decomposition(
    bt,  # TODO: Typing
    tensor: BlockSparseTensor,
    split_axis: int,
    max_singular_values: Optional[int] = None,
    max_truncation_error: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
  """Computes the singular value decomposition (SVD) of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:split_axis]
  right_dims = tensor.shape[split_axis:]

  matrix = bt.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])

  flat_charges = matrix.indices[0]._charges + matrix.indices[1]._charges
  flat_flows = matrix.flat_flows
  partition = len(matrix.indices[0].flat_charges)
  blocks, charges, shapes = _find_diagonal_sparse_blocks(
      flat_charges, flat_flows, partition)

  u_blocks = []
  singvals = []
  v_blocks = []
  for n in range(len(blocks)):
    out = np.linalg.svd(
        np.reshape(matrix.data[blocks[n]], shapes[:, n]),
        full_matrices=True,
        compute_uv=True)
    u_blocks.append(out[0])
    singvals.append(out[1])
    v_blocks.append(out[2])
  discarded_singvals = np.zeros(0, dtype=singvals[0].dtype)
  if (max_truncation_error is not None) or (max_singular_values is not None):

    max_D = np.max([len(s) for s in singvals])
    #fill with zeros
    extended_singvals = np.stack([
        np.append(s, np.zeros(max_D - len(s), dtype=s.dtype)) for s in singvals
    ],
                                 axis=1)
    extended_flat_singvals = np.ravel(extended_singvals)

    inds = np.argsort(extended_flat_singvals)
    discarded_inds = np.zeros(0, dtype=np.uint32)

    maxind = inds[-1]
    if max_truncation_error is not None:
      kept_inds_mask = np.cumsum(np.square(
          extended_flat_singvals[inds])) > max_truncation_error
      trunc_inds_mask = np.logical_not(kept_inds_mask)
      discarded_inds = inds[trunc_inds_mask]
      inds = inds[kept_inds_mask]

    if max_singular_values is not None:
      if max_singular_values < len(inds):
        discarded_inds = np.append(discarded_inds, inds[:-max_singular_values])
        inds = inds[-max_singular_values::]

    if len(inds) == 0:
      #special case of truncation to 0 dimension;
      warnings.warn("svd_decomposition truncated to 0 dimenions. "
                    "Adjusting to `max_singular_values = 1`")
      inds = np.asarray([maxind])
    keep = np.divmod(inds, extended_singvals.shape[1])
    singvals = [
        extended_singvals[keep[0][keep[1] == n], keep[1][keep[1] == n]]
        for n in range(extended_singvals.shape[1])
    ]
    discarded_singvals = extended_flat_singvals[discarded_inds]

  left_singval_charge_labels = np.concatenate([
      np.full(singvals[n].shape[0], fill_value=n, dtype=np.int16)
      for n in range(len(singvals))
  ])

  right_singval_charge_labels = np.concatenate([
      np.full(len(singvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(singvals))
  ])
  left_singval_charge = charges[left_singval_charge_labels]
  right_singval_charge = charges[right_singval_charge_labels]
  #Note: introducing a convention
  #TODO: think about this convention!
  indices_s = [
      Index(left_singval_charge, False),
      Index(right_singval_charge, True)
  ]
  S = BlockSparseTensor(
      np.concatenate([np.ravel(np.diag(s)) for s in singvals]), indices_s)
  #define the new charges on the two central bonds
  left_charge_labels = np.concatenate([
      np.full(len(singvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(u_blocks))
  ])
  right_charge_labels = np.concatenate([
      np.full(len(singvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(v_blocks))
  ])
  new_left_charge = charges[left_charge_labels]
  new_right_charge = charges[right_charge_labels]

  #get the indices of the new tensors U,S and V
  indices_u = [Index(new_left_charge, True), matrix.indices[0]]
  indices_v = [Index(new_right_charge, False), matrix.indices[1]]
  #We fill in data into the transposed U
  #TODO: reuse data from _find_diagonal_sparse_blocks above
  #to avoid the transpose

  U = BlockSparseTensor(
      np.concatenate([
          np.ravel(np.transpose(u_blocks[n][:, 0:len(singvals[n])]))
          for n in range(len(u_blocks))
      ]), indices_u).transpose((1, 0))

  V = BlockSparseTensor(
      np.concatenate([
          np.ravel(v_blocks[n][0:len(singvals[n]), :])
          for n in range(len(v_blocks))
      ]), indices_v)

  left_shape = left_dims + (S.shape[0],)
  right_shape = (S.shape[1],) + right_dims
  return U.reshape(left_shape), S, V.reshape(right_shape), discarded_singvals[
      discarded_singvals > 0.0]


def qr_decomposition(
    bt,  # TODO: Typing
    tensor: BlockSparseTensor,
    split_axis: int,
) -> Tuple[Tensor, Tensor]:
  """Computes the QR decomposition of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:split_axis]
  right_dims = tensor.shape[split_axis:]
  tensor = bt.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])
  q, r = bt.qr(tensor)
  center_dim = q.shape[1]
  q = bt.reshape(q, list(left_dims) + [center_dim])
  r = bt.reshape(r, [center_dim] + list(right_dims))
  return q, r


def rq_decomposition(
    bt,  # TODO: Typing
    tensor: BlockSparseTensor,
    split_axis: int,
) -> Tuple[Tensor, Tensor]:
  """Computes the RQ (reversed QR) decomposition of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:split_axis]
  right_dims = tensor.shape[split_axis:]
  tensor = bt.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])
  q, r = bt.qr(bt.conj(bt.transpose(tensor, (1, 0))))
  r, q = bt.conj(bt.transpose(r, (1, 0))), bt.conj(bt.transpose(
      q, (1, 0)))  #M=r*q at this point
  center_dim = r.shape[1]
  r = bt.reshape(r, list(left_dims) + [center_dim])
  q = bt.reshape(q, [center_dim] + list(right_dims))
  return r, q
