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
import numpy as np
import warnings
from typing import Optional, Any, Tuple
from tensornetwork.block_sparse.blocksparse_utils import (
    _find_transposed_diagonal_sparse_blocks)
from tensornetwork.block_sparse.utils import get_real_dtype
from tensornetwork.block_sparse.sizetypes import SIZE_T
from tensornetwork.block_sparse.blocksparsetensor import (BlockSparseTensor,
                                                          ChargeArray)
Tensor = Any


def svd(
    bt,
    tensor: BlockSparseTensor,
    pivot_axis: int,
    max_singular_values: Optional[int] = None,
    max_truncation_error: Optional[float] = None,
    relative: Optional[bool] = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
  """
  Computes the singular value decomposition (SVD) of a tensor.
  See tensornetwork.backends.tensorflow.decompositions for details.
  """

  left_dims = tensor.shape[:pivot_axis]
  right_dims = tensor.shape[pivot_axis:]

  matrix = bt.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])

  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  u_blocks = []
  singvals = []
  v_blocks = []
  for n, b in enumerate(blocks):
    out = np.linalg.svd(
        np.reshape(matrix.data[b], shapes[:, n]),
        full_matrices=False,
        compute_uv=True)
    u_blocks.append(out[0])
    singvals.append(out[1])
    v_blocks.append(out[2])

  orig_num_singvals = np.int64(np.sum([len(s) for s in singvals]))
  orig_block_size = [len(s) for s in singvals]
  discarded_singvals = np.zeros(0, dtype=get_real_dtype(tensor.dtype))
  if (max_singular_values
      is not None) and (max_singular_values >= orig_num_singvals):
    max_singular_values = None

  if (max_truncation_error is not None) or (max_singular_values is not None):
    max_D = np.max([len(s) for s in singvals]) if len(singvals) > 0 else 0

    #extend singvals of all blocks into a matrix by padding each block with 0
    if len(singvals) > 0:
      extended_singvals = np.stack([
          np.append(s, np.zeros(max_D - len(s), dtype=s.dtype))
          for s in singvals
      ],
                                   axis=1)
    else:
      extended_singvals = np.empty((0, 0), dtype=get_real_dtype(tensor.dtype))

    extended_flat_singvals = np.ravel(extended_singvals)
    #sort singular values
    inds = np.argsort(extended_flat_singvals, kind='stable')
    discarded_inds = np.zeros(0, dtype=SIZE_T)
    if inds.shape[0] > 0:
      maxind = inds[-1]
    else:
      maxind = 0
    if max_truncation_error is not None:
      if relative and (len(singvals) > 0):
        max_truncation_error = max_truncation_error * np.max(
            [s[0] for s in singvals])

      kept_inds_mask = np.sqrt(
          np.cumsum(np.square(
              extended_flat_singvals[inds]))) > max_truncation_error
      trunc_inds_mask = np.logical_not(kept_inds_mask)
      discarded_inds = inds[trunc_inds_mask]
      inds = inds[kept_inds_mask]
    if max_singular_values is not None:
      #if the original number of non-zero singular values
      #is smaller than `max_singular_values` we need to reset
      #`max_singular_values` (we were filling in 0.0 into singular
      #value blocks to facilitate trunction steps, thus we could end up
      #with more singular values than originally there).
      if max_singular_values > orig_num_singvals:
        max_singular_values = orig_num_singvals
      if max_singular_values < len(inds):
        discarded_inds = np.append(discarded_inds,
                                   inds[:(-1) * max_singular_values])
        inds = inds[(-1) * max_singular_values::]

    if len(inds) == 0:
      #special case of truncation to 0 dimension;
      warnings.warn("svd_decomposition truncated to 0 dimensions. "
                    "Adjusting to `max_singular_values = 1`")
      inds = np.asarray([maxind])

    if extended_singvals.shape[1] > 0:
      #pylint: disable=no-member
      keep = np.divmod(inds, extended_singvals.shape[1])
      disc = np.divmod(discarded_inds, extended_singvals.shape[1])
    else:
      keep = (np.zeros(1, dtype=SIZE_T), np.zeros(1, dtype=SIZE_T))
      disc = (np.zeros(0, dtype=SIZE_T), np.zeros(0, dtype=SIZE_T))
    newsingvals = [
        extended_singvals[keep[0][keep[1] == n], keep[1][keep[1] == n]][::-1]
        for n in range(extended_singvals.shape[1])
    ]
    discsingvals = [
        extended_singvals[disc[0][disc[1] == n], disc[1][disc[1] == n]][::-1]
        for n in range(extended_singvals.shape[1])
    ]
    new_block_size = [len(s) for s in newsingvals]
    discsingvals = [
        d[:(orig_block_size[n] - new_block_size[n])]
        for n, d in enumerate(discsingvals)
    ]
    singvals = newsingvals
    discarded_singvals = discsingvals
  if len(singvals) > 0:
    left_singval_charge_labels = np.concatenate([
        np.full(singvals[n].shape[0], fill_value=n, dtype=np.int16)
        for n in range(len(singvals))
    ])
    all_singvals = np.concatenate(singvals)
    #define the new charges on the two central bonds
    left_charge_labels = np.concatenate([
        np.full(len(singvals[n]), fill_value=n, dtype=np.int16)
        for n in range(len(u_blocks))
    ])
    right_charge_labels = np.concatenate([
        np.full(len(singvals[n]), fill_value=n, dtype=np.int16)
        for n in range(len(v_blocks))
    ])
    all_ublocks = np.concatenate([
        np.ravel(np.transpose(u_blocks[n][:, 0:len(singvals[n])]))
        for n in range(len(u_blocks))
    ])
    all_vblocks = np.concatenate([
        np.ravel(v_blocks[n][0:len(singvals[n]), :])
        for n in range(len(v_blocks))
    ])
  else:
    left_singval_charge_labels = np.empty(0, dtype=np.int16)
    all_singvals = np.empty(0, dtype=get_real_dtype(tensor.dtype))
    left_charge_labels = np.empty(0, dtype=np.int16)
    right_charge_labels = np.empty(0, dtype=np.int16)
    all_ublocks = np.empty(0, dtype=get_real_dtype(tensor.dtype))
    all_vblocks = np.empty(0, dtype=get_real_dtype(tensor.dtype))

  if len(discarded_singvals) > 0:
    tmp_labels = [
        np.full(discarded_singvals[n].shape[0], fill_value=n, dtype=np.int16)
        for n in range(len(discarded_singvals))
    ]
    left_discarded_singval_charge_labels = np.concatenate(tmp_labels)
    all_discarded_singvals = np.concatenate(discarded_singvals)

  else:
    left_discarded_singval_charge_labels = np.empty(0, dtype=np.int16)
    all_discarded_singvals = np.empty(0, dtype=get_real_dtype(tensor.dtype))


  left_singval_charge = charges[left_singval_charge_labels]
  S = ChargeArray(all_singvals, [left_singval_charge], [False])

  left_discarded_singval_charge = charges[left_discarded_singval_charge_labels]
  Sdisc = ChargeArray(all_discarded_singvals, [left_discarded_singval_charge],
                      [False])

  new_left_charge = charges[left_charge_labels]
  new_right_charge = charges[right_charge_labels]

  #get the indices of the new tensors U,S and V
  charges_u = [new_left_charge] + [matrix._charges[o] for o in matrix._order[0]]
  order_u = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
  flows_u = [True] + [matrix._flows[o] for o in matrix._order[0]]
  charges_v = [new_right_charge
              ] + [matrix._charges[o] for o in matrix._order[1]]
  flows_v = [False] + [matrix._flows[o] for o in matrix._order[1]]
  order_v = [[0]] + [list(np.arange(1, len(matrix._order[1]) + 1))]

  #We fill in data into the transposed U
  U = BlockSparseTensor(
      all_ublocks,
      charges=charges_u,
      flows=flows_u,
      order=order_u,
      check_consistency=False).transpose((1, 0))

  V = BlockSparseTensor(
      all_vblocks,
      charges=charges_v,
      flows=flows_v,
      order=order_v,
      check_consistency=False)
  left_shape = left_dims + (S.shape[0],)
  right_shape = (S.shape[0],) + right_dims
  return U.reshape(left_shape), S, V.reshape(right_shape), Sdisc


def qr(bt, tensor: BlockSparseTensor, pivot_axis: int) -> Tuple[Tensor, Tensor]:
  """Computes the QR decomposition of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:pivot_axis]
  right_dims = tensor.shape[pivot_axis:]
  tensor = bt.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])
  q, r = bt.qr(tensor)
  center_dim = q.shape[1]
  q = bt.reshape(q, list(left_dims) + [center_dim])
  r = bt.reshape(r, [center_dim] + list(right_dims))
  return q, r


def rq(bt, tensor: BlockSparseTensor, pivot_axis: int) -> Tuple[Tensor, Tensor]:
  """Computes the RQ (reversed QR) decomposition of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:pivot_axis]
  right_dims = tensor.shape[pivot_axis:]
  tensor = bt.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])
  q, r = bt.qr(bt.conj(bt.transpose(tensor, (1, 0))))
  r, q = bt.conj(bt.transpose(r, (1, 0))), bt.conj(bt.transpose(
      q, (1, 0)))  #M=r*q at this point
  center_dim = r.shape[1]
  r = bt.reshape(r, list(left_dims) + [center_dim])
  q = bt.reshape(q, [center_dim] + list(right_dims))
  return r, q
