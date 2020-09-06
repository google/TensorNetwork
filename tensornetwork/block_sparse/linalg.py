# copyright 2019 The TensorNetwork Authors
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
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import (BlockSparseTensor,
                                                          ChargeArray,
                                                          tensordot)
from tensornetwork.block_sparse.utils import (intersect, flatten,
                                              get_real_dtype, _randn, _random)
from tensornetwork.block_sparse.blocksparse_utils import (
    _find_transposed_diagonal_sparse_blocks, _find_diagonal_sparse_blocks,
    compute_num_nonzero, compute_sparse_lookup)
from typing import List, Union, Any, Tuple, Type, Optional, Text, Sequence
from tensornetwork.block_sparse.initialization import empty_like

def norm(tensor: BlockSparseTensor) -> float:
  """
  The norm of the tensor.
  """
  return np.linalg.norm(tensor.data)


def diag(tensor: ChargeArray) -> Any:
  """
  Return a diagonal `BlockSparseTensor` from a `ChargeArray`, or 
  return the diagonal of a `BlockSparseTensor` as a `ChargeArray`.
  For input of type `BlockSparseTensor`:
    The full diagonal is obtained from finding the diagonal blocks of the 
    `BlockSparseTensor`, taking the diagonal elements of those and packing
    the result into a ChargeArray. Note that the computed diagonal elements 
    are usually different from the  diagonal elements obtained from 
    converting the `BlockSparseTensor` to dense storage and taking the diagonal.
    Note that the flow of the resulting 1d `ChargeArray` object is `False`.
  Args:
    tensor: A `ChargeArray`.
  Returns:
    ChargeArray: A 1d `CharggeArray` containing the diagonal of `tensor`, 
      or a diagonal matrix of type `BlockSparseTensor` containing `tensor` 
      on its diagonal.

  """
  if tensor.ndim > 2:
    raise ValueError("`diag` currently only implemented for matrices, "
                     "found `ndim={}".format(tensor.ndim))
  if not isinstance(tensor, BlockSparseTensor):
    if tensor.ndim > 1:
      raise ValueError(
          "`diag` currently only implemented for `ChargeArray` with ndim=1, "
          "found `ndim={}`".format(tensor.ndim))
    flat_charges = tensor._charges + tensor._charges
    flat_flows = list(tensor._flows) + list(np.logical_not(tensor._flows))
    flat_order = list(tensor.flat_order) + list(
        np.asarray(tensor.flat_order) + len(tensor._charges))
    tr_partition = len(tensor._order[0])
    blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
        flat_charges, flat_flows, tr_partition, flat_order)
    data = np.zeros(
        np.int64(np.sum(np.prod(shapes, axis=0))), dtype=tensor.dtype)
    lookup, unique, labels = compute_sparse_lookup(tensor._charges,
                                                   tensor._flows, charges)
    for n, block in enumerate(blocks):
      label = labels[np.nonzero(unique == charges[n])[0][0]]
      data[block] = np.ravel(
          np.diag(tensor.data[np.nonzero(lookup == label)[0]]))

    order = [
        tensor._order[0],
        list(np.asarray(tensor._order[0]) + len(tensor._charges))
    ]
    new_charges = [tensor._charges[0].copy(), tensor._charges[0].copy()]
    return BlockSparseTensor(
        data,
        charges=new_charges,
        flows=list(tensor._flows) + list(np.logical_not(tensor._flows)),
        order=order,
        check_consistency=False)

  flat_charges = tensor._charges
  flat_flows = tensor._flows
  flat_order = tensor.flat_order
  tr_partition = len(tensor._order[0])
  sparse_blocks, charges, block_shapes = _find_transposed_diagonal_sparse_blocks(  #pylint: disable=line-too-long
      flat_charges, flat_flows, tr_partition, flat_order)

  shapes = np.min(block_shapes, axis=0)
  if len(sparse_blocks) > 0:
    data = np.concatenate([
        np.diag(np.reshape(tensor.data[sparse_blocks[n]], block_shapes[:, n]))
        for n in range(len(sparse_blocks))
    ])
    charge_labels = np.concatenate([
        np.full(shapes[n], fill_value=n, dtype=np.int16)
        for n in range(len(sparse_blocks))
    ])

  else:
    data = np.empty(0, dtype=tensor.dtype)
    charge_labels = np.empty(0, dtype=np.int16)
  newcharges = [charges[charge_labels]]
  flows = [False]
  return ChargeArray(data, newcharges, flows)


def reshape(tensor: ChargeArray, shape: Sequence[Union[Index,
                                                       int]]) -> ChargeArray:
  """
  Reshape `tensor` into `shape.
  `ChargeArray.reshape` works the same as the dense 
  version, with the notable exception that the tensor can only be 
  reshaped into a form compatible with its elementary shape. 
  The elementary shape is the shape determined by ChargeArray._charges.
  For example, while the following reshaping is possible for regular 
  dense numpy tensor,
  ```
  A = np.random.rand(6,6,6)
  np.reshape(A, (2,3,6,6))
  ```
  the same code for ChargeArray
  ```
  q1 = U1Charge(np.random.randint(0,10,6))
  q2 = U1Charge(np.random.randint(0,10,6))
  q3 = U1Charge(np.random.randint(0,10,6))
  i1 = Index(charges=q1,flow=False)
  i2 = Index(charges=q2,flow=True)
  i3 = Index(charges=q3,flow=False)
  A = ChargeArray.randn(indices=[i1,i2,i3])
  print(A.shape) #prints (6,6,6)
  A.reshape((2,3,6,6)) #raises ValueError
  ```
  raises a `ValueError` since (2,3,6,6)
  is incompatible with the elementary shape (6,6,6) of the tensor.
  
  Args:
    tensor: A symmetric tensor.
    shape: The new shape. Can either be a list of `Index` 
      or a list of `int`.
  Returns:
    ChargeArray: A new tensor reshaped into `shape`
  """

  return tensor.reshape(shape)


def conj(tensor: ChargeArray) -> ChargeArray:
  """
  Return the complex conjugate of `tensor` in a new 
  `ChargeArray`.
  Args:
    tensor: A `ChargeArray` object.
  Returns:
    ChargeArray
  """
  return tensor.conj()


def transpose(tensor: ChargeArray,
              order: Sequence[int] = np.asarray([1, 0]),
              shuffle: Optional[bool] = False) -> ChargeArray:
  """
  Transpose the tensor into the new order `order`. If `shuffle=False`
  no data-reshuffling is done.
  Args:
    order: The new order of indices.
    shuffle: If `True`, reshuffle data.
  Returns:
    ChargeArray: The transposed tensor.
  """
  return tensor.transpose(order, shuffle)


def svd(matrix: BlockSparseTensor,
        full_matrices: Optional[bool] = True,
        compute_uv: Optional[bool] = True,
        hermitian: Optional[bool] = False) -> Any:
  """
  Compute the singular value decomposition of `matrix`.
  The matrix if factorized into `u * s * vh`, with 
  `u` and `vh` the left and right singular vectors of `matrix`,
  and `s` its singular values.
  Args:
    matrix: A matrix (i.e. an order-2 tensor) of type  `BlockSparseTensor`
    full_matrices: If `True`, expand `u` and `v` to square matrices
      If `False` return the "economic" svd, i.e. `u.shape[1]=s.shape[0]`
      and `v.shape[0]=s.shape[1]`
    compute_uv: If `True`, return `u` and `v`.
    hermitian: If `True`, assume hermiticity of `matrix`.
  Returns:
    If `compute_uv` is `True`: Three BlockSparseTensors `U,S,V`.
    If `compute_uv` is `False`: A BlockSparseTensors `S` containing the 
      singular values.
  """

  if matrix.ndim != 2:
    raise NotImplementedError("svd currently supports only tensors of order 2.")

  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  u_blocks = []
  singvals = []
  v_blocks = []
  for n, block in enumerate(blocks):
    out = np.linalg.svd(
        np.reshape(matrix.data[block], shapes[:, n]), full_matrices, compute_uv,
        hermitian)
    if compute_uv:
      u_blocks.append(out[0])
      singvals.append(out[1])
      v_blocks.append(out[2])

    else:
      singvals.append(out)

  tmp_labels = [
      np.full(len(singvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(singvals))
  ]
  if len(tmp_labels) > 0:
    left_singval_charge_labels = np.concatenate(tmp_labels)
  else:

    left_singval_charge_labels = np.empty(0, dtype=np.int16)
  left_singval_charge = charges[left_singval_charge_labels]
  if len(singvals) > 0:
    all_singvals = np.concatenate(singvals)
  else:
    all_singvals = np.empty(0, dtype=get_real_dtype(matrix.dtype))
  S = ChargeArray(all_singvals, [left_singval_charge], [False])

  if compute_uv:
    #define the new charges on the two central bonds
    tmp_left_labels = [
        np.full(u_blocks[n].shape[1], fill_value=n, dtype=np.int16)
        for n in range(len(u_blocks))
    ]
    if len(tmp_left_labels) > 0:
      left_charge_labels = np.concatenate(tmp_left_labels)
    else:
      left_charge_labels = np.empty(0, dtype=np.int16)

    tmp_right_labels = [
        np.full(v_blocks[n].shape[0], fill_value=n, dtype=np.int16)
        for n in range(len(v_blocks))
    ]
    if len(tmp_right_labels) > 0:
      right_charge_labels = np.concatenate(tmp_right_labels)
    else:
      right_charge_labels = np.empty(0, dtype=np.int16)
    new_left_charge = charges[left_charge_labels]
    new_right_charge = charges[right_charge_labels]

    charges_u = [new_left_charge
                ] + [matrix._charges[o] for o in matrix._order[0]]
    order_u = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
    flows_u = [True] + [matrix._flows[o] for o in matrix._order[0]]
    charges_v = [new_right_charge
                ] + [matrix._charges[o] for o in matrix._order[1]]
    flows_v = [False] + [matrix._flows[o] for o in matrix._order[1]]
    order_v = [[0]] + [list(np.arange(1, len(matrix._order[1]) + 1))]
    # We fill in data into the transposed U
    # note that transposing is essentially free
    if len(u_blocks) > 0:
      all_u_blocks = np.concatenate([np.ravel(u.T) for u in u_blocks])
      all_v_blocks = np.concatenate([np.ravel(v) for v in v_blocks])
    else:
      all_u_blocks = np.empty(0, dtype=matrix.dtype)
      all_v_blocks = np.empty(0, dtype=matrix.dtype)

    return BlockSparseTensor(
        all_u_blocks,
        charges=charges_u,
        flows=flows_u,
        order=order_u,
        check_consistency=False).transpose((1, 0)), S, BlockSparseTensor(
            all_v_blocks,
            charges=charges_v,
            flows=flows_v,
            order=order_v,
            check_consistency=False)

  return S


def qr(matrix: BlockSparseTensor, mode: Text = 'reduced') -> Any:
  """
  Compute the qr decomposition of an `M` by `N` matrix `matrix`.
  The matrix is factorized into `q*r`, with 
  `q` an orthogonal matrix and `r` an upper triangular matrix.
  Args:
    matrix: A matrix (i.e. a rank-2 tensor) of type  `BlockSparseTensor`
    mode : Can take values {'reduced', 'complete', 'r', 'raw'}.
    If K = min(M, N), then

    * 'reduced'  : returns q, r with dimensions (M, K), (K, N) (default)
    * 'complete' : returns q, r with dimensions (M, M), (M, N)
    * 'r'        : returns r only with dimensions (K, N)

  Returns:
    (BlockSparseTensor,BlockSparseTensor): If mode = `reduced` or `complete`
    BlockSparseTensor: If mode = `r`.
  """
  if matrix.ndim != 2:
    raise NotImplementedError("qr currently supports only rank-2 tensors.")
  if mode not in ('reduced', 'complete', 'raw', 'r'):
    raise ValueError('unknown value {} for input `mode`'.format(mode))
  if mode == 'raw':
    raise NotImplementedError('mode `raw` currenntly not supported')

  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  q_blocks = []
  r_blocks = []
  for n, block in enumerate(blocks):
    out = np.linalg.qr(np.reshape(matrix.data[block], shapes[:, n]), mode)
    if mode in ('reduced', 'complete'):
      q_blocks.append(out[0])
      r_blocks.append(out[1])
    else:
      r_blocks.append(out)


  tmp_r_charge_labels = [
      np.full(r_blocks[n].shape[0], fill_value=n, dtype=np.int16)
      for n in range(len(r_blocks))
  ]
  if len(tmp_r_charge_labels) > 0:
    left_r_charge_labels = np.concatenate(tmp_r_charge_labels)
  else:
    left_r_charge_labels = np.empty(0, dtype=np.int16)

  left_r_charge = charges[left_r_charge_labels]
  charges_r = [left_r_charge] + [matrix._charges[o] for o in matrix._order[1]]
  flows_r = [False] + [matrix._flows[o] for o in matrix._order[1]]
  order_r = [[0]] + [list(np.arange(1, len(matrix._order[1]) + 1))]
  if len(r_blocks) > 0:
    all_r_blocks = np.concatenate([np.ravel(r) for r in r_blocks])
  else:
    all_r_blocks = np.empty(0, dtype=matrix.dtype)
  R = BlockSparseTensor(
      all_r_blocks,
      charges=charges_r,
      flows=flows_r,
      order=order_r,
      check_consistency=False)

  if mode in ('reduced', 'complete'):
    tmp_right_q_charge_labels = [
        np.full(q_blocks[n].shape[1], fill_value=n, dtype=np.int16)
        for n in range(len(q_blocks))
    ]
    if len(tmp_right_q_charge_labels) > 0:
      right_q_charge_labels = np.concatenate(tmp_right_q_charge_labels)
    else:
      right_q_charge_labels = np.empty(0, dtype=np.int16)

    right_q_charge = charges[right_q_charge_labels]
    charges_q = [
        right_q_charge,
    ] + [matrix._charges[o] for o in matrix._order[0]]
    order_q = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
    flows_q = [True] + [matrix._flows[o] for o in matrix._order[0]]
    if len(q_blocks) > 0:
      all_q_blocks = np.concatenate([np.ravel(q.T) for q in q_blocks])
    else:
      all_q_blocks = np.empty(0, dtype=matrix.dtype)
    return BlockSparseTensor(
        all_q_blocks,
        charges=charges_q,
        flows=flows_q,
        order=order_q,
        check_consistency=False).transpose((1, 0)), R
  return R

def eigh(matrix: BlockSparseTensor,
         UPLO: Optional[Text] = 'L') -> Tuple[ChargeArray, BlockSparseTensor]:
  """
  Compute the eigen decomposition of a hermitian `M` by `M` matrix `matrix`.
  Args:
    matrix: A matrix (i.e. a rank-2 tensor) of type  `BlockSparseTensor`

  Returns:
    (ChargeArray,BlockSparseTensor): The eigenvalues and eigenvectors

  """
  if matrix.ndim != 2:
    raise NotImplementedError("eigh currently supports only rank-2 tensors.")

  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  eigvals = []
  v_blocks = []
  for n, block in enumerate(blocks):
    e, v = np.linalg.eigh(np.reshape(matrix.data[block], shapes[:, n]), UPLO)
    eigvals.append(e)
    v_blocks.append(v)

  tmp_labels = [
      np.full(len(eigvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(eigvals))
  ]
  if len(tmp_labels) > 0:
    eigvalscharge_labels = np.concatenate(tmp_labels)
  else:
    eigvalscharge_labels = np.empty(0, dtype=np.int16)
  eigvalscharge = charges[eigvalscharge_labels]
  if len(eigvals) > 0:
    all_eigvals = np.concatenate(eigvals)
  else:
    all_eigvals = np.empty(0, dtype=get_real_dtype(matrix.dtype))
  E = ChargeArray(all_eigvals, [eigvalscharge], [False])
  charges_v = [eigvalscharge] + [matrix._charges[o] for o in matrix._order[0]]
  order_v = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
  flows_v = [True] + [matrix._flows[o] for o in matrix._order[0]]
  if len(v_blocks) > 0:
    all_v_blocks = np.concatenate([np.ravel(v.T) for v in v_blocks])
  else:
    all_v_blocks = np.empty(0, dtype=matrix.dtype)
  V = BlockSparseTensor(
      all_v_blocks,
      charges=charges_v,
      flows=flows_v,
      order=order_v,
      check_consistency=False).transpose()

  return E, V  #pytype: disable=bad-return-type


def eig(matrix: BlockSparseTensor) -> Tuple[ChargeArray, BlockSparseTensor]:
  """
  Compute the eigen decomposition of an `M` by `M` matrix `matrix`.
  Args:
    matrix: A matrix (i.e. a rank-2 tensor) of type  `BlockSparseTensor`

  Returns:
    (ChargeArray,BlockSparseTensor): The eigenvalues and eigenvectors

  """
  if matrix.ndim != 2:
    raise NotImplementedError("eig currently supports only rank-2 tensors.")

  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  eigvals = []
  v_blocks = []
  for n, block in enumerate(blocks):
    e, v = np.linalg.eig(np.reshape(matrix.data[block], shapes[:, n]))
    eigvals.append(e)
    v_blocks.append(v)
  tmp_labels = [
      np.full(len(eigvals[n]), fill_value=n, dtype=np.int16)
      for n in range(len(eigvals))
  ]
  if len(tmp_labels) > 0:
    eigvalscharge_labels = np.concatenate(tmp_labels)
  else:
    eigvalscharge_labels = np.empty(0, dtype=np.int16)

  eigvalscharge = charges[eigvalscharge_labels]

  if len(eigvals) > 0:
    all_eigvals = np.concatenate(eigvals)
  else:
    all_eigvals = np.empty(0, dtype=get_real_dtype(matrix.dtype))

  E = ChargeArray(all_eigvals, [eigvalscharge], [False])
  charges_v = [eigvalscharge] + [matrix._charges[o] for o in matrix._order[0]]
  order_v = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
  flows_v = [True] + [matrix._flows[o] for o in matrix._order[0]]
  if len(v_blocks) > 0:
    all_v_blocks = np.concatenate([np.ravel(v.T) for v in v_blocks])
  else:
    all_v_blocks = np.empty(0, dtype=matrix.dtype)

  V = BlockSparseTensor(
      all_v_blocks,
      charges=charges_v,
      flows=flows_v,
      order=order_v,
      check_consistency=False).transpose()

  return E, V  #pytype: disable=bad-return-type


def inv(matrix: BlockSparseTensor) -> BlockSparseTensor:
  """
  Compute the matrix inverse of `matrix`.
  Returns:
    BlockSparseTensor: The inverse of `matrix`.
  """
  if matrix.ndim != 2:
    raise ValueError("`inv` can only be taken for matrices, "
                     "found tensor.ndim={}".format(matrix.ndim))
  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, _, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  data = np.empty(np.sum(np.prod(shapes, axis=0)), dtype=matrix.dtype)
  for n, block in enumerate(blocks):
    data[block] = np.ravel(
        np.linalg.inv(np.reshape(matrix.data[block], shapes[:, n])).T)
  #pylint: disable=line-too-long
  return BlockSparseTensor(
      data=data,
      charges=matrix._charges,
      flows=np.logical_not(matrix._flows),
      order=matrix._order,
      check_consistency=False).transpose((1, 0))  #pytype: disable=bad-return-type


def sqrt(
    tensor: Union[BlockSparseTensor, ChargeArray]
) -> Union[ChargeArray, BlockSparseTensor]:
  obj = tensor.__new__(type(tensor))
  obj.__init__(
      np.sqrt(tensor.data),
      charges=tensor._charges,
      flows=tensor._flows,
      order=tensor._order,
      check_consistency=False)
  return obj


def eye(column_index: Index,
        row_index: Optional[Index] = None,
        dtype: Optional[Type[np.number]] = None) -> BlockSparseTensor:
  """
  Return an identity matrix.
  Args:
    column_index: The column index of the matrix.
    row_index: The row index of the matrix.
    dtype: The dtype of the matrix.
  Returns:
    BlockSparseTensor
  """
  if row_index is None:
    row_index = column_index.copy().flip_flow()
  if dtype is None:
    dtype = np.float64

  blocks, _, shapes = _find_diagonal_sparse_blocks(
      column_index.flat_charges + row_index.flat_charges,
      column_index.flat_flows + row_index.flat_flows,
      len(column_index.flat_charges))
  data = np.empty(np.int64(np.sum(np.prod(shapes, axis=0))), dtype=dtype)
  for n, block in enumerate(blocks):
    data[block] = np.ravel(np.eye(shapes[0, n], shapes[1, n], dtype=dtype))
  order = [list(np.arange(0, len(column_index.flat_charges)))] + [
      list(
          np.arange(
              len(column_index.flat_charges),
              len(column_index.flat_charges) + len(row_index.flat_charges)))
  ]
  return BlockSparseTensor(
      data=data,
      charges=column_index.flat_charges + row_index.flat_charges,
      flows=column_index.flat_flows + row_index.flat_flows,
      order=order,
      check_consistency=False)


def trace(tensor: BlockSparseTensor,
          axes: Optional[Sequence[int]] = None) -> BlockSparseTensor:
  """
  Compute the trace of a matrix or tensor. If input has `ndim>2`, take
  the trace over the last two dimensions.
  Args:
    tensor: A `BlockSparseTensor`.
    axes: The axes over which the trace should be computed.
      Defaults to the last two indices of the tensor.
  Returns:
    BlockSparseTensor: The result of taking the trace.
  """
  if tensor.ndim > 1:
    if axes is None:
      axes = (tensor.ndim - 2, tensor.ndim - 1)
    if len(axes) != 2:
      raise ValueError(f"`len(axes)` has to be 2, found `axes = {axes}`")
    if not np.array_equal(tensor.flows[axes[0]],
                          np.logical_not(tensor.flows[axes[1]])):
      raise ValueError(
          f"trace indices for axes {axes} have non-matching flows.")

    sparse_shape = tensor.sparse_shape
    if sparse_shape[axes[0]].copy().flip_flow() != sparse_shape[axes[1]]:
      raise ValueError(f"trace indices for axes {axes} are not matching")

    #flatten the shape of `tensor`
    out = tensor.reshape(
        flatten([[tensor._charges[n].dim for n in o] for o in tensor._order]))
    _, _, labels0 = np.intersect1d(
        tensor._order[axes[0]], flatten(out._order), return_indices=True)
    _, _, labels1 = np.intersect1d(
        tensor._order[axes[1]], flatten(out._order), return_indices=True)

    a0 = list(labels0[np.argsort(tensor._order[axes[0]])])
    a1 = list(labels1[np.argsort(tensor._order[axes[1]])])

    while len(a0) > 0:
      i = a0.pop(0)
      j = a1.pop(0)
      identity = eye(
          Index([out._charges[out._order[i][0]]],
                [not out._flows[out._order[i][0]]]))
      #pylint: disable=line-too-long
      out = tensordot(out, identity, ([i, j], [0, 1]))  # pytype: disable=wrong-arg-types
      a0ar = np.asarray(a0)

      mask_min = a0ar > np.min([i, j])
      mask_max = a0ar > np.max([i, j])
      a0ar[np.logical_and(mask_min, mask_max)] -= 2
      a0ar[np.logical_xor(mask_min, mask_max)] -= 1

      a1ar = np.asarray(a1)
      mask_min = a1ar > np.min([i, j])
      mask_max = a1ar > np.max([i, j])
      a1ar[np.logical_and(mask_min, mask_max)] -= 2
      a1ar[np.logical_xor(mask_min, mask_max)] -= 1
      a0 = list(a0ar)
      a1 = list(a1ar)
    if out.ndim == 0:
      return out.item()
    return out  # pytype: disable=bad-return-type
  raise ValueError("trace can only be taken for tensors with ndim > 1")


def pinv(matrix: BlockSparseTensor,
         rcond: Optional[float] = 1E-15,
         hermitian: Optional[bool] = False) -> BlockSparseTensor:
  """
  Compute the Moore-Penrose pseudo inverse of `matrix`.
  Args:
    rcond: Pseudo inverse cutoff.
  Returns:
    BlockSparseTensor: The pseudo inverse of `matrix`.
  """
  if matrix.ndim != 2:
    raise ValueError("`pinv` can only be taken for matrices, "
                     "found tensor.ndim={}".format(matrix.ndim))

  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, _, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  data = np.empty(np.sum(np.prod(shapes, axis=0)), dtype=matrix.dtype)
  for n, block in enumerate(blocks):
    data[block] = np.ravel(
        np.linalg.pinv(
            np.reshape(matrix.data[block], shapes[:, n]),
            rcond=rcond,
            hermitian=hermitian).T)
  #pylint: disable=line-too-long
  return BlockSparseTensor(
      data=data,
      charges=matrix._charges,
      flows=np.logical_not(matrix._flows),
      order=matrix._order,
      check_consistency=False).transpose((1, 0)) #pytype: disable=bad-return-type

def abs(tensor: BlockSparseTensor) -> BlockSparseTensor: #pylint: disable=redefined-builtin
  result = empty_like(tensor)
  result.data = np.abs(tensor.data)
  return result

def sign(tensor: BlockSparseTensor) -> BlockSparseTensor:
  result = empty_like(tensor)
  result.data = np.sign(tensor.data)
  return result
