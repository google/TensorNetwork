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
"""Tensor Decomposition Numpy Implementation."""

from typing import Optional, Any, Tuple
import numpy
Tensor = Any


def svd(
    np,  # TODO: Typing
    tensor: Tensor,
    pivot_axis: int,
    max_singular_values: Optional[int] = None,
    max_truncation_error: Optional[float] = None,
    relative: Optional[bool] = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
  """Computes the singular value decomposition (SVD) of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:pivot_axis]
  right_dims = tensor.shape[pivot_axis:]

  tensor = np.reshape(tensor, [numpy.prod(left_dims), numpy.prod(right_dims)])
  u, s, vh = np.linalg.svd(tensor, full_matrices=False)

  if max_singular_values is None:
    max_singular_values = np.size(s)

  if max_truncation_error is not None:
    # Cumulative norms of singular values in ascending order.
    trunc_errs = np.sqrt(np.cumsum(np.square(s[::-1])))
    # If relative is true, rescale max_truncation error with the largest
    # singular value to yield the absolute maximal truncation error.
    if relative:
      abs_max_truncation_error = max_truncation_error * s[0]
    else:
      abs_max_truncation_error = max_truncation_error
    # We must keep at least this many singular values to ensure the
    # truncation error is <= abs_max_truncation_error.
    num_sing_vals_err = np.count_nonzero(
        (trunc_errs > abs_max_truncation_error).astype(np.int32))
  else:
    num_sing_vals_err = max_singular_values

  num_sing_vals_keep = min(max_singular_values, num_sing_vals_err)

  # tf.svd() always returns the singular values as a vector of float{32,64}.
  # since tf.math_ops.real is automatically applied to s. This causes
  # s to possibly not be the same dtype as the original tensor, which can cause
  # issues for later contractions. To fix it, we recast to the original dtype.
  s = s.astype(tensor.dtype)

  s_rest = s[num_sing_vals_keep:]
  s = s[:num_sing_vals_keep]
  u = u[:, :num_sing_vals_keep]
  vh = vh[:num_sing_vals_keep, :]

  dim_s = s.shape[0]
  u = np.reshape(u, list(left_dims) + [dim_s])
  vh = np.reshape(vh, [dim_s] + list(right_dims))

  return u, s, vh, s_rest


def qr(
    np,  # TODO: Typing
    tensor: Tensor,
    pivot_axis: int,
    non_negative_diagonal: bool
) -> Tuple[Tensor, Tensor]:
  """Computes the QR decomposition of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:pivot_axis]
  right_dims = tensor.shape[pivot_axis:]
  tensor = np.reshape(tensor, [numpy.prod(left_dims), numpy.prod(right_dims)])
  q, r = np.linalg.qr(tensor)
  if non_negative_diagonal:
    phases = np.sign(np.diagonal(r))
    q = q * phases
    r = phases.conj()[:, None] * r
  center_dim = q.shape[1]
  q = np.reshape(q, list(left_dims) + [center_dim])
  r = np.reshape(r, [center_dim] + list(right_dims))
  return q, r


def rq(
    np,  # TODO: Typing
    tensor: Tensor,
    pivot_axis: int,
    non_negative_diagonal: bool
) -> Tuple[Tensor, Tensor]:
  """Computes the RQ (reversed QR) decomposition of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:pivot_axis]
  right_dims = tensor.shape[pivot_axis:]
  tensor = np.reshape(tensor, [numpy.prod(left_dims), numpy.prod(right_dims)])
  q, r = np.linalg.qr(np.conj(np.transpose(tensor)))
  if non_negative_diagonal:
    phases = np.sign(np.diagonal(r))
    q = q * phases
    r = phases.conj()[:, None] * r
  r, q = np.conj(np.transpose(r)), np.conj(
      np.transpose(q))  #M=r*q at this point
  center_dim = r.shape[1]
  r = np.reshape(r, list(left_dims) + [center_dim])
  q = np.reshape(q, [center_dim] + list(right_dims))
  return r, q
