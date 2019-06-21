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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional, Tuple, Any

Tensor = Any


def svd_decomposition(tf: Any,
                      tensor: Tensor,
                      split_axis: int,
                      max_singular_values: Optional[int] = None,
                      max_truncation_error: Optional[float] = None
                     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
  """Computes the singular value decomposition (SVD) of a tensor.

  The SVD is performed by treating the tensor as a matrix, with an effective
  left (row) index resulting from combining the axes `tensor.shape[:split_axis]`
  and an effective right (column) index resulting from combining the axes
  `tensor.shape[split_axis:]`.

  For example, if `tensor` had a shape (2, 3, 4, 5) and `split_axis` was 2, then
  `u` would have shape (2, 3, 6), `s` would have shape (6), and `vh` would
  have shape (6, 4, 5).

  If `max_singular_values` is set to an integer, the SVD is truncated to keep
  at most this many singular values.

  If `max_truncation_error > 0`, as many singular values will be truncated as
  possible, so that the truncation error (the norm of discarded singular
  values) is at most `max_truncation_error`.

  If both `max_singular_values` snd `max_truncation_error` are specified, the
  number of retained singular values will be
  `min(max_singular_values, nsv_auto_trunc)`, where `nsv_auto_trunc` is the
  number of singular values that must be kept to maintain a truncation error
  smaller than `max_truncation_error`.

  The output consists of three tensors `u, s, vh` such that:
  ```python
    u[i1,...,iN, j] * s[j] * vh[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]
  ```
  Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

  Args:
    tf: The tensorflow module.
    tensor: A tensor to be decomposed.
    split_axis: Where to split the tensor's axes before flattening into a
      matrix.
    max_singular_values: The number of singular values to keep, or `None` to
      keep them all.
    max_truncation_error: The maximum allowed truncation error or `None` to not
      do any truncation.

  Returns:
    u: Left tensor factor.
    s: Vector of ordered singular values from largest to smallest.
    vh: Right tensor factor.
    s_rest: Vector of discarded singular values (length zero if no
            truncation).
  """
  left_dims = tf.shape(tensor)[:split_axis]
  right_dims = tf.shape(tensor)[split_axis:]

  tensor = tf.reshape(tensor,
                      [tf.reduce_prod(left_dims),
                       tf.reduce_prod(right_dims)])
  s, u, v = tf.linalg.svd(tensor)

  if max_singular_values is None:
    max_singular_values = tf.size(s, out_type=tf.int64)
  else:
    max_singular_values = tf.constant(max_singular_values, dtype=tf.int64)

  if max_truncation_error is not None:
    # Cumulative norms of singular values in ascending order.
    trunc_errs = tf.sqrt(tf.cumsum(tf.square(s), reverse=True))
    # We must keep at least this many singular values to ensure the
    # truncation error is <= max_truncation_error.
    num_sing_vals_err = tf.math.count_nonzero(
        tf.cast(trunc_errs > max_truncation_error, dtype=tf.int32))
  else:
    num_sing_vals_err = max_singular_values

  num_sing_vals_keep = tf.minimum(max_singular_values, num_sing_vals_err)

  # tf.svd() always returns the singular values as a vector of float{32,64}.
  # since tf.math_ops.real is automatically applied to s. This causes
  # s to possibly not be the same dtype as the original tensor, which can cause
  # issues for later contractions. To fix it, we recast to the original dtype.
  s = tf.cast(s, tensor.dtype)

  s_rest = s[num_sing_vals_keep:]
  s = s[:num_sing_vals_keep]
  u = u[:, :num_sing_vals_keep]
  v = v[:, :num_sing_vals_keep]

  vh = tf.linalg.adjoint(v)

  dim_s = tf.shape(s)[0]  # must use tf.shape (not s.shape) to compile
  u = tf.reshape(u, tf.concat([left_dims, [dim_s]], axis=-1))
  vh = tf.reshape(vh, tf.concat([[dim_s], right_dims], axis=-1))

  return u, s, vh, s_rest
