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
import numpy as np

Tensor = Any


def svd_decomposition(torch: Any,
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
  left_dims = list(tensor.shape)[:split_axis]
  right_dims = list(tensor.shape)[split_axis:]

  tensor = torch.reshape(tensor, (np.prod(left_dims), np.prod(right_dims)))
  u, s, v = torch.svd(tensor)

  if max_singular_values is None:
    max_singular_values = s.nelement()
  else:
    max_singular_values = max_singular_values

  if max_truncation_error is not None:
    # Cumulative norms of singular values in ascending order
    s_sorted, _ = torch.sort(s ** 2)
    trunc_errs = torch.sqrt(torch.cumsum(s_sorted, 0))
    # We must keep at least this many singular values to ensure the
    # truncation error is <= max_truncation_error.
    num_sing_vals_err = torch.nonzero(trunc_errs
                                      > max_truncation_error).nelement()
  else:
    num_sing_vals_err = max_singular_values

  num_sing_vals_keep = min(max_singular_values, num_sing_vals_err)

  # we recast to the original dtype.
  s = s.type(tensor.type())

  s_rest = s[num_sing_vals_keep:]
  s = s[:num_sing_vals_keep]
  u = u[:, :num_sing_vals_keep]
  v = v[:, :num_sing_vals_keep]

  vh = torch.transpose(v, 0, 1)

  dim_s = s.shape[0]
  u = torch.reshape(u, left_dims + [dim_s])
  vh = torch.reshape(vh, [dim_s] + right_dims)

  return u, s, vh, s_rest
