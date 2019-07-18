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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional, Any, Tuple
import numpy
Tensor = Any


def svd_decomposition(np, # TODO: Typing
                      tensor: Tensor,
                      split_axis: int,
                      max_singular_values: Optional[int] = None,
                      max_truncation_error: Optional[float] = None,
                     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
  """Computes the singular value decomposition (SVD) of a tensor.

  See tensornetwork.backends.tensorflow.decompositions for details.
  """
  left_dims = tensor.shape[:split_axis]
  right_dims = tensor.shape[split_axis:]

  tensor = np.reshape(tensor, [numpy.prod(left_dims), numpy.prod(right_dims)])
  u, s, vh = np.linalg.svd(tensor)

  if max_singular_values is None:
    max_singular_values = np.size(s)

  if max_truncation_error is not None:
    # Cumulative norms of singular values in ascending order.
    trunc_errs = np.sqrt(np.cumsum(np.square(s[::-1])))
    # We must keep at least this many singular values to ensure the
    # truncation error is <= max_truncation_error.
    num_sing_vals_err = np.count_nonzero(
        (trunc_errs > max_truncation_error).astype(np.int32))
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
