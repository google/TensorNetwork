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
"""Tensor network tools for TensorFlow. This module isd deprecated and will be removed 
in future versions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from tensornetwork import ncon, ncon_network


def svd_tensor(t, left_axes, right_axes, nsv_max=None, auto_trunc_max_err=0.0):
  """Computes the singular value decomposition (SVD) of a tensor.

    The SVD is performed by treating the tensor as a matrix, with an effective
    left (row) index resulting from combining the `left_axes` of the input
    tensor `t` and an effective right (column) index resulting from combining
    the `right_axes`. Transposition is used to move axes of the input tensor
    into position as as required. The output retains the full index structure
    of the original tensor.

    If `nsv_max` is set to an integer, the SVD is truncated to keep 
    `min(nsv_max, nsv)` singular values, where `nsv` is the number of singular 
    values returned by the SVD.

    If `auto_trunc_max_err > 0`, as many singular values will be truncated as 
    possible, so that the truncation error (the norm of discarded singular
    values) is at most `auto_trunc_max_err`. 

    If both `nsv_max` snd `auto_trunc_max_err` are specified, the number
    of retained singular values will be `min(nsv_max, nsv_auto_trunc)`, where
    `nsv_auto_trunc` is the number of singular values that must be kept to
    maintain a truncation error smaller than `auto_trunc_max_err`.

    The output consists of three tensors `u, s, vh` such that:
    ```python
      u[i1,...,iN, j] * s[j] * vh[j, k1,...,kM] == t_tr[i1,...,iN, k1,...,kM]
    ```
    where ```t_tr == tf.transpose(t, (*left_axes, *right_axes))```.

    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    Args:
      t: A tensor to be decomposed.
      left_axes: The axes of `t` to be treated as the left index.
      right_axes: The axes of `t` to be treated as the right index.
      nsv_max: The number of singular values to keep, or `None` to keep them 
               all.
      auto_trunc_max_err: The maximum allowed truncation error.

    Returns:
      u: Left tensor factor.
      s: Vector of singular values.
      vh: Right tensor factor.
      s_rest: Vector of discarded singular values (length zero if no 
              truncation).
    """
  t_shp = tf.shape(t)
  left_dims = [t_shp[i] for i in left_axes]
  right_dims = [t_shp[i] for i in right_axes]

  t_t = tf.transpose(t, (*left_axes, *right_axes))
  t_tr = tf.reshape(t_t, (np.prod(left_dims), np.prod(right_dims)))
  s, u, v = tf.svd(t_tr)

  if nsv_max is None:
    nsv_max = tf.size(s, out_type=tf.int64)
  else:
    nsv_max = tf.cast(nsv_max, tf.int64)

  # Cumulative norms of singular values in ascending order.
  trunc_errs = tf.sqrt(tf.cumsum(tf.square(s), reverse=True))
  # We must keep at least this many singular values to ensure the
  # truncation error is <= auto_trunc_max_err.
  nsv_err = tf.count_nonzero(trunc_errs > auto_trunc_max_err)

  nsv_keep = tf.minimum(nsv_max, nsv_err)

  # tf.svd() always returns the singular values as a vector of real.
  # Generally, however, we want to contract s with Tensors of the original
  # input type. To make this easy, cast here!
  s = tf.cast(s, t.dtype)

  s_rest = s[nsv_keep:]
  s = s[:nsv_keep]
  u = u[:, :nsv_keep]
  v = v[:, :nsv_keep]

  vh = tf.linalg.adjoint(v)

  dim_s = tf.shape(s)[0]  # must use tf.shape (not s.shape) to compile
  u = tf.reshape(u, (*left_dims, dim_s))
  vh = tf.reshape(vh, (dim_s, *right_dims))

  return u, s, vh, s_rest
