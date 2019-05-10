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

"""A modified version of TensorFlow's tensordot operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf


def tensordot(a, b, axes, name=None):
  r"""Tensor contraction of a and b along specified axes.
  Tensordot (also known as tensor contraction) sums the product of elements
  from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
  The lists `a_axes` and `b_axes` specify those pairs of axes along which to
  contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
  as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
  `a_axes` and `b_axes` must have identical length and consist of unique
  integers that specify valid axes for each of the tensors.
  This operation corresponds to `numpy.tensordot(a, b, axes)`.
  Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
  is equivalent to matrix multiplication.
  Example 2: When `a` and `b` are matrices (order 2), the case
  `axes = [[1], [0]]` is equivalent to matrix multiplication.
  Example 3: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
  tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
  \\(c_{jklm}\\) whose entry
  corresponding to the indices \\((j,k,l,m)\\) is given by:
  \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).
  In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.
  Args:
    a: `Tensor` of type `float32` or `float64`.
    b: `Tensor` with the same type as `a`.
    axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
      If axes is a scalar, sum over the last N axes of a and the first N axes of
      b in order. If axes is a list or `Tensor` the first and second row contain
      the set of unique integers specifying axes along which the contraction is
      computed, for `a` and `b`, respectively. The number of axes for `a` and
      `b` must be equal.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with the same type as `a`.
  Raises:
    ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
    IndexError: If the values in axes exceed the rank of the corresponding
      tensor.
  """

  def _tensordot_reshape(a, axes, flip_default=False):
    """Helper method to perform transpose and reshape for contraction op.
    This method is helpful in reducing `math_ops.tensordot` to `math_ops.matmul`
    using `array_ops.transpose` and `array_ops.reshape`. The method takes a
    tensor and performs the correct transpose and reshape operation for a given
    set of indices. It returns the reshaped tensor as well as a list of indices
    necessary to reshape the tensor again after matrix multiplication.
    Args:
      a: `Tensor`.
      axes: List or `int32` `Tensor` of unique indices specifying valid axes of
       `a`.
    Returns:
      A tuple `(reshaped_a, free_dims, free_dims_static, flipped)` where `reshaped_a` is
      the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
      either a list of integers or an `int32` `Tensor`, depending on whether
      the shape of a is fully specified, and free_dims_static is either a list
      of integers and None values, or None, representing the inferred
      static shape of the free dimensions. `flipped` indicates whether
      `reshaped_a` must be transposed, or not, when calling `matmul`.
    """
    if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
      shape_a = a.get_shape().as_list()
      axes = [i if i >= 0 else i + len(shape_a) for i in axes]
      free = [i for i in range(len(shape_a)) if i not in axes]

      # Determine whether to flip by minimizing the average distance the 
      # indices would have to move under a permutation.
      if len(axes) > 0 and len(free) > 0:
        flipped = bool(np.mean(axes) < np.mean(free))
      else:
        flipped = False

      free_dims = [shape_a[i] for i in free]
      prod_free = int(np.prod([shape_a[i] for i in free]))
      prod_axes = int(np.prod([shape_a[i] for i in axes]))
      perm = list(axes) + free if flipped else free + list(axes)
      new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]

      # Skip the transpose op if possible. Although the graph optimizer should
      # kill trivial transposes, it is best not to add them in the first place!
      if perm == list(range(len(perm))):
        transposed_a = a
      else:
        transposed_a = tf.transpose(a, perm)
      reshaped_a = tf.reshape(transposed_a, new_shape)
      return reshaped_a, free_dims, free_dims, flipped
    else:
      if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
        shape_a = a.get_shape().as_list()
        axes = [i if i >= 0 else i + len(shape_a) for i in axes]
        free = [i for i in range(len(shape_a)) if i not in axes]

        # Determine whether to flip by minimizing the average distance the 
        # indices would have to move under a permutation.
        if len(axes) > 0 and len(free) > 0:
            flipped = bool(np.mean(axes) < np.mean(free))
        else:
            flipped = False

        axes_dims = [shape_a[i] for i in axes]
        free_dims = [shape_a[i] for i in free]
        free_dims_static = free_dims
        axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32, name="axes")
        free = tf.convert_to_tensor(free, dtype=tf.dtypes.int32, name="free")
        shape_a = tf.shape(a)
      else:
        free_dims_static = None
        shape_a = tf.shape(a)
        rank_a = tf.rank(a)
        axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32, name="axes")
        axes = tf.where(axes >= 0, axes, axes + rank_a)
        free, _ = tf.setdiff1d(tf.range(rank_a), axes)
        # Matmul does not accept tensors for its transpose arguments, so fall 
        # back to the previous, fixed behavior.
        flipped = flip_default

      free_dims = tf.gather(shape_a, free)
      axes_dims = tf.gather(shape_a, axes)
      prod_free_dims = tf.reduce_prod(free_dims)
      prod_axes_dims = tf.reduce_prod(axes_dims)

      if flipped:
        perm = tf.concat([axes, free], 0)
        new_shape = tf.stack([prod_axes_dims, prod_free_dims])
      else:
        perm = tf.concat([free, axes], 0)
        new_shape = tf.stack([prod_free_dims, prod_axes_dims])
      reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
      return reshaped_a, free_dims, free_dims_static, flipped

  def _tensordot_axes(a, axes):
    """Generates two sets of contraction axes for the two tensor arguments."""
    a_shape = a.get_shape()
    if isinstance(axes, tf.compat.integral_types):
      if axes < 0:
        raise ValueError("'axes' must be at least 0.")
      if a_shape.ndims is not None:
        if axes > a_shape.ndims:
          raise ValueError("'axes' must not be larger than the number of "
                           "dimensions of tensor %s." % a)
        return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                list(range(axes)))
      else:
        rank = tf.rank(a)
        return (range(rank - axes, rank, dtype=tf.int32),
                range(axes, dtype=tf.int32))
    elif isinstance(axes, (list, tuple)):
      if len(axes) != 2:
        raise ValueError("'axes' must be an integer or have length 2.")
      a_axes = axes[0]
      b_axes = axes[1]
      if isinstance(a_axes, tf.compat.integral_types) and \
          isinstance(b_axes, tf.compat.integral_types):
        a_axes = [a_axes]
        b_axes = [b_axes]
      if len(a_axes) != len(b_axes):
        raise ValueError(
            "Different number of contraction axes 'a' and 'b', %s != %s." %
            (len(a_axes), len(b_axes)))

      # The contraction indices do not need to be permuted.
      # Sort axes to avoid unnecessary permutations of a.
      a_axes, b_axes = list(zip(*sorted(zip(a_axes, b_axes))))

      return a_axes, b_axes
    else:
      axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
      return axes[0], axes[1]

  with tf.name_scope(name, "Tensordot", [a, b, axes]) as name:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    a_axes, b_axes = _tensordot_axes(a, axes)
    a_reshape, a_free_dims, a_free_dims_static, a_flip = _tensordot_reshape(
      a, a_axes)
    b_reshape, b_free_dims, b_free_dims_static, b_flip = _tensordot_reshape(
      b, b_axes, flip_default=True)

    ab_matmul = tf.matmul(
        a_reshape,
        b_reshape,
        transpose_a=a_flip,
        transpose_b=not b_flip)

    if isinstance(a_free_dims, list) and isinstance(b_free_dims, list):
      return tf.reshape(ab_matmul, a_free_dims + b_free_dims, name=name)
    else:
      a_free_dims = tf.convert_to_tensor(a_free_dims, dtype=tf.dtypes.int32)
      b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.dtypes.int32)
      product = tf.reshape(
          ab_matmul, tf.concat([a_free_dims, b_free_dims], 0), name=name)
      if a_free_dims_static is not None and b_free_dims_static is not None:
        product.set_shape(a_free_dims_static + b_free_dims_static)
      return product
