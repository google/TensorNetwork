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
from typing import Optional, Union, Text, Sequence, Tuple, List
import numpy as np
import tensorflow as tf

AXES_TYPE = Union[int, tf.Tensor, Sequence[Union[int, Sequence[int]]]]
AXES_ENTRY_TYPE = Union[Sequence[int], tf.Tensor]

def tensordot(a: tf.Tensor, b: tf.Tensor, axes: AXES_TYPE,
  name: Optional[Text] = None) -> tf.Tensor:
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

  def _tensordot_should_flip(contraction_axes: List[int],
    free_axes: List[int]) -> bool:
    """Helper method to determine axis ordering.
    We minimize the average distance the indices would have to move under the
    transposition.
    Args:
      contraction_axes: The axes to be contracted.
      free_axes: The free axes.
    Returns:
      should_flip: `True` if `contraction_axes` should be moved to the left,
        `False` if they should be moved to the right.
    """
    # NOTE: This will fail if the arguments contain any Tensors.
    if contraction_axes and free_axes:
      return bool(np.mean(contraction_axes) < np.mean(free_axes))
    return False

  def _tranpose_if_necessary(tensor: tf.Tensor, perm: List[int]) -> tf.Tensor:
    """Like transpose(), but avoids creating a new tensor if possible.
    Although the graph optimizer should kill trivial transposes, it is best not
    to add them in the first place!
    """
    if perm == list(range(len(perm))):
      return tensor
    return tf.transpose(tensor, perm)

  def _reshape_if_necessary(tensor: tf.Tensor, new_shape: List[int]
    ) -> tf.Tensor:
    """Like reshape(), but avoids creating a new tensor if possible.
    Assumes shapes are both fully specified."""
    cur_shape = tensor.get_shape().as_list()
    if (len(new_shape) == len(cur_shape) and
        all(d0 == d1 for d0, d1 in zip(cur_shape, new_shape))):
      return tensor
    return tf.reshape(tensor, new_shape)

  def _tensordot_reshape(a: tf.Tensor, axes: Union[Sequence[int], tf.Tensor],
    is_right_term=False) -> Tuple[
      tf.Tensor, Union[List[int], tf.Tensor], Optional[List[int]], bool]:
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
      is_right_term: Whether `a` is the right (second) argument to `matmul`.
    Returns:
      A tuple `(reshaped_a, free_dims, free_dims_static, transpose_needed)`
      where `reshaped_a` is the tensor `a` reshaped to allow contraction via
      `matmul`, `free_dims` is either a list of integers or an `int32`
      `Tensor`, depending on whether the shape of a is fully specified, and
      free_dims_static is either a list of integers and None values, or None,
      representing the inferred static shape of the free dimensions. 
      `transpose_needed` indicates whether `reshaped_a` must be transposed,
      or not, when calling `matmul`.
    """
    if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
      shape_a = a.get_shape().as_list()
      # NOTE: This will fail if axes contains any tensors
      axes = [i if i >= 0 else i + len(shape_a) for i in axes]
      free = [i for i in range(len(shape_a)) if i not in axes]
      flipped = _tensordot_should_flip(axes, free)

      free_dims = [shape_a[i] for i in free]
      prod_free = int(np.prod([shape_a[i] for i in free]))
      prod_axes = int(np.prod([shape_a[i] for i in axes]))
      perm = axes + free if flipped else free + axes
      new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
      transposed_a = _tranpose_if_necessary(a, perm)
      reshaped_a = _reshape_if_necessary(transposed_a, new_shape)
      transpose_needed = (not flipped) if is_right_term else flipped
      return reshaped_a, free_dims, free_dims, transpose_needed
    if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
      shape_a = a.get_shape().as_list()
      axes = [i if i >= 0 else i + len(shape_a) for i in axes]
      free = [i for i in range(len(shape_a)) if i not in axes]
      flipped = _tensordot_should_flip(axes, free)
      perm = axes + free if flipped else free + axes

      axes_dims = [shape_a[i] for i in axes]
      free_dims = [shape_a[i] for i in free]
      free_dims_static = free_dims
      axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32, name="axes")
      free = tf.convert_to_tensor(free, dtype=tf.dtypes.int32, name="free")
      shape_a = tf.shape(a)
      transposed_a = _tranpose_if_necessary(a, perm)
    else:
      free_dims_static = None
      shape_a = tf.shape(a)
      rank_a = tf.rank(a)
      axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32, name="axes")
      axes = tf.where(axes >= 0, axes, axes + rank_a)
      free, _ = tf.compat.v1.setdiff1d(tf.range(rank_a), axes)
      # Matmul does not accept tensors for its transpose arguments, so fall
      # back to the previous, fixed behavior.
      # NOTE(amilsted): With a suitable wrapper for `matmul` using e.g. `case`
      #   to match transpose arguments to tensor values, we could also avoid
      #   unneeded tranposes in this case at the expense of a somewhat more
      #   complicated graph. Unclear whether this would be beneficial overall.
      flipped = is_right_term
      perm = (
        tf.concat([axes, free], 0) if flipped else tf.concat([free, axes], 0))
      transposed_a = tf.transpose(a, perm)

    free_dims = tf.gather(shape_a, free)
    axes_dims = tf.gather(shape_a, axes)
    prod_free_dims = tf.reduce_prod(free_dims)
    prod_axes_dims = tf.reduce_prod(axes_dims)

    if flipped:
      new_shape = tf.stack([prod_axes_dims, prod_free_dims])
    else:
      new_shape = tf.stack([prod_free_dims, prod_axes_dims])
    reshaped_a = tf.reshape(transposed_a, new_shape)
    transpose_needed = (not flipped) if is_right_term else flipped
    return reshaped_a, free_dims, free_dims_static, transpose_needed

  def _tensordot_axes(a: tf.Tensor, axes: AXES_TYPE
    ) -> Tuple[AXES_ENTRY_TYPE, AXES_ENTRY_TYPE]:
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
      rank = tf.rank(a)
      return (tf.range(rank - axes, rank, dtype=tf.int32),
              tf.range(axes, dtype=tf.int32))
    if isinstance(axes, (list, tuple)):
      if len(axes) != 2:
        raise ValueError("'axes' must be an integer or have length 2.")
      a_axes = axes[0]
      b_axes = axes[1]
      if isinstance(a_axes, tf.compat.integral_types) and \
          isinstance(b_axes, tf.compat.integral_types):
        a_axes = [a_axes]
        b_axes = [b_axes]
      # NOTE: This fails if either a_axes and b_axes are Tensors.
      if len(a_axes) != len(b_axes):
        raise ValueError(
            "Different number of contraction axes 'a' and 'b', %s != %s." %
            (len(a_axes), len(b_axes)))

      # The contraction indices do not need to be permuted.
      # Sort axes to avoid unnecessary permutations of a.
      # NOTE: This fails if either a_axes and b_axes contain Tensors.
      # pylint: disable=len-as-condition
      if len(a_axes) > 0:
        a_axes, b_axes = list(zip(*sorted(zip(a_axes, b_axes))))

      return a_axes, b_axes
    axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
    return axes[0], axes[1]

  with tf.compat.v1.name_scope(name, "Tensordot", [a, b, axes]) as _name:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    a_axes, b_axes = _tensordot_axes(a, axes)
    a_reshape, a_free_dims, a_free_dims_static, a_transp = _tensordot_reshape(
      a, a_axes)
    b_reshape, b_free_dims, b_free_dims_static, b_transp = _tensordot_reshape(
      b, b_axes, is_right_term=True)

    ab_matmul = tf.matmul(
        a_reshape,
        b_reshape,
        transpose_a=a_transp,
        transpose_b=b_transp)

    if isinstance(a_free_dims, list) and isinstance(b_free_dims, list):
      return tf.reshape(ab_matmul, a_free_dims + b_free_dims, name=_name)
    a_free_dims = tf.convert_to_tensor(a_free_dims, dtype=tf.dtypes.int32)
    b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.dtypes.int32)
    product = tf.reshape(
        ab_matmul, tf.concat([a_free_dims, b_free_dims], 0), name=_name)
    if a_free_dims_static is not None and b_free_dims_static is not None:
      product.set_shape(a_free_dims_static + b_free_dims_static)
    return product
