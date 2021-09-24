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
# pylint: disable=line-too-long
from typing import Optional, Any, Sequence, Tuple, Type, Callable, List
from typing import Union
from tensornetwork.backends import abstract_backend
from tensornetwork.backends.tensorflow import decompositions
import functools as fct
import operator as op

# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
import numpy as np
Tensor = Any

# pylint: disable=abstract-method


class TensorFlowBackend(abstract_backend.AbstractBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self) -> None:
    # pylint: disable=global-variable-undefined
    global tf
    super().__init__()
    try:
      # pylint: disable=import-outside-toplevel
      import tensorflow
    except ImportError as err:
      raise ImportError("Tensorflow not installed, please switch to a "
                        "different backend or install Tensorflow.") from err
    tf = tensorflow
    self.name = "tensorflow"

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
    return tf.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
    return tf.reshape(tensor, shape)

  def transpose(self, tensor, perm=None) -> Tensor:
    return tf.transpose(tensor, perm)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    return tf.slice(tensor, start_indices, slice_sizes)

  def svd(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd(
        tf,
        tensor,
        pivot_axis,
        max_singular_values,
        max_truncation_error,
        relative=relative)

  def qr(self, tensor: Tensor, pivot_axis: int = -1,
         non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
    return decompositions.qr(tf, tensor, pivot_axis, non_negative_diagonal)

  def rq(self, tensor: Tensor, pivot_axis: int = -1,
         non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
    return decompositions.rq(tf, tensor, pivot_axis, non_negative_diagonal)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return tf.concat(values, axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return tf.shape(tensor)

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tuple(tensor.shape.as_list())

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return self.shape_tuple(tensor)

  def shape_prod(self, values: Tensor) -> Tensor:
    return tf.reduce_prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return tf.sqrt(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    result = tf.convert_to_tensor(tensor)
    return result

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tf.tensordot(tensor1, tensor2, 0)

  #pylint: disable=unused-argument
  def einsum(self,
             expression: str,
             *tensors: Tensor,
             optimize: bool = True) -> Tensor:
    return tf.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return tf.linalg.norm(tensor)

  def eye(self,
          N: int,
          dtype: Optional[Type[np.number]] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else tf.float64
    return tf.eye(num_rows=N, num_columns=M, dtype=dtype)

  def ones(self,
           shape: Tuple[int, ...],
           dtype: Optional[Type[np.number]] = None) -> Tensor:
    dtype = dtype if dtype is not None else tf.float64
    return tf.ones(shape=shape, dtype=dtype)

  def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None) -> Tensor:
    dtype = dtype if dtype is not None else tf.float64
    return tf.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None,
            seed: Optional[int] = None) -> Tensor:
    if seed:
      tf.random.set_seed(seed)

    dtype = dtype if dtype is not None else tf.float64
    if (dtype is tf.complex128) or (dtype is tf.complex64):
      return tf.complex(
          tf.random.normal(shape=shape, dtype=dtype.real_dtype),
          tf.random.normal(shape=shape, dtype=dtype.real_dtype))
    return tf.random.normal(shape=shape, dtype=dtype)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[Type[np.number]] = None,
                     seed: Optional[int] = None) -> Tensor:
    if seed:
      tf.random.set_seed(seed)

    dtype = dtype if dtype is not None else tf.float64
    if (dtype is tf.complex128) or (dtype is tf.complex64):
      #pylint: disable=unexpected-keyword-arg
      return tf.complex(
          tf.random.uniform(
              shape=shape,
              minval=boundaries[0],
              maxval=boundaries[1],
              dtype=dtype.real_dtype),
          tf.random.uniform(
              shape=shape,
              minval=boundaries[0],
              maxval=boundaries[1],
              dtype=dtype.real_dtype))
    tf.random.set_seed(10)
    #pylint: disable=unexpected-keyword-arg
    a = tf.random.uniform(
        shape=shape, minval=boundaries[0], maxval=boundaries[1], dtype=dtype)
    return a

  def conj(self, tensor: Tensor) -> Tensor:
    return tf.math.conj(tensor)

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return tf.linalg.eigh(matrix)

  def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 + tensor2

  def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 - tensor2

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 * tensor2

  def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 / tensor2

  def index_update(self, tensor: Tensor, mask: Tensor,
                   assignee: Tensor) -> Tensor:
    #returns a copy (unfortunately)
    return tf.where(mask, assignee, tensor)

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError("input to tensorflow backend method `inv` has shape {}. "
                       "Only matrices are supported.".format(tf.shape(matrix)))
    return tf.linalg.inv(matrix)

  def broadcast_right_multiplication(self, tensor1: Tensor,
                                     tensor2: Tensor) -> Tensor:
    if len(tensor2.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor2`, "
                       "found `tensor2.shape = {}`".format(tf.shape(tensor2)))

    return tensor1 * tensor2

  def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tf.shape(tensor1)))

    t1_broadcast_shape = self.shape_concat(
        [self.shape_tensor(tensor1), [1] * (len(tensor2.shape) - 1)], axis=-1)
    return tensor2 * self.reshape(tensor1, t1_broadcast_shape)

  def sin(self, tensor: Tensor) -> Tensor:
    return tf.math.sin(tensor)

  def cos(self, tensor: Tensor) -> Tensor:
    return tf.math.cos(tensor)

  def exp(self, tensor: Tensor) -> Tensor:
    return tf.math.exp(tensor)

  def log(self, tensor: Tensor) -> Tensor:
    return tf.math.log(tensor)

  def expm(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) != 2:
      raise ValueError("input to tensorflow backend method `expm` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError("input to tensorflow backend method `expm` only supports"
                       "N*N matrix, {x}*{y} matrix is given".format(
                           x=matrix.shape[0], y=matrix.shape[1]))
    return tf.linalg.expm(matrix)

  def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
    # tf.function is slow and bad.
    return fun

  def sum(self,
          tensor: Tensor,
          axis: Optional[Sequence[int]] = None,
          keepdims: bool = False) -> Tensor:
    return tf.math.reduce_sum(tensor, axis=axis, keepdims=keepdims)

  def matmul(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    if (tensor1.ndim <= 1) or (tensor2.ndim <= 1):
      raise ValueError("inputs to `matmul` have to be a tensors of order > 1,")

    return tf.matmul(tensor1, tensor2)

  def diagonal(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
               axis2: int = -1) -> Tensor:
    """Return specified diagonals.

    If tensor is 2-D, returns the diagonal of tensor with the given offset,
    i.e., the collection of elements of the form a[i, i+offset].
    If a has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose diagonal is
    returned. The shape of the resulting array can be determined by removing
    axis1 and axis2 and appending an index to the right equal to the size of the
    resulting diagonals.

    This function only extracts diagonals. If you
    wish to create diagonal matrices from vectors, use diagflat.

    Args:
      tensor: A tensor.
      offset: Offset of the diagonal from the main diagonal.
      axis1, axis2: Axis to be used as the first/second axis of the 2D
                    sub-arrays from which the diagonals should be taken.
                    Defaults to second-last and last axis (note this
                    differs from the NumPy defaults).

                    These arguments are not supported in the TensorFlow
                    backend and an error will be raised if they are
                    specified.
    Returns:
      array_of_diagonals: A dim = min(1, tensor.ndim - 2) tensor storing
                          the batched diagonals.
    """
    if axis1 != -2 or axis2 != -1:
      errstr = (f"axis1={axis1}, axis2={axis2} must be -2, -1 (the defaults)"
                f"with TensorFlow backend.")
      raise NotImplementedError(errstr)
    #pylint: disable=unexpected-keyword-arg
    return tf.linalg.diag_part(tensor, k=offset)

  def diagflat(self, tensor: Tensor, k: int = 0) -> Tensor:
    """ Flattens tensor and creates a new matrix of zeros with its elements
    on the k'th diagonal.
    Args:
      tensor: A tensor.
      k     : The diagonal upon which to place its elements.
    Returns:
      tensor: A new tensor with all zeros save the specified diagonal.
    """
    #pylint: disable=unexpected-keyword-arg
    return tf.linalg.diag(tensor, k=k)

  def trace(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
            axis2: int = -1) -> Tensor:
    """Return summed entries along diagonals.

    If tensor is 2-D, the sum is over the
    diagonal of tensor with the given offset,
    i.e., the collection of elements of the form a[i, i+offset].
    If a has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose diagonal is
    summed.

    Args:
      tensor: A tensor.
      offset: Offset of the diagonal from the main diagonal.
              This argument is not supported in the TensorFlow
              backend and an error will be raised if they are
              specified.
      axis1, axis2: Axis to be used as the first/second axis of the 2D
                    sub-arrays from which the diagonals should be taken.
                    Defaults to first/second axis.
                    These arguments are not supported in the TensorFlow
                    backend and an error will be raised if they are
                    specified.
    Returns:
      array_of_diagonals: The batched summed diagonals.
    """
    if offset != 0:
      errstr = (f"offset = {offset} must be 0 (the default)"
                f"with TensorFlow backend.")
      raise NotImplementedError(errstr)
    if axis1 == axis2:
      raise ValueError(f"axis1 = {axis1} cannot equal axis2 = {axis2}")
    N = len(tensor.shape)
    if N > 25:
      raise ValueError(f"Currently only tensors with ndim <= 25 can be traced"
                       f"in the TensorFlow backend (yours was {N})")

    if axis1 < 0:
      axis1 = N+axis1
    if axis2 < 0:
      axis2 = N+axis2

    inds = list(map(chr, range(98, 98+N)))
    indsout = [i for n, i in enumerate(inds) if n not in (axis1, axis2)]
    inds[axis1] = 'a'
    inds[axis2] = 'a'
    return tf.einsum(''.join(inds) + '->' +''.join(indsout), tensor)

  def abs(self, tensor: Tensor) -> Tensor:
    """
    Returns the elementwise absolute value of tensor.
    Args:
      tensor: An input tensor.
    Returns:
      tensor: Its elementwise absolute value.
    """
    return tf.math.abs(tensor)

  def sign(self, tensor: Tensor) -> Tensor:
    """
    Returns an elementwise tensor with entries
    y[i] = 1, 0, -1 where tensor[i] > 0, == 0, and < 0 respectively.

    For complex input the behaviour of this function may depend on the backend.
    The TensorFlow version returns y[i] = x[i] / abs(x[i]).

    Args:
      tensor: The input tensor.
    """
    return tf.math.sign(tensor)

  def item(self, tensor):
    numel = 0
    if len(tensor.shape) > 0:
      numel = fct.reduce(op.mul, tensor.shape)
      if numel != 1:
        raise ValueError(f"expected tensor with one element, "
                         f"got {tensor.shape}")

    if numel == 1:
      return tensor[0]
    return tensor

  def power(self, a: Tensor, b: Union[Tensor, float]) -> Tensor:
    """
    Returns the exponentiation of tensor a raised to b.
      If b is a tensor, then the exponentiation is element-wise
        between the two tensors, with a as the base and b as the power.
        Note that a and b must be broadcastable to the same shape if
        b is a tensor.
      If b is a scalar, then the exponentiation is each value in a
        raised to the power of b.

    Args:
      a: The tensor containing the bases.
      b: The tensor containing the powers; or a single scalar as the power.

    Returns:
      The tensor that is each element of a raised to the
        power of b.  Note that the shape of the returned tensor
        is that produced by the broadcast of a and b.
    """
    return tf.math.pow(a, b)

  def eps(self, dtype: Type[np.number]) -> float:
    """
    Return machine epsilon for given `dtype`

    Args:
      dtype: A dtype.

    Returns:
      float: Machine epsilon.
    """
    return tf.experimental.numpy.finfo(dtype).eps
