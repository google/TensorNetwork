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

from typing import Any, Union, Text, Optional, List, Sequence, Tuple
from tensornetwork.tensor import Tensor

def _check_backends(tensors: Sequence[Tensor], fname: str) -> Tuple[bool, str]:
  """ Checks that each of tensors has the same backend, returning True and an
      empty string if so, or False and an error string if not.
  Args:
    tensors: The list of tensors whose backends to check.
    fname: The name of the calling function, which will go into the errstring.
  Returns:
    (flag, errstr): Whether all backends agree, and an error message if not.
  """
  backend = tensors[0].backend
  backend_names = [tensor.backend.name for tensor in tensors]
  all_backends_same = True
  for name in backend_names[1:]:
    all_backends_same = all_backends_same and name == backend.name
  errstr = ""
  if not all_backends_same:
    errstr = "All Tensors fed to " + fname + "must have the same backend."
    errstr += "Backends were: \n"
    errstr += str([name + "\n" for name in backend_names])
  return (all_backends_same, errstr)


def tensordot(a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]) -> Tensor:
  """Do a tensordot (contraction) of Tensors `a` and `b` over the given axes.
  The behaviour of this function largely matches that of np.tensordot.

  Args:
    a: A Tensor.
    b: Another Tensor.
    axes: Two lists of integers. These values are the contraction
          axes.
  Raises:
    ValueError, if a and b have different backends.
  Returns:
    The result of the tensordot, a Tensor.
  """
  if a.backend.name != b.backend.name:
    errstr = "Tried to Tensordot Tensors with differing backends \n"
    errstr += a.backend.name + "and " + b.backend.name + "."
    raise ValueError(errstr)
  out_array = a.backend.tensordot(a.array, b.array, axes)
  out_tensor = Tensor(out_array, backend=a.backend)
  return out_tensor


def reshape(tensor: Tensor, new_shape: Sequence[int]) -> Tensor:
  """Reshape Tensor to the given shape.

  Args:
    tensor: Tensor to reshape.
    new_shape: The new shape.
  Returns:
    The reshaped Tensor.
  """
  return tensor.reshape(new_shape)


def transpose(tensor: Tensor, perm: Optional[Sequence[int]] = None) -> Tensor:
  """ Return a new `Tensor` transposed according to the permutation set
  by `axes`. By default the axes are reversed.
  Args:
    axes: The permutation. If None (default) the index order is reversed.
  Returns:
    The transposed `Tensor`.
  """
  return tensor.transpose(perm=perm)


def take_slice(tensor: Tensor, start_indices: Tuple[int, ...],
               slice_sizes: Tuple[int, ...]) -> Tensor:
  """Obtains a slice of a Tensor based on start_indices and slice_sizes.

  Args:
    Tensor: A Tensor.
    start_indices: Tuple of integers denoting start indices of slice.
    slice_sizes: Tuple of integers denoting size of slice along each axis.
  Returns:
    The slice, a Tensor.
  """
  sliced = tensor.backend.slice(tensor.array, start_indices, slice_sizes)
  sliced_tensor = Tensor(sliced, backend=tensor.backend)
  return sliced_tensor


def concatenate(tensors: Sequence[Tensor],
                axis: Optional[int] = 0) -> Tensor:
  """Concatenate a sequence of Tensors together about the given axis.
  Args:
    tensors: The list of Tensors to concatenate.
    axis   : The axis to concatenate on. Default 0.
  Returns:
    The concatenated Tensor.
  """
  if len(tensors) <= 1:
    raise ValueError("Must supply at least two Tensors to concatenate.")
  all_backends_same, errstr = _check_backends(tensors, "concatenate")
  if not all_backends_same:
    raise ValueError(errstr)
  arrays = [tensor.array for tensor in tensors]
  backend = tensors[0].backend
  concat_array = backend.shape_concat(arrays, axis)
  return Tensor(concat_array, backend=backend)


def shape(tensor: Tensor) -> Tuple[Optional[int], ...]:
  """Get the shape of a Tensor as a tuple of integers.

  Args:
    Tensor: A Tensor.
  Returns:
    The shape of the input Tensor.
  """
  return tensor.shape


def prod(values: Tensor) -> Tensor:
  """Take the product of all of the elements in values"""
  out_array = values.backend.shape_prod(values.array)
  return Tensor(out_array, backend=values.backend)


def sqrt(tensor: Tensor) -> Tensor:
  """Take the square root (element wise) of a given Tensor."""
  out_array = tensor.backend.sqrt(tensor.array)
  return Tensor(out_array, backend=tensor.backend)


def outer(tensor1: Tensor, tensor2: Tensor) -> Tensor:
  """Calculate the outer product of the two given Tensors."""
  if tensor1.backend.name != tensor2.backend.name:
    errstr = "Tried to Tensordot Tensors with differing backends \n"
    errstr += tensor1.backend.name + "and " + tensor2.backend.name + "."
    raise ValueError(errstr)
  out_data = tensor1.backend.outer_product(tensor1.array, tensor2.array)
  return Tensor(out_data, backend=tensor1.backend)


def einsum(expression: str, *tensors: Tensor, optimize: bool) -> Tensor:
  """Calculate sum of products of Tensors according to expression."""
  all_backends_same, errstr = _check_backends(tensors, "einsum")
  if not all_backends_same:
    raise ValueError(errstr)
  backend = tensors[0].backend
  arrays = [tensor.array for tensor in tensors]
  result_data = backend.einsum(expression, *arrays, optimize=optimize)
  return Tensor(result_data, backend=backend)


def conj(tensor: Tensor) -> Tensor:
  """
  Return the complex conjugate of `Tensor`
  Args:
    Tensor: A Tensor.
  Returns:
    The complex conjugated Tensor.
  """
  return tensor.conj()


def hconj(tensor: Tensor, perm: Optional[Sequence[int]] = None) -> Tensor:
  """ The Hermitian conjugated tensor; e.g. the complex conjugate tranposed
  by the permutation set be `axes`. By default the axes are reversed.
  Args:
    tensor: The Tensor to conjugate.
    axes: The permutation. If None (default) the index order is reversed.
  Returns:
    The Hermitian conjugated `Tensor`.
  """
  return tensor.hconj(perm=perm)


def sin(tensor: Tensor) -> Tensor:
  """
  Return sin of `Tensor`.
  Args:
    Tensor: A Tensor.
  Returns:
    Tensor
  """
  out_array = tensor.backend.sin(tensor.array)
  return Tensor(out_array, backend=tensor.backend)


def cos(tensor: Tensor) -> Tensor:
  """
  Return cos of `Tensor`.
  Args:
    Tensor: A Tensor.
  Returns:
    Tensor
  """
  out_array = tensor.backend.cos(tensor.array)
  return Tensor(out_array, backend=tensor.backend)


def exp(tensor: Tensor) -> Tensor:
  """
  Return elementwise exp of `Tensor`.
  Args:
    Tensor: A Tensor.
  Returns:
    Tensor
  """
  out_array = tensor.backend.exp(tensor.array)
  return Tensor(out_array, backend=tensor.backend)


def log(tensor: Tensor) -> Tensor:
  """
  Return elementwise natural logarithm of `Tensor`.
  Args:
    Tensor: A Tensor.
  Returns:
    Tensor
  """
  out_array = tensor.backend.log(tensor.array)
  return Tensor(out_array, backend=tensor.backend)


def diagonal(tensor: Tensor, offset: Optional[int] = 0,
             axis1=0, axis2=1) -> Tensor:
  """
  Extract diagonals from a Tensor.

  Args:
    tensor: A Tensor.
    offset     : Offset of the diagonal from the main diagonal. If tensor is 1D,
  """
  raise NotImplementedError()


def diagflat(tensor: Tensor, k: Optional[int] = 0) -> Tensor:
  """
  Construct a two-dimensional Tensor with the flattened input from tensor on
  the diagonal.
  """
  raise NotImplementedError()


def trace(tensor: Tensor, offset=0, axis1=0, axis2=1) -> Tensor:
  """Calculate the sum along diagonal entries of the given Tensor."""
  raise NotImplementedError()
