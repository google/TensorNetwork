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
from tensornetwork import ncon_interface


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
  return all_backends_same, errstr


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


def shape(tensor: Tensor) -> Tuple[Optional[int], ...]:
  """Get the shape of a Tensor as a tuple of integers.

  Args:
    Tensor: A Tensor.
  Returns:
    The shape of the input Tensor.
  """
  return tensor.shape


def sqrt(tensor: Tensor) -> Tensor:
  """Take the square root (element wise) of a given Tensor."""
  out_array = tensor.backend.sqrt(tensor.array)
  return Tensor(out_array, backend=tensor.backend)


def outer(tensor1: Tensor, tensor2: Tensor) -> Tensor:
  """Calculate the outer product of the two given Tensors."""
  tensors = [tensor1, tensor2]
  all_backends_same, errstr = _check_backends(tensors, "outer")
  if not all_backends_same:
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


def diagonal(tensor: Tensor, offset: int = 0, axis1: int = -2,
             axis2: int = -1) -> Tensor:
  """
  Extracts the offset'th diagonal from the matrix slice of tensor indexed
  by (axis1, axis2).

  Args:
    tensor: A Tensor.
    offset: Offset of the diagonal from the main diagonal.
    axis1, axis2: Indices of the matrix slice to extract from.

  Returns:
    out  : A 1D Tensor storing the elements of the selected diagonal.
  """
  backend = tensor.backend
  result = backend.diagonal(tensor.array, offset=offset, axis1=axis1,
                            axis2=axis2)
  return Tensor(result, backend=backend)


def diagflat(tensor: Tensor, k: int = 0) -> Tensor:
  """
  Flattens tensor and places its elements at the k'th diagonal of a new
  (tensor.size + k, tensor.size + k) `Tensor` of zeros.

  Args:
    tensor: A Tensor.
    k     : The elements of tensor will be stored at this diagonal.
  Returns:
    out   : A (tensor.size + k, tensor.size + k) `Tensor` with the elements
            of tensor on its kth diagonal.
  """
  backend = tensor.backend
  result = backend.diagflat(tensor.array, k=k)
  return Tensor(result, backend=backend)


def trace(tensor: Tensor, offset: int = 0, axis1: int = -2,
          axis2: int = -1) -> Tensor:
  """Calculate the sum along diagonal entries of the given Tensor. The
     entries of the offset`th diagonal of the matrix slice of tensor indexed by
     (axis1, axis2) are summed.

  Args:
    tensor: A Tensor.
    offset: Offset of the diagonal from the main diagonal.
    axis1, axis2: Indices of the matrix slice to extract from.

  Returns:
    out: The trace.
  """
  backend = tensor.backend
  result = backend.trace(tensor.array, offset=offset, axis1=axis1,
                         axis2=axis2)
  return Tensor(result, backend=backend)


def ncon(
    tensors: Sequence[Tensor],
    network_structure: Sequence[Sequence[Union[str, int]]],
    con_order: Optional[Sequence] = None,
    out_order: Optional[Sequence] = None,
    check_network: bool = True,
) -> Tensor:
  r"""Contracts a list of tn.Tensor according to a tensor network
    specification.

    The network is provided as a list of lists, one for each
    tensor, specifying the labels for the edges connected to that tensor.

    Labels can be any numbers or strings. Negative number-type labels
    and string-type labels with a prepended hyphen ('-') are open labels
    and remain uncontracted.

    Positive number-type labels and string-type labels with no prepended
    hyphen ('-') are closed labels and are contracted.

    Any open label appearing more than once is treated as an open
    batch label. Any closed label appearing more than once is treated as
    a closed batch label.

    Upon finishing the contraction, all open batch labels will have been
    collapsed into a single dimension, and all closed batch labels will
    have been summed over.

    If `out_order = None`, output labels are ordered according to descending
    number ordering and ascending ASCII ordering, with number labels always
    appearing before string labels. Example:
    network_structure = [[-1, 1, '-rick', '2',-2], [-2, '2', 1, '-morty']]
    results in an output order of [-1, -2, '-morty', '-rick'].

    If `out_order` is given, the indices of the resulting tensor will be
    transposed into this order.

    If `con_order = None`, `ncon` will first contract all number labels
    in ascending order followed by all string labels in ascending ASCII
    order.
    If `con_order` is given, `ncon` will contract according to this order.

    For example, matrix multiplication:

    .. code-block:: python

      A = np.array([[1.0, 2.0], [3.0, 4.0]])
      B = np.array([[1.0, 1.0], [0.0, 1.0]])
      ncon([A,B], [(-1, 1), (1, -2)])

    Matrix trace:

    .. code-block:: python

      A = np.array([[1.0, 2.0], [3.0, 4.0]])
      ncon([A], [(1, 1)]) # 5.0

    Note:
      Disallowing `0` as an edge label is legacy behaviour, see
      `original NCON implementation`_.
    .. _original NCON implementation:
      https://arxiv.org/abs/1402.0939
    Args:
      tensors: List of `Tensors`.
      network_structure: List of lists specifying the tensor network structure.
      con_order: List of edge labels specifying the contraction order.
      out_order: List of edge labels specifying the output order.
      check_network: Boolean flag. If `True` check the network.
      backend: String specifying the backend to use. Defaults to
        `tensornetwork.backend_contextmanager.get_default_backend`.

    Returns:
      The result of the contraction. The result is returned as a `Node`
      if all elements of `tensors` are `AbstractNode` objects, else
      it is returned as a `Tensor` object.
    """
  all_backends_same, errstr = _check_backends(tensors, "ncon")
  if not all_backends_same:
    raise ValueError(errstr)
  backend = tensors[0].backend
  arrays = [tensor.array for tensor in tensors]
  res = ncon_interface.ncon(arrays, network_structure, con_order=con_order,
                            out_order=out_order, check_network=check_network,
                            backend=backend)
  output = Tensor(res, backend=backend)
  return output


def sign(tensor: Tensor) -> Tensor:
  """ Returns the sign of the elements of Tensor.
  """
  backend = tensor.backend
  result = backend.sign(tensor.array)
  return Tensor(result, backend=backend)


# pylint: disable=redefined-builtin
def abs(tensor: Tensor) -> Tensor:
  """ Returns the absolute value of the elements of Tensor.
  """
  backend = tensor.backend
  result = backend.abs(tensor.array)
  return Tensor(result, backend=backend)
