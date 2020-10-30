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

import math
import copy
import warnings
from typing import Any, Union, Text, Optional, List, Sequence
import numpy as np
from tensornetwork.backends import abstract_backend
from tensornetwork import backends, backend_contextmanager

BaseBackend = abstract_backend.AbstractBackend

class Tensor():
  def __init__(self,
               array: Any,
               backend: Optional[Union[Text, BaseBackend]] = None) -> None:
    if backend is None:
      backend = backend_contextmanager.get_default_backend()
    backend_obj = backends.backend_factory.get_backend(backend)
    self.backend = backend_obj
    self.array = self.backend.convert_to_tensor(array)
    self.shape = array.shape
    self.size = np.prod(self.shape)
    self.ndim = len(self.shape)

  @property
  def dtype(self) -> Any: # To maintain backend independence
    """ Returns: The dtype of the backend array.
    """
    return self.array.dtype

  @property
  def T(self) -> "Tensor":
    """ The `Tensor` with reversed axes.
    Returns:
      The transposed `Tensor`.
    """
    return self.transpose()

  @property
  def H(self) -> "Tensor":
    """ The conjugate `Tensor` with reversed axes.
    """
    star = self.backend.conj(self.array)
    array_H = self.backend.transpose(star)
    return Tensor(array_H, backend=self.backend)

  def conj(self) -> "Tensor":
    """ Returns: The complex-conjugated `Tensor`.
    """
    star = self.backend.conj(self.array)
    return Tensor(star, backend=self.backend)

  def conjugate(self) -> "Tensor":
    """ Returns: The complex-conjugated `Tensor`.
    """
    return self.conj()

  def copy(self) -> "Tensor":
    """ Returns: A copy of the `Tensor`.
    """
    return copy.deepcopy(self)

  def flatten(self):
    """Return a new `Tensor` with the same number of elements but only one
    dimension. This function differs from ravel in some cases at the backend
    level. Notably, with the numpy backend, flatten always returns a copy,
    wheras ravel returns a view when possible.
    """
    size = self.size
    flat = self.reshape([size,]).copy()
    return flat

  def hconj(self, perm: Optional[Sequence[int]] = None) -> "Tensor":
    """ The Hermitian conjugated tensor; e.g. the complex conjugate tranposed
    by the permutation set be `axes`. By default the axes are reversed.
    Args:
      perm: The permutation. If None (default) the index order is reversed.  
    Returns:
      The Hermitian conjugated `Tensor`.
    """
    dag = self.conj().transpose(perm=perm)
    return dag

  def ravel(self):
    """Return a new `Tensor` with the same number of elements but only one
    dimension.
    """
    size = self.size
    flat = self.reshape(shape=[size,])
    return flat

  def reshape(self, shape: Sequence[int]) -> "Tensor":
    """Return a new `Tensor` with the same data but a new shape `shape`.
    Args:
      shape: The new shape.
    Returns:
      The reshaped `Tensor`.
    """
    reshaped = self.backend.reshape(self.array, shape)
    return Tensor(reshaped, backend=self.backend)

  def squeeze(self):
    """Return a new `Tensor` with all axes of size 1 eliminated.
    """
    shape = self.shape
    squeezed_shape = [d for d in shape if d != 1]
    return self.reshape(squeezed_shape)

  def transpose(self, perm: Optional[Sequence[int]] = None) -> "Tensor":
    """ Return a new `Tensor` transposed according to the permutation set
    by `axes`. By default the axes are reversed.
    Args:
      axes: The permutation. If None (default) the index order is reversed.
    Returns:
      The transposed `Tensor`.
    """
    array_T = self.backend.transpose(self.array, perm=perm)
    return Tensor(array_T, backend=self.backend)

  def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
    if isinstance(other, Tensor):
      if self.backend.name != other.backend.name:
        errstr = (f"Given backens are inconsistent. Found '{self.backend.name}'"
                  f"and '{other.backend.name}'")
        raise ValueError(errstr)
      other = other.array
    array = self.backend.multiply(self.array, other)
    return Tensor(array, backend=self.backend)

  __rmul__ = __mul__

  def __truediv__(self, other: Union["Tensor", float]) -> "Tensor":
    if isinstance(other, Tensor):
      if self.backend.name != other.backend.name:
        errstr = (f"Given backens are inconsistent. Found '{self.backend.name}'"
                  f"and '{other.backend.name}'")
        raise ValueError(errstr)
      other = other.array
    array = self.backend.divide(self.array, other)
    return Tensor(array, backend=self.backend)

  def __sub__(self, other: Union["Tensor", float]) -> "Tensor":
    if isinstance(other, Tensor):
      if self.backend.name != other.backend.name:
        errstr = (f"Given backens are inconsistent. Found '{self.backend.name}'"
                  f"and '{other.backend.name}'")
        raise ValueError(errstr)
      other = other.array
    array = self.backend.subtraction(self.array, other)
    return Tensor(array, backend=self.backend)

  def __rsub__(self, other: float) -> "Tensor":
    array = self.backend.subtraction(other, self.array)
    return Tensor(array, backend=self.backend)

  def __add__(self, other: Union["Tensor", float]) -> "Tensor":
    if isinstance(other, Tensor):
      if self.backend.name != other.backend.name:
        errstr = (f"Given backens are inconsistent. Found '{self.backend.name}'"
                  f"and '{other.backend.name}'")
        raise ValueError(errstr)
      other = other.array
    array = self.backend.addition(self.array, other)
    return Tensor(array, backend=self.backend)

  __radd__ = __add__

  def __matmul__(self, other: "Tensor") -> "Tensor":
    if self.backend.name != other.backend.name:
      errstr = (f"Backends {self.backend.name} and {other.backend.name} did"
                f"not agree.")
      raise ValueError(errstr)
    array = self.backend.matmul(self.array, other.array)
    return Tensor(array, backend=self.backend)

  def __call__(self, *args):
    return NconBuilder([self], [list(args)]) 


class NconBuilder():
  def __init__(self, tensors, axes):
    self.tensors = tensors[:] # Forces a copy.
    self.axes = axes[:]

  def __matmul__(self, other: "NconBuilder") -> "NconBuilder":
    assert isinstance(other, NconBuilder)
    return NconBuilder(
        self.tensors + other.tensors,
        self.axes + other.axes)
