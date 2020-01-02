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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensornetwork.network_components import Node, contract, contract_between
# pylint: disable=line-too-long
from tensornetwork.backends import backend_factory
import copy
from typing import List, Union, Any, Optional, Tuple, Text


def _copy_charges(charges):
  cs = []
  for n in range(len(charges)):
    c = type(charges[n]).__new__(type(
        charges[n]))  #create a new charge object of type type(other)
    c.__init__(charges[n].charges.copy())
    cs.append(c)
  return cs


class BaseCharge:
  """
  Base class for fundamental charges (i.e. for symmetries that 
  are not products of smaller groups)
  """

  def __init__(self, charges: np.ndarray) -> None:
    if not isinstance(charges, np.ndarray):
      raise TypeError("only np.ndarray allowed for argument `charges` "
                      "in BaseCharge.__init__(charges)")

    self.charges = np.asarray(charges)

  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    raise NotImplementedError("`__add__` is not implemented for `BaseCharge`")

  def __mul__(self, number: int) -> "BaseCharge":
    raise NotImplementedError("`__mul__` is not implemented for `BaseCharge`")

  def __rmul__(self, number: int) -> "BaseCharge":
    raise NotImplementedError("`__rmul__` is not implemented for `BaseCharge`")

  def __matmul__(self, other: "BaseCharge") -> "Charge":
    raise NotImplementedError(
        "`__matmul__` is not implemented for `BaseCharge`")

  def __len__(self) -> int:
    return len(self.charges)

  def __repr__(self) -> str:
    return self.charges.__repr__()

  @property
  def dual_charges(self) -> np.ndarray:
    raise NotImplementedError(
        "`dual_charges` is not implemented for `BaseCharge`")

  def get_charges(self, dual: bool) -> np.ndarray:
    if dual:
      return self.dual_charges
    return self.charges


class U1ChargeMerged:
  """
  A simple charge class for a single U1 symmetry.
  """

  def __init__(self,
               charges: List[np.ndarray],
               offsets: Optional[np.ndarray] = None,
               shifts: Optional[np.ndarray] = None) -> None:
    if len(charges) > 1:
      if offsets is not None:
        raise ValueError("If `offsets` is passed, only a single charge array "
                         "can be passed. Got len(charges) = {}".format(
                             len(charges)))
      if shifts is not None:
        raise ValueError("If `shifts` is passed, only a single charge array "
                         "can be passed. Got len(charges) = {}".format(
                             len(charges)))

    self._itemsizes = [c.dtype.itemsize for c in charges]
    if np.sum(self._itemsizes) > 8:
      raise TypeError("number of bits required to store all charges "
                      "in a single int is larger than 64")

    if len(charges) > 1:
      dtype = np.int8
      if np.sum(self._itemsizes) > 1:
        dtype = np.int16
      if np.sum(self._itemsizes) > 2:
        dtype = np.int32
      if np.sum(self._itemsizes) > 4:
        dtype = np.int64
      #multiply by eight to get number of bits
      self.shifts = 8 * np.flip(
          np.append(0, np.cumsum(np.flip(self._itemsizes[1::])))).astype(dtype)
      dtype_charges = [c.astype(dtype) for c in charges]
      offsets = [
          np.min([0, np.min(dtype_charges[n])])
          for n in range(len(dtype_charges))
      ]
      pos_charges = [
          dtype_charges[n] - offsets[n] for n in range(len(dtype_charges))
      ]
      self.offsets = np.sum([
          np.left_shift(offsets[n], self.shifts[n])
          for n in range(len(dtype_charges))
      ],
                            axis=0).astype(dtype)
      self._charges = np.sum([
          np.left_shift(pos_charges[n], self.shifts[n])
          for n in range(len(dtype_charges))
      ],
                             axis=0).astype(dtype)
    else:
      if shifts is None:
        shifts = np.asarray([0]).astype(charges[0].dtype)
      self.shifts = shifts
      if offsets is None:
        self.offsets = np.min([0, np.min(charges[0])])
        self._charges = charges[0] - self.offsets
      else:
        #we assume that `charges` are in this case already
        #positive
        self.offsets = offsets
        if np.min(charges[0]) < 0:
          raise ValueError("Expected all charges to be >= 0, "
                           "but found negative charges")
        self._charges = charges[0]

  @property
  def num_symmetries(self):
    return len(self.shifts)

  def __add__(self, other: "U1ChargeMerged") -> "U1ChargeMerged":
    """
    Fuse the charges of `self` with charges of `other`, and 
    return a new `U1Charge` object holding the result.
    Args: 
      other: A `U1ChargeMerged` object.
    Returns:
      U1ChargeMerged: The result of fusing `self` with `other`.
    """
    if self.num_symmetries != other.num_symmetries:
      raise ValueError(
          "cannot fuse charges with different number of symmetries")

    if not np.all(self.shifts == other.shifts):
      raise ValueError(
          "Cannot fuse U1-charges with different shifts {} and {}".format(
              self.shifts, other.shifts))
    offsets = np.sum([self.offsets, other.offsets])
    fused = np.reshape(self._charges[:, None] + other._charges[None, :],
                       len(self._charges) * len(other._charges))
    return U1ChargeMerged(charges=[fused], offsets=offsets, shifts=self.shifts)

  def __repr__(self):
    return 'U1-charge: \n' + 'shifts: ' + self.shifts.__repr__(
    ) + '\n' + 'offsets: ' + self.offsets.__repr__(
    ) + '\n' + 'charges: ' + self._charges.__repr__()

  def __matmul__(self, other: Union["U1ChargeMerged", "U1ChargeMerged"]
                ) -> "U1ChargeMerged":
    itemsize = np.sum(self._itemsizes + other._itemsizes)
    if itemsize > 8:
      raise TypeError("number of bits required to store all charges "
                      "in a single int is larger than 64")
    dtype = np.int16  #need at least np.int16 to store two charges
    if itemsize > 2:
      dtype = np.int32
    if itemsize > 4:
      dtype = np.int64

    charges = np.left_shift(
        self._charges.astype(dtype),
        8 * np.sum(other._itemsizes)) + other._charges.astype(dtype)

    offsets = np.left_shift(
        self.offsets.astype(dtype),
        8 * np.sum(other._itemsizes)) + other.offsets.astype(dtype)
    shifts = np.append(self.shifts + 8 * np.sum(other._itemsizes), other.shifts)
    return U1ChargeMerged(charges=[charges], offsets=offsets, shifts=shifts)

  @property
  def dual_charges(self) -> np.ndarray:
    #the dual of a U1 charge is its negative value
    return (self._charges + self.offsets) * self._charges.dtype.type(-1)

  @property
  def charges(self) -> np.ndarray:
    if self.offsets != 0:
      return self._charges + self.offsets
    return self._charges

  def get_charges(self, dual: bool) -> np.ndarray:
    if dual:
      return self.dual_charges
    return self.charges

  def nonzero(self, target_charges: Union[List, np.ndarray]) -> np.ndarray:
    if len(target_charges) != len(self.shifts):
      raise ValueError("len(target_charges) = {} is different "
                       "from len(U1ChargeMerged.shifts) = {}".format(
                           len(target_charges), len(self.shifts)))
    _target_charges = np.asarray(target_charges).astype(self._charges.dtype)
    target = np.sum([
        np.left_shift(_target_charges[n], self.shifts[n])
        for n in range(len(self.shifts))
    ])
    return np.nonzero(self.charges == target)[0]


class U1Charge(BaseCharge):
  """
  A simple charge class for a single U1 symmetry.
  """

  def __init__(self, charges: np.ndarray) -> None:
    super().__init__(charges)

  def __add__(self, other: "U1Charge") -> "U1Charge":
    """
    Fuse the charges of `self` with charges of `other`, and 
    return a new `U1Charge` object holding the result.
    Args: 
      other: A `U1Charge` object.
    Returns:
      U1Charge: The result of fusing `self` with `other`.
    """
    fused = np.reshape(self.charges[:, None] + other.charges[None, :],
                       len(self.charges) * len(other.charges))

    return U1Charge(charges=fused)

  def __mul__(self, number: int) -> "U1Charge":
    return U1Charge(charges=self.charges * number)

  def __rmul__(self, number: int) -> "U1Charge":
    return U1Charge(charges=self.charges * number)

  def __matmul__(self, other: Union["U1Charge", "Charge"]) -> "Charge":
    c1 = U1Charge(self.charges.copy())  #make a copy of the charges (np.ndarray)
    if isinstance(other, U1Charge):
      c2 = type(other).__new__(
          type(other))  #create a new charge object of type type(other)
      c2.__init__(other.charges.copy())
      return Charge([c1, c2])
    #`other` should be of type `Charge`.
    return Charge([c1] + _copy_charges(other.charges))

  @property
  def dual_charges(self):
    #the dual of a U1 charge is its negative value
    return self.charges * self.charges.dtype.type(-1)


class Z2Charge(BaseCharge):
  """
  A simple charge class for a single Z2 symmetry.
  """

  def __init__(self, charges: np.ndarray) -> None:
    if charges.dtype is not np.dtype(np.bool):
      raise TypeError("Z2 charges have to be boolian")
    super().__init__(charges)

  def __add__(self, other: "U1Charge") -> "U1Charge":
    """
    Fuse the charges of `self` with charges of `other`, and 
    return a new `U1Charge` object holding the result.
    Args: 
      other: A `U1Charge` object.
    Returns:
      U1Charge: The result of fusing `self` with `other`.
    """
    fused = np.reshape(
        np.logical_xor(self.charges[:, None], other.charges[None, :]),
        len(self.charges) * len(other.charges))

    return U1Charge(charges=fused)

  def __mul__(self, number: int) -> "U1Charge":
    return U1Charge(charges=self.charges * number)

  def __rmul__(self, number: int) -> "U1Charge":
    return U1Charge(charges=self.charges * number)

  def __matmul__(self, other: Union["U1Charge", "Charge"]) -> "Charge":
    c1 = U1Charge(self.charges.copy())  #make a copy of the charges (np.ndarray)
    if isinstance(other, U1Charge):
      c2 = type(other).__new__(
          type(other))  #create a new charge object of type type(other)
      c2.__init__(other.charges.copy())
      return Charge([c1, c2])
    #`other` should be of type `Charge`.
    return Charge([c1] + _copy_charges(other.charges))

  @property
  def dual_charges(self):
    #Z2 charges are self-dual
    return self.charges


class Charge:

  def __init__(self, charges: List[Union[np.ndarray, BaseCharge]]) -> None:
    if not isinstance(charges, list):
      raise TypeError("only list allowed for argument `charges` "
                      "in BaseCharge.__init__(charges)")
    if not np.all([len(c) == len(charges[0]) for c in charges]):
      raise ValueError("not all charges have the same length. "
                       "Got lengths = {}".format([len(c) for c in charges]))
    for n in range(len(charges)):
      if not isinstance(charges[n], BaseCharge):
        raise TypeError(
            "`Charge` can only be initialized with a list of `BaseCharge`. Found {} instead"
            .format(type(charges[n])))

    self.charges = charges

  def __add__(self, other: "Charge") -> "Charge":
    """
    Fuse `self` with `other`.
    Args:
      other: A `Charge` object.
    Returns:
      Charge: The result of fusing `self` with `other`.
    """
    return Charge([c1 + c2 for c1, c2 in zip(self.charges, other.charges)])

  def __matmul__(self, other: Union["Charge", BaseCharge]) -> "Charge":
    """
    Product of `self` with `other` (group product).
    Args:
      other: A `BaseCharge` or `Charge` object.
    Returns:
      Charge: The resulting charge of the product of `self` with `other`.

    """
    if isinstance(other, BaseCharge):
      c = type(other).__new__(
          type(other))  #create a new charge object of type type(other)
      c.__init__(other.charges.copy())
      return Charge(self.charges + [c])
    elif isinstance(other, Charge):
      return Charge(_copy_charges(self.charges) + _copy_charges(other.charges))

    raise TypeError("datatype not understood")

  def __mul__(self, number: int) -> "Charge":
    return Charge(charges=[c * number for c in self.charges])

  def __rmul__(self, number: int) -> "Charge":
    return Charge(charges=[c * number for c in self.charges])

  def __len__(self):
    return len(self.charges[0])

  def __repr__(self):
    return self.charges.__repr__()

  def get_charges(self, dual: bool) -> List[np.ndarray]:
    return [c.get_charges(dual) for c in self.charges]
