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
import warnings
from typing import List, Union, Any, Optional, Tuple, Text, Iterable, Type


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

  def __init__(self,
               charges: Optional[Union[List[np.ndarray], np.ndarray]] = None,
               shifts: Optional[Union[List[int], np.ndarray]] = None) -> None:
    """
    Initialize a BaseCharge object.
    Args:
      charges: Optional `np.ndarray` or list of `np.ndarray` of type `int` holdingn
        the physical charges. If a list of `np,ndarray` is passed, the arrays are merged
        into a single `np.ndarray` by `np.left_shift`-ing and adding up charges. The amount
        of left-shift per `np,ndarray` is determined by its `dtype`. E.g. an `np,ndarray` of
        `dtype=np.int16` is shifted by 16 bits. Charges are shifted and added moving from 
         small to large indices in `charges`. `BaseCharge` can hold at most 8 individual 
         charges of `dtype=np.int8` on 64-bit architectures.
      shifts: An optional list of shifts, used for initializing a `BaseCharge` object from 
        an existing `BaseCharge` object.
    """
    if charges is not None:
      if isinstance(charges, np.ndarray):
        charges = [charges]
      self._itemsizes = [c.dtype.itemsize for c in charges]
      if np.sum(self._itemsizes) > 8:
        raise TypeError("number of bits required to store all charges "
                        "in a single int is larger than 64")

      if len(charges) > 1:
        if shifts is not None:
          raise ValueError("If `shifts` is passed, only a single charge array "
                           "can be passed. Got len(charges) = {}".format(
                               len(charges)))
      if shifts is None:
        dtype = np.int8
        if np.sum(self._itemsizes) > 1:
          dtype = np.int16
        if np.sum(self._itemsizes) > 2:
          dtype = np.int32
        if np.sum(self._itemsizes) > 4:
          dtype = np.int64
        #multiply by eight to get number of bits
        self.shifts = 8 * np.flip(
            np.append(0, np.cumsum(np.flip(
                self._itemsizes[1::])))).astype(dtype)
        dtype_charges = [c.astype(dtype) for c in charges]
        self.charges = np.sum([
            np.left_shift(dtype_charges[n], self.shifts[n])
            for n in range(len(dtype_charges))
        ],
                              axis=0).astype(dtype)
      else:
        if np.max(shifts) >= charges[0].dtype.itemsize * 8:
          raise TypeError("shifts {} are incompatible with dtype {}".format(
              shifts, charges[0].dtype))
        self.shifts = np.asarray(shifts)
        self.charges = charges[0]
    else:
      self.charges = np.asarray([])
      self.shifts = np.asarray([])

  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    """
    Fuse the charges of two `BaseCharge` objects and return a new 
    `BaseCharge` holding the result.
    Args:
      other: A `BaseChare` object.
    Returns:
      BaseCharge: The result of fusing `self` with `other`.
    """
    raise NotImplementedError("`__add__` is not implemented for `BaseCharge`")

  def __sub__(self, other: "BaseCharge") -> "BaseCharge":
    """
    Subtract the charges of `other` from `self.
    Returns a `BaseCharge` holding the result.
    Args:
      other: A `BaseChare` object.
    Returns:
      BaseCharge: The result subtracting `other` from `self`.
    """

    raise NotImplementedError("`__sub__` is not implemented for `BaseCharge`")

  def __matmul__(self, other: "BaseCharge") -> "BaseCharge":
    """
    Build the direct product of two charges and return 
    it in a new `BaseCharge` object.
    Args:
      other: A `BaseCharge` object.
    Returns:
      BaseCharge: The direct product of `self` and `other`.
    """
    raise NotImplementedError(
        "`__matmul__` is not implemented for `BaseCharge`")

  def get_item(self, n: Union[np.ndarray, int]) -> np.ndarray:
    """
    Return the charge-element at position `n`.
    Args: 
      n: An integer or `np.ndarray`.
    Returns:
      np.ndarray: The charges at `n`.
    """
    return self.charges[n]

  def get_item_ndarray(self, n: Union[np.ndarray, int]) -> np.ndarray:
    """
    Return the charge-element at position `n`.
    Needed to provide a common interface with `ChargeCollection`.
    Args: 
      n: An integer or `np.ndarray`.
    Returns:
      np.ndarray: The charges at `n`.

    """

    return self.get_item(n)

  def __getitem__(self, n: Union[np.ndarray, int]) -> "BaseCharge":
    """
    Return the charge-element at position `n`, wrapped into a `BaseCharge`
    object.
    Args: 
      n: An integer or `np.ndarray`.
    Returns:
      BaseCharge: The charges at `n`.
    """

    if isinstance(n, (np.integer, int)):
      n = np.asarray([n])
    charges = self.charges[n]
    obj = self.__new__(type(self))
    obj.__init__(charges=[charges], shifts=self.shifts)
    return obj

  @property
  def num_symmetries(self):
    """
    The number of individual symmetries stored in this object.
    """
    return len(self.shifts)

  def __len__(self) -> int:
    return np.prod(self.charges.shape)

  def __repr__(self):
    return str(type(self)) + '\nshifts: ' + self.shifts.__repr__(
    ) + '\n' + 'charges: ' + self.charges.__repr__() + '\n'

  @property
  def dual_charges(self) -> np.ndarray:
    raise NotImplementedError(
        "`dual_charges` is not implemented for `BaseCharge`")

  def __mul__(self, number: Union[bool, int]) -> "BaseCharge":
    """
    Multiply `self` with `number` from the left. 
    `number` can take values in `1,-1, 0, True, False`.
    This multiplication is used to transform between charges and dual-charges.
    Args:
      number: Can can take values in `1,-1, 0, True, False`. 
        If `1,True`, return the original object
        If `-1, 0, False` return a new `BaseCharge` holding the 
        dual-charges.
    Returns:
      BaseCharge: The result of `self * number`
    """
    raise NotImplementedError("`__mul__` is not implemented for `BaseCharge`")

  def __rmul__(self, number: Union[bool, int]) -> "BaseCharge":
    """
    Multiply `self` with `number` from the right. 
    `number` can take values in `1,-1, 0, True, False`.
    This multiplication is used to transform between charges and dual-charges.
    Args:
      number: Can can take values in `1,-1, 0, True, False`. 
        If `1,True`, return the original object
        If `-1, 0, False` return a new `BaseCharge` holding the 
        dual-charges.
    Returns:
      BaseCharge: The result of `number * self`.
    """

    raise NotImplementedError("`__rmul__` is not implemented for `BaseCharge`")

  @property
  def dtype(self):
    return self.charges.dtype

  def unique(self,
             return_index=False,
             return_inverse=False,
             return_counts=False
            ) -> Tuple["BaseCharge", np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the unique charges in `BaseCharge`.
    See np.unique for a more detailed explanation. This function
    does the same but instead of a np.ndarray, it returns the unique
    elements in a `BaseCharge` object.
    Args:
      return_index: If `True`, also return the indices of `self.charges` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
      return_inverse: If `True`, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `self.charges`.
      return_counts: If `True`, also return the number of times each unique item appears
        in `self.charges`.
    Returns:
      BaseCharge: The sorted unique values.
      np.ndarray: The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
      np.ndarray: The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
      np.ndarray: The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.      
    """
    result = np.unique(
        self.charges,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts)
    if not (return_index or return_inverse or return_counts):
      out = self.__new__(type(self))
      out.__init__([result], self.shifts)
      return out
    else:
      out = self.__new__(type(self))
      out.__init__([result[0]], self.shifts)
      return tuple([out] + [result[n] for n in range(1, len(result))])

  def isin(self, targets: Union[int, Iterable, "BaseCharge"]) -> np.ndarray:
    """
    Test each element of `BaseCharge` if it is in `targets`. Returns 
    an `np.ndarray` of `dtype=bool`.
    Args:
      targets: The test elements 
    Returns:
      np.ndarray: An array of `bool` type holding the result of the comparison.
    """
    if isinstance(targets, type(self)):
      if not np.all(self.shifts == targets.shifts):
        raise ValueError(
            "Cannot compare charges with different shifts {} and {}".format(
                self.shifts, tpargets.shifts))

      targets = targets.charges
    targets = np.asarray(targets)
    return np.isin(self.charges, targets)

  def __contains__(self, target: Union[int, Iterable, "BaseCharge"]) -> bool:
    """
    Test each element of `BaseCharge` if it is in `targets`. Returns 
    an `np.ndarray` of `dtype=bool`.
    Args:
      targets: The test elements 
    Returns:
      np.ndarray: An array of `bool` type holding the result of the comparison.
    """

    if isinstance(target, type(self)):
      if not np.all(self.shifts == target.shifts):
        raise ValueError(
            "Cannot compare charges with different shifts {} and {}".format(
                self.shifts, tparget.shifts))
      target = target.charges
    target = np.asarray(target)
    return target in self.charges

  def equals(self, target_charges: Iterable) -> np.ndarray:
    """
    Find indices where `BaseCharge` equals `target_charges`.
    `target_charges` has to be an array of the same lenghts 
    as `BaseCharge.shifts`, containing one integer per symmetry of 
    `BaseCharge`
    Args:
      target_charges: np.ndarray of integers encoding charges.
    Returns:
      np.ndarray: Boolean array with `True` where `BaseCharge` equals
      `target_charges` and `False` everywhere else.
    """
    if len(target_charges) != len(self.shifts):
      raise ValueError("len(target_charges) = {} is different "
                       "from len(shifts) = {}".format(
                           len(target_charges), len(self.shifts)))
    _target_charges = np.asarray(target_charges).astype(self.charges.dtype)
    target = np.sum([
        np.left_shift(_target_charges[n], self.shifts[n])
        for n in range(len(self.shifts))
    ])
    return self.charges == target

  def __eq__(self, target: Union[int, Iterable]) -> np.ndarray:
    """
    Find indices where `BaseCharge` equals `target_charges`.
    `target` is a single integer encoding all symmetries of
    `BaseCharge`
    Args:
      target: integerger encoding charges.
    Returns:
      np.ndarray: Boolean array with `True` where `BaseCharge.charges` equals
      `target` and `False` everywhere else.
    """
    if isinstance(target, type(self)):
      return self.charges == target.charges
    return self.charges == np.asarray(target)

  def concatenate(self, others: Union["BaseCharge", List["BaseCharge"]]):
    """
    Concatenate `self.charges` with `others.charges`.
    Args: 
      others: List of `BaseCharge` objects.
    Returns:
      BaseCharge: The concatenated charges.
    """
    if isinstance(others, type(self)):
      others = [others]
    for o in others:
      if not np.all(self.shifts == o.shifts):
        raise ValueError(
            "Cannot fuse charges with different shifts {} and {}".format(
                self.shifts, o.shifts))

    charges = np.concatenate(
        [self.charges] + [o.charges for o in others], axis=0)
    out = self.__new__(type(self))
    out.__init__([charges], self.shifts)
    return out

  @property
  def dtype(self):
    return self.charges.dtype

  @property
  def zero_charge(self):
    obj = self.__new__(type(self))
    obj.__init__(charges=[np.asarray([self.dtype.type(0)])], shifts=self.shifts)
    return obj

  def __iter__(self):
    return iter(self.charges)


class U1Charge(BaseCharge):
  """
  A simple charge class for a single U1 symmetry.
  This class can store multiple U1 charges in a single 
  np.ndarray of integer dtype. Depending on the dtype of
  the individual symmetries, this class can store:
  * 8 np.int8 
  * 4 np.int16
  * 2 np.int32
  * 1 np.int64
  or any suitable combination of dtypes, such that their 
  bite-sum remains below 64.
  """

  def __init__(self,
               charges: List[np.ndarray],
               shifts: Optional[np.ndarray] = None) -> None:
    super().__init__(charges=charges, shifts=shifts)

  def __add__(self, other: "U1Charge") -> "U1Charge":
    """
    Fuse the charges of `self` with charges of `other`, and 
    return a new `U1Charge` object holding the result.
    Args: 
      other: A `U1Charge` object.
    Returns:
      U1Charge: The result of fusing `self` with `other`.
    """
    if self.num_symmetries != other.num_symmetries:
      raise ValueError(
          "cannot fuse charges with different number of symmetries")

    if not np.all(self.shifts == other.shifts):
      raise ValueError(
          "Cannot fuse U1-charges with different shifts {} and {}".format(
              self.shifts, other.shifts))
    if not isinstance(other, U1Charge):
      raise TypeError(
          "can only add objects of identical types, found {} and {} instead"
          .format(type(self), type(other)))
    fused = np.reshape(self.charges[:, None] + other.charges[None, :],
                       len(self.charges) * len(other.charges))
    return U1Charge(charges=[fused], shifts=self.shifts)

  def __sub__(self, other: "U1Charge") -> "U1Charge":
    """
    Subtract the charges of `other` from charges of `self` and 
    return a new `U1Charge` object holding the result.
    Args: 
      other: A `U1Charge` object.
    Returns:
      U1Charge: The result of fusing `self` with `other`.
    """
    if self.num_symmetries != other.num_symmetries:
      raise ValueError(
          "cannot fuse charges with different number of symmetries")

    if not np.all(self.shifts == other.shifts):
      raise ValueError(
          "Cannot fuse U1-charges with different shifts {} and {}".format(
              self.shifts, other.shifts))
    if not isinstance(other, U1Charge):
      raise TypeError(
          "can only subtract objects of identical types, found {} and {} instead"
          .format(type(self), type(other)))

    fused = np.reshape(self.charges[:, None] - other.charges[None, :],
                       len(self.charges) * len(other.charges))
    return U1Charge(charges=[fused], shifts=self.shifts)

  def __matmul__(self, other: Union["U1Charge", "U1Charge"]) -> "U1Charge":
    itemsize = np.sum(self._itemsizes + other._itemsizes)
    if itemsize > 8:
      raise TypeError("Number of bits required to store all charges "
                      "in a single int is larger than 64")
    dtype = np.int16  #need at least np.int16 to store two charges
    if itemsize > 2:
      dtype = np.int32
    if itemsize > 4:
      dtype = np.int64

    charges = np.left_shift(
        self.charges.astype(dtype),
        8 * np.sum(other._itemsizes)) + other.charges.astype(dtype)

    shifts = np.append(self.shifts + 8 * np.sum(other._itemsizes), other.shifts)
    return U1Charge(charges=[charges], shifts=shifts)

  def __mul__(self, number: Union[bool, int]) -> "U1Charge":
    if number not in (True, False, 0, 1, -1):
      raise ValueError(
          "can only multiply by `True`, `False`, `1` or `0`, found {}".format(
              number))
    #outflowing charges
    if number in (0, False, -1):
      charges = self.dtype.type(-1) * self.charges
      shifts = self.shifts
      return U1Charge(charges=[charges], shifts=shifts)
    #inflowing charges
    if number in (1, True):
      #Note: the returned U1Charge shares its data with self
      return U1Charge(charges=[self.charges], shifts=self.shifts)

  # def __rmul__(self, number: Union[bool, int]) -> "U1Charge":
  #   raise
  #   print(number not in (True, False, 0, 1, -1))
  #   if number not in (True, False, 0, 1, -1):
  #     raise ValueError(
  #         "can only multiply by `True`, `False`, `1` or `0`, found {}".format(
  #             number))
  #   return self.__mul__(number)

  @property
  def dual_charges(self) -> np.ndarray:
    #the dual of a U1 charge is its negative value
    return self.charges * self.dtype.type(-1)


class Z2Charge(BaseCharge):
  """
  A simple charge class for Z2 symmetries.
  """

  def __init__(self,
               charges: List[np.ndarray],
               shifts: Optional[np.ndarray] = None) -> None:
    if isinstance(charges, np.ndarray):
      charges = [charges]

    if shifts is None:
      itemsizes = [c.dtype.itemsize for c in charges]
      if not np.all([i == 1 for i in itemsizes]):
        # martin: This error could come back at us, but I'll leave it for now
        warnings.warn(
            "Z2 charges can be entirely stored in "
            "np.int8, but found dtypes = {}. Converting to np.int8.".format(
                [c.dtype for c in charges]))

      charges = [c.astype(np.int8) for c in charges]

    super().__init__(charges, shifts)

  def __add__(self, other: "Z2Charge") -> "Z2Charge":
    """
    Fuse the charges of `self` with charges of `other`, and 
    return a new `Z2Charge` object holding the result.
    Args: 
      other: A `Z2Charge` object.
    Returns:
      Z2Charge: The result of fusing `self` with `other`.
    """
    if not np.all(self.shifts == other.shifts):
      raise ValueError(
          "Cannot fuse Z2-charges with different shifts {} and {}".format(
              self.shifts, other.shifts))
    if not isinstance(other, Z2Charge):
      raise TypeError(
          "can only add objects of identical types, found {} and {} instead"
          .format(type(self), type(other)))

    fused = np.reshape(
        np.bitwise_xor(self.charges[:, None], other.charges[None, :]),
        len(self.charges) * len(other.charges))

    return Z2Charge(charges=[fused], shifts=self.shifts)

  def __sub__(self, other: "Z2Charge") -> "Z2Charge":
    """
    Subtract charges of `other` from charges of `self` and 
    return a new `Z2Charge` object holding the result.
    Note that ofr Z2 charges, subtraction and addition are identical
    Args: 
      other: A `Z2Charge` object.
    Returns:
      Z2Charge: The result of fusing `self` with `other`.
    """
    if not np.all(self.shifts == other.shifts):
      raise ValueError(
          "Cannot fuse Z2-charges with different shifts {} and {}".format(
              self.shifts, other.shifts))
    if not isinstance(other, Z2Charge):
      raise TypeError(
          "can only subtract objects of identical types, found {} and {} instead"
          .format(type(self), type(other)))

    return self.__add__(other)

  def __matmul__(self, other: Union["Z2Charge", "Z2Charge"]) -> "Z2Charge":
    itemsize = np.sum(self._itemsizes + other._itemsizes)
    if itemsize > 8:
      raise TypeError("Number of bits required to store all charges "
                      "in a single int is larger than 64")
    dtype = np.int16  #need at least np.int16 to store two charges
    if itemsize > 2:
      dtype = np.int32
    if itemsize > 4:
      dtype = np.int64

    charges = np.left_shift(
        self.charges.astype(dtype),
        8 * np.sum(other._itemsizes)) + other.charges.astype(dtype)

    shifts = np.append(self.shifts + 8 * np.sum(other._itemsizes), other.shifts)
    return Z2Charge(charges=[charges], shifts=shifts)

  def __mul__(self, number: Union[bool, int]) -> "Z2Charge":
    if number not in (True, False, 0, 1, -1):
      raise ValueError(
          "can only multiply by `True`, `False`, `1` or `0`, found {}".format(
              number))
    #Z2 is self-dual
    return U1Charge(charges=[self.charges], shifts=self.shifts)

  def __rmul__(self, number: Union[bool, int]) -> "Z2Charge":
    if number not in (True, False, 0, 1, -1):
      raise ValueError(
          "can only multiply by `True`, `False`, `1` or `0`, found {}".format(
              number))

    return self.__mul__(number)

  @property
  def dual_charges(self):
    #Z2 charges are self-dual
    return self.charges

  def equals(self, target_charges: Iterable) -> np.ndarray:
    if not np.all(np.isin(target_charges, np.asarray([0, 1]))):
      raise ValueError("Z2-charges can only be 0 or 1, found charges {}".format(
          np.unique(target_charges)))
    return super().equals(target_charges)


class ChargeCollection:
  """

  """

  class Iterator:

    def __init__(self, data: np.ndarray):
      self.n = 0
      self.data = data

    def __next__(self):
      if self.n < self.data.shape[0]:
        result = self.data[self.n, :]
        self.n += 1
        return tuple(result)  #this makes a copy!
      else:
        raise StopIteration

  def __init__(self,
               charges: List[BaseCharge],
               shifts: Optional[List[np.ndarray]] = None,
               stacked_charges: Optional[np.ndarray] = None) -> None:
    if not isinstance(charges, list):
      raise TypeError("only list allowed for argument `charges` "
                      "in BaseCharge.__init__(charges)")
    if (shifts is not None) and (stacked_charges is None):
      raise ValueError(
          "Found `shifts == None` and `stacked_charges != None`."
          "`shifts` and `stacked_charges` can only be passed together.")
    if (shifts is None) and (stacked_charges is not None):
      raise ValueError(
          "Found `shifts != None` and `stacked_charges == None`."
          "`shifts` and `stacked_charges` can only be passed together.")
    self.charges = []
    if stacked_charges is None:
      if not np.all([len(c) == len(charges[0]) for c in charges]):
        raise ValueError("not all charges have the same length. "
                         "Got lengths = {}".format([len(c) for c in charges]))
      for n in range(len(charges)):
        if not isinstance(charges[n], BaseCharge):
          raise TypeError(
              "`ChargeCollection` can only be initialized "
              "with a list of `BaseCharge`. Found {} instead".format(
                  [type(charges[n]) for n in range(len(charges))]))

      self._stacked_charges = np.stack([c.charges for c in charges], axis=1)
      for n in range(len(charges)):
        charge = charges[n].__new__(type(charges[n]))
        charge.__init__(self._stacked_charges[:, n], shifts=charges[n].shifts)
        self.charges.append(charge)
    else:
      if len(shifts) != stacked_charges.shape[1]:
        raise ValueError("`len(shifts)` = {} is different from "
                         "`stacked_charges.shape[1]` = {}".format(
                             len(shifts), stacked_charges.shape[1]))

      if stacked_charges.shape[1] != len(charges):
        raise ValueError("`len(charges) and shape[1] of `stacked_charges` "
                         "have to be the same.")
      for n in range(len(charges)):
        charge = charges[n].__new__(type(charges[n]))
        charge.__init__(stacked_charges[:, n], shifts=shifts[n])
        self.charges.append(charge)
      self._stacked_charges = stacked_charges

  @classmethod
  def from_charge_types(cls, charge_types: Type, shifts: List[np.ndarray],
                        stacked_charges: np.ndarray):
    if len(charge_types) != stacked_charges.shape[1]:
      raise ValueError("`len(charge_types) and shape[1] of `stacked_charges` "
                       "have to be the same.")
    if len(charge_types) != len(shifts):
      raise ValueError(
          "`len(charge_types) and  `len(shifts)` have to be the same.")
    charges = [
        charge_types[n].__new__(charge_types[n])
        for n in range(len(charge_types))
    ]
    return cls(charges=charges, stacked_charges=stacked_charges, shifts=shifts)

  @property
  def num_charges(self) -> int:
    """
    Return the number of different charges in `ChargeCollection`.
    """
    return self._stacked_charges.shape[1]

  def get_item(self, n: int) -> Tuple:
    """
    Returns the `n-th` charge-tuple of ChargeCollection in a tuple.
    """
    if isinstance(n, (np.integer, int)):
      n = np.asarray([n])
    return tuple(self._stacked_charges[n, :].flat)

  def get_item_ndarray(self, n: Union[np.ndarray, int]) -> np.ndarray:
    """
    Returns the `n-th` charge-tuples of ChargeCollection in an np.ndarray.
    """
    if isinstance(n, (np.integer, int)):
      n = np.asarray([n])
    return self._stacked_charges[n, :]

  def __getitem__(self, n: Union[np.ndarray, int]) -> "ChargeCollection":

    if isinstance(n, (np.integer, int)):
      n = np.asarray([n])

    array = self._stacked_charges[n, :]

    return self.from_charge_types(
        charge_types=[type(c) for c in self.charges],
        shifts=[c.shifts for c in self.charges],
        stacked_charges=array)
    # if self.num_charges == 1:
    #   array = np.expand_dims(array, 0)

    # if len(array.shape) == 2:
    #   if array.shape[1] == 1:
    #     array = np.squeeze(array, axis=1)
    # if len(array.shape) == 0:
    #   array = np.asarray([array])

    # charges = []
    # if np.prod(array.shape) == 0:
    #   for n in range(len(self.charges)):
    #     charge = self.charges[n].__new__(type(self.charges[n]))
    #     charge.__init__(
    #         charges=[np.empty(0, dtype=self.charges[n].dtype)],
    #         shifts=self.charges[n].shifts)
    #     charges.append(charge)

    #   obj = self.__new__(type(self))
    #   obj.__init__(charges=charges)
    #   return obj

    # if len(array.shape) == 1:
    #   array = np.expand_dims(array, 1)

    # for m in range(len(self.charges)):
    #   charge = self.charges[m].__new__(type(self.charges[m]))
    #   charge.__init__(charges=[array[:, m]], shifts=self.charges[m].shifts)
    #   charges.append(charge)

    # obj = self.__new__(type(self))
    # obj.__init__(charges=charges)
    # return obj

  def __iter__(self):
    return self.Iterator(self._stacked_charges)

  def __add__(self, other: "Charge") -> "Charge":
    """
    Fuse `self` with `other`.
    Args:
      other: A `ChargeCollection` object.
    Returns:
      Charge: The result of fusing `self` with `other`.
    """
    return ChargeCollection(
        [c1 + c2 for c1, c2 in zip(self.charges, other.charges)])

  def __sub__(self, other: "Charge") -> "Charge":
    """
    Subtract `other` from `self`.
    Args:
      other: A `ChargeCollection` object.
    Returns:
      Charge: The result of fusing `self` with `other`.
    """
    return ChargeCollection(
        [c1 - c2 for c1, c2 in zip(self.charges, other.charges)])

  def __repr__(self):
    text = str(type(self)) + '\n '
    for n in range(len(self.charges)):
      tmp = self.charges[n].__repr__()
      tmp = tmp.replace('\n', '\n\t')
      text += (tmp + '\n')
    return text

  def __len__(self):
    return len(self.charges[0])

  def __mul__(self, number: Union[bool, int]) -> "Charge":
    if number not in (True, False, 0, 1, -1):
      raise ValueError(
          "can only multiply by `True`, `False`, `1` or `0`, found {}".format(
              number))
    return ChargeCollection(charges=[number * c for c in self.charges])

  def __rmul__(self, number: Union[bool, int]) -> "Charge":
    if number not in (True, False, 0, 1, -1):
      raise ValueError(
          "can only multiply by `True`, `False`, `1` or `0`, found {}".format(
              number))

    return self.__mul__(number)

  def isin(self, targets: Union[Iterable, "ChargeCollection"]):
    if isinstance(targets, type(self)):
      _targets = [t for t in targets]
    return np.logical_or.reduce([
        np.logical_and.reduce([
            np.isin(self._stacked_charges[:, n], _targets[m][n])
            for n in range(len(_targets[m]))
        ])
        for m in range(len(_targets))
    ])

  def __contains__(self, targets: Union[Iterable, "ChargeCollection"]):
    if isinstance(targets, type(self)):
      if len(targets) > 1:
        raise ValueError(
            '__contains__ expects a single input, found {} inputs'.format(
                len(targets)))

      _targets = targets.get_item(0)
    return np.any(
        np.logical_and.reduce([
            np.isin(self._stacked_charges[:, n], _targets[n])
            for n in range(len(_targets))
        ]))

  def unique(
      self,
      return_index=False,
      return_inverse=False,
      return_counts=False,
  ) -> Tuple["ChargeCollection", np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the unique charges in `BaseCharge`.
    See np.unique for a more detailed explanation. This function
    does the same but instead of a np.ndarray, it returns the unique
    elements in a `BaseCharge` object.
    Args:
      return_index: If `True`, also return the indices of `self.charges` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
      return_inverse: If `True`, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `self.charges`.
      return_counts: If `True`, also return the number of times each unique item appears
        in `self.charges`.
    Returns:
      BaseCharge: The sorted unique values.
      np.ndarray: The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
      np.ndarray: The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
      np.ndarray: The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.      
    """

    result = np.unique(
        np.stack([self.charges[n].charges for n in range(len(self.charges))],
                 axis=1),
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=0)
    charges = []
    if not (return_index or return_inverse or return_counts):
      for n in range(len(self.charges)):
        obj = self.charges[n].__new__(type(self.charges[n]))
        obj.__init__(charges=[result[:, n]], shifts=self.charges[n].shifts)
        charges.append(obj)
      return ChargeCollection(charges)
    for n in range(len(self.charges)):
      obj = self.charges[n].__new__(type(self.charges[n]))
      obj.__init__(charges=[result[0][:, n]], shifts=self.charges[n].shifts)
      charges.append(obj)
      out = ChargeCollection(charges)
    return tuple([out] + [result[n] for n in range(1, len(result))])

  def equals(self, target_charges: List[Union[List, np.ndarray]]) -> np.ndarray:
    if len(target_charges) != len(self.charges):
      raise ValueError(
          "len(target_charges) ={} is different from len(ChargeCollection.charges) = {}"
          .format(len(target_charges), len(self.charges)))
    return np.logical_and.reduce([
        self.charges[n].equals(target_charges[n])
        for n in range(len(target_charges))
    ])

  def __eq__(self, target_charges: Iterable):
    if isinstance(target_charges, type(self)):
      target_charges = np.stack([c.charges for c in target_charges.charges],
                                axis=1)
    target_charges = np.asarray(target_charges)
    if target_charges.ndim == 1:
      target_charges = np.expand_dims(target_charges, 0)
    if target_charges.shape[1] != len(self.charges):
      raise ValueError(
          "len(target_charges) ={} is different from len(ChargeCollection.charges) = {}"
          .format(len(target_charges), len(self.charges)))
    return np.logical_and.reduce(
        self._stacked_charges == target_charges, axis=1)

  def concatenate(self,
                  others: Union["ChargeCollection", List["ChargeCollection"]]):
    """
    Concatenate `self.charges` with `others.charges`.
    Args: 
      others: List of `BaseCharge` objects.
    Returns:
      BaseCharge: The concatenated charges.
    """
    if isinstance(others, type(self)):
      others = [others]

    charges = [
        self.charges[n].concatenate([o.charges[n]
                                     for o in others])
        for n in range(len(self.charges))
    ]
    return ChargeCollection(charges)

  @property
  def dtype(self):
    return np.result_type(*[c.dtype for c in self.charges])

  @property
  def zero_charge(self):
    obj = self.__new__(type(self))
    obj.__init__(charges=[c.zero_charge for c in self.charges])
    return obj


def fuse_charges(
    charges: List[Union[BaseCharge, ChargeCollection]],
    flows: List[Union[bool, int]]) -> Union[BaseCharge, ChargeCollection]:
  """
  Fuse all `charges` into a new charge.
  Charges are fused from "right to left", 
  in accordance with row-major order.

  Args:
    charges: A list of charges to be fused.
    flows: A list of flows, one for each element in `charges`.
  Returns:
    ChargeCollection: The result of fusing `charges`.
  """
  if len(charges) != len(flows):
    raise ValueError(
        "`charges` and `flows` are of unequal lengths {} != {}".format(
            len(charges), len(flows)))
  fused_charges = charges[0] * flows[0]
  for n in range(1, len(charges)):
    fused_charges = fused_charges + charges[n] * flows[n]
  return fused_charges


def fuse_degeneracies(degen1: Union[List, np.ndarray],
                      degen2: Union[List, np.ndarray]) -> np.ndarray:
  """
  Fuse degeneracies `degen1` and `degen2` of two leg-charges 
  by simple kronecker product. `degen1` and `degen2` typically belong to two 
  consecutive legs of `BlockSparseTensor`.
  Given `degen1 = [1, 2, 3]` and `degen2 = [10, 100]`, this returns
  `[10, 100, 20, 200, 30, 300]`.
  When using row-major ordering of indices in `BlockSparseTensor`, 
  the position of `degen1` should be "to the left" of the position of `degen2`.
  Args:
    degen1: Iterable of integers
    degen2: Iterable of integers
  Returns:
    np.ndarray: The result of fusing `dege1` with `degen2`.
  """
  return np.reshape(degen1[:, None] * degen2[None, :],
                    len(degen1) * len(degen2))
