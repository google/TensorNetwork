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


class BaseCharge:

  def __init__(self,
               charges: np.ndarray,
               charge_labels: Optional[np.ndarray] = None) -> None:
    if charges.dtype is not np.int16:
      raise TypeError("`charges` have to be of dtype `np.int16`")
    if charge_labels.dtype is not np.int16:
      raise TypeError("`charge_labels` have to be of dtype `np.int16`")

    if charge_labels is None:
      self.unique_charges, charge_labels = np.unique(
          charges, return_inverse=True)
      self.charge_labels = charge_labels.astype(np.uint16)

    else:
      self.unique_charges = charges
      self.charge_labels = charge_labels.astype(np.uint16)

  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    # fuse the unique charges from each index, then compute new unique charges
    comb_qnums = self.fuse(self.unique_charges, other.unique_charges)
    [unique_charges, new_labels] = np.unique(comb_qnums, return_inverse=True)
    new_labels = new_labels.reshape(
        len(self.unique_charges), len(other.unique_charges)).astype(np.uint16)

    # find new labels using broadcasting (could use np.tile but less efficient)
    charge_labels = new_labels[(
        self.charge_labels[:, None] + np.zeros([1, len(other)], dtype=np.uint16)
    ).ravel(), (other.charge_labels[None, :] +
                np.zeros([len(self), 1], dtype=np.uint16)).ravel()]
    obj = self.__new__(type(self))
    obj.__init__(unique_charges, charge_labels)
    return obj

  def __len__(self):
    return len(self.charge_labels)

  @property
  def charges(self) -> np.ndarray:
    return self.unique_charges[self.charge_labels]

  @property
  def dtype(self):
    return self.unique_charges.dtype

  def __repr__(self):
    return str(type(self)) + '\n' + 'charges: ' + self.charges.__repr__() + '\n'

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
    obj = self.__new__(type(self))
    obj.__init__(
        self.unique_charges,
        charge_labels=np.arange(len(self.unique_charges), dtype=np.uint16))

    out = [obj]
    if return_index:
      _, index = np.unique(self.charge_labels, return_index=True)
      out.append(index)
    if return_inverse:
      out.append(self.charge_labels)
    if return_counts:
      _, cnts = np.unique(self.charge_labels, return_counts=True)
      out.append(cnts)
    if len(out) == 1:
      return out[0]
    if len(out) == 2:
      return out[0], out[1]
    if len(out) == 3:
      return out[0], out[1], out[2]

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
      targets = targets.unique_charges
    targets = np.asarray(targets)
    common, label_to_unique, label_to_targets = np.intersect1d(
        self.unique_charges, targets, return_indices=True)
    if len(common) == 0:
      return np.full(len(self.charge_labels), fill_value=False, dtype=np.bool)
    return np.isin(self.charge_labels, label_to_unique)

  def __contains__(self, target: Union[int, Iterable, "BaseCharge"]) -> bool:
    """
    """

    if isinstance(target, type(self)):
      target = target.unique_charges
    target = np.asarray(target)
    return target in self.unique_charges

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
      target = target.charges
    elif isinstance(target, (np.integer, int)):
      target = np.asarray([target])
    target = np.asarray(target)
    tmp = np.full(len(target), fill_value=-1, dtype=np.int16)

    _, label_to_unique, label_to_target = np.intersect1d(
        self.unique_charges, target, return_indices=True)
    tmp[label_to_target] = label_to_unique
    return np.squeeze(
        np.expand_dims(self.charge_labels, 1) == np.expand_dims(tmp, 0))

  @property
  def zero_charge(self):
    obj = self.__new__(type(self))
    obj.__init__(
        np.asarray([self.dtype.type(0)]), np.asarray([0], dtype=np.uint16))
    return obj

  def __iter__(self):
    return iter(self.charges)

  def intersect(self,
                other: "BaseCharge",
                return_indices: Optional[bool] = False) -> "BaseCharge":
    if return_indices:
      charges, comm1, comm2 = np.intersect1d(
          self.charges, other.charges, return_indices=return_indices)
    else:
      charges = np.intersect1d(self.charges, other.charges)

    obj = self.__new__(type(self))
    obj.__init__(charges, np.arange(len(charges), dtype=np.uint16))
    if return_indices:
      return obj, comm1.astype(np.uint16), comm2.astype(np.uint16)
    return obj

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
    obj = self.__new__(type(self))
    obj.__init__(self.unique_charges, self.charge_labels[n])
    return obj

  def get_item(self, n: Union[np.ndarray, int]) -> np.ndarray:
    """
    Return the charge-element at position `n`.
    Args: 
      n: An integer or `np.ndarray`.
    Returns:
      np.ndarray: The charges at `n`.
    """
    return self.charges[n]

  def __mul__(self, number: Union[bool, int]) -> "U1Charge":
    if number not in (True, False, 0, 1, -1):
      raise ValueError(
          "can only multiply by `True`, `False`, `1` or `0`, found {}".format(
              number))
    #outflowing charges
    if number in (0, False, -1):
      return U1Charge(
          self.dual_charges(self.unique_charges), self.charge_labels)
    #inflowing charges
    if number in (1, True):
      return U1Charge(self.unique_charges, self.charge_labels)

  @property
  def dual(self, charges):
    return self.dual_charges


class U1Charge(BaseCharge):

  def __init__(self,
               charges: np.ndarray,
               charge_labels: Optional[np.ndarray] = None) -> None:
    super().__init__(charges, charge_labels)

  @staticmethod
  def fuse(charge1, charge2):
    return np.add.outer(charge1, charge2).ravel()

  @staticmethod
  def dual_charges(charges):
    return charges * charges.dtype.type(-1)


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
    return Z2Charge(charges=[self.charges], shifts=self.shifts)

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
    return ChargeCollection(charges=[c * number for c in self.charges])

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
    raise NotImplementedError()
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

  def intersect(self,
                other: "ChargeCollection",
                return_indices: Optional[bool] = False) -> "ChargeCollection":
    if return_indices:
      ua, ia = self.unique(return_index=True)
      ub, ib = other.unique(return_index=True)
      conc = ua.concatenate(ub)
      uab, iab, cntab = conc.unique(return_index=True, return_counts=True)
      intersection = uab[cntab == 2]
      comm1 = np.argmax(
          np.logical_and.reduce(
              np.repeat(
                  np.expand_dims(self._stacked_charges, 2),
                  intersection._stacked_charges.shape[0],
                  axis=2) == np.expand_dims(intersection._stacked_charges.T, 0),
              axis=1),
          axis=0)
      comm2 = np.argmax(
          np.logical_and.reduce(
              np.repeat(
                  np.expand_dims(other._stacked_charges, 2),
                  intersection._stacked_charges.shape[0],
                  axis=2) == np.expand_dims(intersection._stacked_charges.T, 0),
              axis=1),
          axis=0)
      return intersection, comm1, comm2

    else:
      self_unique = self.unique()
      other_unique = other.unique()
      concatenated = self_unique.concatenate(other_unique)
      tmp_unique, counts = concatenated.unique(return_counts=True)
      return tmp_unique[counts == 2]


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
