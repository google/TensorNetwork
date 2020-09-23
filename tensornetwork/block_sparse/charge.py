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

import numpy as np
from tensornetwork.block_sparse.utils import intersect, unique
from typing import (List, Optional, Type, Any, Union, Callable)

#TODO (mganahl): clean up implementation of identity charges

class BaseCharge:
  """
  Base class for charges of BlockSparseTensor. All user defined charges 
  should be derived from this class.
  Attributes:
    * unique_charges: np.ndarray of shape `(m,n)` with `m`
      the number of charge types, and `n` the number of unique charges.
    * charge_labels: np.ndarray of dtype np.int16. Used for identifying 
      charges with integer labels. `unique_charges[:, charge_labels] 
      is the np.ndarray of actual charges.
    * charge_types: A list of `type` objects. Stored the different charge types,
      on for each row in `unique_charges`.
      
  """

  class Iterator:
    def __init__(self, charges: np.ndarray):
      self.n = 0
      self.charges = charges

    def __next__(self):
      if self.n < self.charges.shape[0]:
        out = self.charges[self.n, :]
        self.n += 1
        return out
      raise StopIteration

  def __init__(self,
               charges: Union[List, np.ndarray],
               charge_labels: Optional[np.ndarray] = None,
               charge_types: Optional[List[Type["BaseCharge"]]] = None,
               charge_dtype: Optional[Type[np.number]] = np.int16) -> None:
    charges = np.asarray(charges)
    if charges.ndim == 1:
      charges = charges[:, None]
    if (charge_types is not None) and (len(charge_types) != charges.shape[1]):
      raise ValueError(
          "`len(charge_types) = {}` does not match `charges.shape[1]={}`"
          .format(len(charge_types), charges.shape[1]))
    self.num_symmetries = charges.shape[1]
    if charges.shape[1] < 3:
      self.label_dtype = np.int16
    else:
      self.label_dtype = np.int32
    if charge_types is None:
      charge_types = [type(self)] * self.num_symmetries
    self.charge_types = charge_types

    if charge_labels is None:
      self._unique_charges = None
      self._charge_labels = None
      self._charges = charges.astype(charge_dtype)


    else:
      self._charge_labels = np.asarray(charge_labels, dtype=self.label_dtype)
      self._unique_charges = charges.astype(charge_dtype)
      self._charges = None

  @property
  def unique_charges(self):
    if self._unique_charges is None:
      self._unique_charges, self._charge_labels = unique(
          self.charges, return_inverse=True)
      self._charges = None
    return self._unique_charges

  @property
  def charge_labels(self):
    if self._charge_labels is None:
      self._unique_charges, self._charge_labels = unique(
          self.charges, return_inverse=True)
      self._charges = None
    return self._charge_labels

  @property
  def charges(self):
    if self._charges is not None:
      return self._charges
    return self._unique_charges[self._charge_labels, :]

  @staticmethod
  def fuse(charge1, charge2):
    raise NotImplementedError("`fuse` has to be implemented in derived classes")

  @staticmethod
  def dual_charges(charges):
    raise NotImplementedError(
        "`dual_charges` has to be implemented in derived classes")

  @staticmethod
  def identity_charge():
    raise NotImplementedError(
        "`identity_charge` has to be implemented in derived classes")

  @classmethod
  def random(cls, dimension: int, minval: int, maxval: int):
    raise NotImplementedError(
        "`random` has to be implemented in derived classes")

  @property
  def dim(self):
    if self._charge_labels is not None:
      return len(self._charge_labels)
    return self._charges.shape[0]

  @property
  def num_unique(self) -> int:
    """
    Return the number of different charges in `ChargeCollection`.
    """
    return self.unique_charges.shape[0]

  def copy(self):
    """
    Return a copy of `BaseCharge`.
    """
    if self._unique_charges is not None:
      charges = self._unique_charges.copy()
      labels = self._charge_labels.copy()
    else:
      charges = self._charges.copy()
      labels = None

    obj = self.__new__(type(self))
    obj.__init__(
        charges=charges,
        charge_labels=labels,
        charge_types=self.charge_types,
        charge_dtype=self.dtype)
    return obj

  @property
  def dtype(self):
    if self._unique_charges is not None:
      return self._unique_charges.dtype
    return self._charges.dtype

  def __repr__(self):
    return 'BaseCharge object:' + '\n   charge types: ' + self.names + \
        '\n   unique charges:' + str(self.charges.T).replace('\n', '\n\t\t  ')\
        + '\n'

  def __iter__(self):
    return self.Iterator(self.charges)

  def __len__(self):
    if self._charges is not None:
      return self.charges.shape[0]
    return self._charge_labels.shape[0]

  def __eq__(self, target_charges: Union[np.ndarray,
                                         "BaseCharge"]) -> np.ndarray:
    if isinstance(target_charges, type(self)):
      if len(target_charges) == 0:
        raise ValueError('input to __eq__ cannot be an empty charge')
      targets = target_charges.charges
    else:
      if target_charges.ndim == 1:
        target_charges = target_charges[:, None]
      if target_charges.shape[0] == 0:
        raise ValueError('input to __eq__ cannot be an empty np.ndarray')
      if target_charges.shape[1] != self.num_symmetries:
        raise ValueError("shape of `target_charges = {}` is incompatible with "
                         "`self.num_symmetries = {}".format(
                             target_charges.shape, self.num_symmetries))
      targets = target_charges
    return np.logical_and.reduce(
        self.charges[:, :, None] == targets.T[None, :, :], axis=1)

  def identity_charges(self, dim: int = 1) -> "BaseCharge":
    """
    Returns the identity charge.
    Returns:
      BaseCharge: The identity charge.
    """
    charges = np.concatenate(
        [
            np.asarray([ct.identity_charge() for ct in self.charge_types],
                       dtype=self.dtype)[None, :]
        ] * dim,
        axis=0)
    obj = self.__new__(type(self))
    obj.__init__(
        charges=charges, charge_labels=None, charge_types=self.charge_types)
    return obj

  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    """
    Fuse `self` with `other`.
    Args:
      other: A `BaseCharge` object.
    Returns:
      BaseCharge: The result of fusing `self` with `other`.
    """
    # fuse the unique charges from each index, then compute new unique charges
    fused_charges = fuse_ndarray_charges(self.charges, other.charges,
                                         self.charge_types)
    obj = self.__new__(type(self))
    obj.__init__(fused_charges, charge_types=self.charge_types)
    return obj

  def dual(self, take_dual: Optional[bool] = False) -> "BaseCharge":
    """
    Return the charges of `BaseCharge`, possibly conjugated.
    Args:
      take_dual: If `True` return the dual charges. If `False` return 
        regular charges.
    Returns:
      BaseCharge
    """
    if take_dual:
      if self._unique_charges is not None:
        unique_dual_charges = np.stack([
            self.charge_types[n].dual_charges(self._unique_charges[:, n])
            for n in range(len(self.charge_types))
        ],
                                       axis=1)

        obj = self.__new__(type(self))
        obj.__init__(
            unique_dual_charges,
            charge_labels=self.charge_labels,
            charge_types=self.charge_types)
        return obj
      dual_charges = np.stack([
          self.charge_types[n].dual_charges(self._charges[:, n])
          for n in range(len(self.charge_types))
      ],
                              axis=1)

      obj = self.__new__(type(self))
      obj.__init__(
          dual_charges, charge_labels=None, charge_types=self.charge_types)
      return obj

    return self

  def __matmul__(self, other):
    #some checks
    if len(self) != len(other):
      raise ValueError(
          '__matmul__ requires charges to have the same number of elements')
    charges = np.concatenate([self.charges, other.charges], axis=1)
    charge_types = self.charge_types + other.charge_types
    return BaseCharge(
        charges=charges, charge_labels=None, charge_types=charge_types)

  def __mul__(self, number: bool) -> "BaseCharge":
    if not isinstance(number, (bool, np.bool_)):
      raise ValueError(
          "can only multiply by `True` or `False`, found {}".format(number))
    return self.dual(number)

  def intersect(self, other, assume_unique=False, return_indices=False) -> Any:
    """
    Compute the intersection of `self` with `other`. See also np.intersect1d.

    Args:
      other: A BaseCharge object.
      assume_unique: If `True` assume that elements are unique.
      return_indices: If `True`, return index-labels.

    Returns:
      If `return_indices=True`:
        BaseCharge
        np.ndarray: The indices of the first occurrences of the 
          common values in `self`.
        np.ndarray: The indices of the first occurrences of the 
          common values in `other`.
      If `return_indices=False`:
        BaseCharge
    """
    if isinstance(other, type(self)):
      out = intersect(
          self.charges,
          other.charges,
          assume_unique=assume_unique,
          axis=0,
          return_indices=return_indices)
    else:
      if other.ndim == 1:
        other = other[:, None]
      out = intersect(
          self.charges,
          np.asarray(other),
          axis=0,
          assume_unique=assume_unique,
          return_indices=return_indices)
    obj = self.__new__(type(self))
    if return_indices:
      obj.__init__(
          charges=out[0],
          charge_labels=np.arange(out[0].shape[0], dtype=self.label_dtype),
          charge_types=self.charge_types,
      )
      return obj, out[1], out[2]
    obj.__init__(
        charges=out,
        charge_labels=np.arange(out.shape[0], dtype=self.label_dtype),
        charge_types=self.charge_types,
    )
    return obj

  def unique(self, #pylint: disable=inconsistent-return-statements
             return_index: bool = False,
             return_inverse: bool = False,
             return_counts: bool = False) -> Any:
    """
    Compute the unique charges in `BaseCharge`.
    See unique for a more detailed explanation. This function
    does the same but instead of a np.ndarray, it returns the unique
    elements (not neccessarily sorted in standard order) in a `BaseCharge` 
    object.

    Args:
      return_index: If `True`, also return the indices of `self.charges` 
        (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
      return_inverse: If `True`, also return the indices of the unique array 
        (for the specified
        axis, if provided) that can be used to reconstruct `self.charges`.
      return_counts: If `True`, also return the number of times each unique 
        item appears in `self.charges`.

    Returns:
      BaseCharge: The sorted unique values.
      np.ndarray: The indices of the first occurrences of the unique values 
        in the original array. Only provided if `return_index` is True.
      np.ndarray: The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
      np.ndarray: The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.      
    """

    obj = self.__new__(type(self))
    if self._charges is not None:
      tmp = unique(
          self._charges,
          return_index=return_index,
          return_inverse=return_inverse,
          return_counts=return_counts)

      if any([return_index, return_inverse, return_counts]):
        unique_charges = tmp[0]
        obj.__init__(
            charges=unique_charges,
            charge_labels=np.arange(
                unique_charges.shape[0], dtype=self.label_dtype),
            charge_types=self.charge_types)
        tmp[0] = obj
      else:
        obj.__init__(
            charges=tmp,
            charge_labels=np.arange(tmp.shape[0], dtype=self.label_dtype),
            charge_types=self.charge_types)
        tmp = obj
      return tmp

    if self._unique_charges is not None:
      if not return_index:
        obj.__init__(
            charges=self._unique_charges,
            charge_labels=np.arange(
                self._unique_charges.shape[0], dtype=self.label_dtype),
            charge_types=self.charge_types)

        out = [obj]
        if return_inverse:
          out.append(self._charge_labels)

        if return_counts:
          _, cnts = unique(self._charge_labels, return_counts=True)
          out.append(cnts)
        if len(out) > 1:
          return out
        return out[0]
      tmp = unique(
          self._charge_labels,
          return_index=return_index,
          return_inverse=return_inverse,
          return_counts=return_counts)

      unique_charges = self._unique_charges[tmp[0], :]
      obj.__init__(
          charges=unique_charges,
          charge_labels=np.arange(
              unique_charges.shape[0], dtype=self.label_dtype),
          charge_types=self.charge_types)
      tmp[0] = obj
      return tmp

  def reduce(self,
             target_charges: Union[int, np.ndarray],
             return_locations: bool = False,
             strides: Optional[int] = 1) -> Any:
    """
    Reduce the dimension of a 
    charge to keep only the charge values that intersect target_charges
    Args:
      target_charges: array of unique charges to keep.
      return_locations: If `True`, also return the locations of 
        target values within `BaseCharge`.
      strides: An optional stride value.
    Returns:
      BaseCharge: charge of reduced dimension.
      np.ndarray: If `return_locations = True`; the index locations 
        of target values.
    """
    if isinstance(target_charges, (np.integer, int)):
      target_charges = np.asarray([target_charges], dtype=self.dtype)
    if target_charges.ndim == 1:
      target_charges = target_charges[:, None]
    target_charges = np.asarray(target_charges, dtype=self.dtype)
    # find intersection of index charges and target charges
    reduced_charges, label_to_unique, _ = intersect(
        self.unique_charges, target_charges, axis=0, return_indices=True)
    num_unique = len(label_to_unique)

    # construct the map to the reduced charges
    map_to_reduced = np.full(self.dim, fill_value=-1, dtype=self.label_dtype)
    map_to_reduced[label_to_unique] = np.arange(
        num_unique, dtype=self.label_dtype)

    # construct the map to the reduced charges
    reduced_ind_labels = map_to_reduced[self.charge_labels]
    reduced_locs = reduced_ind_labels >= 0
    new_ind_labels = reduced_ind_labels[reduced_locs].astype(self.label_dtype)
    obj = self.__new__(type(self))
    obj.__init__(reduced_charges, new_ind_labels, self.charge_types)

    if return_locations:
      return obj, strides * np.flatnonzero(reduced_locs).astype(np.uint32)
    return obj

  def __getitem__(self, n: Union[List[int], np.ndarray, int]) -> "BaseCharge":
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
    n = np.asarray(n)
    obj = self.__new__(type(self))
    if self._unique_charges is not None:
      labels = self.charge_labels[n]
      unique_labels, new_labels = unique(labels, return_inverse=True)
      unique_charges = self.unique_charges[unique_labels, :]
      obj.__init__(unique_charges, new_labels, self.charge_types)
      return obj
    obj.__init__(
        self._charges[n, :], charge_labels=None, charge_types=self.charge_types)
    return obj

  @property
  def names(self):
    return repr([ct.__new__(ct).__class__.__name__ for ct in self.charge_types])


class U1Charge(BaseCharge):
  """Charge Class for the U1 symmetry group."""

  @staticmethod
  def fuse(charge1: np.ndarray, charge2: np.ndarray) -> np.ndarray:
    return np.add.outer(charge1, charge2).ravel()

  @staticmethod
  def dual_charges(charges: np.ndarray) -> np.ndarray:
    return charges * charges.dtype.type(-1)

  @staticmethod
  def identity_charge() -> np.ndarray:
    return np.int16(0)

  @classmethod
  def random(cls, dimension: int, minval: int, maxval: int) -> BaseCharge:
    charges = np.random.randint(minval, maxval + 1, dimension, dtype=np.int16)
    return cls(charges=charges)


class Z2Charge(BaseCharge):
  """Charge Class for the Z2 symmetry group."""

  def __init__(self,
               charges: Union[List, np.ndarray],
               charge_labels: Optional[np.ndarray] = None,
               charge_types: Optional[List[Type["BaseCharge"]]] = None,
               charge_dtype: Optional[Type[np.number]] = np.int16) -> None:
    #do some checks before calling the base class constructor
    unique_charges = unique(np.ravel(charges))
    if not np.all(np.isin(unique_charges, [0, 1])):
      raise ValueError("Z2 charges can only be 0 or 1, found {}".format(unique))
    super().__init__(
        charges,
        charge_labels,
        charge_types=[type(self)],
        charge_dtype=charge_dtype)

  @staticmethod
  def fuse(charge1: np.ndarray, charge2: np.ndarray) -> np.ndarray:
    #pylint: disable=no-member
    return np.bitwise_xor.outer(charge1, charge2).ravel()

  @staticmethod
  def dual_charges(charges: np.ndarray) -> np.ndarray:
    return charges

  @staticmethod
  def identity_charge() -> np.ndarray:
    return np.int16(0)

  @classmethod
  def random(cls,
             dimension: int,
             minval: int = 0,
             maxval: int = 1) -> BaseCharge:
    if minval != 0 or maxval != 1:
      raise ValueError("Z2 charges can only take values 0 or 1")

    charges = np.random.randint(0, 2, dimension, dtype=np.int16)
    return cls(charges=charges)


def ZNCharge(n: int) -> Callable:
  """Contstructor for charge classes of the ZN symmetry groups.

  Args:
    n: The module of the symmetry group.
  Returns:
    A charge class of your given ZN symmetry group.
  """
  if n < 2:
    raise ValueError(f"n must be >= 2, found {n}")

  class ModularCharge(BaseCharge):

    def __init__(self,
                 charges: Union[List, np.ndarray],
                 charge_labels: Optional[np.ndarray] = None,
                 charge_types: Optional[List[Type["BaseCharge"]]] = None,
                 charge_dtype: Optional[Type[np.number]] = np.int16) -> None:
      unique_charges = unique(np.ravel(charges))
      if not np.all(np.isin(unique_charges, list(range(n)))):
        raise ValueError(f"Z{n} charges must be in range({n}), found: {unique}")
      super().__init__(
          charges,
          charge_labels,
          charge_types=[type(self)],
          charge_dtype=charge_dtype)

    @staticmethod
    def fuse(charge1: np.ndarray, charge2: np.ndarray) -> np.ndarray:
      return np.add.outer(charge1, charge2).ravel() % n

    @staticmethod
    def dual_charges(charges: np.ndarray) -> np.ndarray:
      return (n - charges) % n

    @staticmethod
    def identity_charge() -> np.ndarray:
      return np.int16(0)

    @classmethod
    def random(cls,
               dimension: int,
               minval: int = 0,
               maxval: int = n - 1) -> BaseCharge:
      if maxval >= n:
        raise ValueError(f"maxval must be less than n={n}, got {maxval}")
      if minval < 0:
        raise ValueError(f"minval must be greater than 0, found {minval}")
      # No need for the mod due to the checks above.
      charges = np.random.randint(minval, maxval + 1, dimension, dtype=np.int16)
      return cls(charges=charges)

  return ModularCharge


def fuse_ndarray_charges(charges_A: np.ndarray, charges_B: np.ndarray,
                         charge_types: List[Type[BaseCharge]]) -> np.ndarray:
  """
  Fuse the quantum numbers of two indices under their kronecker addition.
  Args:
    charges_A (np.ndarray): n-by-D1 dimensional array integers encoding charges,
      with n the number of symmetries and D1 the index dimension.
    charges__B (np.ndarray): n-by-D2 dimensional array of charges.
    charge_types: A list of types of the charges.
  Returns:
    np.ndarray: n-by-(D1 * D2) dimensional array of the fused charges.
  """
  comb_charges = [0] * len(charge_types)
  for n, ct in enumerate(charge_types):
    comb_charges[n] = ct.fuse(charges_A[:, n], charges_B[:, n])[:, None]
  return np.concatenate(comb_charges, axis=1)


def fuse_charges(charges: List[BaseCharge], flows: List[bool]) -> BaseCharge:
  """
  Fuse all `charges` into a new charge.
  Charges are fused from "right to left",
  in accordance with row-major order.

  Args:
    charges: A list of charges to be fused.
    flows: A list of flows, one for each element in `charges`.
  Returns:
    BaseCharge: The result of fusing `charges`.
  """
  if len(charges) != len(flows):
    raise ValueError(
        "`charges` and `flows` are of unequal lengths {} != {}".format(
            len(charges), len(flows)))
  fused_charges = charges[0] * flows[0]
  for n in range(1, len(charges)):
    fused_charges = fused_charges + charges[n] * flows[n]
  return fused_charges


def charge_equal(c1: BaseCharge, c2: BaseCharge) -> bool:
  """
  Compare two BaseCharges `c1` and `c2`.
  Return `True` if they are equal, else `False`.
  """
  res = True
  if c1.dim != c2.dim:
    return False

  res = True      
  if c1._unique_charges is not None and c2._unique_charges is not None:
    if c1._unique_charges.shape != c2._unique_charges.shape:
      res = False
    elif not np.all(c1._unique_charges == c2._unique_charges):
      res = False
    elif not np.all(c1.charge_labels == c2.charge_labels):
      res = False
    return res

  if c1._charges is not None and c2._charges is not None:
    if c1._charges.shape != c2._charges.shape:
      res = False
    elif not np.all(c1._charges == c2._charges):
      res = False
    return res
  if c1.charges.shape != c2.charges.shape:
    res = False
  elif not np.all(c1.charges == c2.charges):
    res = False
  return res
